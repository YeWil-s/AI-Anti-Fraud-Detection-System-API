"""
综合性能评测基准 (Benchmark Evaluation Suite)
=============================================
- 音频/文本: 直接调用模型推理，不写数据库
- 融合视频: 参考 detection_tasks.py 流程，调用模型+LLM，不写数据库

覆盖硬性指标:
  1. 多模态融合识别准确率 > 90%
  2. 正常社交行为误报率 < 5%
  3. 精准识别 ≥ 10 种典型诈骗剧本
  4. 文本/图片平均响应时间 < 10 秒
  5. 连续运行测试无崩溃

使用方法:
  python eval_benchmark.py --all
  python eval_benchmark.py --audio
  python eval_benchmark.py --text
  python eval_benchmark.py --video
  python eval_benchmark.py --fusion
  python eval_benchmark.py --stability -n 30

数据目录（dataset/ 常被 gitignore，需本地放置）:
  audio/fake|real/**/*.wav  text/fraud|normal/*.txt
  video/fake|real/*.mp4     fusion/fraud|safe/*.mp4（优先）或仅 *.txt（自动走仅文本融合）
"""

import os
import io
import sys
import json
import time
import base64
import asyncio
import tempfile
import argparse
import traceback
import warnings
from unittest.mock import MagicMock

# --- 禁用 ChromaDB / PostHog 遥测（必须在任何库 import 之前） ---
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"
os.environ["CHROMA_DISABLE_TELEMETRY"] = "true"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

_mock_ph = MagicMock()
sys.modules['posthog'] = _mock_ph
sys.modules['chromadb.telemetry.posthog'] = MagicMock()

# 抑制 protobuf SymbolDatabase 的 DeprecationWarning
warnings.filterwarnings("ignore", message=".*SymbolDatabase.*deprecated.*")

# 强制 stdout 无缓冲，确保实时输出
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# 数据集路径
# ---------------------------------------------------------------------------
DATASET_ROOT = Path("dataset")
AUDIO_FAKE_DIR = DATASET_ROOT / "audio" / "fake"
AUDIO_REAL_DIR = DATASET_ROOT / "audio" / "real"
TEXT_FRAUD_DIR = DATASET_ROOT / "text" / "fraud"
TEXT_NORMAL_DIR = DATASET_ROOT / "text" / "normal"
FUSION_FRAUD_DIR = DATASET_ROOT / "fusion" / "fraud"
FUSION_SAFE_DIR = DATASET_ROOT / "fusion" / "safe"
VIDEO_FAKE_DIR = DATASET_ROOT / "video" / "fake"
VIDEO_REAL_DIR = DATASET_ROOT / "video" / "real"

SCRIPT_COVERAGE_TARGET = 10


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def safe_div(a, b, default=0.0):
    return a / b if b > 0 else default


def print_section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def print_divider():
    print(f"  {'-' * 76}")


# ---- 视频工具 ----
_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".m4v"}


def extract_audio_from_video(video_path: str) -> Optional[np.ndarray]:
    """从视频提取 16kHz mono 音频。视频容器优先走 ffmpeg，避免 librosa 直接读 mp4 触发 PySoundFile/audioread 告警且易失败。"""
    import librosa
    import subprocess

    path = Path(video_path)
    suffix = path.suffix.lower()

    def _load_wav_file(wav_path: str) -> Optional[np.ndarray]:
        try:
            y, _ = librosa.load(wav_path, sr=16000, mono=True)
            return y if y is not None and len(y) > 0 else None
        except Exception:
            return None

    # 1) 常见视频后缀：先用 ffmpeg 解出 PCM WAV（比 librosa 直接读 mp4 更稳）
    if suffix in _VIDEO_EXTS:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            proc = subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-i", str(path),
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1",
                    "-y", tmp_path,
                ],
                capture_output=True,
                timeout=120,
            )
            if proc.returncode == 0 and os.path.isfile(tmp_path) and os.path.getsize(tmp_path) >= 64:
                y = _load_wav_file(tmp_path)
                if y is not None:
                    return y
        except FileNotFoundError:
            pass
        except Exception:
            pass
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # 2) 回退：直接 librosa（纯音频、ffmpeg 失败或无音轨时偶发仍能读到）
    try:
        y, _ = librosa.load(str(path), sr=16000, mono=True)
        return y if y is not None and len(y) > 0 else None
    except Exception:
        pass

    # 3) 最后再试一次 ffmpeg（无后缀或非典型命名）
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        proc = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", str(path), "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1", "-y", tmp_path,
            ],
            capture_output=True,
            timeout=120,
        )
        if proc.returncode == 0 and os.path.isfile(tmp_path) and os.path.getsize(tmp_path) >= 64:
            return _load_wav_file(tmp_path)
    except FileNotFoundError:
        return None
    except Exception:
        pass
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return None


def slice_audio(y: np.ndarray, sr: int = 16000, slice_sec: float = 5.0) -> List[bytes]:
    """将音频切成 4-6s 的 WAV bytes 切片"""
    import soundfile as sf
    slice_len = int(sr * slice_sec)
    total = len(y)
    slices = []
    start = 0
    while start < total:
        end = min(start + slice_len, total)
        chunk = y[start:end]
        if len(chunk) < sr * 2:
            break
        buf = io.BytesIO()
        sf.write(buf, chunk, sr, format="WAV")
        buf.seek(0)
        slices.append(buf.read())
        start = end
    return slices


def audio_array_to_wav_bytes(y: np.ndarray, sr: int = 16000) -> bytes:
    """numpy 音频数组 → WAV bytes（整段，用于 ASR 等）"""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def aggregate_video_window_confs(confs: List[float], mode: str, trim: float) -> float:
    """
    将一段视频上多次滑窗推理的 fake 概率合成单一标量，用于离线整段标签。
    mode: trimmed_mean | iqr_mean | mean | median | latest
    """
    arr = np.asarray(confs, dtype=np.float64)
    n = arr.size
    if n == 0:
        return 0.0
    mode = (mode or "trimmed_mean").strip().lower()
    if mode == "latest":
        return float(arr[-1])
    if mode == "mean":
        return float(np.mean(arr))
    if mode == "median":
        return float(np.median(arr))
    if mode == "trimmed_mean":
        if n <= 2:
            return float(np.mean(arr))
        k = max(1, int(n * float(trim)))
        if n <= 2 * k:
            return float(np.median(arr))
        s = np.sort(arr)
        return float(np.mean(s[k : n - k]))
    if mode == "iqr_mean":
        if n < 4:
            return float(np.median(arr))
        q1, q3 = np.percentile(arr, [25.0, 75.0])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        kept = arr[(arr >= lo) & (arr <= hi)]
        if kept.size == 0:
            return float(np.median(arr))
        return float(np.mean(kept))
    return float(np.mean(arr))


async def replay_video_like_detection_api(video_path: str, video_processor, user_id: int = 0):
    """
    与 app/api/detection.py 视频 WebSocket 分支一致：
    整帧 JPG(base64) → VideoProcessor.process_frame → status==ready 时
    detect_video_task 同款 payload，并 clear_buffer 再积下一窗。

    返回按时间顺序的窗列表；线上下一次推理会覆盖 Redis latest_video_conf，
    故整段视频的「当前分数」应对应列表中最后一次推理（最后一窗）。
    """
    import cv2

    batches = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return batches
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 与 tests/test_video_flow.py 一致：整帧 JPEG，不经由 _extract_face_image(raw)
        _, buf = cv2.imencode(".jpg", frame)
        b64 = base64.b64encode(buf).decode("utf-8")
        result = await video_processor.process_frame(b64, user_id)
        if result.get("status") == "ready":
            batches.append(result["celery_payload"])
            video_processor.clear_buffer(user_id)
    cap.release()
    return batches


# ===========================================================================
# 评测 1: 音频伪造检测 (参考 eval.py，直接加载 AASIST 模型)
# ===========================================================================
def _load_aasist_model():
    """eval.py 方式加载 AASIST"""
    import torch, json
    from importlib import import_module

    model_path = "./models/aasist/weights/best.pth"
    config_path = "./models/aasist/config/AASIST.conf"
    code_path = "./models/aasist"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if code_path not in sys.path:
        sys.path.insert(0, code_path)

    with open(config_path, "r") as f:
        full_config = json.load(f)
    model_config = full_config["model_config"]

    module = import_module("models.AASIST")
    model = module.Model(model_config).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model, device


def _preprocess_audio(audio_path, cut=64600):
    """eval.py 预处理：trim → center-crop / tile-repeat"""
    import librosa as _librosa
    x, sr = _librosa.load(audio_path, sr=16000)
    x, _ = _librosa.effects.trim(x)
    x_len = x.shape[0]
    if x_len >= cut:
        start = (x_len - cut) // 2
        x_pad = x[start: start + cut]
    else:
        num_repeats = int(cut / x_len) + 1
        x_pad = np.tile(x, num_repeats)[:cut]
    import torch
    return torch.from_numpy(x_pad.astype(np.float32)).unsqueeze(0)


def evaluate_audio():
    import torch
    real_files = sorted(AUDIO_REAL_DIR.rglob("*.wav"))
    fake_files = sorted(AUDIO_FAKE_DIR.rglob("*.wav"))
    test_cases = [(f, "real") for f in real_files] + [(f, "fake") for f in fake_files]

    if not test_cases:
        print("  ⚠ 未找到音频文件，跳过")
        return {}

    print_section(f"音频伪造检测评测 ({len(fake_files)} fake + {len(real_files)} real = {len(test_cases)} 条)")
    print("  使用 eval.py 方式: 直接加载 AASIST 模型, 阈值=0.5")

    model, device = _load_aasist_model()
    print(f"  设备: {device}")

    print(f"\n  {'序号':>5}  {'标签':<6} {'文件名':<30} {'fake_prob':>10} {'判定':>8}  {'正确':>4} {'耗时':>7}")
    print_divider()

    y_true, y_pred = [], []
    latencies = []

    for i, (fp, label) in enumerate(test_cases, 1):
        true_int = 1 if label == "fake" else 0

        t0 = time.perf_counter()
        try:
            x = _preprocess_audio(fp)
            if x is None:
                raise ValueError("preprocess failed")
            x = x.to(device)
            with torch.no_grad():
                _, output = model(x, Freq_aug=False)
                probs = torch.softmax(output, dim=1)[0]
                fake_prob = float(probs[0])
            is_fake = fake_prob > 0.5
        except Exception as e:
            print(f"  {i:>5}  [{label:<4}] {fp.name:<30} ERROR: {e}")
            continue
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        pred_int = 1 if is_fake else 0
        correct = true_int == pred_int
        mark = "✓" if correct else "✗"

        y_true.append(true_int)
        y_pred.append(pred_int)

        fake_str = "FAKE" if is_fake else "REAL"
        print(f"  {i:>5}  [{label:<4}] {fp.name:<30} {fake_prob:>10.4f} {fake_str:>8}  {mark:>4} {elapsed:>6.2f}s")

    if not y_true:
        return {}
    return _print_binary_summary("音频伪造检测", y_true, y_pred, latencies, "fake", "real")


# ===========================================================================
# 评测 2: 文本诈骗检测 (调用 model_service.predict_text，含 BERT + LLM)
# ===========================================================================
def evaluate_text():
    fraud_files = sorted(TEXT_FRAUD_DIR.glob("*.txt"))
    normal_files = sorted(TEXT_NORMAL_DIR.glob("*.txt"))
    test_cases = [(f, "fraud") for f in fraud_files] + [(f, "normal") for f in normal_files]

    if not test_cases:
        print("  ⚠ 未找到文本文件，跳过")
        return {}

    print_section(f"文本诈骗检测评测 ({len(fraud_files)} fraud + {len(normal_files)} normal = {len(test_cases)} 条)")
    print("  调用 model_service.predict_text() (BERT + LLM 融合)")

    return asyncio.run(_run_text_eval(test_cases))


async def _run_text_eval(test_cases):
    """在单个事件循环中完成所有文本预测，避免 httpx Event loop is closed 警告"""
    from app.services.model_service import ModelService
    ms = ModelService()

    print(f"\n  {'序号':>4} {'标签':<7} {'文件':<10} {'文本摘要':<28} {'置信度':>6} {'BERT':>6} {'LLM诈骗':>7} {'诈骗类型':<18} {'判定':>4} {'耗时':>6}")
    print_divider()

    y_true, y_pred = [], []
    latencies = []
    identified_scripts = set()

    for i, (fp, label) in enumerate(test_cases, 1):
        text = fp.read_text(encoding="utf-8").strip()
        if not text:
            continue

        preview = text[:26].replace("\n", " ") + ("…" if len(text) > 26 else "")
        true_int = 1 if label == "fraud" else 0

        t0 = time.perf_counter()
        try:
            result = await ms.predict_text(text)
        except Exception as e:
            print(f"  {i:>4} [{label:<6}] {fp.name:<10} ERROR: {e}")
            continue
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        conf = result.get("confidence", 0.0)
        bert_conf = result.get("bert_confidence", conf)
        llm_fraud = result.get("llm_is_fraud", result.get("is_fraud", False))
        fraud_type = result.get("fraud_type", "-") or "-"
        is_fraud = conf >= 0.5 or llm_fraud

        pred_int = 1 if is_fraud else 0
        correct = true_int == pred_int
        mark = "✓" if correct else "✗"

        y_true.append(true_int)
        y_pred.append(pred_int)

        if true_int == 1 and fraud_type and fraud_type not in ("-", "", "其他", "无"):
            identified_scripts.add(fraud_type)

        ft_display = fraud_type[:16] if fraud_type else "-"
        llm_str = "是" if llm_fraud else "否"
        print(f"  {i:>4} [{label:<6}] {fp.name:<10} {preview:<28} {conf:>6.3f} {bert_conf:>6.3f} {llm_str:>7} {ft_display:<18} {mark:>4} {elapsed:>5.1f}s")

    if not y_true:
        return {}

    metrics = _print_binary_summary("文本诈骗检测", y_true, y_pred, latencies, "fraud", "normal")
    metrics["identified_scripts"] = sorted(identified_scripts)

    coverage = len(identified_scripts)
    print(f"\n  诈骗剧本覆盖: {coverage} 种")
    print(f"  已识别剧本: {sorted(identified_scripts) if identified_scripts else '无'}")
    metrics["script_coverage"] = coverage
    return metrics


# ===========================================================================
# 评测 2b: 视频换脸检测 (dataset/video/fake vs real，仅视觉模态)
# ===========================================================================
def evaluate_video():
    fake_files = sorted(VIDEO_FAKE_DIR.glob("*.mp4"))
    real_files = sorted(VIDEO_REAL_DIR.glob("*.mp4"))
    test_cases = [(f, "fake") for f in fake_files] + [(f, "real") for f in real_files]

    if not test_cases:
        print("  ⚠ 未找到视频文件，跳过")
        return {}

    from app.core.config import settings

    print_section(f"视频换脸检测评测 ({len(fake_files)} fake + {len(real_files)} real = {len(test_cases)} 条)")
    print(
        f"  多窗聚合={settings.VIDEO_WINDOWS_AGGREGATE} (trim={settings.VIDEO_WINDOWS_TRIM})；"
        f"整段判定: agg 与阈值 {settings.VIDEO_DETECTION_THRESHOLD} 比较（同单点 fake 概率含义）"
    )
    print("  agg=聚合分 latest=最后一窗 peak=峰值（参考）")

    return asyncio.run(_run_video_eval(test_cases))


async def _run_video_eval(test_cases):
    from app.services.model_service import ModelService
    from app.services.video_processor import VideoProcessor
    from app.core.config import settings

    ms = ModelService()
    thr = float(settings.VIDEO_DETECTION_THRESHOLD)
    agg_mode = settings.VIDEO_WINDOWS_AGGREGATE
    agg_trim = float(settings.VIDEO_WINDOWS_TRIM)

    print(f"\n  {'序号':>5} {'标签':<6} {'文件名':<24} {'agg':>10} {'latest':>8} {'peak':>8} {'判定':>8}  {'正确':>4} {'耗时':>7}")
    print_divider()

    y_true, y_pred = [], []
    latencies = []

    for i, (fp, label) in enumerate(test_cases, 1):
        true_int = 1 if label == "fake" else 0
        t0 = time.perf_counter()

        # 每个视频独立 Processor，避免缓冲串线；链路同 WebSocket
        vp = VideoProcessor()
        batches = await replay_video_like_detection_api(str(fp), vp, user_id=0)

        pred_int = 0
        latest = 0.0
        peak = 0.0
        agg = 0.0

        if batches:
            batch_confs = []
            for batch in batches:
                try:
                    video_tensor = VideoProcessor.preprocess_batch(batch)
                    v_result = await ms.predict_video(video_tensor)
                    batch_confs.append(v_result.get("confidence", 0.0))
                except Exception as e:
                    print(f"  {i:>5}  [{label:<4}] {fp.name:<24} batch ERR: {e}")
            if batch_confs:
                latest = float(batch_confs[-1])
                peak = float(np.max(batch_confs))
                agg = aggregate_video_window_confs(batch_confs, agg_mode, agg_trim)
                pred_int = 1 if agg >= thr else 0
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        if not batches:
            print(f"  {i:>5}  [{label:<4}] {fp.name:<24} {'(无窗)':>10} {'—':>8} {'—':>8} {'SKIP':>8}    —  {elapsed:>6.2f}s")
            continue

        correct = true_int == pred_int
        mark = "✓" if correct else "✗"
        pred_str = "FAKE" if pred_int == 1 else "REAL"

        y_true.append(true_int)
        y_pred.append(pred_int)

        print(f"  {i:>5}  [{label:<4}] {fp.name:<24} {agg:>10.4f} {latest:>8.4f} {peak:>8.4f} {pred_str:>8}  {mark:>4} {elapsed:>6.2f}s")

    if not y_true:
        return {}
    return _print_binary_summary("视频换脸检测", y_true, y_pred, latencies, "fake", "real")


# ===========================================================================
# 评测 3: 多模态融合场景
# 参考 detection_tasks.py 流程:
#   1. predict_voice → audio_conf
#   2. predict_video → video_conf
#   3. ASR → transcript
#   4. predict_text → text_conf
#   5. llm_service.analyze_multimodal_risk → llm_result
#   6. fusion_engine.calculate_score → fused_score
# ===========================================================================
def evaluate_fusion():
    fraud_videos = sorted(FUSION_FRAUD_DIR.glob("*.mp4"))
    safe_videos = sorted(FUSION_SAFE_DIR.glob("*.mp4"))
    if fraud_videos or safe_videos:
        test_cases = [(f, "fraud") for f in fraud_videos] + [(f, "safe") for f in safe_videos]
        print_section(
            f"多模态融合场景评测 ({len(fraud_videos)} fraud + {len(safe_videos)} safe = {len(test_cases)} 条)"
        )
        print("  参考 detection_tasks.py 流程: 音频→视频→ASR→文本→LLM融合→评分")
        return asyncio.run(_run_fusion(test_cases, text_only=False))

    fraud_txts = sorted(FUSION_FRAUD_DIR.glob("*.txt"))
    safe_txts = sorted(FUSION_SAFE_DIR.glob("*.txt"))
    test_cases = [(f, "fraud") for f in fraud_txts] + [(f, "safe") for f in safe_txts]
    if not test_cases:
        print("  ⚠ 未找到融合测试数据（fusion/fraud|safe 下无 .mp4 且无 .txt），跳过")
        return {}

    print_section(
        f"多模态融合场景评测 [仅文本] ({len(fraud_txts)} fraud + {len(safe_txts)} safe = {len(test_cases)} 条)"
    )
    print("  无视频时仅跑文本+LLM+融合评分（audio_conf=0, video_conf=0）")

    return asyncio.run(_run_fusion(test_cases, text_only=True))


async def _run_fusion(test_cases, text_only: bool = False):
    from app.services.model_service import ModelService
    from app.services.video_processor import VideoProcessor
    from app.services.risk_fusion_engine import RiskFusionEngine
    from app.services.llm_service import LLMService
    from app.core.config import settings

    ms = ModelService()
    fe = RiskFusionEngine()
    llm = LLMService()

    # ASR (可选，仅视频模式)
    asr_service = None
    if not text_only:
        try:
            from app.services.asr_service import ASRService
            asr_service = ASRService()
            print("  ASR 引擎加载成功")
        except Exception as e:
            print(f"  ⚠ ASR 不可用: {e}，将跳过语音转文本")

    y_true, y_pred = [], []
    latencies = []
    identified_scripts = set()

    for i, (video_path, label) in enumerate(test_cases, 1):
        true_int = 1 if label == "fraud" else 0
        t0_total = time.perf_counter()

        print(f"\n{'='*70}")
        print(f"  [{i:02d}/{len(test_cases)}] {video_path.name}  (标签: {label})")
        print(f"{'='*70}")

        audio_conf = 0.0
        video_conf = 0.0
        audio_bytes = None
        transcript = ""

        if text_only:
            transcript = video_path.read_text(encoding="utf-8").strip()
            t_preview = transcript[:50].replace("\n", " ") + ("..." if len(transcript) > 50 else "")
            print(f"  [1] 音频伪造检测   (仅文本模式，跳过)")
            print(f"  [2] 视频换脸检测   (仅文本模式，跳过)")
            print(f"  [3] 文本(标注)     \"{t_preview}\"")
        else:
            # ---- 1. 音频模态: predict_voice (切片检测) ----
            audio_array = extract_audio_from_video(str(video_path))
            audio_bytes = audio_array_to_wav_bytes(audio_array) if audio_array is not None else None
            if audio_array is not None and len(audio_array) > 16000:
                slices = slice_audio(audio_array, sr=16000, slice_sec=5.0)
                if slices:
                    slice_confs = []
                    slice_fakes = []
                    for s_bytes in slices:
                        try:
                            a_result = await ms.predict_voice(s_bytes)
                            slice_confs.append(a_result.get("confidence", 0.0))
                            slice_fakes.append(a_result.get("is_fake", False))
                        except Exception:
                            pass
                    if slice_confs:
                        audio_conf = float(np.max(slice_confs))
                        fake_cnt = sum(slice_fakes)
                        print(f"  [1] 音频伪造检测   {len(slices)} 切片, max_conf={audio_conf:.4f}, fake={fake_cnt}/{len(slices)}")
                    else:
                        print(f"  [1] 音频伪造检测   切片推理全部失败")
                else:
                    print(f"  [1] 音频伪造检测   音频过短，无法切片")
            else:
                if audio_array is None:
                    print(f"  [1] 音频伪造检测   跳过(未提取到音频；请确认已安装 ffmpeg 且视频含音轨)")
                else:
                    n = len(audio_array)
                    print(f"  [1] 音频伪造检测   跳过(音频过短 {n} 样本，需 >16000 即约 1s@16kHz)")

            # ---- 2. 视频模态: predict_video（与 detection.py → detect_video_task 相同数据源）----
            vp_vid = VideoProcessor()
            batches = await replay_video_like_detection_api(str(video_path), vp_vid, user_id=0)
            if batches:
                batch_confs = []
                for batch in batches:
                    try:
                        video_tensor = VideoProcessor.preprocess_batch(batch)
                        v_result = await ms.predict_video(video_tensor)
                        batch_confs.append(v_result.get("confidence", 0.0))
                    except Exception as e:
                        print(f"  [2] 视频批次错误: {e}")
                video_conf = aggregate_video_window_confs(
                    batch_confs,
                    settings.VIDEO_WINDOWS_AGGREGATE,
                    float(settings.VIDEO_WINDOWS_TRIM),
                )
                peak_conf = float(np.max(batch_confs)) if batch_confs else 0.0
                print(
                    f"  [2] 视频换脸检测   conf(agg)={video_conf:.4f} peak={peak_conf:.4f} "
                    f"[{settings.VIDEO_WINDOWS_AGGREGATE}] ({len(batches)} 窗)"
                )
            else:
                print(f"  [2] 视频换脸检测   跳过(未积满人脸窗)")

            # ---- 3. 文本获取：优先读同名 .txt 标注，否则走 ASR ----
            txt_path = video_path.with_suffix(".txt")
            if txt_path.exists():
                transcript = txt_path.read_text(encoding="utf-8").strip()
                t_preview = transcript[:50].replace("\n", " ") + ("..." if len(transcript) > 50 else "")
                print(f"  [3] 文本(标注)     \"{t_preview}\"")
            elif asr_service and audio_bytes:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                try:
                    transcript = await asr_service.transcribe_audio_file(tmp_path)
                finally:
                    os.unlink(tmp_path)
                t_preview = transcript[:50].replace("\n", " ") + ("..." if len(transcript) > 50 else "")
                print(f"  [3] 文本(ASR)      \"{t_preview}\"")
            else:
                print(f"  [3] 文本           无标注且 ASR 不可用")

        # ---- 4. 文本模态: predict_text → text_conf ----
        text_conf = 0.0
        if transcript and len(transcript.strip()) > 2:
            try:
                t_result = await ms.predict_text(transcript)
                text_conf = t_result.get("confidence", 0.5)
                print(f"  [4] 文本诈骗模型   text_conf={text_conf:.4f}")
            except Exception as e:
                text_conf = 0.5
                print(f"  [4] 文本诈骗模型   ERROR: {e} (默认 0.5)")

        # ---- 5. LLM 多模态融合分析 ----
        llm_result = {"is_fraud": False, "risk_level": "safe", "match_script": "无", "fraud_type": "其他"}
        if transcript and len(transcript.strip()) > 5:
            try:
                llm_result = await llm.analyze_multimodal_risk(
                    user_input=transcript,
                    chat_history="暂无历史上下文。",
                    user_profile="普通人群",
                    call_type="视频通话场景",
                    audio_conf=audio_conf,
                    video_conf=f"{video_conf:.4f}",
                    text_conf=text_conf,
                )
                print(f"  [5] LLM 融合推理")
                print(f"       is_fraud  = {llm_result.get('is_fraud')}")
                print(f"       risk      = {llm_result.get('risk_level')}")
                print(f"       fraud_type= {llm_result.get('fraud_type', '-')}")
                print(f"       script    = {llm_result.get('match_script', '-')}")
            except Exception as e:
                print(f"  [5] LLM 融合推理   ERROR: {e}")
        else:
            print(f"  [5] LLM 融合推理   跳过(无有效文本)")

        # ---- 6. 融合评分 ----
        fused_score = fe.calculate_score(
            llm_classification=llm_result,
            local_text_conf=text_conf,
            audio_conf=audio_conf,
            video_conf=video_conf,
        )

        # ---- 最终判定 ----
        is_fraud_pred = llm_result.get("is_fraud", False)
        risk_level = llm_result.get("risk_level", "safe")
        pred_int = 1 if (fused_score >= 50.0 or is_fraud_pred or risk_level in ("fake", "suspicious")) else 0
        correct = pred_int == true_int
        mark = "OK" if correct else "WRONG"
        elapsed_total = time.perf_counter() - t0_total

        pred_str = "FRAUD" if pred_int == 1 else "SAFE"
        print(f"  {'─'*60}")
        print(f"  [6] 融合评分       {fused_score:.1f}")
        print(f"      音频={audio_conf:.4f}  视频={video_conf:.4f}  文本={text_conf:.4f}")
        print(f"  >>> 判定: {pred_str}  真实: {label}  {mark}  耗时: {elapsed_total:.1f}s")

        latencies.append(elapsed_total)
        y_true.append(true_int)
        y_pred.append(pred_int)

        if true_int == 1:
            ms_val = llm_result.get("match_script", "无")
            ft_val = llm_result.get("fraud_type", "其他")
            if ms_val and ms_val != "无":
                identified_scripts.add(ms_val)
            if ft_val and ft_val not in ("其他", ""):
                identified_scripts.add(ft_val)

    if not y_true:
        return {}

    metrics = _print_binary_summary("多模态融合", y_true, y_pred, latencies, "fraud", "safe")
    metrics["identified_scripts"] = sorted(identified_scripts)
    metrics["fusion_mode"] = "text_only" if text_only else "full"

    coverage = len(identified_scripts)
    print(f"\n  融合场景剧本覆盖: {coverage} 种")
    print(f"  已识别剧本: {sorted(identified_scripts) if identified_scripts else '无'}")
    metrics["script_coverage"] = coverage
    return metrics


# ===========================================================================
# 评测 4: 连续运行稳定性 (纯模型推理，不写 DB)
# ===========================================================================
def evaluate_stability(n_rounds: int):
    real_files = sorted(AUDIO_REAL_DIR.rglob("*.wav"))[:10]
    fake_files = sorted(AUDIO_FAKE_DIR.rglob("*.wav"))[:10]
    audio_files = real_files + fake_files

    text_files = sorted(TEXT_FRAUD_DIR.glob("*.txt"))[:5] + sorted(TEXT_NORMAL_DIR.glob("*.txt"))[:5]

    if not audio_files and not text_files:
        print("  ⚠ 无测试数据，跳过")
        return {}

    print_section(f"连续运行稳定性测试 ({n_rounds} 轮, 每轮 {len(audio_files)} 音频 + {len(text_files)} 文本)")

    return asyncio.run(_run_stability(n_rounds, audio_files, text_files))


async def _run_stability(n_rounds, audio_files, text_files):
    """在单个事件循环中完成所有稳定性测试"""
    from app.services.model_service import ModelService
    ms = ModelService()

    total_ops = 0
    crashes = 0
    crash_details = []

    for rd in range(1, n_rounds + 1):
        round_errors = 0
        t0 = time.perf_counter()

        for fp in audio_files:
            try:
                await ms.predict_voice(fp.read_bytes())
                total_ops += 1
            except Exception as e:
                round_errors += 1
                crash_details.append(f"R{rd} audio {fp.name}: {e}")

        for fp in text_files:
            try:
                text = fp.read_text(encoding="utf-8").strip()
                if text:
                    await ms.predict_text(text)
                    total_ops += 1
            except Exception as e:
                round_errors += 1
                crash_details.append(f"R{rd} text {fp.name}: {e}")

        elapsed = time.perf_counter() - t0
        crashes += round_errors
        status = "✓" if round_errors == 0 else f"✗ ({round_errors} errors)"
        print(f"  轮次 {rd:>3}/{n_rounds}  ops={len(audio_files) + len(text_files)}  {status}  ({elapsed:.1f}s)")

    print_divider()
    passed = crashes == 0
    print(f"  总操作: {total_ops},  崩溃次数: {crashes}")
    print(f"  结果: {'✅ PASS (零崩溃)' if passed else '❌ FAIL'}")
    if crash_details:
        for d in crash_details[:5]:
            print(f"    - {d}")

    return {"total_ops": total_ops, "crashes": crashes, "passed": passed}


# ===========================================================================
# 共用: 二分类汇总
# ===========================================================================
def _print_binary_summary(title, y_true, y_pred, latencies, pos_name, neg_name):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)

    avg_lat = float(np.mean(latencies)) if latencies else 0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0

    print_divider()
    print(f"  ═══ {title} 汇总 ═══")
    print(f"  准确率 (Accuracy):  {acc:.4f}  {'✅ ≥0.90' if acc >= 0.90 else '❌ <0.90'}")
    print(f"  精确率 (Precision): {prec:.4f}")
    print(f"  召回率 (Recall):    {rec:.4f}")
    print(f"  F1 分数:            {f1:.4f}")
    print(f"  误报率 (FPR):       {fpr:.4f}  {'✅ <5%' if fpr < 0.05 else '❌ ≥5%'}")
    print(f"  漏报率 (FNR):       {fnr:.4f}")
    print(f"  混淆矩阵: [{neg_name}] TN={tn} FP={fp} | [{pos_name}] FN={fn} TP={tp}")
    print(f"  平均耗时: {avg_lat:.3f}s  P95: {p95_lat:.3f}s  {'✅ <10s' if avg_lat < 10 else '❌ ≥10s'}")

    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "fpr": fpr, "fnr": fnr, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "avg_latency": avg_lat, "p95_latency": p95_lat,
    }




# ===========================================================================
# 总报告
# ===========================================================================
def print_final_report(all_metrics: dict):
    print_section("最 终 达 标 判 定")

    checks = []

    fusion_acc = all_metrics.get("fusion", {}).get("accuracy")
    if fusion_acc is not None:
        checks.append(("融合识别准确率 > 90%", fusion_acc, 0.90, fusion_acc >= 0.90))

    for name, key in [
        ("融合安全场景", "fusion"),
        ("文本正常消息", "text"),
        ("音频真实语音", "audio"),
        ("视频真实场景", "video"),
    ]:
        fpr = all_metrics.get(key, {}).get("fpr")
        if fpr is not None:
            checks.append((f"{name} FPR < 5%", fpr, 0.05, fpr < 0.05))

    all_scripts = set()
    for key in ("text", "fusion"):
        all_scripts.update(all_metrics.get(key, {}).get("identified_scripts", []))
    if all_scripts:
        total_cov = len(all_scripts)
        checks.append(("诈骗剧本覆盖 ≥ 10 种", float(total_cov), 10.0, total_cov >= 10))

    text_lat = all_metrics.get("text", {}).get("avg_latency")
    if text_lat is not None:
        checks.append(("文本平均响应时间 < 10s", text_lat, 10.0, text_lat < 10.0))

    stab = all_metrics.get("stability", {})
    if stab:
        checks.append(("连续运行零崩溃", float(stab.get("crashes", -1)), 0.0, stab.get("passed", False)))

    print(f"\n  {'指标':<30} {'实际值':>10} {'阈值':>10} {'判定':>8}")
    print(f"  {'-' * 62}")
    for name, val, thresh, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:<30} {val:>10.4f} {thresh:>10.4f} {status:>8}")

    passed_count = sum(1 for _, _, _, p in checks if p)
    print(f"\n  总计: {passed_count}/{len(checks)} 项达标")

    if all_scripts:
        print(f"\n  全部已识别诈骗剧本 ({len(all_scripts)} 种):")
        for s in sorted(all_scripts):
            print(f"    - {s}")

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (set, list)) or isinstance(vv, list)}
                    for k, v in all_metrics.items() if isinstance(v, dict)},
        "checks": [{"name": n, "value": v, "threshold": t, "passed": p} for n, v, t, p in checks],
    }
    with open("benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  详细报告已保存至 benchmark_report.json")


# ===========================================================================
# 主入口
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="综合性能评测基准")
    parser.add_argument("--all", action="store_true", help="运行全部测试")
    parser.add_argument("--audio", action="store_true", help="音频伪造检测")
    parser.add_argument("--text", action="store_true", help="文本诈骗检测 (BERT+LLM)")
    parser.add_argument("--fusion", action="store_true", help="多模态融合场景")
    parser.add_argument("--video", action="store_true", help="视频换脸检测 (dataset/video)")
    parser.add_argument("--stability", action="store_true", help="连续运行稳定性")
    parser.add_argument("-n", "--rounds", type=int, default=20, help="稳定性测试轮次")
    args = parser.parse_args()

    run_all = args.all or not any([args.audio, args.text, args.fusion, args.stability, args.video])

    print("=" * 80)
    print("  综合性能评测基准")
    print("  音频/文本: 直接调用模型，不写数据库")
    print("  融合: 参考 detection_tasks.py 流程，调用模型+LLM")
    print("=" * 80)

    all_metrics = {}

    if run_all or args.audio:
        all_metrics["audio"] = evaluate_audio()

    if run_all or args.text:
        all_metrics["text"] = evaluate_text()

    if run_all or args.video:
        all_metrics["video"] = evaluate_video()

    if run_all or args.fusion:
        result = evaluate_fusion()
        if result:
            all_metrics["fusion"] = result

    if run_all or args.stability:
        all_metrics["stability"] = evaluate_stability(args.rounds)

    print_final_report(all_metrics)


if __name__ == "__main__":
    main()
