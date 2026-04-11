"""
综合性能评测基准 (Benchmark Evaluation Suite)
=============================================
- 单模态 audio/text/video: 默认与线上一致走 app ModelService（可选音频直连 AASIST 离线权重）
- 多模态 fusion: 与线上一致走「规则引擎 + RAG + ONNX 文本 + LLM 多模态 + 环境融合引擎 + MDP」
  （对齐 app/tasks/detection_tasks.py 中 detect_text_task 融合段，不经 Celery，不写业务库）

覆盖硬性指标:
  1. 多模态融合识别准确率 > 90%
  2. 正常社交行为误报率 < 5%
  3. 精准识别 ≥ 10 种典型诈骗剧本
  4. 文本/图片平均响应时间 < 10 秒
  5. 连续运行测试无崩溃

使用方法:
  python eval_benchmark.py --all
  python eval_benchmark.py --audio [--audio-backend aasist|service]
  python eval_benchmark.py --text
  python eval_benchmark.py --video
  python eval_benchmark.py --fusion [--detail-json PATH]
  python eval_benchmark.py --stability -n 30

数据目录（dataset/ 常被 gitignore，需本地放置）:
  audio/fake|real/**/*.wav
  text/fraud|normal/*.txt
  video/fake|real/*.mp4
  fusion/fraud|safe/<同一 stem>：stem.mp4（视频） + stem.txt（转写） + stem.wav（与视频一致的音轨，优先用于语音模态）
  若缺少 mp4 仅有 txt，则自动按「仅文本融合」跑一条（audio/video 置 0）
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

# 融合评测用合成 call_id 区间，避免与真实通话冲突（仅内存态环境缓存）
_FUSION_CALL_ID_BASE = 880_000


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def discover_fusion_cases() -> List[Dict]:
    """
    按 stem 对齐 fusion/fraud|safe 下的 mp4、txt、wav。
    返回列表元素: stem, label, mp4|None, txt|None, wav|None
    """
    cases: List[Dict] = []
    for subdir, label in ((FUSION_FRAUD_DIR, "fraud"), (FUSION_SAFE_DIR, "safe")):
        if not subdir.is_dir():
            continue
        stems = set()
        for p in subdir.iterdir():
            if p.is_file():
                stems.add(p.stem)
        for stem in sorted(stems):
            mp4 = subdir / f"{stem}.mp4"
            txt = subdir / f"{stem}.txt"
            wav = subdir / f"{stem}.wav"
            c_mp4 = mp4 if mp4.is_file() else None
            c_txt = txt if txt.is_file() else None
            c_wav = wav if wav.is_file() else None
            if c_mp4 is None and c_txt is None:
                continue
            cases.append(
                {
                    "stem": stem,
                    "label": label,
                    "mp4": c_mp4,
                    "txt": c_txt,
                    "wav": c_wav,
                }
            )
    return cases


def load_fusion_audio_array(mp4: Optional[Path], wav: Optional[Path]) -> Optional[np.ndarray]:
    """优先 dataset 提供的 wav（与线上一致：视频轨转录），否则从 mp4 抽取。"""
    import librosa as _librosa

    if wav is not None and wav.is_file():
        try:
            y, _ = _librosa.load(str(wav), sr=16000, mono=True)
            return y if y is not None and len(y) > 0 else None
        except Exception:
            pass
    if mp4 is not None and mp4.is_file():
        return extract_audio_from_video(str(mp4))
    return None


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


async def evaluate_audio(audio_backend: str = "service"):
    """
    audio_backend:
      - service: 与线上一致，走 app.services.model_service.ModelService.predict_voice（切片）
      - aasist:  离线直接加载 models/aasist 权重（旧 eval.py 方式）

    须由主入口与其它异步评测共用同一 asyncio 事件循环（避免多次 asyncio.run 导致 httpx 关闭时报 Event loop is closed）。
    """
    real_files = sorted(AUDIO_REAL_DIR.rglob("*.wav"))
    fake_files = sorted(AUDIO_FAKE_DIR.rglob("*.wav"))
    test_cases = [(f, "real") for f in real_files] + [(f, "fake") for f in fake_files]

    if not test_cases:
        print("  ⚠ 未找到音频文件，跳过")
        return {}

    print_section(f"音频伪造检测评测 ({len(fake_files)} fake + {len(real_files)} real = {len(test_cases)} 条)")
    if audio_backend == "aasist":
        print("  模式: 直接加载 AASIST 权重, 阈值=0.5")
        return _evaluate_audio_aasist(test_cases)
    print("  模式: ModelService.predict_voice（与线上一致）, 阈值=0.5")
    return await _evaluate_audio_service(test_cases)


def _evaluate_audio_aasist(test_cases):
    import torch

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


async def _evaluate_audio_service(test_cases):
    import librosa as _librosa
    from app.services.model_service import ModelService

    ms = ModelService()
    thr = 0.5

    print(f"\n  {'序号':>5} {'标签':<6} {'文件名':<28} {'max_conf':>10} {'判定':>8} {'正确':>4} {'耗时':>7}")
    print_divider()

    y_true, y_pred = [], []
    latencies = []

    for i, (fp, label) in enumerate(test_cases, 1):
        true_int = 1 if label == "fake" else 0
        t0 = time.perf_counter()
        try:
            y, _ = _librosa.load(str(fp), sr=16000, mono=True)
        except Exception as e:
            print(f"  {i:>5} [{label:<4}] {fp.name:<28} ERROR: {e}")
            continue
        slices = slice_audio(y, sr=16000, slice_sec=5.0) if y is not None and len(y) > 16000 else []
        fake_prob = 0.0
        is_fake = False
        if slices:
            max_conf = 0.0
            for s_bytes in slices:
                r = await ms.predict_voice(s_bytes)
                c = float(r.get("confidence", 0.0))
                max_conf = max(max_conf, c)
            fake_prob = max_conf
            is_fake = fake_prob > thr
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        pred_int = 1 if is_fake else 0
        correct = true_int == pred_int
        mark = "✓" if correct else "✗"
        y_true.append(true_int)
        y_pred.append(pred_int)
        fake_str = "FAKE" if is_fake else "REAL"
        print(f"  {i:>5} [{label:<4}] {fp.name:<28} {fake_prob:>10.4f} {fake_str:>8} {mark:>4} {elapsed:>6.2f}s")

    if not y_true:
        return {}
    return _print_binary_summary("音频伪造检测", y_true, y_pred, latencies, "fake", "real")


# ===========================================================================
# 评测 2: 文本诈骗检测 (调用 model_service.predict_text，含 BERT + LLM)
# ===========================================================================
async def evaluate_text():
    fraud_files = sorted(TEXT_FRAUD_DIR.glob("*.txt"))
    normal_files = sorted(TEXT_NORMAL_DIR.glob("*.txt"))
    test_cases = [(f, "fraud") for f in fraud_files] + [(f, "normal") for f in normal_files]

    if not test_cases:
        print("  ⚠ 未找到文本文件，跳过")
        return {}

    print_section(f"文本诈骗检测评测 ({len(fraud_files)} fraud + {len(normal_files)} normal = {len(test_cases)} 条)")
    print("  调用 model_service.predict_text() (BERT + LLM 融合)")

    return await _run_text_eval(test_cases)


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
async def evaluate_video():
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

    return await _run_video_eval(test_cases)


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
async def evaluate_fusion(detail_json_path: str = "benchmark_detailed_report.json"):
    cases = discover_fusion_cases()
    if not cases:
        print("  ⚠ 未找到融合测试数据（fusion/fraud|safe 下无可用 txt/mp4），跳过")
        return {}

    n_fraud = sum(1 for c in cases if c["label"] == "fraud")
    n_safe = len(cases) - n_fraud
    print_section(f"多模态融合场景评测 ({n_fraud} fraud + {n_safe} safe = {len(cases)} 条，按 stem 对齐 mp4/txt/wav)")
    print(
        "  对齐线上 detect_text_task 融合段: 规则引擎 + RAG 检索 + ONNX 文本 + "
        "LLM 多模态 + environment_fusion_engine + MDP（不经 Celery，不写业务库）"
    )
    print(f"  详细轨迹 JSON: {detail_json_path}")
    return await _run_fusion_pipeline(cases, detail_json_path)


def _truncate(s: str, n: int = 4000) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n] + "…(truncated)"


async def _run_fusion_pipeline(cases: List[Dict], detail_json_path: str):
    """
    与 detection_tasks.detect_text_task 融合核心一致（不含 ONNX 快路径提前 return，便于评测必跑 LLM）。
    """
    from app.core.config import settings
    from app.services.model_service import ModelService
    from app.services.video_processor import VideoProcessor
    from app.services.llm_service import llm_service
    from app.services.risk_fusion_engine import environment_fusion_engine
    from app.services.security_service import security_service
    from app.services.vector_db_service import vector_db
    from app.services.mdp_defense.dynamic_defense_agent import DynamicDefenseAgent
    from app.services.mdp_defense.mdp_types import MDPObservation

    ms = ModelService()
    mdp_agent = DynamicDefenseAgent()

    try:
        from app.services.asr_service import ASRService

        asr_service = ASRService()
        print("  ASR 引擎加载成功（无 txt 时回退转写）")
    except Exception as e:
        asr_service = None
        print(f"  ⚠ ASR 不可用: {e}")

    y_true, y_pred = [], []
    latencies = []
    identified_scripts = set()
    detail_cases: List[Dict] = []

    for i, case in enumerate(cases, 1):
        stem = case["stem"]
        label = case["label"]
        mp4 = case["mp4"]
        txt_path = case["txt"]
        wav_path = case["wav"]
        true_int = 1 if label == "fraud" else 0
        call_id = _FUSION_CALL_ID_BASE + i
        call_id_str = str(call_id)

        t0_total = time.perf_counter()

        print(f"\n{'='*70}")
        print(f"  [{i:02d}/{len(cases)}] stem={stem}  (标签: {label})")
        print(f"{'='*70}")
        if mp4:
            print(f"  文件: mp4={mp4.name} txt={txt_path.name if txt_path else '-'} wav={wav_path.name if wav_path else '-'}")
        else:
            print(f"  文件: (无 mp4) txt={txt_path.name if txt_path else '-'}")

        environment_fusion_engine.clear_environment_cache(call_id_str)
        if mp4 is not None:
            environment_fusion_engine.set_call_environment(call_id_str, "video_call", False)
        else:
            environment_fusion_engine.set_call_environment(call_id_str, "unknown", False)

        audio_conf = 0.0
        video_conf = 0.0
        audio_high_risk = False
        video_high_risk = False
        transcript = ""
        text_only = mp4 is None

        # ---- 文本：优先 dataset 的 txt ----
        if txt_path is not None:
            transcript = txt_path.read_text(encoding="utf-8").strip()
            t_preview = transcript[:50].replace("\n", " ") + ("..." if len(transcript) > 50 else "")
            print(f"  [0] 文本(标注文件) \"{t_preview}\"")
        elif not text_only:
            transcript = ""

        # ---- 1. 音频：优先 wav，否则从 mp4 抽轨 ----
        audio_array = load_fusion_audio_array(mp4, wav_path)
        audio_bytes = audio_array_to_wav_bytes(audio_array) if audio_array is not None else None
        if audio_array is not None and len(audio_array) > 16000:
            slices = slice_audio(audio_array, sr=16000, slice_sec=5.0)
            if slices:
                slice_confs = []
                for s_bytes in slices:
                    try:
                        a_result = await ms.predict_voice(s_bytes)
                        slice_confs.append(float(a_result.get("confidence", 0.0)))
                        if a_result.get("is_fake") and float(a_result.get("confidence", 0.0)) >= 0.95:
                            audio_high_risk = True
                    except Exception:
                        pass
                if slice_confs:
                    audio_conf = float(np.max(slice_confs))
                    print(f"  [1] 音频伪造检测   {len(slices)} 切片, max_conf={audio_conf:.4f} high_risk={audio_high_risk}")
            else:
                print(f"  [1] 音频伪造检测   音频过短，无法切片")
        else:
            print(f"  [1] 音频伪造检测   跳过(无有效 wav/mp4 音轨)")

        # ---- 2. 视频 ----
        if not text_only and mp4 is not None:
            vp_vid = VideoProcessor()
            batches = await replay_video_like_detection_api(str(mp4), vp_vid, user_id=0)
            if batches:
                batch_confs = []
                for batch in batches:
                    try:
                        video_tensor = VideoProcessor.preprocess_batch(batch)
                        v_result = await ms.predict_video(video_tensor)
                        raw_conf = float(v_result.get("confidence", 0.0))
                        batch_confs.append(raw_conf)
                    except Exception as e:
                        print(f"  [2] 视频批次错误: {e}")
                video_conf = aggregate_video_window_confs(
                    batch_confs,
                    settings.VIDEO_WINDOWS_AGGREGATE,
                    float(settings.VIDEO_WINDOWS_TRIM),
                )
                peak_conf = float(np.max(batch_confs)) if batch_confs else 0.0
                video_high_risk = peak_conf >= 0.95
                print(
                    f"  [2] 视频换脸检测   conf(agg)={video_conf:.4f} peak={peak_conf:.4f} "
                    f"high_risk={video_high_risk} ({len(batches)} 窗)"
                )
            else:
                print(f"  [2] 视频换脸检测   跳过(未积满人脸窗)")
        else:
            print(f"  [2] 视频换脸检测   (仅文本模式，跳过)")

        # ---- 3. 无 txt 时 ASR ----
        if (not transcript or len(transcript.strip()) < 2) and asr_service and audio_bytes:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                transcript = await asr_service.transcribe_audio_file(tmp_path)
            finally:
                os.unlink(tmp_path)
            t_preview = (transcript or "")[:50].replace("\n", " ") + ("..." if len(transcript or "") > 50 else "")
            print(f"  [3] 文本(ASR)      \"{t_preview}\"")

        # ---- 规则引擎（与线上一致）----
        rule_hit = await security_service.match_risk_rules(transcript) if transcript else None
        if rule_hit:
            print(f"  [4] 规则引擎        命中: {rule_hit.get('keyword', '')} level={rule_hit.get('risk_level', '-')}")
        else:
            print(f"  [4] 规则引擎        未命中")

        # ---- RAG（Agent 侧「检索」步骤，与 LLM 内调用一致）----
        rag_context = ""
        if transcript and len(transcript.strip()) > 2:
            try:
                rag_context = await asyncio.to_thread(vector_db.get_context_for_llm, transcript, 2)
            except Exception as e:
                rag_context = f"(rag_error: {e})"

        # ---- ONNX 文本 ----
        text_conf = 0.5
        text_result: Dict = {}
        if transcript and len(transcript.strip()) > 2:
            try:
                text_result = await ms.predict_text(transcript)
                text_conf = float(text_result.get("confidence", 0.5))
                print(f"  [5] ONNX/BERT 文本   text_conf={text_conf:.4f}")
            except Exception as e:
                print(f"  [5] ONNX/BERT 文本   ERROR: {e} (默认 0.5)")
        if rule_hit and rule_hit.get("risk_level", 0) >= 4:
            text_conf = max(text_conf, 0.95)
        if audio_high_risk and text_conf < 0.5:
            text_conf = max(text_conf, 0.5)
        if video_high_risk and text_conf < 0.5:
            text_conf = max(text_conf, 0.5)

        env_info = environment_fusion_engine.get_environment_info(call_id_str)
        env_type = env_info.get("environment_type", "unknown")
        call_type_desc = {
            "text_chat": "QQ/微信文字聊天场景",
            "voice_chat": "QQ/微信语音聊天场景",
            "phone_call": "电话通话场景",
            "video_call": "视频通话场景",
            "unknown": "视频通话场景" if mp4 else "纯语音/未知场景",
        }.get(env_type, "视频通话场景" if mp4 else "未知场景")

        # ---- LLM 多模态（与 llm_service.analyze_multimodal_risk 一致）----
        llm_result = {
            "is_fraud": False,
            "risk_level": "safe",
            "match_script": "无",
            "fraud_type": "其他",
            "analysis": "",
            "advice": "",
        }
        if transcript and len(transcript.strip()) > 5:
            try:
                raw_video_conf = f"{video_conf:.4f}" if mp4 else "N/A"
                llm_result = await llm_service.analyze_multimodal_risk(
                    user_input=transcript,
                    chat_history="暂无历史上下文。",
                    user_profile="普通人群",
                    call_type=call_type_desc,
                    audio_conf=audio_conf,
                    video_conf=raw_video_conf,
                    text_conf=text_conf,
                )
                print(f"  [6] LLM 多模态")
                print(f"       is_fraud   = {llm_result.get('is_fraud')}")
                print(f"       risk_level = {llm_result.get('risk_level')}")
                print(f"       script     = {llm_result.get('match_script', '-')}")
                adv = llm_result.get("advice", "") or ""
                print(f"       advice     = {_truncate(adv, 120)}")
            except Exception as e:
                print(f"  [6] LLM 多模态       ERROR: {e}")
        else:
            print(f"  [6] LLM 多模态       跳过(无有效文本)")

        # ---- 环境融合引擎评分（与线上 calculate_score）----
        fused_score = environment_fusion_engine.calculate_score(
            llm_classification=llm_result,
            local_text_conf=text_conf,
            audio_conf=audio_conf,
            video_conf=video_conf,
            call_id=call_id_str,
        )

        msg_count = 1
        mdp_observation = MDPObservation(
            user_id=0,
            call_id=call_id,
            risk_score=fused_score,
            message_count=msg_count,
            environment_type=env_type,
            audio_risk_flag=audio_high_risk,
            video_risk_flag=video_high_risk,
            rule_hit_level=int(rule_hit.get("risk_level", 0)) if rule_hit else 0,
            text_conf=text_conf,
            audio_conf=audio_conf,
            video_conf=video_conf,
            llm_result=llm_result,
            rule_hit=rule_hit or {},
        )
        decision = mdp_agent.select_action(None, mdp_observation)
        action_level = decision.action.value

        is_fraud_pred = bool(llm_result.get("is_fraud", False))
        risk_level = llm_result.get("risk_level", "safe")
        pred_int = 1 if (
            fused_score >= 50.0
            or is_fraud_pred
            or risk_level in ("fake", "suspicious")
            or action_level >= 1
        ) else 0

        elapsed_total = time.perf_counter() - t0_total
        correct = pred_int == true_int
        mark = "OK" if correct else "WRONG"
        pred_str = "FRAUD" if pred_int == 1 else "SAFE"

        print(f"  {'─'*60}")
        print(f"  [7] 融合分         {fused_score:.1f}  (environment_fusion_engine)")
        print(f"      模态置信度     音频={audio_conf:.4f}  视频={video_conf:.4f}  文本={text_conf:.4f}")
        print(f"  [8] MDP 决策       action_level={action_level} ({decision.reason_codes})")
        print(f"  >>> 二分类判定   {pred_str}  真实: {label}  {mark}  耗时: {elapsed_total:.1f}s")

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

        detail_cases.append(
            {
                "stem": stem,
                "label": label,
                "paths": {
                    "mp4": str(mp4) if mp4 else None,
                    "txt": str(txt_path) if txt_path else None,
                    "wav": str(wav_path) if wav_path else None,
                },
                "call_id_synthetic": call_id,
                "latency_sec": round(elapsed_total, 3),
                "modalities": {
                    "audio_conf": audio_conf,
                    "video_conf": video_conf,
                    "text_conf": text_conf,
                },
                "agent_trace": {
                    "rule_engine": rule_hit,
                    "rag_retrieval": _truncate(rag_context, 4000),
                    "onnx_text": {k: text_result.get(k) for k in ("confidence", "bert_confidence", "label", "model_type") if k in text_result},
                    "llm_multimodal": {
                        "is_fraud": llm_result.get("is_fraud"),
                        "risk_level": llm_result.get("risk_level"),
                        "match_script": llm_result.get("match_script"),
                        "fraud_type": llm_result.get("fraud_type"),
                        "intent": llm_result.get("intent"),
                        "analysis": llm_result.get("analysis"),
                        "advice": llm_result.get("advice"),
                        "tts_enabled": llm_result.get("tts_enabled"),
                        "tts_text": llm_result.get("tts_text"),
                    },
                    "fusion": {
                        "fused_score_0_100": fused_score,
                        "engine": "environment_fusion_engine",
                        "environment_type": env_type,
                    },
                    "mdp": {
                        "action_level": action_level,
                        "policy_version": decision.policy_version,
                        "reason_codes": decision.reason_codes,
                        "fallback_used": decision.fallback_used,
                        "state_key": decision.state.policy_key(),
                    },
                },
                "binary_eval": {
                    "y_true": true_int,
                    "y_pred": pred_int,
                    "correct": correct,
                },
            }
        )

    if not y_true:
        return {}

    try:
        with open(detail_json_path, "w", encoding="utf-8") as dj:
            json.dump(
                {
                    "generated_at": datetime.now().isoformat(),
                    "pipeline": "fusion_benchmark_alignment_detect_text_task",
                    "cases": detail_cases,
                },
                dj,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        print(f"\n  已写入详细轨迹: {detail_json_path}")
    except Exception as e:
        print(f"\n  ⚠ 详细轨迹写入失败: {e}")

    metrics = _print_binary_summary(
        "多模态融合", y_true, y_pred, latencies, "fraud", "safe", include_latency=False
    )
    metrics["identified_scripts"] = sorted(identified_scripts)
    metrics["fusion_mode"] = "stem_aligned_mp4_txt_wav"
    metrics["detail_report_path"] = detail_json_path

    coverage = len(identified_scripts)
    print(f"\n  融合场景剧本覆盖: {coverage} 种")
    print(f"  已识别剧本: {sorted(identified_scripts) if identified_scripts else '无'}")
    metrics["script_coverage"] = coverage
    return metrics


# ===========================================================================
# 评测 4: 连续运行稳定性 (纯模型推理，不写 DB)
# ===========================================================================
async def evaluate_stability(n_rounds: int):
    real_files = sorted(AUDIO_REAL_DIR.rglob("*.wav"))[:10]
    fake_files = sorted(AUDIO_FAKE_DIR.rglob("*.wav"))[:10]
    audio_files = real_files + fake_files

    text_files = sorted(TEXT_FRAUD_DIR.glob("*.txt"))[:5] + sorted(TEXT_NORMAL_DIR.glob("*.txt"))[:5]

    if not audio_files and not text_files:
        print("  ⚠ 无测试数据，跳过")
        return {}

    print_section(f"连续运行稳定性测试 ({n_rounds} 轮, 每轮 {len(audio_files)} 音频 + {len(text_files)} 文本)")

    return await _run_stability(n_rounds, audio_files, text_files)


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
def _print_binary_summary(
    title,
    y_true,
    y_pred,
    latencies,
    pos_name,
    neg_name,
    *,
    include_latency: bool = True,
):
    """
    include_latency: 单模态推理延迟（<10s）指标；多模态融合整段含多模型与 LLM，不适用该耗时口径。
    """
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
    if include_latency:
        print(f"  平均耗时: {avg_lat:.3f}s  P95: {p95_lat:.3f}s  {'✅ <10s' if avg_lat < 10 else '❌ ≥10s'}")

    out = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    if include_latency:
        out["avg_latency"] = avg_lat
        out["p95_latency"] = p95_lat
    return out




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
    print(f"\n  汇总报告已保存至 benchmark_report.json")
    fusion_detail = all_metrics.get("fusion", {}).get("detail_report_path")
    if fusion_detail:
        print(f"  融合逐条轨迹: {fusion_detail}")


# ===========================================================================
# 主入口
# ===========================================================================
async def run_all_benchmarks(args) -> Dict:
    """
    所有含 ModelService / LLM / httpx 的评测在同一次 asyncio.run 内顺序执行，
    避免多次关闭事件循环导致 AsyncClient.aclose 报 Event loop is closed。
    """
    run_all = args.all or not any(
        [args.audio, args.text, args.fusion, args.stability, args.video]
    )

    print("=" * 80)
    print("  综合性能评测基准")
    print("  单模态: 默认走 ModelService（音频可用 --audio-backend aasist）")
    print("  融合: 对齐 detect_text_task 融合段 + 输出详细 JSON")
    print("=" * 80)

    all_metrics: Dict = {}

    if run_all or args.audio:
        all_metrics["audio"] = await evaluate_audio(audio_backend=args.audio_backend)

    if run_all or args.text:
        all_metrics["text"] = await evaluate_text()

    if run_all or args.video:
        all_metrics["video"] = await evaluate_video()

    if run_all or args.fusion:
        result = await evaluate_fusion(detail_json_path=args.detail_json)
        if result:
            all_metrics["fusion"] = result

    if run_all or args.stability:
        all_metrics["stability"] = await evaluate_stability(args.rounds)

    # 在事件循环仍运行时释放 SQLAlchemy 异步引擎与连接池，避免进程退出时 aiomysql Connection.__del__
    # 在已关闭的 loop 上 close → Exception ignored / RuntimeError: Event loop is closed
    try:
        from app.db.database import engine

        await engine.dispose()
    except Exception:
        pass

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="综合性能评测基准")
    parser.add_argument("--all", action="store_true", help="运行全部测试")
    parser.add_argument("--audio", action="store_true", help="音频伪造检测")
    parser.add_argument("--text", action="store_true", help="文本诈骗检测 (BERT+LLM)")
    parser.add_argument("--fusion", action="store_true", help="多模态融合场景")
    parser.add_argument("--video", action="store_true", help="视频换脸检测 (dataset/video)")
    parser.add_argument("--stability", action="store_true", help="连续运行稳定性")
    parser.add_argument("-n", "--rounds", type=int, default=20, help="稳定性测试轮次")
    parser.add_argument(
        "--audio-backend",
        choices=["service", "aasist"],
        default="service",
        help="音频评测：service=与线上一致 ModelService；aasist=直连本地 AASIST 权重",
    )
    parser.add_argument(
        "--detail-json",
        default="benchmark_detailed_report.json",
        help="融合评测每条样本的完整轨迹（规则/RAG/LLM 建议/MDP）输出路径",
    )
    args = parser.parse_args()

    all_metrics = asyncio.run(run_all_benchmarks(args))
    print_final_report(all_metrics)


if __name__ == "__main__":
    main()
