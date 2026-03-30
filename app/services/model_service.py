"""
AI模型服务层
"""
import os
import onnxruntime as ort
import joblib
import numpy as np
import librosa  # [新增]
import io       # [新增]
import random
import sys
from typing import Dict, Optional
from pathlib import Path
from transformers import BertTokenizer
from app.core.config import settings
from app.core.logger import get_logger

# 初始化模块级 logger
logger = get_logger(__name__)

class MockOnnxSession:
    """Mock Session (保持不变)"""
    def __init__(self, model_type="unknown"):
        self.model_type = model_type
    def get_inputs(self):
        class Node: pass
        n = Node(); n.name = "input"; return [n]
    def run(self, output_names, input_feed):
        # 模拟返回
        return [np.array([[0.1, 0.9]], dtype=np.float32)]

class ModelService:
    """AI模型加载与推理服务"""
    
    # 模型版本标识
    MODEL_VERSIONS = {
        "audio": "aasist-finetuned-v1.0",
        "audio_fallback": "dual-stream-v1.0",
        "video": "resnet-lstm-v2.0",
        "text": "bert-finetuned-v1.0",
        "text_fallback": "bert-base-v1.0",
        "text_onnx": "bert-onnx-v1.0"
    }
    
    def __init__(self):
        # --- 音频模型组件 ---
        self.voice_session = None  # ONNX深度流模型
        self.gnb_model = None      # 统计流GNB
        self.nmf_model = None      # 统计流NMF
        self.svm_model = None      # 统计流SVM
        self.aasist_model = None   # AASIST微调模型（主模型）
        self.aasist_config = None
        self.aasist_eer = None     # 模型EER指标
        self.aasist_model_source = None  # 模型来源: speechfake_finetuned 或 aasist_finetuned
        
        # --- 文本模型组件 ---
        self.text_session = None   # ONNX文本模型（备用）
        self.text_model_torch = None  # PyTorch BERT模型（主模型）
        self.tokenizer = None
        
        # --- 视频模型组件 ---
        self.video_session = None
        
        # --- 模型加载状态 ---
        self.audio_model_type = "unknown"
        self.text_model_type = "unknown"
        
        self._load_models()
    
    def _load_models(self):
        """加载模型文件 - 优先加载微调模型，降级到基础模型"""
        try:
            # ============================================================
            # 1. 加载音频检测模型 (优先级: AASIST微调 > 双流融合 > Mock)
            # ============================================================
            self._load_audio_models()
            
            # ============================================================
            # 2. 加载视频检测模型
            # ============================================================
            self._load_video_models()
            
            # ============================================================
            # 3. 加载文本检测模型 (优先级: BERT微调 > BERT基础 > ONNX > Mock)
            # ============================================================
            self._load_text_models()
            
            logger.info(f"模型加载完成: audio={self.audio_model_type}, text={self.text_model_type}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
    
    def _load_audio_models(self):
        """加载音频检测模型"""
        # 1. 优先加载微调后的 AASIST 模型
        aasist_path = Path(settings.AASIST_MODEL_PATH)
        logger.info(f"检查 AASIST 模型: {aasist_path}, 存在: {aasist_path.exists()}")
        
        if aasist_path.exists():
            try:
                import torch
                from collections import OrderedDict
                
                # 添加 AASIST 源码路径
                aasist_code_path = str(Path(settings.AASIST_CODE_PATH))
                if aasist_code_path not in sys.path:
                    sys.path.insert(0, aasist_code_path)
                
                from models.AASIST import Model as AASISTModel
                
                # 加载checkpoint
                checkpoint = torch.load(aasist_path, map_location='cpu', weights_only=False)
                
                # 判断checkpoint格式：完整checkpoint或纯state_dict
                if isinstance(checkpoint, OrderedDict) or isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint:
                    # 纯state_dict格式（如SpeechFake训练的新模型）
                    logger.info("检测到纯state_dict格式模型，使用配置文件中参数初始化")
                    state_dict = checkpoint if isinstance(checkpoint, OrderedDict) else checkpoint
                    self.aasist_config = settings.AASIST_CONFIG if hasattr(settings, 'AASIST_CONFIG') else self._get_default_aasist_config()
                    self.aasist_model = AASISTModel(self.aasist_config)
                    self.aasist_model.load_state_dict(state_dict)
                    self.aasist_eer = None
                    best_acc = 'N/A'
                    model_source = "speechfake_finetuned"
                else:
                    # 完整checkpoint格式（包含model_state_dict, config, best_eer等）
                    logger.info("检测到完整checkpoint格式模型")
                    if hasattr(settings, 'AASIST_CONFIG') and settings.AASIST_CONFIG:
                        self.aasist_config = settings.AASIST_CONFIG
                        logger.info(f"使用配置文件中的AASIST参数: first_conv={self.aasist_config.get('first_conv')}")
                    else:
                        self.aasist_config = checkpoint.get('config', self._get_default_aasist_config())
                        logger.info(f"使用checkpoint中的AASIST参数")
                    
                    self.aasist_model = AASISTModel(self.aasist_config)
                    self.aasist_model.load_state_dict(checkpoint['model_state_dict'])
                    self.aasist_eer = checkpoint.get('best_eer', None)
                    best_acc = checkpoint.get('best_acc', 'N/A')
                    model_source = "aasist_finetuned"
                
                self.aasist_model.eval()
                self.audio_model_type = model_source
                self.aasist_model_source = model_source  # 记录模型来源
                # AASIST模型统一语义: Index 0 = 伪造, Index 1 = 真人
                logger.info(f"✅ AASIST模型加载成功 [{model_source}] (Acc: {best_acc}, EER: {self.aasist_eer})")
                return
                
            except Exception as e:
                logger.warning(f"⚠️ AASIST模型加载失败: {e}，尝试加载备用模型")
                self.aasist_model = None
        
        # 2. 降级到双流融合模型
        self._load_fallback_audio_models()
    
    def _get_default_aasist_config(self) -> dict:
        """获取默认AASIST配置（与AASIST.conf一致）"""
        return {
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],  # 第一个元素是初始通道数
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.7, 0.5, 0.5],  # 4个元素，与SpeechFake训练配置一致
            "temperatures": [2.0, 2.0, 100.0, 100.0],  # 4个元素，与SpeechFake训练配置一致
            "first_conv": 128,
            "nb_samp": 64600,
        }
    
    def _load_fallback_audio_models(self):
        """加载备用音频模型（双流融合）"""
        # 加载深度流 ONNX 模型
        if Path(settings.VOICE_MODEL_PATH).exists():
            self.voice_session = ort.InferenceSession(
                settings.VOICE_MODEL_PATH,
                providers=['CPUExecutionProvider']
            )
            logger.info(f"✓ 音频深度流模型加载: {settings.VOICE_MODEL_PATH}")
        else:
            logger.warning(f"⚠ 音频深度流模型不存在: {settings.VOICE_MODEL_PATH}")
            self.voice_session = MockOnnxSession("voice")
        
        # 加载统计流模型
        ml_path = Path("./models/ml")
        try:
            if (ml_path / "gnb.pkl").exists():
                self.gnb_model = joblib.load(ml_path / "gnb.pkl")
                self.nmf_model = joblib.load(ml_path / "nmf.pkl")
                self.svm_model = joblib.load(ml_path / "svm.pkl")
                logger.info("✓ 音频统计流模型加载成功")
            else:
                logger.warning("⚠ 音频统计流模型不存在")
        except Exception as e:
            logger.error(f"统计流模型加载失败: {e}")
        
        # 设置模型类型
        if self.voice_session and not isinstance(self.voice_session, MockOnnxSession):
            if self.gnb_model:
                self.audio_model_type = "dual_stream_fusion"
            else:
                self.audio_model_type = "deep_learning_only"
        elif self.gnb_model:
            self.audio_model_type = "statistical_only"
        else:
            self.audio_model_type = "mock"
            logger.warning("⚠ 所有音频模型加载失败，使用Mock模型")
    
    def _load_video_models(self):
        """加载视频检测模型"""
        if Path(settings.VIDEO_MODEL_PATH).exists():
            self.video_session = ort.InferenceSession(
                settings.VIDEO_MODEL_PATH, 
                providers=['CPUExecutionProvider']
            )
            logger.info(f"✅ 视频模型加载成功: {settings.VIDEO_MODEL_PATH}")
        else:
            logger.warning(f"⚠️ 视频模型不存在: {settings.VIDEO_MODEL_PATH}")
            self.video_session = MockOnnxSession("video")
    
    def _load_text_models(self):
        """加载文本检测模型"""
        # 1. 优先加载微调后的BERT模型
        finetuned_path = Path(settings.BERT_FINETUNED_PATH)
        logger.info(f"检查微调BERT模型: {finetuned_path}, 存在: {finetuned_path.exists()}")
        
        if finetuned_path.exists():
            try:
                import torch
                from transformers import BertTokenizer, BertForSequenceClassification
                
                logger.info(f"加载微调后的BERT模型: {finetuned_path}")
                self.tokenizer = BertTokenizer.from_pretrained(str(finetuned_path))
                self.text_model_torch = BertForSequenceClassification.from_pretrained(str(finetuned_path))
                self.text_model_torch.eval()
                
                self.text_model_type = "bert_finetuned"
                logger.info(f"✅ 微调BERT模型加载成功")
                return
                
            except Exception as e:
                logger.warning(f"微调BERT加载失败: {e}，尝试加载基础模型")
                self.text_model_torch = None
        
        # 2. 尝试加载基础BERT backbone（需要添加分类头）
        base_path = Path(settings.BERT_BASE_PATH)
        if base_path.exists():
            try:
                import torch
                from transformers import BertTokenizer, BertForSequenceClassification
                
                logger.info(f"加载基础BERT模型: {base_path}")
                self.tokenizer = BertTokenizer.from_pretrained(str(base_path))
                
                # 加载基础模型，添加分类头（num_labels=2）
                self.text_model_torch = BertForSequenceClassification.from_pretrained(
                    str(base_path),
                    num_labels=2,
                    ignore_mismatched_sizes=True
                )
                self.text_model_torch.eval()
                
                self.text_model_type = "bert_base"
                logger.warning(f"⚠️ 使用基础BERT模型（未微调，效果可能不佳）")
                return
                
            except Exception as e:
                logger.warning(f"基础BERT加载失败: {e}，尝试ONNX模型")
                self.text_model_torch = None
        
        # 3. 降级到ONNX模型
        if Path(settings.TEXT_MODEL_PATH).exists() and Path(settings.TEXT_VOCAB_PATH).exists():
            try:
                abs_path = os.path.abspath(settings.TEXT_MODEL_PATH)
                logger.info(f"加载ONNX文本模型: {abs_path}")
                
                size = os.path.getsize(settings.TEXT_MODEL_PATH) / (1024 * 1024)
                logger.info(f"ONNX模型大小: {size:.2f} MB")
                
                self.tokenizer = BertTokenizer.from_pretrained(settings.TEXT_VOCAB_PATH)
                self.text_session = ort.InferenceSession(
                    settings.TEXT_MODEL_PATH, 
                    providers=['CPUExecutionProvider']
                )
                
                input_names = [i.name for i in self.text_session.get_inputs()]
                logger.info(f"ONNX输入节点: {input_names}")
                
                self.text_model_type = "onnx"
                logger.info(f"✅ ONNX文本模型加载成功")
                return
                
            except Exception as e:
                logger.error(f"ONNX文本模型加载失败: {e}")
                self.text_session = MockOnnxSession("text")
        
        # 4. 所有模型都失败
        self.text_model_type = "mock"
        logger.warning("⚠️ 所有文本模型加载失败，使用Mock模型")

    def _calculate_risk_level(self, probability: float, threshold: float) -> str:
        if probability >= 0.9: return "critical"
        elif probability >= threshold: return "high"
        elif probability >= (threshold / 2): return "medium"
        else: return "low"

    def _detect_audio_format(self, audio_bytes: bytes) -> str:
        """检测音频文件的实际格式"""
        header = audio_bytes[:16]
        
        # RIFF/WAV 格式
        if header.startswith(b'RIFF'):
            return 'wav'
        # MP3 格式 (ID3 标签或帧同步)
        elif header.startswith(b'ID3') or header[0:2] == b'\xff\xfb' or header[0:2] == b'\xff\xf3':
            return 'mp3'
        # OGG 格式
        elif header.startswith(b'OggS'):
            return 'ogg'
        # FLAC 格式
        elif header.startswith(b'fLaC'):
            return 'flac'
        # MP4/M4A 格式 (ftyp 在偏移4处)
        elif b'ftyp' in header[:8]:
            # 检查具体类型
            if b'mp4' in header or b'M4A' in header or b'mp42' in header:
                return 'm4a'
            return 'mp4'
        # AMR 格式
        elif header.startswith(b'#!AMR'):
            return 'amr'
        # PCM/raw (无头部，假设是常见格式)
        else:
            return 'unknown'

    def _load_audio_with_fallback(self, audio_bytes: bytes) -> tuple:
        """
        尝试多种方式加载音频，提高格式兼容性
        返回: (audio_array, sample_rate)
        """
        detected_format = self._detect_audio_format(audio_bytes)
        logger.debug(f"检测到音频格式: {detected_format}")
        
        # 方法1: 尝试直接使用 soundfile (支持 WAV, FLAC, OGG, MP3 等)
        try:
            import soundfile as sf
            y, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
            
            # 转换为单声道
            if len(y.shape) > 1:
                y = y.mean(axis=1)
            
            # 重采样到 16kHz
            if sr != 16000:
                from scipy import signal
                y = signal.resample(y, int(len(y) * 16000 / sr))
                sr = 16000
            
            return y, sr
        except Exception as e:
            logger.debug(f"soundfile 加载失败: {e}")
            pass
        
        # 方法1b: 尝试使用 librosa (soundfile后端) - 支持多种格式
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
            return y, sr
        except Exception as e:
            logger.debug(f"librosa 加载失败: {e}")
            pass
        
        # 方法2: 使用 pydub 加载 (支持 MP3, WAV, OGG, FLAC 等)
        try:
            from pydub import AudioSegment
            
            # 根据检测到的格式指定格式参数
            format_map = {
                'mp3': 'mp3',
                'wav': 'wav',
                'ogg': 'ogg',
                'flac': 'flac',
                'm4a': 'm4a'
            }
            file_format = format_map.get(detected_format, None)
            
            if file_format:
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=file_format)
            else:
                # 让 pydub 自动检测
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            # 转换为单声道、16kHz
            audio = audio.set_channels(1).set_frame_rate(16000)
            # 转换为numpy数组
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            # 归一化到 [-1, 1]
            max_val = float(2**15) if audio.sample_width == 2 else float(2**31)
            samples = samples / max_val
            return samples, 16000
        except Exception as e:
            logger.debug(f"pydub 加载失败: {e}")
            pass
        
        # 方法3: 如果是MP3格式，尝试使用 audioread 解码
        if detected_format == 'mp3':
            try:
                import audioread
                with audioread.audio_open(io.BytesIO(audio_bytes)) as f:
                    # 读取所有音频数据
                    raw_data = b''.join(f)
                    # 转换为numpy数组 (16-bit PCM)
                    samples = np.frombuffer(raw_data, dtype=np.int16)
                    sr = f.samplerate
                    
                    # 转换为单声道
                    if f.channels > 1:
                        samples = samples.reshape(-1, f.channels).mean(axis=1)
                    
                    # 归一化到 [-1, 1]
                    samples = samples.astype(np.float32) / 32768.0
                    
                    # 重采样到 16kHz
                    if sr != 16000:
                        from scipy import signal
                        samples = signal.resample(samples, int(len(samples) * 16000 / sr))
                    
                    return samples, 16000
            except Exception as e:
                logger.debug(f"audioread 加载失败: {e}")
                pass
            
            # 方法3b: 尝试使用 pydub (需要ffmpeg)
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                audio = audio.set_channels(1).set_frame_rate(16000)
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                max_val = float(2**15) if audio.sample_width == 2 else float(2**31)
                samples = samples / max_val
                return samples, 16000
            except Exception as e:
                logger.debug(f"pydub mp3 加载失败: {e}")
                pass
        
        # 方法4: 如果是M4A/MP4格式
        if detected_format == 'm4a':
            try:
                import audioread
                with audioread.audio_open(io.BytesIO(audio_bytes)) as f:
                    raw_data = b''.join(f)
                    samples = np.frombuffer(raw_data, dtype=np.int16)
                    sr = f.samplerate
                    
                    if f.channels > 1:
                        samples = samples.reshape(-1, f.channels).mean(axis=1)
                    
                    samples = samples.astype(np.float32) / 32768.0
                    
                    if sr != 16000:
                        from scipy import signal
                        samples = signal.resample(samples, int(len(samples) * 16000 / sr))
                    
                    return samples, 16000
            except Exception as e:
                logger.debug(f"audioread m4a 加载失败: {e}")
                pass
        
        # 方法4: 使用标准库 wave 加载标准WAV
        if detected_format == 'wav':
            try:
                import wave
                with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
                    n_channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    sample_rate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()
                    
                    raw_data = wav_file.readframes(n_frames)
                    
                    # 根据采样宽度解析数据
                    if sample_width == 2:
                        samples = np.frombuffer(raw_data, dtype=np.int16)
                    elif sample_width == 4:
                        samples = np.frombuffer(raw_data, dtype=np.int32)
                    else:
                        raise ValueError(f"Unsupported sample width: {sample_width}")
                    
                    # 转换为单声道
                    if n_channels > 1:
                        samples = samples.reshape(-1, n_channels).mean(axis=1)
                    
                    # 归一化到 [-1, 1]
                    samples = samples.astype(np.float32) / (2**15 if sample_width == 2 else 2**31)
                    
                    # 重采样到 16kHz
                    if sample_rate != 16000:
                        from scipy import signal
                        samples = signal.resample(samples, int(len(samples) * 16000 / sample_rate))
                    
                    return samples, 16000
            except Exception as e:
                logger.debug(f"wave 加载失败: {e}")
                pass
        
        # 如果所有方法都失败，提供有用的错误信息
        error_msg = f"无法加载音频文件，格式不支持 (检测到: {detected_format})"
        if detected_format in ['m4a', 'mp4']:
            error_msg += "\n提示: M4A/MP4 格式需要安装 ffmpeg。请访问 https://ffmpeg.org/download.html 下载并添加到系统 PATH"
        elif detected_format == 'mp3':
            error_msg += "\n提示: MP3 解码失败，请确保安装了最新版本的 soundfile: pip install -U soundfile"
        raise ValueError(error_msg)

    async def predict_voice(self, audio_bytes: bytes) -> Dict:
        """
        声音伪造检测 (接收原始音频 bytes -> 内部提取特征 -> 推理)
        优先使用微调后的 AASIST 模型，降级到双流融合
        """
        try:
            # === 0. 统一预处理 ===
            y, sr = self._load_audio_with_fallback(audio_bytes)
            if len(y) < 100:
                return {
                    "confidence": 0.0, 
                    "is_fake": False, 
                    "error": "Audio too short",
                    "model_type": self.audio_model_type,
                    "model_version": self.MODEL_VERSIONS.get("audio_fallback", "unknown")
                }

            # === 优先使用 AASIST 模型 ===
            if self.aasist_model is not None:
                return await self._predict_voice_aasist(y)
            
            # === 降级到双流融合模型 ===
            return await self._predict_voice_dual_stream(y)

        except Exception as e:
            logger.error(f"音频预测失败: {e}", exc_info=True)
            return {
                "confidence": 0.0, 
                "is_fake": False, 
                "error": str(e),
                "model_type": self.audio_model_type,
                "model_version": "error"
            }
    
    async def _predict_voice_aasist(self, y: np.ndarray) -> Dict:
        """使用微调后的 AASIST 模型进行预测"""
        import torch
        
        try:
            # 获取配置中的音频长度
            target_len = self.aasist_config.get('nb_samp', 64600) if self.aasist_config else 64600
            
            # 1. 准备输入 tensor
            if len(y) < target_len:
                # 填充到目标长度
                y = np.pad(y, (0, target_len - len(y)), mode='constant')
            else:
                # 截断到目标长度
                y = y[:target_len]
            
            waveform = torch.FloatTensor(y).unsqueeze(0)  # (1, target_len)
            
            # 2. 推理
            with torch.no_grad():
                _, outputs = self.aasist_model(waveform, Freq_aug=False)
                
                # AASIST模型输出语义（所有模型统一）：
                # Index 0 = 伪造概率, Index 1 = 真人概率
                import torch.nn.functional as F
                probs = F.softmax(outputs, dim=1)[0]
                
                # 伪造置信度 = softmax(outputs)[0]
                score = float(probs[0])
                
                # 保留原始信息用于调试
                spoof_logit = float(outputs[0][0])
                bona_logit = float(outputs[0][1])
            
            # 3. 结果判定
            threshold = settings.VOICE_DETECTION_THRESHOLD
            is_fake = score > threshold
            
            return {
                "confidence": round(score, 4),
                "is_fake": is_fake,
                "risk_level": self._calculate_risk_level(score, threshold),
                "method": "aasist_finetuned",
                "model_type": self.audio_model_type,
                "model_version": self.MODEL_VERSIONS["audio"],
                "details": {
                    "model": "AASIST",
                    "eer": self.aasist_eer,
                    "audio_length": len(y),
                    "threshold": threshold
                }
            }
            
        except Exception as e:
            logger.error(f"AASIST预测失败: {e}")
            # 降级到双流融合
            return await self._predict_voice_dual_stream(y)
    
    async def _predict_voice_dual_stream(self, y: np.ndarray) -> Dict:
        """使用双流融合模型进行预测（降级方案）"""
        score_dl = 0.5
        score_ml = 0.5
        has_dl = False
        has_ml = False
        method = "unknown"

        try:
            # === Stream A: 深度流 (ResNet ONNX) ===
            if self.voice_session and not isinstance(self.voice_session, MockOnnxSession):
                # 提取 Mel 谱图
                mel_spec = librosa.feature.melspectrogram(
                    y=y, sr=16000, n_mels=64, n_fft=1024, hop_length=512
                )
                log_mel = librosa.power_to_db(mel_spec, ref=np.max)
                
                # 构造输入 Tensor (Batch=1, Channel=1, Freq=64, Time=...)
                dl_input = log_mel[np.newaxis, np.newaxis, :, :].astype(np.float32)
                
                # 推理
                input_name = self.voice_session.get_inputs()[0].name
                output = self.voice_session.run(None, {input_name: dl_input})
                
                # Softmax
                logits = output[0]
                probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
                score_dl = float(probs[0][1])  # Index 1 是 Fake
                has_dl = True

            # === Stream B: 统计流 (GNB -> NMF -> SVM) ===
            if self.gnb_model and self.nmf_model and self.svm_model:
                # 提取 MFCC
                mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=20)
                
                # 计算统计特征 (Mean + Std) -> 40维向量
                mfcc_mean = np.mean(mfcc, axis=1)
                mfcc_std = np.std(mfcc, axis=1)
                ml_input = np.concatenate([mfcc_mean, mfcc_std]).reshape(1, -1)
                
                # 流水线推理
                prob_feats = self.gnb_model.predict_proba(ml_input)
                nmf_feats = self.nmf_model.transform(prob_feats)
                score_ml = float(self.svm_model.predict_proba(nmf_feats)[0][1])
                has_ml = True

            # === Fusion: 决策融合 ===
            if has_dl and has_ml:
                final_score = 0.7 * score_dl + 0.3 * score_ml
                method = "dual_stream_fusion"
            elif has_dl:
                final_score = score_dl
                method = "deep_learning_only"
            elif has_ml:
                final_score = score_ml
                method = "statistical_only"
            else:
                final_score = 0.0
                method = "mock"

            # 结果判定
            threshold = settings.VOICE_DETECTION_THRESHOLD
            return {
                "confidence": round(final_score, 4),
                "is_fake": final_score > threshold,
                "risk_level": self._calculate_risk_level(final_score, threshold),
                "method": method,
                "model_type": self.audio_model_type,
                "model_version": self.MODEL_VERSIONS.get("audio_fallback", "unknown"),
                "details": {"dl_score": round(score_dl, 4), "ml_score": round(score_ml, 4)}
            }

        except Exception as e:
            logger.error(f"双流融合预测失败: {e}")
            return {
                "confidence": 0.0, 
                "is_fake": False, 
                "error": str(e),
                "method": "error",
                "model_type": self.audio_model_type
            }

    async def predict_video(self, video_tensor: np.ndarray) -> Dict:
        """
        视频Deepfake检测 (ResNet+LSTM 时序检测)
        """
        if self.video_session is None:
            return {
                "confidence": 0.0, 
                "is_deepfake": False, 
                "message": "Video model not loaded",
                "model_version": "none"
            }
        
        try:
            # 1. 准备输入
            input_name = self.video_session.get_inputs()[0].name
            
            # 强制转 float32
            video_tensor = video_tensor.astype(np.float32)
            
            # 2. 执行推理
            outputs = self.video_session.run(None, {input_name: video_tensor})
            logits = outputs[0]
            
            logger.debug(f"[Video Model] Raw Logits: {logits}")

            # 3. Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            fake_prob = float(probs[0][0]) 
            
            logger.debug(f"[Video Result] Fake Prob: {fake_prob:.6f}") 
            
            threshold = settings.VIDEO_DETECTION_THRESHOLD
            is_deepfake = fake_prob > threshold
            risk_level = self._calculate_risk_level(fake_prob, threshold)
            
            return {
                "confidence": fake_prob,
                "is_deepfake": is_deepfake,
                "risk_level": risk_level,
                "threshold": threshold,
                "model_version": self.MODEL_VERSIONS["video"],
                "raw_logits": logits.tolist()
            }
            
        except Exception as e:
            logger.error(f"Video prediction failed: {e}", exc_info=True)
            return {
                "confidence": 0.0, 
                "is_deepfake": False, 
                "error": str(e),
                "model_version": "error"
            }

    async def predict_text(self, text: str) -> dict:
        """文本诈骗检测 - 支持PyTorch和ONNX模型"""
        # 1. 检查模型是否加载
        if not self.tokenizer:
            return {
                "risk_level": "unknown", 
                "confidence": 0.0, 
                "error": "Text Model not loaded",
                "model_type": self.text_model_type,
                "model_version": "none"
            }

        try:
            # 使用PyTorch模型（优先）
            if self.text_model_torch is not None:
                return await self._predict_text_torch(text)
            
            # 使用ONNX模型
            if self.text_session is not None and not isinstance(self.text_session, MockOnnxSession):
                return await self._predict_text_onnx(text)
            
            return {
                "risk_level": "unknown", 
                "confidence": 0.0, 
                "error": "No text model available",
                "model_type": self.text_model_type,
                "model_version": "none"
            }

        except Exception as e:
            logger.error(f"Text prediction error: {e}")
            return {
                "risk_level": "error", 
                "details": str(e),
                "model_type": self.text_model_type,
                "model_version": "error"
            }
    
    async def _predict_text_torch(self, text: str) -> dict:
        """使用PyTorch BERT模型进行预测，并结合LLM进行增强判断"""
        import torch
        
        # 1. BERT 基础预测
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256
        )
        
        with torch.no_grad():
            outputs = self.text_model_torch(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            bert_prob = probs[0][1].item()
        
        # 2. LLM 增强判断
        llm_result = await self._llm_text_analysis(text)
        
        # 3. 融合 BERT + LLM 结果
        if llm_result.get("is_fraud"):
            # LLM 判定为诈骗，使用 LLM 的置信度（通常更准确）
            fraud_prob = max(bert_prob, 0.7)  # 至少 0.7
            fraud_prob = max(fraud_prob, llm_result.get("confidence", 0.7))
        else:
            # LLM 判定为正常，降低 BERT 的诈骗概率
            fraud_prob = bert_prob * 0.5  # 降低权重
        
        # 结果判定
        threshold = settings.TEXT_DETECTION_THRESHOLD
        is_fraud = fraud_prob > threshold or llm_result.get("is_fraud", False)
        
        # 确定模型版本
        model_version = self.MODEL_VERSIONS["text"] if self.text_model_type == "bert_finetuned" else self.MODEL_VERSIONS["text_fallback"]
        
        return {
            "risk_level": self._calculate_risk_level(fraud_prob, threshold),
            "label": "fraud" if is_fraud else "normal",
            "confidence": round(fraud_prob, 4),
            "threshold": threshold,
            "keywords": llm_result.get("keywords", []),
            "fraud_type": llm_result.get("fraud_type", ""),
            "analysis": llm_result.get("analysis", ""),
            "bert_confidence": round(bert_prob, 4),
            "llm_is_fraud": llm_result.get("is_fraud", False),
            "model_type": self.text_model_type,
            "model_version": model_version
        }
    
    async def _llm_text_analysis(self, text: str) -> dict:
        """使用 LLM 进行文本诈骗分析"""
        try:
            # 延迟导入避免循环依赖
            from app.services.llm_service import LLMService
            from langchain_core.messages import SystemMessage, HumanMessage
            
            llm_service = LLMService()
            
            # 简化的 Prompt 用于文本检测
            prompt = f"""你是一个专业的反诈文本分析专家。请分析以下文本是否存在诈骗风险。

【待分析文本】：
{text}

请输出 JSON 格式结果：
{{
    "is_fraud": true/false,
    "confidence": 0.0-1.0,
    "fraud_type": "诈骗类型（如：刷单返利、冒充公检法、虚假投资等）",
    "keywords": ["关键词1", "关键词2"],
    "analysis": "简要分析理由"
}}

判断标准：
- 涉及转账、汇款、验证码、密码 → 高风险
- 冒充公检法、银行、客服 → 高风险  
- 中奖、返利、投资高回报 → 高风险
- 正常社交、工作沟通 → 安全"""

            # 调用 LLM
            llm = llm_service._create_llm()
            response = await llm.ainvoke([
                SystemMessage(content="你是一个专业的反诈文本分析专家，只输出JSON格式结果。"),
                HumanMessage(content=prompt),
            ])
            
            # 解析 JSON
            import json
            content = response.content
            
            # 提取 JSON 部分
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            
            return {
                "is_fraud": result.get("is_fraud", False),
                "confidence": result.get("confidence", 0.5),
                "fraud_type": result.get("fraud_type", ""),
                "keywords": result.get("keywords", []),
                "analysis": result.get("analysis", "")
            }
            
        except Exception as e:
            logger.warning(f"LLM 文本分析失败: {e}")
            # 返回默认结果，降级到纯 BERT
            return {
                "is_fraud": False,
                "confidence": 0.5,
                "fraud_type": "",
                "keywords": [],
                "analysis": ""
            }
    
    async def _predict_text_onnx(self, text: str) -> dict:
        """使用ONNX模型进行预测"""
        # 预处理
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=256
        )
        
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }

        # 推理
        logits = self.text_session.run(None, ort_inputs)[0]
        
        # Softmax
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        probs = softmax(logits[0])
        fraud_prob = float(probs[1])

        # 结果判定
        threshold = settings.TEXT_DETECTION_THRESHOLD
        is_fraud = fraud_prob > threshold
        
        return {
            "risk_level": self._calculate_risk_level(fraud_prob, threshold),
            "label": "fraud" if is_fraud else "normal",
            "confidence": round(fraud_prob, 4),
            "threshold": threshold,
            "keywords": ["高风险语义"] if is_fraud else [],
            "model_type": self.text_model_type,
            "model_version": self.MODEL_VERSIONS["text_onnx"]
        }

model_service = ModelService()