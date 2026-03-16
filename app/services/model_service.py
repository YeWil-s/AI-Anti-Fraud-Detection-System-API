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
    
    def __init__(self):
        # --- 音频模型组件 ---
        self.voice_session = None
        self.gnb_model = None
        self.nmf_model = None
        self.svm_model = None
        
        # --- 其他模型 ---
        self.video_session = None
        self.text_session = None
        self.tokenizer = None
        
        self._load_models()
    
    def _load_models(self):
        """加载模型文件"""
        try:
            # 1. 加载深度流 (ONNX)
            if Path(settings.VOICE_MODEL_PATH).exists():
                self.voice_session = ort.InferenceSession(
                    settings.VOICE_MODEL_PATH,
                    providers=['CPUExecutionProvider']
                )
                logger.info(f"✓ Voice Deep Model loaded: {settings.VOICE_MODEL_PATH}")
            else:
                logger.warning(f"⚠ Voice model not found at {settings.VOICE_MODEL_PATH}, using Mock.")
                self.voice_session = MockOnnxSession("voice")

            # 2. 加载统计流 (PKL)
            ml_path = Path("./models/ml") # 确保路径对应
            try:
                if (ml_path / "gnb.pkl").exists():
                    self.gnb_model = joblib.load(ml_path / "gnb.pkl")
                    self.nmf_model = joblib.load(ml_path / "nmf.pkl")
                    self.svm_model = joblib.load(ml_path / "svm.pkl")
                    logger.info("✓ Voice Statistical Models loaded")
                else:
                    logger.warning("⚠ ML models not found in models/ml/")
            except Exception as e:
                logger.error(f"Failed to load ML models: {e}")

            # === 3. 加载视频模型 (ONNX) ===
            if Path(settings.VIDEO_MODEL_PATH).exists():
                self.video_session = ort.InferenceSession(
                    settings.VIDEO_MODEL_PATH, 
                    providers=['CPUExecutionProvider']
                )
                logger.info(f"✅ Video Model loaded: {settings.VIDEO_MODEL_PATH}")
            else:
                logger.warning(f"⚠️ Video model not found, using Mock.")
                self.video_session = MockOnnxSession("video")
                
            # === 4. 加载文本模型 (BERT ONNX + Tokenizer) ===
            # [新增调试代码] 打印绝对路径，看看 Celery 到底在读哪里
            abs_path = os.path.abspath(settings.TEXT_MODEL_PATH)
            logger.info(f"[DEBUG] Loading Text Model from: {abs_path}")
            
            if Path(settings.TEXT_MODEL_PATH).exists() and Path(settings.TEXT_VOCAB_PATH).exists():
                try:
                    # [新增调试代码] 打印文件大小
                    size = os.path.getsize(settings.TEXT_MODEL_PATH) / (1024 * 1024)
                    logger.info(f"[DEBUG] File Size: {size:.2f} MB")

                    self.tokenizer = BertTokenizer.from_pretrained(settings.TEXT_VOCAB_PATH)
                    
                    self.text_session = ort.InferenceSession(
                        settings.TEXT_MODEL_PATH, 
                        providers=['CPUExecutionProvider']
                    )

                    input_names = [i.name for i in self.text_session.get_inputs()]
                    logger.info(f"[DEBUG] Actual Input Names in Memory: {input_names}")
                    
                    logger.info(f"✅ Text Fraud Model loaded")
                except Exception as e:
                    logger.error(f"❌ Error loading Text model details: {e}")
                    self.text_session = MockOnnxSession("text")

        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def _calculate_risk_level(self, probability: float, threshold: float) -> str:
        if probability >= 0.9: return "critical"
        elif probability >= threshold: return "high"
        elif probability >= (threshold / 2): return "medium"
        else: return "low"

    async def predict_voice(self, audio_bytes: bytes) -> Dict:
        """
        声音伪造检测 (接收原始音频 bytes -> 内部提取双流特征 -> 融合)
        """
        score_dl = 0.5
        score_ml = 0.5
        has_dl = False
        has_ml = False

        try:
            # === 0. 统一预处理 ===
            # 使用 librosa 加载 (重采样到 16k)
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
            if len(y) < 100:
                return {"confidence": 0.0, "is_fake": False, "error": "Audio too short"}

            # === Stream A: 深度流 (ResNet) ===
            # 需求: Log-Mel Spectrogram (1, 1, 64, Time)
            if self.voice_session:
                # 1. 提取 Mel 谱图 (参数需与训练时 dataset_loader.py 一致)
                mel_spec = librosa.feature.melspectrogram(
                    y=y, sr=16000, n_mels=64, n_fft=1024, hop_length=512
                )
                # 2. 转 Log 刻度 (AmplitudeToDB)
                log_mel = librosa.power_to_db(mel_spec, ref=np.max)
                
                # 3. 构造输入 Tensor (Batch=1, Channel=1, Freq=64, Time=...)
                # 注意: ONNX 输入是 float32
                dl_input = log_mel[np.newaxis, np.newaxis, :, :].astype(np.float32)
                
                # 4. 推理
                input_name = self.voice_session.get_inputs()[0].name
                output = self.voice_session.run(None, {input_name: dl_input})
                
                # 5. Softmax
                logits = output[0]
                probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
                score_dl = float(probs[0][1]) # Index 1 是 Fake
                has_dl = True

            # === Stream B: 统计流 (GNB -> NMF -> SVM) ===
            # 需求: MFCC (Time, 20) -> Mean/Std 统计量
            if self.gnb_model and self.nmf_model and self.svm_model:
                # 1. 提取 MFCC
                mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=20)
                
                # 2. 计算统计特征 (Mean + Std) -> 40维向量
                # 注意: 这里必须与 train_ml_stream.py 里的 extract_features 逻辑完全一致
                mfcc_mean = np.mean(mfcc, axis=1)
                mfcc_std = np.std(mfcc, axis=1)
                ml_input = np.concatenate([mfcc_mean, mfcc_std]).reshape(1, -1)
                
                # 3. 流水线推理
                prob_feats = self.gnb_model.predict_proba(ml_input) # GNB
                nmf_feats = self.nmf_model.transform(prob_feats)    # NMF
                score_ml = float(self.svm_model.predict_proba(nmf_feats)[0][1]) # SVM
                has_ml = True

            # === Fusion: 决策融合 ===
            if has_dl and has_ml:
                # 深度流权重 ，统计流 
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
                "confidence": final_score,
                "is_fake": final_score > threshold,
                "risk_level": self._calculate_risk_level(final_score, threshold),
                "method": method,
                "details": {"dl_score": score_dl, "ml_score": score_ml}
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return {"confidence": 0.0, "is_fake": False, "error": str(e)}

    async def predict_video(self, video_tensor: np.ndarray) -> Dict:
        """
        视频Deepfake检测 (ResNet+LSTM 时序检测)
        """
        if self.video_session is None:
            return {"confidence": 0.0, "is_deepfake": False, "message": "Video model not loaded"}
        
        try:
            # 1. 准备输入
            input_name = self.video_session.get_inputs()[0].name
            
            # 强制转 float32
            video_tensor = video_tensor.astype(np.float32)
            
            # 2. 执行推理
            outputs = self.video_session.run(None, {input_name: video_tensor})
            logits = outputs[0]
            
            # ================= [Debug 日志] =================
            logger.info(f"🧠 [Model Output] Raw Logits: {logits}")
            # ====================================================

            # 3. Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            fake_prob = float(probs[0][0]) 
            
            logger.info(f"📊 [Result] Fake Prob: {fake_prob:.6f}") 
            
            threshold = settings.VIDEO_DETECTION_THRESHOLD
            is_deepfake = fake_prob > threshold
            risk_level = self._calculate_risk_level(fake_prob, threshold)
            
            return {
                "confidence": fake_prob,
                "is_deepfake": is_deepfake,
                "risk_level": risk_level,
                "threshold": threshold,
                "model_version": "v2.0-lstm",
                "raw_logits": logits.tolist() # 把原始分也返回回去
            }
            
        except Exception as e:
            logger.error(f"Video prediction failed: {e}", exc_info=True)
            return {"confidence": 0.0, "is_deepfake": False, "error": str(e)}

    async def predict_text(self, text: str) -> dict:
        """文本诈骗检测"""
        # 1. 检查模型是否加载
        if not self.text_session or not self.tokenizer or isinstance(self.text_session, MockOnnxSession):
            return {"risk_level": "unknown", "confidence": 0.0, "error": "Text Model not loaded"}

        try:
            # 2. 预处理 (Tokenization)
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

            # 3. 推理
            logits = self.text_session.run(None, ort_inputs)[0]
            
            # 4. Softmax
            def softmax(x):
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()

            probs = softmax(logits[0])
            fraud_prob = float(probs[1]) # Label 1 是诈骗

            # 5. 结果判定 (使用配置文件中的阈值)
            threshold = settings.TEXT_DETECTION_THRESHOLD
            is_fraud = fraud_prob > threshold
            
            return {
                "risk_level": self._calculate_risk_level(fraud_prob, threshold),
                "label": "fraud" if is_fraud else "normal",
                "confidence": round(fraud_prob, 4),
                "threshold": threshold,
                "keywords": ["高风险语义"] if is_fraud else []
            }

        except Exception as e:
            logger.error(f"Text prediction error: {e}")
            return {"risk_level": "error", "details": str(e)}

model_service = ModelService()