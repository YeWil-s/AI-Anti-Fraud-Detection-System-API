"""
AI模型服务层
"""
import onnxruntime as ort
import joblib  # [新增] 用于加载 sklearn 模型 (GNB, NMF, SVM)
import numpy as np
from typing import Dict, Optional
from pathlib import Path
from transformers import AutoTokenizer  # [新增] 用于文本分词
from app.core.config import settings
from app.core.logger import get_logger

# 初始化模块级 logger
logger = get_logger(__name__)

class ModelService:
    """AI模型加载与推理服务"""
    
    def __init__(self):
        """初始化模型服务"""
        # --- 音频模型组件 ---
        self.voice_session: Optional[ort.InferenceSession] = None # 深度流 (ResNet/RawNet)
        self.gnb_model = None     # 统计流 (GNB)
        self.nmf_model = None     # 统计流 (NMF)
        self.svm_model = None     # 统计流 (SVM)
        
        # --- 视频模型组件 ---
        self.video_session: Optional[ort.InferenceSession] = None # 时空流 (ResNet+LSTM)
        
        # --- 文本模型组件 ---
        self.text_session: Optional[ort.InferenceSession] = None  # 文本流 (BERT ONNX)
        self.tokenizer = None     # BERT 分词器
        
        # 加载所有模型
        self._load_models()
    
    def _load_models(self):
        """加载AI模型 (ONNX & Sklearn & Transformers)"""
        try:
            # 1. 加载声音检测模型 (混合架构)
            # A. 深度学习流 (ONNX)
            if Path(settings.VOICE_MODEL_PATH).exists():
                self.voice_session = ort.InferenceSession(
                    settings.VOICE_MODEL_PATH,
                    providers=['CPUExecutionProvider']
                )
                logger.info(f"✓ Voice Deep Model loaded: {settings.VOICE_MODEL_PATH}")
            else:
                logger.warning(f"⚠ Voice Deep model not found: {settings.VOICE_MODEL_PATH}")
            
            # B. 统计特征流 (Sklearn: GNB/NMF/SVM)
            # 假设模型文件存放在 ./models/ml/ 目录下
            ml_path = Path("./models/ml")
            try:
                if (ml_path / "gnb.pkl").exists():
                    self.gnb_model = joblib.load(ml_path / "gnb.pkl")
                    self.nmf_model = joblib.load(ml_path / "nmf.pkl")
                    self.svm_model = joblib.load(ml_path / "svm.pkl")
                    logger.info("✓ Voice Statistical Models (GNB/NMF) loaded")
                else:
                    logger.warning(f"⚠ Voice Statistical models not found in {ml_path}")
            except Exception as e:
                logger.error(f"Failed to load statistical models: {e}")

            # 2. 加载视频检测模型 (ResNet+LSTM ONNX)
            if Path(settings.VIDEO_MODEL_PATH).exists():
                self.video_session = ort.InferenceSession(
                    settings.VIDEO_MODEL_PATH,
                    providers=['CPUExecutionProvider']
                )
                logger.info(f"✓ Video Model loaded: {settings.VIDEO_MODEL_PATH}")
            else:
                logger.warning(f"⚠ Video model not found: {settings.VIDEO_MODEL_PATH}")
            
            # 3. 加载文本检测模型 (BERT ONNX + Tokenizer)
            text_path = Path(settings.TEXT_MODEL_PATH) # 例如 ./models/text_bert_onnx
            onnx_file = text_path / "model.onnx"
            
            if text_path.exists():
                try:
                    # 加载分词器 (从本地文件夹)
                    self.tokenizer = AutoTokenizer.from_pretrained(str(text_path))
                    
                    if onnx_file.exists():
                        self.text_session = ort.InferenceSession(
                            str(onnx_file),
                            providers=['CPUExecutionProvider']
                        )
                        logger.info(f"✓ Text Model loaded: {text_path}")
                    else:
                        logger.warning(f"⚠ Text ONNX file missing: {onnx_file}")
                except Exception as e:
                    logger.error(f"Failed to load Text model components: {e}")
            else:
                logger.warning(f"⚠ Text model path not found: {settings.TEXT_MODEL_PATH}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
    
    async def predict_voice(self, features: np.ndarray) -> Dict:
        """
        声音伪造检测 (双流融合: 深度学习 + 统计特征)
        Args:
            features: MFCC特征矩阵, 形状 (n_frames, 20)
        """
        # 结果初始化
        score_ml = 0.5    # 统计流分数
        score_dl = 0.5    # 深度流分数
        has_ml = False
        has_dl = False

        try:
            # 1. 统计流推理 (GNB -> NMF -> SVM)
            if self.gnb_model and self.nmf_model and self.svm_model:
                # GNB: 特征 -> 概率空间 (n_frames, 2)
                prob_feats = self.gnb_model.predict_proba(features)
                # NMF: 概率空间 -> 潜在模式 (n_frames, n_components)
                nmf_feats = self.nmf_model.transform(prob_feats)
                # SVM: 对整段音频的平均特征进行分类
                avg_feat = np.mean(nmf_feats, axis=0, keepdims=True)
                score_ml = float(self.svm_model.predict_proba(avg_feat)[0][1]) # Index 1 是 Fake
                has_ml = True

            # 2. 深度流推理 (ResNet/RawNet ONNX)
            if self.voice_session:
                # 注意：深度模型通常需要特定的输入形状，例如 (1, 1, n_mels, time)
                # 这里假设 features 已经适配或模型支持该输入，实际中可能需要 reshape
                input_name = self.voice_session.get_inputs()[0].name
                # 为了演示，这里假设模型接受 (Batch, Time, Feat)
                dl_input = features[np.newaxis, ...].astype(np.float32)
                output = self.voice_session.run(None, {input_name: dl_input})
                
                # Softmax
                logits = output[0]
                probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
                score_dl = float(probs[0][1])
                has_dl = True

            # 3. 决策融合
            if not has_ml and not has_dl:
                return {"confidence": 0.0, "is_fake": False, "message": "No models loaded"}
            
            if has_ml and has_dl:
                # 加权融合: 0.4 * ML + 0.6 * DL
                final_score = 0.4 * score_ml + 0.6 * score_dl
                method = "hybrid_fusion"
            elif has_ml:
                final_score = score_ml
                method = "statistical_only"
            else:
                final_score = score_dl
                method = "deep_learning_only"

            return {
                "confidence": final_score,
                "is_fake": final_score > 0.5, # 阈值
                "model_version": "v2.0-hybrid",
                "method": method,
                "details": {"ml_score": score_ml, "dl_score": score_dl}
            }
            
        except Exception as e:
            logger.error(f"Voice prediction failed: {e}", exc_info=True)
            return {"confidence": 0.0, "is_fake": False, "error": str(e)}
    
    async def predict_video(self, video_tensor: np.ndarray) -> Dict:
        """
        视频Deepfake检测 (ResNet+LSTM 时序检测)
        Args:
            video_tensor: 5D Tensor, 形状 (1, 10, 3, 224, 224)
        """
        if self.video_session is None:
            logger.debug("Video model not loaded")
            return {"confidence": 0.0, "is_deepfake": False, "message": "Video model not loaded"}
        
        try:
            # 1. 准备输入
            input_name = self.video_session.get_inputs()[0].name
            
            # 2. 执行推理 (ONNX)
            # 输出通常是 Logits (1, 2)
            outputs = self.video_session.run(None, {input_name: video_tensor})
            logits = outputs[0]
            
            # 3. Softmax 计算概率
            # exp(x) / sum(exp(x))
            exp_logits = np.exp(logits - np.max(logits)) # 减去最大值防止溢出
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            fake_prob = float(probs[0][1]) # 假设 Index 1 是 Fake 类
            
            return {
                "confidence": fake_prob,
                "is_deepfake": fake_prob > 0.6, # 视频检测阈值通常设高一点
                "model_version": "v2.0-lstm",
                "detection_type": "spatio_temporal"
            }
            
        except Exception as e:
            logger.error(f"Video prediction failed: {e}", exc_info=True)
            return {"confidence": 0.0, "is_deepfake": False, "error": str(e)}
    
    async def predict_text(self, text: str) -> Dict:
        """
        文本诈骗话术检测 (BERT)
        Args:
            text: 待检测文本
        """
        if self.text_session is None or self.tokenizer is None:
            return {"confidence": 0.0, "is_scam": False, "message": "Text model not loaded"}
        
        try:
            # 1. 分词与编码
            # return_tensors="np" 直接返回 numpy 数组供 ONNX 使用
            inputs = self.tokenizer(
                text,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # 2. 准备 ONNX 输入
            # BERT ONNX 通常需要 input_ids, attention_mask, token_type_ids
            # 类型必须是 int64
            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }
            
            # 部分模型可能不需要 token_type_ids，视导出配置而定
            if "token_type_ids" in inputs:
                ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

            # 3. 执行推理
            # 输出通常是 Logits (Batch, 2)
            outputs = self.text_session.run(None, ort_inputs)
            logits = outputs[0]
            
            # 4. Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            scam_prob = float(probs[0][1]) # 假设 Index 1 是 Scam
            
            return {
                "confidence": scam_prob,
                "is_scam": scam_prob > 0.7, # 文本误报较多，建议阈值高些
                "model_version": "bert-base-finetuned",
                "keywords": [] # 后续可接关键词匹配逻辑
            }
            
        except Exception as e:
            logger.error(f"Text prediction failed: {e}", exc_info=True)
            return {"confidence": 0.0, "is_scam": False, "error": str(e)}
    
    def get_model_info(self) -> Dict:
        """获取模型加载状态信息"""
        return {
            "voice_deep_loaded": self.voice_session is not None,
            "voice_ml_loaded": self.gnb_model is not None,
            "video_loaded": self.video_session is not None,
            "text_loaded": self.text_session is not None,
            "paths": {
                "voice": settings.VOICE_MODEL_PATH,
                "video": settings.VIDEO_MODEL_PATH,
                "text": settings.TEXT_MODEL_PATH
            }
        }

# 全局模型服务实例
model_service = ModelService()