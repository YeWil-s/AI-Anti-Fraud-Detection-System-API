"""
AI模型服务层
"""
import onnxruntime as ort
import joblib  # 用于加载 sklearn 模型 (GNB, NMF, SVM)
import numpy as np
import random
from typing import Dict, Optional
from pathlib import Path
from transformers import AutoTokenizer  # 用于文本分词
from app.core.config import settings
from app.core.logger import get_logger

# 初始化模块级 logger
logger = get_logger(__name__)

class MockInput:
    """模拟 ONNX 输入节点信息"""
    def __init__(self, name="input"):
        self.name = name

class MockOnnxSession:
    """
    伪造的 ONNX InferenceSession。
    当真实模型文件不存在时，使用此类生成随机结果。
    """
    def __init__(self, model_type="unknown"):
        self.model_type = model_type
        
    def get_inputs(self):
        """模拟获取输入节点名称"""
        # 返回一个带有 name 属性的对象列表，模拟 model.get_inputs()[0].name
        return [MockInput(name="mock_input")]

    def run(self, output_names, input_feed):
        """
        模拟推理过程
        Args:
            output_names: 不需要关注
            input_feed: 字典 {input_name: input_tensor}
        Returns:
            list: [logits] (模拟原始输出)
        """
        # 1. 获取 Batch Size (通常是输入的第一维)
        input_tensor = list(input_feed.values())[0]
        batch_size = input_tensor.shape[0] if hasattr(input_tensor, 'shape') else 1
        
        # 2. 生成随机 Logits
        # 假设是二分类 (Real vs Fake)，输出 shape 为 (Batch, 2)
        # 生成两个随机数
        val1 = random.uniform(-1, 1)
        val2 = random.uniform(-1, 1)
        
        # 模拟 logits
        mock_logits = np.array([[val1, val2]] * batch_size, dtype=np.float32)
        
        return [mock_logits]

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
                logger.warning(f"⚠ Voice Deep model not found at {settings.VOICE_MODEL_PATH}, using MOCK.")
                self.voice_session = MockOnnxSession(model_type="voice")
            
            # B. 统计特征流 (Sklearn: GNB/NMF/SVM)
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
                logger.warning(f"⚠ Video model not found at {settings.VIDEO_MODEL_PATH}, using MOCK.")
                self.video_session = MockOnnxSession(model_type="video")
            
            # 3. 加载文本检测模型 (BERT ONNX + Tokenizer)
            text_path = Path(settings.TEXT_MODEL_PATH)
            onnx_file = text_path / "model.onnx"
            
            # 默认设为 Mock
            self.text_session = MockOnnxSession(model_type="text") 

            if text_path.exists():
                try:
                    # 尝试加载真实 Tokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(str(text_path))
                    
                    if onnx_file.exists():
                        # 尝试加载真实 Session
                        self.text_session = ort.InferenceSession(
                            str(onnx_file),
                            providers=['CPUExecutionProvider']
                        )
                        logger.info(f"✓ Text Model loaded: {text_path}")
                    else:
                        logger.warning(f"⚠ Text ONNX file missing, using Mock session.")
                except Exception as e:
                    logger.error(f"Failed to load Text model components: {e}, using Mock.")
            else:
                logger.warning(f"⚠ Text model path not found: {settings.TEXT_MODEL_PATH}, using Mock.")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)

    # [新增] 辅助方法：计算风险等级
    def _calculate_risk_level(self, probability: float, threshold: float) -> str:
        """
        根据概率和阈值计算风险等级
        """
        if probability >= 0.9:
            return "critical"  # 极高风险
        elif probability >= threshold:
            return "high"      # 高风险 (判定为 Fake)
        elif probability >= (threshold / 2):
            return "medium"    # 中等风险 (可疑但未达阈值)
        else:
            return "low"       # 低风险
    
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
                prob_feats = self.gnb_model.predict_proba(features)
                nmf_feats = self.nmf_model.transform(prob_feats)
                avg_feat = np.mean(nmf_feats, axis=0, keepdims=True)
                score_ml = float(self.svm_model.predict_proba(avg_feat)[0][1])
                has_ml = True

            # 2. 深度流推理 (ResNet/RawNet ONNX)
            if self.voice_session:
                input_name = self.voice_session.get_inputs()[0].name
                dl_input = features[np.newaxis, ...].astype(np.float32)
                output = self.voice_session.run(None, {input_name: dl_input})
                
                logits = output[0]
                probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
                score_dl = float(probs[0][1])
                has_dl = True

            # 3. 决策融合
            if not has_ml and not has_dl:
                return {"confidence": 0.0, "is_fake": False, "message": "No models loaded"}
            
            if has_ml and has_dl:
                # 加权融合
                final_score = 0.4 * score_ml + 0.6 * score_dl
                method = "hybrid_fusion"
            elif has_ml:
                final_score = score_ml
                method = "statistical_only"
            else:
                final_score = score_dl
                method = "deep_learning_only"

            # [修改] 使用配置阈值 & 计算风险等级
            threshold = settings.VOICE_DETECTION_THRESHOLD
            is_fake = final_score > threshold
            risk_level = self._calculate_risk_level(final_score, threshold)

            return {
                "confidence": final_score,
                "is_fake": is_fake,
                "risk_level": risk_level,    # [新增]
                "threshold": threshold,      # [新增]
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
            outputs = self.video_session.run(None, {input_name: video_tensor})
            logits = outputs[0]
            
            # 3. Softmax 计算概率
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            fake_prob = float(probs[0][1]) # 假设 Index 1 是 Fake 类
            
            # [修改] 使用配置阈值 & 计算风险等级
            threshold = settings.VIDEO_DETECTION_THRESHOLD
            is_deepfake = fake_prob > threshold
            risk_level = self._calculate_risk_level(fake_prob, threshold)
            
            return {
                "confidence": fake_prob,
                "is_deepfake": is_deepfake,
                "risk_level": risk_level,    # [新增]
                "threshold": threshold,      # [新增]
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
        # Mock 模式拦截
        if isinstance(self.text_session, MockOnnxSession):
             # 模拟随机结果
            import random
            prob = random.random()
            
            # [修改] Mock 模式也使用配置阈值
            threshold = settings.TEXT_DETECTION_THRESHOLD
            is_scam = prob > threshold
            risk_level = self._calculate_risk_level(prob, threshold)
            
            return {
                "confidence": prob,
                "is_scam": is_scam,
                "risk_level": risk_level,
                "threshold": threshold,
                "model_version": "mock-text-v1",
                "keywords": ["mock_risk"] if is_scam else []
            }

        # 如果真的没加载，才返回错误
        if self.text_session is None or self.tokenizer is None:
            return {"confidence": 0.0, "is_scam": False, "message": "Text model not loaded"}
        
        try:
            # 1. 分词与编码
            inputs = self.tokenizer(
                text,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # 2. 准备 ONNX 输入
            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }
            if "token_type_ids" in inputs:
                ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

            # 3. 执行推理
            outputs = self.text_session.run(None, ort_inputs)
            logits = outputs[0]
            
            # 4. Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            scam_prob = float(probs[0][1])
            
            # [修改] 使用配置阈值 & 计算风险等级
            threshold = settings.TEXT_DETECTION_THRESHOLD
            is_scam = scam_prob > threshold
            risk_level = self._calculate_risk_level(scam_prob, threshold)
            
            return {
                "confidence": scam_prob,
                "is_scam": is_scam,
                "risk_level": risk_level,    # [新增]
                "threshold": threshold,      # [新增]
                "model_version": "bert-base-finetuned",
                "keywords": [] 
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