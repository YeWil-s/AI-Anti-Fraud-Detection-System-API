from app.core.logger import get_logger

logger = get_logger(__name__)

class RiskFusionEngine:
    def __init__(self):
        # 默认权重：文本意图占主导(50%)，视觉(30%)，音频(20%)
        self.weights = {"text": 0.5, "vision": 0.3, "audio": 0.2}

    def calculate_score(self, llm_classification: dict, local_text_conf: float, audio_conf: float, video_conf: float) -> float:
        """
        基于 LLM 意图分类和本地模型置信度的多模态动态加权融合算法
        """
        # 1. 熔断机制：触发特定意图+高危模态特征，直接满分预警
        intent = llm_classification.get("intent", "")
        if "屏幕共享" in intent and video_conf > 0.7:
            logger.warning("触发熔断：屏幕共享 + 异常视觉")
            return 95.0
        if "转账" in intent and audio_conf > 0.8:
            logger.warning("触发熔断：诱导转账 + 语音伪造")
            return 95.0

        # 2. 将 LLM 的分类结果映射为基础文本分数
        risk_level = llm_classification.get("risk_level", "safe")
        if risk_level == "fake":
            base_text_score = 85.0
        elif risk_level == "suspicious":
            base_text_score = 60.0
        else:
            base_text_score = 20.0

        # 如果命中了具体的诈骗剧本，文本分数加成 10 分
        if llm_classification.get("match_script", "无") != "无":
            base_text_score = min(100.0, base_text_score + 10.0)

        # 结合本地 Text-ONNX 的打分，平滑文本总分
        final_text_score = (base_text_score * 0.7) + ((local_text_conf * 100) * 0.3)

        # 3. 动态加权计算 (处理某些模态缺失)
        # 例如：如果是纯文本聊天，audio_conf 和 video_conf 为 0
        active_modalities = {"text": final_text_score}
        if audio_conf > 0:
            active_modalities["audio"] = audio_conf * 100
        if video_conf > 0:
            active_modalities["vision"] = video_conf * 100

        total_weight = sum([self.weights[k] for k in active_modalities.keys()])
        
        final_fused_score = 0.0
        for mod, score in active_modalities.items():
            # 归一化当前存在的模态权重
            norm_weight = self.weights[mod] / total_weight
            final_fused_score += score * norm_weight

        logger.info(f"多模态算分完毕 -> T:{final_text_score:.1f}, A:{audio_conf*100:.1f}, V:{video_conf*100:.1f} | 最终分: {final_fused_score:.1f}")
        return round(final_fused_score, 2)

fusion_engine = RiskFusionEngine()