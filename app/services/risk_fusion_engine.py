from typing import Dict, Optional, List
from collections import deque
from app.core.logger import get_logger

logger = get_logger(__name__)


class RiskFusionEngine:
    """基础风险融合引擎（保持向后兼容）"""
    
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


class RiskFusionEngineV2:
    """
    增强版风险融合引擎
    
    新增特性：
    - P0: 场景自适应权重 - 根据检测意图动态调整模态权重
    - P1: 模态协同放大 - 多模态同时异常时风险非线性叠加
    - P2: 时序滑动窗口 - EMA平滑降低瞬时抖动误报
    """
    
    # 场景自适应权重配置 (P0)
    SCENE_WEIGHTS = {
        # 屏幕共享类诈骗：视觉权重提高
        "屏幕共享": {"text": 0.35, "vision": 0.50, "audio": 0.15},
        "远程控制": {"text": 0.35, "vision": 0.50, "audio": 0.15},
        "查看余额": {"text": 0.35, "vision": 0.50, "audio": 0.15},
        
        # 语音冒充类诈骗：音频权重提高
        "转账": {"text": 0.35, "vision": 0.15, "audio": 0.50},
        "汇款": {"text": 0.35, "vision": 0.15, "audio": 0.50},
        "语音冒充": {"text": 0.35, "vision": 0.15, "audio": 0.50},
        "冒充公检法": {"text": 0.35, "vision": 0.15, "audio": 0.50},
        
        # 钓鱼链接类诈骗：文本权重提高
        "钓鱼链接": {"text": 0.65, "vision": 0.25, "audio": 0.10},
        "点击链接": {"text": 0.65, "vision": 0.25, "audio": 0.10},
        "下载APP": {"text": 0.65, "vision": 0.25, "audio": 0.10},
        "验证码": {"text": 0.60, "vision": 0.20, "audio": 0.20},
        
        # 默认权重
        "default": {"text": 0.50, "vision": 0.30, "audio": 0.20}
    }
    
    # 模态协同放大配置 (P1)
    SYNERGY_CONFIG = {
        "threshold": 0.6,  # 异常判定阈值
        "bonus_factor": 1.15,  # 协同加成系数（双模态异常+15%）
        "triple_bonus_factor": 1.30,  # 三模态异常+30%
        "cap": 98.0  # 协同加成上限
    }
    
    # 时序滑动窗口配置 (P2)
    TEMPORAL_CONFIG = {
        "ema_alpha": 0.3,  # EMA平滑因子（0-1，越大越敏感）
        "history_window": 5,  # 历史窗口大小
        "trend_boost": 0.05  # 趋势加成（风险上升时额外加权）
    }
    
    def __init__(self):
        self.default_weights = {"text": 0.5, "vision": 0.3, "audio": 0.2}
        # 每个call_id的时序状态缓存
        self._temporal_cache: Dict[str, dict] = {}
    
    def _get_scene_weights(self, intent: str, match_script: str) -> Dict[str, float]:
        """
        P0: 根据意图和匹配剧本获取场景自适应权重
        """
        # 优先匹配意图关键词
        for keyword, weights in self.SCENE_WEIGHTS.items():
            if keyword in intent and keyword != "default":
                logger.debug(f"场景权重匹配: 意图'{intent}' -> 关键词'{keyword}'")
                return weights.copy()
        
        # 其次匹配剧本名称
        if match_script and match_script != "无":
            for keyword, weights in self.SCENE_WEIGHTS.items():
                if keyword in match_script and keyword != "default":
                    logger.debug(f"场景权重匹配: 剧本'{match_script}' -> 关键词'{keyword}'")
                    return weights.copy()
        
        return self.SCENE_WEIGHTS["default"].copy()
    
    def _calculate_synergy_bonus(
        self, 
        text_score: float, 
        audio_conf: float, 
        video_conf: float,
        weights: Dict[str, float]
    ) -> float:
        """
        P1: 计算模态协同风险放大系数
        
        当多个模态同时超过阈值时，风险非线性叠加
        """
        threshold = self.SYNERGY_CONFIG["threshold"]
        
        # 计算各模态风险度（归一化到0-1）
        text_risk = text_score / 100.0
        audio_risk = audio_conf
        video_risk = video_conf
        
        # 统计超过阈值的模态数
        active_risks = 0
        if text_risk > threshold * 0.8:  # 文本阈值略低（LLM判断更可靠）
            active_risks += 1
        if audio_risk > threshold:
            active_risks += 1
        if video_risk > threshold:
            active_risks += 1
        
        # 应用协同加成
        if active_risks >= 3:
            bonus = self.SYNERGY_CONFIG["triple_bonus_factor"]
            logger.debug(f"三模态协同触发: 加成 {bonus}x")
            return bonus
        elif active_risks >= 2:
            bonus = self.SYNERGY_CONFIG["bonus_factor"]
            logger.debug(f"双模态协同触发: 加成 {bonus}x")
            return bonus
        
        return 1.0
    
    def _apply_temporal_smoothing(self, call_id: str, current_score: float) -> float:
        """
        P2: 应用时序滑动窗口平滑（EMA + 趋势感知）
        """
        if not call_id:
            return current_score
        
        alpha = self.TEMPORAL_CONFIG["ema_alpha"]
        
        if call_id not in self._temporal_cache:
            # 首次检测，初始化缓存
            self._temporal_cache[call_id] = {
                "ema_score": current_score,
                "history": deque([current_score], maxlen=self.TEMPORAL_CONFIG["history_window"]),
                "last_score": current_score
            }
            return current_score
        
        cache = self._temporal_cache[call_id]
        
        # EMA平滑: ema = alpha * current + (1 - alpha) * prev_ema
        ema_score = alpha * current_score + (1 - alpha) * cache["ema_score"]
        
        # 趋势感知：如果风险持续上升，给予额外加权
        trend_bonus = 0.0
        if len(cache["history"]) >= 2:
            recent_scores = list(cache["history"])
            if current_score > cache["last_score"]:
                # 风险上升趋势
                trend_bonus = self.TEMPORAL_CONFIG["trend_boost"] * (
                    (current_score - cache["last_score"]) / 100.0
                )
                logger.debug(f"风险上升趋势 detected: +{trend_bonus:.3f}")
        
        # 更新缓存
        final_score = min(100.0, ema_score * (1 + trend_bonus))
        cache["ema_score"] = ema_score
        cache["last_score"] = current_score
        cache["history"].append(current_score)
        
        logger.debug(f"时序平滑: 原始={current_score:.1f}, EMA={ema_score:.1f}, 最终={final_score:.1f}")
        return final_score
    
    def calculate_score(
        self, 
        llm_classification: dict, 
        local_text_conf: float, 
        audio_conf: float, 
        video_conf: float,
        call_id: Optional[str] = None
    ) -> float:
        """
        增强版多模态风险融合算法 (P0+P1+P2)
        
        Args:
            llm_classification: LLM分类结果
            local_text_conf: 本地文本模型置信度
            audio_conf: 音频置信度
            video_conf: 视频置信度
            call_id: 通话ID（用于时序平滑）
        
        Returns:
            最终融合风险分数 (0-100)
        """
        intent = llm_classification.get("intent", "")
        match_script = llm_classification.get("match_script", "无")
        
        # === P0: 场景自适应权重 ===
        scene_weights = self._get_scene_weights(intent, match_script)
        
        # 1. 熔断机制（保持原有逻辑，但使用场景权重）
        if "屏幕共享" in intent and video_conf > 0.7:
            logger.warning(f"[V2] 触发熔断：屏幕共享 + 异常视觉 (conf={video_conf:.2f})")
            fused_score = 95.0 * scene_weights["vision"] + 85.0 * (1 - scene_weights["vision"])
            return self._apply_temporal_smoothing(call_id, min(98.0, fused_score)) if call_id else min(98.0, fused_score)
            
        if "转账" in intent and audio_conf > 0.8:
            logger.warning(f"[V2] 触发熔断：诱导转账 + 语音伪造 (conf={audio_conf:.2f})")
            fused_score = 95.0 * scene_weights["audio"] + 85.0 * (1 - scene_weights["audio"])
            return self._apply_temporal_smoothing(call_id, min(98.0, fused_score)) if call_id else min(98.0, fused_score)
        
        # 2. 基础文本分数计算
        risk_level = llm_classification.get("risk_level", "safe")
        if risk_level == "fake":
            base_text_score = 85.0
        elif risk_level == "suspicious":
            base_text_score = 60.0
        else:
            base_text_score = 20.0
        
        # 剧本匹配加成
        if match_script != "无":
            base_text_score = min(100.0, base_text_score + 10.0)
        
        # 文本置信度平滑
        final_text_score = (base_text_score * 0.7) + ((local_text_conf * 100) * 0.3)
        
        # 3. 构建活跃模态集合
        active_modalities = {"text": final_text_score}
        if audio_conf > 0:
            active_modalities["audio"] = audio_conf * 100
        if video_conf > 0:
            active_modalities["vision"] = video_conf * 100
        
        # === P1: 模态协同放大 ===
        synergy_bonus = self._calculate_synergy_bonus(
            final_text_score, audio_conf, video_conf, scene_weights
        )
        
        # 4. 动态加权融合（使用场景权重）
        total_weight = sum([scene_weights[k] for k in active_modalities.keys()])
        
        base_fused_score = 0.0
        for mod, score in active_modalities.items():
            norm_weight = scene_weights[mod] / total_weight
            base_fused_score += score * norm_weight
        
        # 应用协同加成
        fused_score = min(self.SYNERGY_CONFIG["cap"], base_fused_score * synergy_bonus)
        
        logger.info(
            f"[V2] 融合计算 -> 场景权重: T{scene_weights['text']:.1f}/V{scene_weights['vision']:.1f}/A{scene_weights['audio']:.1f} | "
            f"协同加成: {synergy_bonus:.2f}x | 基础分: {base_fused_score:.1f} | 加成后: {fused_score:.1f}"
        )
        
        # === P2: 时序滑动窗口平滑 ===
        if call_id:
            final_score = self._apply_temporal_smoothing(call_id, fused_score)
        else:
            final_score = fused_score
        
        return round(final_score, 2)
    
    def clear_temporal_cache(self, call_id: Optional[str] = None):
        """清理时序缓存（通话结束时调用）"""
        if call_id:
            self._temporal_cache.pop(call_id, None)
            logger.debug(f"清理时序缓存: {call_id}")
        else:
            self._temporal_cache.clear()
            logger.debug("清理所有时序缓存")


# 全局实例（保持向后兼容）
fusion_engine = RiskFusionEngine()
fusion_engine_v2 = RiskFusionEngineV2()