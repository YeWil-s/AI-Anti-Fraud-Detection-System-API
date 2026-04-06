"""
图片OCR服务
使用智谱GLM-4V-Flash模型提取聊天记录截图中的文字
支持对话去重和增量识别
"""
import base64
import hashlib
import re
from typing import Optional, Dict, List, Set
import aiohttp
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class ImageOCRService:
    """图片OCR服务类"""
    
    def __init__(self):
        self.api_key = settings.ZHIPU_API_KEY
        self.base_url = settings.ZHIPU_BASE_URL
        self.model = "glm-4v-flash"  # 免费的多模态模型
    
    def _encode_image(self, image_bytes: bytes) -> str:
        """将图片字节编码为base64"""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    async def extract_text_from_image(
        self, 
        image_bytes: bytes,
        image_type: str = "chat_screenshot"
    ) -> Optional[str]:
        """
        从图片中提取文字
        
        Args:
            image_bytes: 图片字节数据
            image_type: 图片类型，默认是聊天记录截图
            
        Returns:
            提取的文字内容，失败返回None
        """
        if not self.api_key:
            logger.warning("智谱API密钥未配置，跳过图片OCR")
            return None
        
        try:
            # 编码图片
            base64_image = self._encode_image(image_bytes)
            
            # 构建请求
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 根据图片类型构建不同的提示词
            if image_type == "chat_screenshot":
                system_prompt = "你是一个专门提取聊天记录的助手。请准确识别图片中的聊天对话内容，按时间顺序输出对话文本，保留发送者信息。只输出对话内容，不要添加解释。"
            else:
                system_prompt = "请识别图片中的文字内容，准确输出所有可见文本。"
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "请提取图片中的文字内容。"
                            }
                        ]
                    }
                ]
            }
            
            # 发送请求
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"智谱API请求失败: {response.status}, {error_text}")
                        return None
                    
                    result = await response.json()
                    
                    # 提取文字
                    if result.get("choices") and len(result["choices"]) > 0:
                        extracted_text = result["choices"][0].get("message", {}).get("content", "")
                        logger.info(f"图片OCR成功，提取文字长度: {len(extracted_text)}")
                        return extracted_text.strip()
                    else:
                        logger.warning("智谱API返回结果为空")
                        return None
                        
        except Exception as e:
            logger.error(f"图片OCR处理失败: {e}")
            return None
    
    async def _post_vision_chat(self, base64_image: str, system_prompt: str, user_text: str) -> Optional[str]:
        """调用智谱多模态 chat/completions，返回 assistant 文本内容。"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"智谱API请求失败: {response.status}, {error_text}")
                    return None
                result = await response.json()
                if result.get("choices") and len(result["choices"]) > 0:
                    return result["choices"][0].get("message", {}).get("content", "")
                logger.warning("智谱API返回结果为空")
                return None

    @staticmethod
    def _parse_json_object(content: str) -> Optional[dict]:
        import json

        try:
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    async def classify_screenshot_environment(self, image_bytes: bytes) -> Optional[dict]:
        """
        第一步：仅识别平台、号码/昵称与沟通方式（文字/语音/视频/电话），不提取对话正文。
        """
        if not self.api_key:
            logger.warning("智谱API密钥未配置，跳过截图环境分类")
            return None

        base64_image = self._encode_image(image_bytes)
        system_prompt = """你是通话/聊天截图分析助手。请仅根据界面判断以下信息，不要编造对话内容。

1. platform（必选）：
   - wechat: 微信界面
   - qq: QQ界面
   - phone: 系统电话拨号/来电/通话中界面
   - video_call: 视频通话全屏或悬浮窗为主
   - other: 其他或无法判断

2. communication_mode（必选）：
   - text: 纯文字聊天（消息列表、气泡为主，非语音通话中界面）
   - voice: 微信/QQ 等语音通话中界面（含通话时长、麦克风、扬声器等）
   - video: 视频通话为主
   - phone: 传统电话通话界面
   - unknown: 无法判断

3. caller_number / target_name：可见则填写，否则空字符串。

请只输出 JSON：
{
  "platform": "wechat|qq|phone|video_call|other",
  "communication_mode": "text|voice|video|phone|unknown",
  "caller_number": "",
  "target_name": ""
}"""

        try:
            content = await self._post_vision_chat(
                base64_image,
                system_prompt,
                "请分析这张截图的平台与沟通方式，只输出 JSON。",
            )
            if not content:
                return None
            parsed = self._parse_json_object(content)
            if not parsed:
                logger.warning(f"环境分类 JSON 解析失败: {content[:200]}")
                return None

            platform = (parsed.get("platform") or "other").lower()
            mode = (parsed.get("communication_mode") or "unknown").lower()
            if mode not in ("text", "voice", "video", "phone", "unknown"):
                mode = "unknown"

            is_text_chat = mode == "text"

            logger.info(
                f"截图环境分类: platform={platform}, communication_mode={mode}, is_text_chat={is_text_chat}"
            )

            return {
                "environment": {
                    "platform": platform,
                    "caller_number": parsed.get("caller_number") or "",
                    "target_name": parsed.get("target_name") or "",
                    "communication_mode": mode,
                },
                "is_text_chat": is_text_chat,
                "should_extract_dialogue": is_text_chat,
            }
        except Exception as e:
            logger.error(f"截图环境分类失败: {e}")
            return None

    async def extract_chat_from_screenshot(self, image_bytes: bytes) -> Optional[dict]:
        """
        第二步：仅提取聊天记录中的文字与结构化对话（在已判定为文字交流后调用）。

        Returns:
            {
                "text": "按行标准格式，供去重与文本检测（每行 user: … / other: …）",
                "dialogue": [
                    {"speaker": "user"|"other", "content": "单条消息"},
                    ...
                ],
            }
        """
        if not self.api_key:
            logger.warning("智谱API密钥未配置，跳过图片OCR")
            return None
        
        try:
            base64_image = self._encode_image(image_bytes)
            
            # 分条 + 左右气泡 -> user/other；与后端 ChatMessage.speaker、增量解析一致
            system_prompt = """你是聊天记录截图 OCR 助手。请按截图中消息从上到下的时间顺序，逐条输出每一条气泡内的完整文字。

【说话人判定（必遵守）】
- 根据消息气泡在页面中的左右位置判断角色（适用于微信/QQ 等常见布局）：
  - 气泡主体在屏幕右侧、或明显属于「自己发出的消息」→ speaker_role 为 "user"（本机用户）。
  - 气泡主体在屏幕左侧、或明显属于「对方发来的消息」→ speaker_role 为 "other"（对方）。
- 同时填写 side：气泡整体更靠左填 "left"，更靠右填 "right"；若无法判断 side，仅依据左右习惯推断 speaker_role 即可。
- 不要合并多条消息；一条气泡对应 messages 中一项。
- 不要编造截图中不存在的文字；看不清的 content 可为空字符串并仍保留该条结构。

【输出格式】只输出一个 JSON 对象，不要 Markdown、不要解释：
{
  "messages": [
    {
      "side": "left|right",
      "speaker_role": "user|other",
      "content": "该条气泡内完整文本"
    }
  ]
}

若图中无任何聊天文字，输出：{"messages": []}"""

            content = await self._post_vision_chat(
                base64_image,
                system_prompt,
                "请逐条提取截图中的聊天内容，只输出 JSON。",
            )
            if not content:
                return None

            try:
                parsed = self._parse_json_object(content)
                if not parsed:
                    raise ValueError("no json object")
                messages = parsed.get("messages")
                if not isinstance(messages, list):
                    # 兼容旧版 dialogue_text
                    dt = (parsed.get("dialogue_text") or "").strip()
                    if dt:
                        return self._fallback_dialogue_from_flat_text(dt)
                    messages = []

                dialogue: List[Dict[str, str]] = []
                lines_out: List[str] = []
                for item in messages:
                    if not isinstance(item, dict):
                        continue
                    raw_content = (item.get("content") or "").strip()
                    if not raw_content:
                        continue
                    side = (item.get("side") or "").lower()
                    role = (item.get("speaker_role") or "").lower()
                    if role not in ("user", "other"):
                        if side == "right":
                            role = "user"
                        elif side == "left":
                            role = "other"
                        else:
                            role = "other"
                    dialogue.append({"speaker": role, "content": raw_content[:2000]})
                    lines_out.append(f"{role}: {raw_content[:2000]}")

                text = "\n".join(lines_out)
                logger.info(
                    f"聊天文字提取成功: {len(dialogue)} 条, 总长度: {len(text)}"
                )
                return {"text": text, "dialogue": dialogue}

            except (ValueError, TypeError) as e:
                logger.warning(f"结构化 JSON 解析失败，回退为纯文本: {e}")
                dialogue_text = content.strip()
                return self._fallback_dialogue_from_flat_text(dialogue_text)

        except Exception as e:
            logger.error(f"图片OCR处理失败: {e}")
            return None

    def _fallback_dialogue_from_flat_text(self, dialogue_text: str) -> dict:
        """旧版 dialogue_text 或模型未按 JSON 输出时的回退。"""
        dialogue_text = (dialogue_text or "").strip()
        dialogue: List[Dict[str, str]] = []
        if dialogue_text:
            for line in dialogue_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if ":" in line or "：" in line:
                    parts = line.replace("：", ":").split(":", 1)
                    if len(parts) == 2:
                        left = parts[0].strip().lower()
                        right = parts[1].strip()
                        sp = "other"
                        if left in ("user", "other"):
                            sp = left
                        dialogue.append({"speaker": sp, "content": right[:2000]})
                    else:
                        dialogue.append({"speaker": "other", "content": line[:2000]})
                else:
                    dialogue.append({"speaker": "other", "content": line[:2000]})
        lines_out = [f"{d['speaker']}: {d['content']}" for d in dialogue]
        text = "\n".join(lines_out)
        return {"text": text, "dialogue": dialogue}

    async def extract_call_environment(self, image_bytes: bytes) -> Optional[dict]:
        """
        专门提取截图中的通话环境信息（平台、号码、昵称）
        """
        result = await self.classify_screenshot_environment(image_bytes)
        if result:
            return result.get("environment")
        return None


    def calculate_dialogue_hash(self, text: str) -> str:
        """
        计算对话内容的哈希值，用于去重
        使用对话内容的简化版本（去除空格、标点）计算哈希
        """
        # 简化文本：去除空格、换行、标点，转为小写
        simplified = re.sub(r'[\s\n\r，。！？、：""''（）【】]', '', text)
        simplified = simplified.lower()
        return hashlib.sha256(simplified.encode('utf-8')).hexdigest()
    
    def extract_new_dialogues(
        self, 
        current_text: str, 
        previous_texts: List[str],
        similarity_threshold: float = 0.8
    ) -> Dict:
        """
        智能识别新增对话内容
        
        Args:
            current_text: 当前OCR提取的完整对话
            previous_texts: 之前提取的对话列表
            similarity_threshold: 相似度阈值，超过则认为重复
            
        Returns:
            {
                "is_duplicate": False,  # 是否完全重复
                "new_content": "新增内容",
                "all_content": "完整内容",
                "new_lines": ["新增行1", "新增行2"],
                "similarity": 0.5  # 与最近一条的相似度
            }
        """
        if not current_text:
            return {"is_duplicate": True, "new_content": "", "all_content": "", "new_lines": [], "similarity": 1.0}
        
        current_hash = self.calculate_dialogue_hash(current_text)
        current_lines = [line.strip() for line in current_text.split('\n') if line.strip()]
        
        # 如果没有历史记录，全部是新内容
        if not previous_texts:
            return {
                "is_duplicate": False,
                "new_content": current_text,
                "all_content": current_text,
                "new_lines": current_lines,
                "similarity": 0.0
            }
        
        # 获取最近的对话
        latest_text = previous_texts[-1]
        latest_hash = self.calculate_dialogue_hash(latest_text)
        
        # 计算相似度（简化版本）
        latest_lines = [line.strip() for line in latest_text.split('\n') if line.strip()]
        
        # 计算共同行数
        common_lines = set(current_lines) & set(latest_lines)
        total_unique_lines = set(current_lines) | set(latest_lines)
        
        if total_unique_lines:
            similarity = len(common_lines) / len(total_unique_lines)
        else:
            similarity = 1.0 if not current_lines and not latest_lines else 0.0
        
        # 如果相似度超过阈值，认为是重复
        if similarity >= similarity_threshold:
            return {
                "is_duplicate": True,
                "new_content": "",
                "all_content": current_text,
                "new_lines": [],
                "similarity": similarity
            }
        
        # 找出新增的行
        new_lines = []
        for line in current_lines:
            if line not in latest_lines:
                new_lines.append(line)
        
        # 如果没有新增行但相似度不高，可能是顺序变化
        if not new_lines and similarity < similarity_threshold:
            new_lines = current_lines[len(latest_lines):] if len(current_lines) > len(latest_lines) else []
        
        new_content = '\n'.join(new_lines) if new_lines else current_text
        
        return {
            "is_duplicate": False,
            "new_content": new_content,
            "all_content": current_text,
            "new_lines": new_lines,
            "similarity": similarity
        }


# 全局单例
image_ocr_service = ImageOCRService()
