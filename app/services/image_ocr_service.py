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
    
    async def extract_chat_from_screenshot(self, image_bytes: bytes) -> Optional[dict]:
        """
        专门处理聊天记录截图，提取结构化对话
        
        Returns:
            {
                "text": "提取的完整对话文本",
                "dialogue": [
                    {"speaker": "发送者", "content": "消息内容"},
                    ...
                ],
                "environment": {
                    "platform": "wechat/qq/phone/other",
                    "caller_number": "来电号码",
                    "target_name": "对方昵称"
                }
            }
        """
        if not self.api_key:
            logger.warning("智谱API密钥未配置，跳过图片OCR")
            return None
        
        try:
            base64_image = self._encode_image(image_bytes)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 增强版提示词：同时提取对话内容和通话环境信息
            system_prompt = """你是一个专门分析聊天截图的助手。请从图片中提取以下信息：

1. 对话内容：按时间顺序提取所有对话，保留发送者信息
2. 平台识别：识别这是哪个平台的截图
   - wechat: 微信聊天界面
   - qq: QQ聊天界面  
   - phone: 电话通话界面（显示来电号码）
   - video_call: 视频通话界面
   - other: 其他平台
3. 号码/昵称：识别对方的电话号码或昵称

请按以下JSON格式输出：
{
  "platform": "wechat/qq/phone/video_call/other",
  "caller_number": "电话号码（如果是电话）",
  "target_name": "对方昵称或名称",
  "dialogue_text": "完整的对话文本"
}

只输出JSON，不要添加其他解释。"""
            
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
                                "text": "请分析这张截图，提取平台信息、号码/昵称和对话内容。"
                            }
                        ]
                    }
                ]
            }
            
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
                    
                    if result.get("choices") and len(result["choices"]) > 0:
                        content = result["choices"][0].get("message", {}).get("content", "")
                        
                        # 尝试解析JSON
                        import json
                        try:
                            # 提取JSON部分（可能被markdown包裹）
                            json_match = re.search(r'\{[\s\S]*\}', content)
                            if json_match:
                                parsed = json.loads(json_match.group())
                            else:
                                parsed = json.loads(content)
                            
                            logger.info(f"图片OCR成功: platform={parsed.get('platform')}, caller={parsed.get('caller_number') or parsed.get('target_name')}")
                            
                            # 解析对话结构
                            dialogue = []
                            dialogue_text = parsed.get("dialogue_text", "")
                            if dialogue_text:
                                for line in dialogue_text.split('\n'):
                                    line = line.strip()
                                    if not line:
                                        continue
                                    if ':' in line or '：' in line:
                                        parts = line.replace('：', ':').split(':', 1)
                                        if len(parts) == 2:
                                            dialogue.append({"speaker": parts[0].strip(), "content": parts[1].strip()})
                                        else:
                                            dialogue.append({"speaker": "未知", "content": line})
                                    else:
                                        dialogue.append({"speaker": "未知", "content": line})
                            
                            return {
                                "text": dialogue_text,
                                "dialogue": dialogue,
                                "environment": {
                                    "platform": parsed.get("platform", "other"),
                                    "caller_number": parsed.get("caller_number", ""),
                                    "target_name": parsed.get("target_name", "")
                                }
                            }
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON解析失败，使用原始文本: {e}")
                            # 回退到原始文本
                            return {
                                "text": content,
                                "dialogue": [],
                                "environment": {
                                    "platform": "other",
                                    "caller_number": "",
                                    "target_name": ""
                                }
                            }
                    else:
                        logger.warning("智谱API返回结果为空")
                        return None
                        
        except Exception as e:
            logger.error(f"图片OCR处理失败: {e}")
            return None

    async def extract_call_environment(self, image_bytes: bytes) -> Optional[dict]:
        """
        专门提取截图中的通话环境信息（平台、号码、昵称）
        
        Returns:
            {
                "platform": "wechat/qq/phone/video_call/other",
                "caller_number": "电话号码",
                "target_name": "对方昵称"
            }
        """
        result = await self.extract_chat_from_screenshot(image_bytes)
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
