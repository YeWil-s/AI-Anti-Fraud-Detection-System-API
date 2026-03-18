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
                ]
            }
        """
        text = await self.extract_text_from_image(image_bytes, "chat_screenshot")
        
        if not text:
            return None
        
        # 简单解析对话结构（可以根据需要进一步优化）
        dialogue = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 尝试识别发送者:内容的格式
            if ':' in line or '：' in line:
                parts = line.replace('：', ':').split(':', 1)
                if len(parts) == 2:
                    speaker = parts[0].strip()
                    content = parts[1].strip()
                    dialogue.append({"speaker": speaker, "content": content})
                else:
                    dialogue.append({"speaker": "未知", "content": line})
            else:
                dialogue.append({"speaker": "未知", "content": line})
        
        return {
            "text": text,
            "dialogue": dialogue
        }


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
