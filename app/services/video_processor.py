"""
视频流处理器
"""
import base64
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from typing import Dict, List, Optional, Union
import asyncio
from app.core.logger import get_logger

# 初始化模块级 logger
logger = get_logger(__name__)


class VideoProcessor:
    """
    视频帧提取和处理 (Scheme A: MediaPipe + Temporal Buffer)
    """
    
    def __init__(self, sequence_length: int = 10):
        """
        初始化视频处理器
        
        Args:
            sequence_length: LSTM 时序长度，默认 10 帧作为一个检测单元
        """
        self.sequence_length = sequence_length
        # 存储每个用户的视频帧缓冲: {user_id: deque([frame1, frame2, ...])}
        # 使用 deque (双端队列) 实现高效的滑动窗口
        self.frame_buffers: Dict[int, deque] = {}
        
        # 初始化 MediaPipe Face Mesh
        # static_image_mode=False: 视频流模式，利用上一帧加速跟踪，更稳定
        # refine_landmarks=True: 提取更精细的眼部和嘴部关键点
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    async def process_frame(self, frame_data: str, user_id: int) -> Dict:
        """
        处理单个视频帧并维护时序缓冲区
        
        Args:
            frame_data: Base64编码的帧数据
            user_id: 用户ID (用于区分不同用户的缓冲区)
            
        Returns:
            Dict: 处理结果
            - status="buffering": 正在积攒帧，无需推理
            - status="ready": 缓冲区已满，返回 input_tensor 供模型推理
            - status="error": 处理出错
        """
        try:
            # 1. 解码 Base64 -> Image
            frame_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.warning(f"Invalid frame data received (User: {user_id})")
                return {"status": "error", "message": "Invalid frame data"}
            
            # 2. 预处理单帧 (MediaPipe Crop + Resize + Normalize)
            # 返回格式: (3, 224, 224) 的 float32 numpy 数组
            processed_frame = self._preprocess_single_frame(img)
            
            if processed_frame is None:
                # 未检测到人脸，跳过此帧 (或者也可以选择清空缓冲)
                return {
                    "status": "skip", 
                    "message": "No face detected",
                    "face_detected": False
                }

            # 3. 维护用户的帧队列
            if user_id not in self.frame_buffers:
                self.frame_buffers[user_id] = deque(maxlen=self.sequence_length)
            
            self.frame_buffers[user_id].append(processed_frame)

            # 4. 判断是否满足推理条件 (队列满)
            current_len = len(self.frame_buffers[user_id])
            
            if current_len == self.sequence_length:
                # 堆叠成 (1, 10, 3, 224, 224)
                # list of (3, 224, 224) -> (10, 3, 224, 224)
                seq_array = np.array(list(self.frame_buffers[user_id]))
                
                # 增加 Batch 维度 -> (1, 10, 3, 224, 224)
                # 这就是送入 ONNX 模型的最终 Tensor
                input_tensor = np.expand_dims(seq_array, axis=0).astype(np.float32)
                
                return {
                    "status": "ready",
                    "input_tensor": input_tensor,
                    "timestamp": asyncio.get_event_loop().time(),
                    "face_detected": True,
                    "frame_shape": img.shape
                }
            
            # 缓冲区未满，继续积攒
            return {
                "status": "buffering", 
                "current_len": current_len,
                "target_len": self.sequence_length,
                "face_detected": True
            }
            
        except Exception as e:
            logger.error(f"Process frame failed: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }

    def _preprocess_single_frame(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        使用 MediaPipe 提取人脸并进行标准化预处理
        """
        h, w, _ = img.shape
        # MediaPipe 需要 RGB 输入
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(rgb_img)
        
        if not results.multi_face_landmarks:
            return None

        # 获取关键点 (Landmarks)
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 计算人脸边界框 (Bounding Box)
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        
        x1, x2 = int(min(x_coords) * w), int(max(x_coords) * w)
        y1, y2 = int(min(y_coords) * h), int(max(y_coords) * h)
        
        # [关键] 增加 20% Padding，确保不切掉下巴或额头，保留部分背景信息有助于判断
        pad_w = int((x2 - x1) * 0.2)
        pad_h = int((y2 - y1) * 0.2)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        # 裁剪人脸
        face_img = rgb_img[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return None

        # Resize 到模型输入大小 (224x224)
        face_resized = cv2.resize(face_img, (224, 224))
        
        # Normalize (0-1)
        face_norm = face_resized.astype(np.float32) / 255.0
        
        # Transpose: HWC (224, 224, 3) -> CHW (3, 224, 224)
        # 这是 PyTorch/ONNX 模型的标准输入格式
        return np.transpose(face_norm, (2, 0, 1))

    # --- 以下方法保留用于兼容旧逻辑或文件上传场景 ---
    
    async def extract_frames(self, video_bytes: bytes, frame_rate: int = 1) -> List[str]:
        """
        从视频文件中提取关键帧 (用于离线检测)
        """
        frames = []
        try:
            # 使用临时文件处理 (简化版)
            # 注意：实际生产中建议使用 tempfile 模块
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(video_bytes)
                tmp_path = tmp_file.name
            
            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            interval = int(fps / frame_rate) if fps > 0 else 30
            
            count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if count % interval == 0:
                    # 编码为 base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    b64_str = base64.b64encode(buffer).decode('utf-8')
                    frames.append(b64_str)
                count += 1
            
            cap.release()
            os.remove(tmp_path) # 清理临时文件
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}", exc_info=True)
            return []
    
    # 兼容旧接口，虽然不再使用
    def get_buffered_frames(self, user_id: int) -> Optional[List[np.ndarray]]:
        if user_id in self.frame_buffers:
            return list(self.frame_buffers[user_id])
        return None
        
    def clear_buffer(self, user_id: int):
        if user_id in self.frame_buffers:
            del self.frame_buffers[user_id]