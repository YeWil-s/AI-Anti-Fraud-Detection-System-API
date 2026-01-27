"""
视频流处理器
"""
import base64
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from typing import Dict, List, Optional, Union, Tuple
import asyncio
from app.core.logger import get_logger
from app.core.config import settings

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
        # 存储每个用户的视频帧缓冲 (存储的是 cropped face uint8 numpy array)
        self.frame_buffers: Dict[int, deque] = {}
        
        # 初始化 MediaPipe Face Mesh
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
        处理单个视频帧: 解码 -> 人脸裁剪 -> 存入缓冲 -> (满) -> 返回Base64列表
        """
        try:
            # 1. 解码 Base64 -> Image
            frame_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.warning(f"Invalid frame data received (User: {user_id})")
                return {"status": "error", "message": "Invalid frame data"}
            
            # 2. 仅提取人脸 (不缩放，不归一化，保持原始 uint8 格式以节省内存)
            # 这一步是为了只保留核心数据，去除无关背景
            face_img = self._extract_face_image(img)
            
            if face_img is None:
                return {
                    "status": "skip", 
                    "message": "No face detected",
                    "face_detected": False
                }

            # 3. 维护缓冲区
            if user_id not in self.frame_buffers:
                self.frame_buffers[user_id] = deque(maxlen=self.sequence_length)
            
            self.frame_buffers[user_id].append(face_img)

            # 4. 判断是否满足推理条件
            current_len = len(self.frame_buffers[user_id])
            
            if current_len == self.sequence_length:
                # [优化关键] 缓冲区满，将 10 张人脸图片编码为 JPG Base64 列表
                # 这样传给 Celery/Redis 的数据量极小 (从 6MB -> 200KB)
                face_batch_base64 = []
                for face in self.frame_buffers[user_id]:
                    # Resize 到配置大小再传输，进一步减少带宽
                    # 虽然 Celery 也会检查 Resize，但在这里先做可以保证传输的一致性
                    face_resized = cv2.resize(face, settings.VIDEO_INPUT_SIZE)
                    
                    # 编码为 JPG
                    _, buffer = cv2.imencode('.jpg', face_resized)
                    face_batch_base64.append(base64.b64encode(buffer).decode('utf-8'))
                
                return {
                    "status": "ready",
                    "celery_payload": face_batch_base64, # [修改] 传递压缩后的数据列表，而不是 Tensor
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

    def _extract_face_image(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        辅助方法: 仅使用 MediaPipe 裁剪人脸，返回 uint8 图像
        """
        h, w, _ = img.shape
        # MediaPipe 需要 RGB 输入
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(rgb_img)
        
        if not results.multi_face_landmarks:
            return None

        # 获取关键点
        landmarks = results.multi_face_landmarks[0].landmark
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        
        x1, x2 = int(min(x_coords) * w), int(max(x_coords) * w)
        y1, y2 = int(min(y_coords) * h), int(max(y_coords) * h)
        
        # Padding
        pad_w = int((x2 - x1) * 0.2)
        pad_h = int((y2 - y1) * 0.2)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        # Crop (注意: 这里返回的是原图 BGR 格式的切片，方便后续 cv2 处理)
        face_img = img[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return None
            
        return face_img

    @staticmethod
    def preprocess_batch(face_list_base64: List[str]) -> np.ndarray:
        """
        [Celery端专用] 将 Base64 图片列表转换为模型需要的 Tensor
        
        步骤: Decode -> Resize -> Normalize -> Transpose -> Stack
        这样做的好处是将计算密集型操作放在 Worker 节点，减少 API 节点负载和 Redis 传输延迟。
        """
        processed_faces = []
        
        # 获取配置中的均值和方差
        mean = np.array(settings.VIDEO_NORM_MEAN, dtype=np.float32)
        std = np.array(settings.VIDEO_NORM_STD, dtype=np.float32)
        
        for b64_str in face_list_base64:
            # 1. Decode
            img_bytes = base64.b64decode(b64_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # BGR
            
            # 2. Resize (再次确保尺寸正确)
            face_resized = cv2.resize(face_img, settings.VIDEO_INPUT_SIZE)
            
            # 3. Normalize (标准化)
            # BGR -> RGB (模型通常是 RGB 训练的)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # 归一化到 0-1
            face_norm = face_rgb.astype(np.float32) / 255.0
            
            # 减均值除方差
            face_norm = (face_norm - mean) / std
            
            # 4. Transpose (HWC -> CHW)
            face_chw = np.transpose(face_norm, (2, 0, 1))
            processed_faces.append(face_chw)
            
        # 5. Stack -> (Batch=10, 3, 224, 224)
        seq_array = np.array(processed_faces)
        
        # 6. Add Batch Dim -> (1, 10, 3, 224, 224)
        # 这就是可以直接送入 ONNX 的 Tensor
        return np.expand_dims(seq_array, axis=0).astype(np.float32)

    # --- 兼容性方法 ---
    
    async def extract_frames(self, video_bytes: bytes, frame_rate: int = 1) -> List[str]:
        """从视频文件中提取关键帧"""
        frames = []
        try:
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
                    _, buffer = cv2.imencode('.jpg', frame)
                    b64_str = base64.b64encode(buffer).decode('utf-8')
                    frames.append(b64_str)
                count += 1
            
            cap.release()
            os.remove(tmp_path)
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}", exc_info=True)
            return []
    
    def get_buffered_frames(self, user_id: int) -> Optional[List[np.ndarray]]:
        if user_id in self.frame_buffers:
            return list(self.frame_buffers[user_id])
        return None
        
    def clear_buffer(self, user_id: int):
        if user_id in self.frame_buffers:
            del self.frame_buffers[user_id]