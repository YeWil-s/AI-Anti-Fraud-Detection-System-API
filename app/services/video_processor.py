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

# 与 Xception 输入一致：传输与预处理均为 299×299 JPG（质量）
_VIDEO_JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 92]


class VideoProcessor:
    """
    视频帧提取和处理 (Scheme A: MediaPipe + Temporal Buffer)
    """
    
    def __init__(self, sequence_length: int = settings.VIDEO_SEQUENCE_LENGTH):
        """
        初始化视频处理器
        Args:
            sequence_length: LSTM 时序长度，默认 10 帧作为一个检测单元
        """
        self.sequence_length = sequence_length
        # 存储每个用户的视频帧缓冲 (存储的是 cropped face uint8 numpy array)
        self.frame_buffers: Dict[int, deque] = {}
        # 观测：按用户统计 process_frame 状态与连续 skip
        self.status_counters: Dict[int, Dict[str, int]] = {}
        self.consecutive_skip_counts: Dict[int, int] = {}
        
        # MediaPipe Face Mesh：稠密关键点，适合正脸裁剪；侧脸/小脸/运动模糊时易漏检
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        # Face Detection：全图人脸框，作为 Mesh 失败时的回退（model_selection=1 适合远景/小脸）
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.3,
        )
    
    async def process_frame(self, frame_data: str, user_id: int) -> Dict:
        """
        处理单个视频帧: 解码 -> 人脸裁剪 -> 存入缓冲 -> (满) -> 返回Base64列表
        """
        try:
            if user_id not in self.status_counters:
                self.status_counters[user_id] = {"skip": 0, "buffering": 0, "ready": 0, "error": 0}

            # 1. 解码 Base64 -> Image
            frame_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                self.status_counters[user_id]["error"] += 1
                logger.warning(f"Invalid frame data received (User: {user_id})")
                return {"status": "error", "message": "Invalid frame data"}
            
            # 2. 仅提取人脸 (不缩放，不归一化，保持原始 uint8 格式以节省内存)
            # 这一步是为了只保留核心数据，去除无关背景
            face_img = self._extract_face_image(img)
            
            if face_img is None:
                self.status_counters[user_id]["skip"] += 1
                self.consecutive_skip_counts[user_id] = self.consecutive_skip_counts.get(user_id, 0) + 1
                skip_count = self.consecutive_skip_counts[user_id]
                # 降低噪声：前几次和每 30 次输出一次
                if skip_count <= 5 or skip_count % 30 == 0:
                    logger.info(
                        f"Video frame skipped (no face). user={user_id}, "
                        f"consecutive_skip={skip_count}, total_skip={self.status_counters[user_id]['skip']}"
                    )
                return {
                    "status": "skip", 
                    "message": "No face detected",
                    "face_detected": False
                }

            # 3. 维护缓冲区
            if user_id not in self.frame_buffers:
                self.frame_buffers[user_id] = deque(maxlen=self.sequence_length)
            
            self.frame_buffers[user_id].append(face_img)
            self.consecutive_skip_counts[user_id] = 0

            # 4. 判断是否满足推理条件
            current_len = len(self.frame_buffers[user_id])
            
            if current_len == self.sequence_length:
                self.status_counters[user_id]["ready"] += 1
                # [优化关键] 缓冲区满，将 10 张人脸图片编码为 JPG Base64 列表
                # 这样传给 Celery/Redis 的数据量极小 (从 6MB -> 200KB)
                face_batch_base64 = []
                for face in self.frame_buffers[user_id]:
                    # 训练输入 299×299，与 VIDEO_INPUT_SIZE / 模型一致
                    face_resized = cv2.resize(face, settings.VIDEO_INPUT_SIZE, interpolation=cv2.INTER_AREA)
                    _, buffer = cv2.imencode(".jpg", face_resized, _VIDEO_JPEG_PARAMS)
                    face_batch_base64.append(base64.b64encode(buffer).decode("utf-8"))

                logger.info(
                    f"Video frame status=ready, user={user_id}, "
                    f"ready_count={self.status_counters[user_id]['ready']}, seq_len={self.sequence_length}"
                )
                
                return {
                    "status": "ready",
                    "celery_payload": face_batch_base64, # [修改] 传递压缩后的数据列表，而不是 Tensor
                    "timestamp": asyncio.get_event_loop().time(),
                    "face_detected": True,
                    "frame_shape": img.shape
                }
            
            # 缓冲区未满，继续积攒
            self.status_counters[user_id]["buffering"] += 1
            if current_len == 1 or current_len == self.sequence_length - 1:
                logger.info(
                    f"Video frame status=buffering, user={user_id}, current_len={current_len}, "
                    f"target_len={self.sequence_length}, buffering_count={self.status_counters[user_id]['buffering']}"
                )
            return {
                "status": "buffering", 
                "current_len": current_len,
                "target_len": self.sequence_length,
                "face_detected": True
            }
            
        except Exception as e:
            if user_id not in self.status_counters:
                self.status_counters[user_id] = {"skip": 0, "buffering": 0, "ready": 0, "error": 0}
            self.status_counters[user_id]["error"] += 1
            logger.error(f"Process frame failed: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }

    def _crop_from_mesh_landmarks(self, img: np.ndarray, rgb_img: np.ndarray) -> Optional[np.ndarray]:
        h, w, _ = img.shape
        results = self.face_mesh.process(rgb_img)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        x1, x2 = int(min(x_coords) * w), int(max(x_coords) * w)
        y1, y2 = int(min(y_coords) * h), int(max(y_coords) * h)
        if x2 <= x1 or y2 <= y1:
            return None
        pad_w = max(int((x2 - x1) * 0.2), 2)
        pad_h = max(int((y2 - y1) * 0.2), 2)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        face_img = img[y1:y2, x1:x2]
        if face_img.size == 0 or face_img.shape[0] < 16 or face_img.shape[1] < 16:
            return None
        return face_img

    def _crop_from_face_detection(self, img: np.ndarray, rgb_img: np.ndarray) -> Optional[np.ndarray]:
        """Face Detection 回退：对侧脸、部分遮挡、小脸更稳。"""
        h, w, _ = img.shape
        det = self.face_detection.process(rgb_img)
        if not det.detections:
            return None
        bbox = det.detections[0].location_data.relative_bounding_box
        bw = bbox.width * w
        bh = bbox.height * h
        if bw <= 0 or bh <= 0:
            return None
        xmin, ymin = bbox.xmin * w, bbox.ymin * h
        pad_x = max(bw * 0.15, 2.0)
        pad_y = max(bh * 0.15, 2.0)
        x1 = int(max(0, xmin - pad_x))
        y1 = int(max(0, ymin - pad_y))
        x2 = int(min(w, xmin + bw + pad_x))
        y2 = int(min(h, ymin + bh + pad_y))
        if x2 <= x1 or y2 <= y1:
            return None
        face_img = img[y1:y2, x1:x2]
        if face_img.size == 0 or face_img.shape[0] < 16 or face_img.shape[1] < 16:
            return None
        return face_img

    def _extract_face_image(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        MediaPipe：优先 Face Mesh 稠密框；失败则用 Face Detection 框。
        返回 BGR uint8 切片，后续统一 resize 为 299×299 JPG。
        """
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_img = self._crop_from_mesh_landmarks(img, rgb_img)
        if face_img is not None:
            return face_img

        face_img = self._crop_from_face_detection(img, rgb_img)
        if face_img is not None:
            logger.debug("Face crop: used Face Detection fallback (Mesh missed)")
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
            if face_img is None:
                logger.warning("Video preprocess: failed to decode face frame, skipping one frame")
                continue
            
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
            
        # 5. Stack -> (Seq, 3, H, W)
        if not processed_faces:
            raise ValueError("Video preprocess got no valid frames")
        seq_array = np.array(processed_faces)
        
        # 6. Add Batch Dim -> (1, Seq, 3, H, W)
        # 这就是可以直接送入 ONNX 的 Tensor
        return np.expand_dims(seq_array, axis=0).astype(np.float32)

    # --- 兼容性方法 ---
    
    async def extract_frames(self, video_bytes: bytes, frame_rate: int = 0) -> List[str]:
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
            target_fps = frame_rate if frame_rate and frame_rate > 0 else int(settings.VIDEO_TARGET_FPS)
            interval = max(1, int(round(fps / target_fps))) if fps > 0 else 1
            
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
        if user_id in self.consecutive_skip_counts:
            del self.consecutive_skip_counts[user_id]
        if user_id in self.status_counters:
            del self.status_counters[user_id]