import cv2
import base64
import json
import asyncio
import websockets
import time

async def simulate_video_stream():
    # 测试配置
    video_path = "tests/assets/fake_audio_text about .mp4"
    user_id = 3
    call_id = 9999
    # 注意：你的 WS 接口需要 JWT token 进行鉴权，请填写一个有效的 token
    token = "YOUR_VALID_JWT_TOKEN" 
    
    uri = f"ws://127.0.0.1:8000/api/detection/ws/{user_id}/{call_id}?token={token}"
    
    print(f"=== 🎬 开始模拟视频流检测 ===")
    print(f"连接到: {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket 连接成功！开始读取视频...")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ 无法打开视频文件: {video_path}")
                return

            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 调整帧率或跳帧（比如每 5 帧发一次，模拟前端真实的采样率）
                frame_count += 1
                if frame_count % 5 != 0:
                    continue
                    
                # 将帧压缩编码为 JPEG
                # 降低分辨率可减少网络传输负担，视你的 VideoProcessor 需求而定
                frame_resized = cv2.resize(frame, (640, 480))
                _, buffer = cv2.imencode('.jpg', frame_resized)
                
                # 转为 base64 字符串
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                # 某些处理逻辑可能需要 data URI 前缀，如 "data:image/jpeg;base64,..."
                
                # 构建 WebSocket payload
                message = {
                    "type": "video",
                    "data": frame_base64
                }
                
                # 发送视频帧
                await websocket.send(json.dumps(message))
                print(f"⬆️ 已发送第 {frame_count} 帧视频数据")
                
                # 接收服务器的 ack 回执
                response = await websocket.recv()
                res_data = json.loads(response)
                
                if res_data.get("status") == "ready":
                    print(f"🚨 [触发检测] 后端已积攒 10 帧，Celery 检测任务已下发！回执: {res_data}")
                else:
                    print(f"⬇️ 收到回执: {res_data.get('status')}...")
                
                # 控制发送速率（模拟 30fps 视频的真实时间流逝）
                await asyncio.sleep(0.03)
                
            cap.release()
            print("✅ 视频读取并发送完毕。")
            
            # 保持连接一段时间，等待接收 Redis 转发回来的 fraud_alerts 报警消息
            print("⏳ 等待接收后端 AI 的报警结果...")
            for _ in range(10):
                try:
                    alert = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    print(f"🔥 [收到报警/通知]: {alert}")
                except asyncio.TimeoutError:
                    pass

    except Exception as e:
        print(f"WebSocket 错误: {e}")

if __name__ == "__main__":
    asyncio.run(simulate_video_stream())