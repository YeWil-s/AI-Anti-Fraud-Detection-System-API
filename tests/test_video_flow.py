import asyncio
import websockets
import json
import cv2
import base64
import httpx
import time

# === 配置 ===
VIDEO_PATH = "./assets/test_fake.mp4"  # 你的测试视频路径
API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"
PHONE = "13800138000"        # 确保数据库里有这个用户
PASSWORD = "123456"

# 全局/局部变量存储最终检测结果
detection_result = None  # 存储最终检测结论
total_frames = 0         # 视频总帧数
sent_frames = 0          # 已发送帧数

async def login():
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(f"{API_URL}/api/users/login", json={
                "phone": PHONE, "password": PASSWORD
            })
            if resp.status_code == 200:
                data = resp.json()
                print("✅ 登录成功")  # 精简登录输出
                return data["access_token"], data["user"]["user_id"]
            else:
                print(f"❌ 登录失败: {resp.text}")
                return None, None
        except Exception as e:
            print(f"❌ 连接API失败: {e}")
            return None, None

async def send_video_stream(token, user_id):
    global detection_result, total_frames, sent_frames
    call_id = int(time.time())
    uri = f"{WS_URL}/api/detection/ws/{user_id}/{call_id}?token={token}"
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎬 开始测试视频: {VIDEO_PATH} (共 {total_frames} 帧)")
    
    async with websockets.connect(uri) as ws:
        # 启动接收任务（仅收集结果，不实时打印）
        receive_task = asyncio.create_task(receive_messages(ws))
        
        # 逐帧发送
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 读不到帧直接退出循环
            
            # 编码发送
            _, buffer = cv2.imencode('.jpg', frame)
            b64_frame = base64.b64encode(buffer).decode('utf-8')
            
            await ws.send(json.dumps({
                "type": "video",
                "data": b64_frame
            }))
            
            sent_frames += 1
            # 仅保留进度条输出
            print(f"\r📤 发送进度: {sent_frames}/{total_frames}", end="", flush=True)
            
            # 控制发送速度（30fps左右，可根据需要调整）
            await asyncio.sleep(0.03) 
        
        # 发送完毕后，等待服务器返回最终结果
        print("\n⏳ 等待服务器处理最终结果...")
        await asyncio.sleep(5)
        
        # 关闭接收任务并清理
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass
        
    cap.release()
    
    # 最后统一输出最终结果
    print("\n" + "="*50)
    print("🏁 测试完成 | 最终结果")
    print(f"📽️  视频总帧数: {total_frames}")
    print(f"📤 实际发送帧数: {sent_frames}")
    print(f"🔍 检测结论: {detection_result if detection_result else '未收到检测结果'}")
    print("="*50)

async def receive_messages(ws):
    """仅收集检测结果，不实时打印调试信息"""
    global detection_result
    try:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            
            msg_type = data.get("type")
            # 只记录关键结果，不打印每条消息
            if msg_type == "alert":
                detection_result = "⚠️  检测到Deepfake伪造视频"
            elif msg_type == "info":
                detection_result = "✅ 视频为真实视频，未检测到伪造"
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        detection_result = f"❌ 接收结果出错: {str(e)}"

if __name__ == "__main__":
    # 初始化结果变量
    detection_result = None
    total_frames = 0
    sent_frames = 0
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    token, user_id = loop.run_until_complete(login())
    if token:
        loop.run_until_complete(send_video_stream(token, user_id))