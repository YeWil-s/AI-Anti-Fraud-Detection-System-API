import asyncio
import websockets
import json
import base64
import httpx
import random
import wave
import io

# === [升级] 生成合法的 WAV 音频 ===
def create_valid_wav(duration_sec=2):
    """生成一段标准的静音/白噪 WAV 数据"""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)      # 单声道
        wav_file.setsampwidth(2)      # 16位
        wav_file.setframerate(16000)  # 16kHz
        # 生成数据 (这里全是0，相当于静音)
        data = b'\x00' * 16000 * 2 * duration_sec 
        wav_file.writeframes(data)
    return buffer.getvalue()

async def run_test():
    base_url = "http://localhost:8000"
    print("🚀 开始测试音频检测全流程 (v2.0)...")

    # 1. 登录
    async with httpx.AsyncClient() as client:
        # 请确保账号密码正确
        login_payload = {"phone": "13800138000", "password": "123456"}
        try:
            resp = await client.post(f"{base_url}/api/users/login", json=login_payload)
            if resp.status_code != 200:
                print(f"❌ 登录失败: {resp.text}")
                return
            token = resp.json()["access_token"]
            user_id = resp.json()["user"]["user_id"]
            print(f"✅ 登录成功! UserID: {user_id}")
        except Exception as e:
            print(f"❌ 连接后端失败: {e}")
            return

    # 2. 连接 WS
    call_id = random.randint(1000, 9999)
    ws_url = f"ws://localhost:8000/api/detection/ws/{user_id}/{call_id}?token={token}"

    async with websockets.connect(ws_url) as ws:
        print(f"✅ WebSocket 连接建立 (CallID: {call_id})")

        # 3. 发送数据
        print("📤 生成并发送合法 WAV 音频...")
        wav_data = create_valid_wav()
        audio_b64 = base64.b64encode(wav_data).decode()
        
        await ws.send(json.dumps({
            "type": "audio",
            "data": audio_b64
        }))

        # 4. 接收结果
        print("⏳ 等待结果 (请确保 Celery 已重启)...")
        while True:
            try:
                res = await asyncio.wait_for(ws.recv(), timeout=15.0)
                msg = json.loads(res)
                
                if msg.get("type") == "ack":
                    print("📩 [ACK] 服务器已确认接收")
                
                elif msg.get("type") == "alert":
                    print(f"⚠️ [ALERT] 发现风险: {msg['message']}")
                    break
                
                elif msg.get("type") == "info":
                    print(f"✅ [INFO] 检测通过: {msg['message']}")
                    print(f"   置信度: {msg.get('confidence')}")
                    break
                    
            except asyncio.TimeoutError:
                print("❌ 等待超时。请检查是否修改了 Celery 代码并重启了 Worker。")
                break

if __name__ == "__main__":
    asyncio.run(run_test())