"""
真人 vs 克隆人 - 对抗测试脚本
存放位置: tests/test_voice_cloning.py
运行方式: python tests/test_voice_cloning.py
"""
import asyncio
import websockets
import json
import base64
import httpx
import os
import random
from pathlib import Path

# === 配置区域 ===
BASE_URL = "http://127.0.0.1:8000"
# 自动定位到 tests/assets 目录
CURRENT_DIR = Path(__file__).parent
ASSETS_DIR = CURRENT_DIR / "assets"

# 定义测试任务
TEST_CASES = [
    {
        "filename": "real_me (2).wav", 
        "description": "【真人母带】", 
        "expect": "Real"
    },
    {
        "filename": "fake_me (2).wav", 
        "description": "【AI克隆】", 
        "expect": "Fake"
    }
]

# 颜色代码 (让输出更漂亮)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

async def test_single_file(file_path: Path, description: str, expect: str, token: str, user_id: int):
    print(f"\n{Colors.HEADER}🎧 正在测试: {description} {Colors.ENDC}")
    print(f"   文件路径: {file_path.name}")
    
    if not file_path.exists():
        print(f"   {Colors.RED}❌ 错误: 文件不存在! 请将音频放入 tests/assets/ 目录{Colors.ENDC}")
        return

    # 读取并转码
    with open(file_path, "rb") as f:
        file_content = f.read()
    audio_b64 = base64.b64encode(file_content).decode()

    # 建立 WebSocket 连接
    call_id = random.randint(10000, 99999)
    ws_url = f"ws://localhost:8000/api/detection/ws/{user_id}/{call_id}?token={token}"

    try:
        async with websockets.connect(ws_url) as ws:
            # 发送数据
            await ws.send(json.dumps({
                "type": "audio",
                "data": audio_b64
            }))
            print("   📤 数据已发送，等待 AI 判决...")

            # 等待结果 (15秒超时)
            try:
                while True:
                    res = await asyncio.wait_for(ws.recv(), timeout=15.0)
                    msg = json.loads(res)
                    
                    # 收到 ACK 忽略，继续等结果
                    if msg.get("type") == "ack":
                        continue

                    # === 收到检测结果 ===
                    if msg.get("type") == "alert":
                        # AI 判定为假
                        confidence = msg.get('confidence', 0.0)
                        print(f"   🤖 模型判定: {Colors.RED}[伪造/FAKE]{Colors.ENDC} (置信度: {confidence:.4f})")
                        
                        if expect == "Fake":
                            print(f"   {Colors.GREEN}✅ 识别正确！(成功抓住了克隆人){Colors.ENDC}")
                        else:
                            print(f"   {Colors.RED}❌ 误报！(真人被冤枉了){Colors.ENDC}")
                        break
                    
                    elif msg.get("type") == "info":
                        # AI 判定为真
                        confidence = msg.get('confidence', 0.0)
                        print(f"   🤖 模型判定: {Colors.GREEN}[真人/REAL]{Colors.ENDC} (置信度: {confidence:.4f})")
                        
                        if expect == "Real":
                            print(f"   {Colors.GREEN}✅ 识别正确！(通过验证){Colors.ENDC}")
                        else:
                            print(f"   {Colors.RED}❌ 漏报！(克隆人混进去了){Colors.ENDC}")
                        break

            except asyncio.TimeoutError:
                print(f"   {Colors.RED}⚠️ 测试超时 (Celery可能没反应){Colors.ENDC}")

    except Exception as e:
        print(f"   {Colors.RED}❌ 连接错误: {e}{Colors.ENDC}")

async def main():
    print(f"{Colors.BOLD}🚀 开始【真人 vs 克隆人】对抗测试{Colors.ENDC}")
    print(f"📂 资源目录: {ASSETS_DIR}")
    
    # 1. 登录获取 Token
    async with httpx.AsyncClient(trust_env=False) as client:
        try:
            # 确保这里使用你数据库中存在的账号
            resp = await client.post(f"{BASE_URL}/api/users/login", 
                                   json={"phone": "13800138000", "password": "123456"})
            if resp.status_code != 200:
                print(f"{Colors.RED}登录失败: {resp.text}{Colors.ENDC}")
                return
            data = resp.json()
            token = data["access_token"]
            user_id = data["user"]["user_id"]
        except Exception as e:
            print(f"{Colors.RED}无法连接后端，请确保 main.py 已启动: {e}{Colors.ENDC}")
            return

    # 2. 遍历测试用例
    for case in TEST_CASES:
        file_path = ASSETS_DIR / case["filename"]
        await test_single_file(
            file_path, 
            case["description"], 
            case["expect"], 
            token, 
            user_id
        )
        await asyncio.sleep(1) # 稍作停顿

    print(f"\n{Colors.BOLD}🏁 测试结束{Colors.ENDC}")

if __name__ == "__main__":
    # Windows 下防止 asyncio 报错
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass