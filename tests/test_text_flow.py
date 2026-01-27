"""
文本诈骗检测 - 实时流对抗测试脚本
存放位置: tests/test_text_flow.py
运行方式: python tests/test_text_flow.py

⚠️ 注意: 运行此测试前，请确保:
1. 后端服务已启动 (main.py)
2. Celery Worker 已启动 (否则收不到检测结果)
"""
import asyncio
import websockets
import json
import httpx
import random

# === 配置区域 ===
BASE_URL = "http://localhost:8000"

# 定义测试任务
TEST_CASES = [
    {
        "text": "你好我是京东客服，你的金条利率过高需要注销，请下载腾讯会议屏幕共享，否则会影响征信。", 
        "description": "【高危诈骗样本】", 
        "expect": "Fraud" # 期望结果: 诈骗
    },
    {
        "text": "妈，今晚我不回家吃饭了，公司要加班，你们先吃吧，不用等我。", 
        "description": "【正常家常对话】", 
        "expect": "Normal" # 期望结果: 正常
    },
    {
        "text": "恭喜您中奖了！点击链接领取您的iPhone 15 Pro Max，名额有限，速点！", 
        "description": "【中奖诱导诈骗】", 
        "expect": "Fraud"
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

async def test_single_text(text: str, description: str, expect: str, token: str, user_id: int):
    print(f"\n{Colors.HEADER}💬 正在测试: {description} {Colors.ENDC}")
    print(f"   内容摘要: {text[:30]}...")
    
    # 建立 WebSocket 连接
    call_id = random.randint(10000, 99999)
    ws_url = f"ws://localhost:8000/api/detection/ws/{user_id}/{call_id}?token={token}"

    try:
        async with websockets.connect(ws_url) as ws:
            # 构造并发送消息
            # 根据 app/api/detection.py 的逻辑，payload 可以是字典或字符串
            message = {
                "type": "text",
                "data": {
                    "text": text
                }
            }
            
            await ws.send(json.dumps(message))
            print("   📤 文本已发送，等待 AI 判决...")

            # 等待结果 (10秒超时)
            try:
                while True:
                    res = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    msg = json.loads(res)
                    
                    # 1. 收到 ACK (确认收到)，忽略并继续等待
                    if msg.get("type") == "ack":
                        # print("   (后端已接收，正在处理...)")
                        continue

                    # 2. 收到心跳或其他消息，忽略
                    if msg.get("type") in ["heartbeat_ack", "ping"]:
                        continue

                    # === 3. 收到检测结果 ===
                    # 假设后端逻辑: risk_level 高 -> alert, risk_level 低 -> info
                    msg_type = msg.get("type")
                    
                    if msg_type == "alert":
                        # AI 判定为诈骗
                        confidence = msg.get('confidence', 0.0)
                        keywords = msg.get('details', {}).get('keywords', [])
                        print(f"   🤖 模型判定: {Colors.RED}[诈骗/FRAUD]{Colors.ENDC} (置信度: {confidence:.4f})")
                        if keywords:
                            print(f"      敏感词: {keywords}")
                        
                        if expect == "Fraud":
                            print(f"   {Colors.GREEN}✅ 测试通过！(成功拦截){Colors.ENDC}")
                        else:
                            print(f"   {Colors.RED}❌ 误报！(正常话术被拦截){Colors.ENDC}")
                        break
                    
                    elif msg_type == "info":
                        # AI 判定为正常
                        confidence = msg.get('confidence', 0.0)
                        print(f"   🤖 模型判定: {Colors.GREEN}[正常/NORMAL]{Colors.ENDC} (置信度: {confidence:.4f})")
                        
                        if expect == "Normal":
                            print(f"   {Colors.GREEN}✅ 测试通过！(正确放行){Colors.ENDC}")
                        else:
                            print(f"   {Colors.RED}❌ 漏报！(诈骗话术未识别){Colors.ENDC}")
                        break

            except asyncio.TimeoutError:
                print(f"   {Colors.RED}⚠️ 测试超时 (Celery可能未启动或处理过慢){Colors.ENDC}")

    except Exception as e:
        print(f"   {Colors.RED}❌ 连接错误: {e}{Colors.ENDC}")

async def main():
    print(f"{Colors.BOLD}🚀 开始【文本反诈】对抗测试{Colors.ENDC}")
    
    # 1. 登录获取 Token (使用默认测试账号)
    async with httpx.AsyncClient() as client:
        try:
            # 确保这里使用你数据库中存在的账号
            login_data = {"phone": "13800138000", "password": "123456"}
            resp = await client.post(f"{BASE_URL}/api/users/login", json=login_data)
            
            if resp.status_code != 200:
                print(f"{Colors.RED}登录失败: {resp.text}{Colors.ENDC}")
                print("请检查数据库中是否存在该用户，或修改脚本中的账号密码。")
                return
                
            data = resp.json()
            token = data["access_token"]
            user_id = data["user"]["user_id"]
            print(f"🔑 登录成功，User ID: {user_id}")
            
        except Exception as e:
            print(f"{Colors.RED}无法连接后端，请确保 main.py 已启动: {e}{Colors.ENDC}")
            return

    # 2. 遍历测试用例
    for case in TEST_CASES:
        await test_single_text(
            case["text"], 
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