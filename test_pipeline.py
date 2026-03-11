import time
import requests

# 配置你的本地服务地址
BASE_URL = "http://127.0.0.1:8000"
TEST_PHONE = "13800138000"
TEST_PASSWORD = "123456"

def test_post_call_audit_api():
    print(f"🚀 开始测试通话事后审计全流程 API: {BASE_URL}")
    
    # ==========================================
    # 1. 登录获取 Token
    # ==========================================
    print("\n[1/5] 正在登录系统...")
    login_url = f"{BASE_URL}/api/users/login"
    
    # 保持账户密码不变
    login_data = {"username": TEST_PHONE, "password": TEST_PASSWORD}
    
    try:
        # 【关键修改】：将 data=login_data 改为 json=login_data
        # 这样 requests 会自动设置 Content-Type 为 application/json
        resp = requests.post(login_url, json=login_data, timeout=10) 
        
        if resp.status_code != 200:
            print(f"❌ 登录失败: HTTP {resp.status_code} - {resp.text}")
            return
            
        token = resp.json().get("access_token")
        headers = {"Authorization": f"Bearer {token}"}
        print("✅ 登录成功，已获取 Token")
    except Exception as e:
        print(f"❌ 登录请求异常: {e}")
        return

    # ==========================================
    # 2. 创建通话
    # ==========================================
    print("\n[2/5] 正在模拟接听来电...")
    start_url = f"{BASE_URL}/api/call-records/start?platform=phone&target_identifier=诈骗测试分子"
    try:
        resp = requests.post(start_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            print(f"❌ 创建通话失败: {resp.text}")
            return
            
        call_id = resp.json().get("call_id")
        print(f"✅ 通话建立成功！获取到 Call ID: {call_id}")
    except Exception as e:
        print(f"❌ 创建通话异常: {e}")
        return

    # ==========================================
    # 3. 模拟通话过程中的等待（关键！）
    # ==========================================
    print("\n[3/5] ⏳ 通话进行中...")
    print("🚨 【极其重要】：为了让大模型有内容可分析，请在接下来的 15 秒内，")
    print("🚨 通过你的 App 或 WebSocket 接口向该通话发送一些测试语音/文本！")
    print("🚨 (如果完全没有任何文本，后端会为了省钱直接跳过大模型调用)")
    
    for i in range(15, 0, -1):
        print(f"剩余 {i} 秒钟挂断...", end="\r")
        time.sleep(1)
    print("\n✅ 通话时间到，准备挂机。")

    # ==========================================
    # 4. 结束通话 (触发后台大模型审计任务)
    # ==========================================
    print("\n[4/5] 正在调用挂断接口，并触发后台大模型审计...")
    end_url = f"{BASE_URL}/api/call-records/{call_id}/end"
    payload = {
        "audio_url": "http://minio/test-audio.wav",
        "video_url": "",
        "cover_image": ""
    }
    try:
        start_time = time.time()
        resp = requests.post(end_url, headers=headers, json=payload, timeout=10)
        if resp.status_code != 200:
            print(f"❌ 挂断通话失败: {resp.text}")
            return
            
        elapsed_time = time.time() - start_time
        print(f"✅ 挂机请求完成！接口耗时: {elapsed_time:.2f} 秒")
        print(f"👉 服务器反馈: {resp.json().get('message')}")
    except Exception as e:
        print(f"❌ 挂断通话异常: {e}")
        return

    # ==========================================
    # 5. 轮询获取审计结果
    # ==========================================
    print("\n[5/5] ⏳ 大模型正在后台思考，开始轮询查询结果 (每 3 秒查一次)...")
    detail_url = f"{BASE_URL}/api/call-records/record/{call_id}"
    
    max_retries = 15 # 最多等 45 秒
    for i in range(max_retries):
        try:
            resp = requests.get(detail_url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get("data", {}).get("call_record", {})
                analysis = data.get("analysis")
                advice = data.get("advice")
                
                # 如果数据库里已经有了大模型写好的评价，说明后台任务完成了！
                if analysis and advice:
                    print("\n" + "="*60)
                    print("🎉 【大模型事后审计报告生成成功！】")
                    print("="*60)
                    print(f"📌 最终风险定性: {data.get('detected_result')}")
                    print(f"🧠 LLM 分析过程:\n{analysis}\n")
                    print(f"💡 防骗处理建议:\n{advice}")
                    print("="*60)
                    return
            
            print(f"🔄 第 {i+1}/{max_retries} 次查询：报告尚未生成，继续等待...")
            time.sleep(3)
        except Exception as e:
            print(f"❌ 查询记录异常: {e}")
            time.sleep(3)
            
    print("\n❌ 轮询超时，大模型未能生成总结。")
    print("👉 诊断原因可能有：")
    print("1. 刚才在步骤 [3/5] 的 15秒等待期内，没有任何文本进入记忆池（chat_history为空）。")
    print("2. LLM 的网络连接超时，请检查后端运行终端里是否有报错。")

if __name__ == "__main__":
    test_post_call_audit_api()