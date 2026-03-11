import requests
import json

# 请根据你实际运行的服务地址修改
BASE_URL = "http://localhost:8000" 

def test_generate_report():
    print("1. 正在尝试登录...")
    login_url = f"{BASE_URL}/api/users/login"
    login_payload = {
        "phone": "13800138000",
        "password": "123456"
    }
    
    try:
        # 登录请求不需要太长超时
        login_response = requests.post(login_url, json=login_payload, timeout=10)
        
        if login_response.status_code != 200:
            print(f"❌ 登录失败: {login_response.text}")
            return
            
        login_data = login_response.json()
        token = login_data.get("access_token")
        # 从返回的用户信息中获取 user_id
        user_id = login_data.get("user", {}).get("user_id")
        
        if not user_id:
            print("❌ 获取 user_id 失败")
            return
            
        print(f"✅ 登录成功！获取到 user_id: {user_id}")
        
    except Exception as e:
        print(f"❌ 登录请求发生异常: {e}")
        return

    # ---------------------------------------------------------
    
    print("\n2. 开始生成个人安全监测报告 (调用大模型中，请耐心等待...)")
    report_url = f"{BASE_URL}/api/users/{user_id}/security-report"
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    try:
        # 【关键】因为大模型生成 Markdown 需要较长时间，这里必须设置较长的超时时间 (例如 120 秒)
        report_response = requests.get(report_url, headers=headers, timeout=120)
        
        if report_response.status_code == 200:
            report_data = report_response.json()
            print("\n✅ 报告生成成功！")
            print("="*50)
            print(f"用户: {report_data.get('username')}")
            print(f"生成时间: {report_data.get('report_generated_at')}")
            print("="*50)
            print("【报告内容 Markdown】:\n")
            print(report_data.get('report_content'))
            print("="*50)
        else:
            print(f"❌ 生成报告失败，状态码: {report_response.status_code}")
            print(f"错误详情: {report_response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ 请求超时！大模型未能在 120 秒内返回结果，建议检查大模型 API 响应速度。")
    except Exception as e:
        print(f"❌ 生成报告请求发生异常: {e}")

if __name__ == "__main__":
    test_generate_report()