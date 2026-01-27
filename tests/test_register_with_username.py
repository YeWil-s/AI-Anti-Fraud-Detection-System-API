"""
测试用户注册(带用户名)功能
"""
import httpx
import asyncio


async def test_register():
    """测试用户注册"""
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        print("=== 测试用户注册(带用户名和密码) ===\n")
        
        # 1. 发送验证码
        phone = "13800138000"
        #print(f"1. 发送验证码到手机号: {phone}")
        #response = await client.post(f"{base_url}/api/users/send-code?phone={phone}")
        #print(f"   状态码: {response.status_code}")
        #print(f"   响应: {response.json()}\n")
        
        # 2. 注册用户
        print("2. 注册新用户")
        register_data = {
            "phone": phone,
            "username": "testuser_1",  # 用户名
            "name": "张三",  # 真实姓名(可选)
            "password": "123456",
            "sms_code": "240417"  # 实际使用时从Redis获取
        }
        print(f"   注册数据: {register_data}")
        
        try:
            response = await client.post(
                f"{base_url}/api/users/register",
                json=register_data
            )
            print(f"   状态码: {response.status_code}")
            print(f"   响应: {response.json()}\n")
            
            if response.status_code == 201:
                print("✓ 注册成功!")
                
                # 3. 登录
                print("\n3. 使用手机号和密码登录")
                login_data = {
                    "phone": phone,
                    "password": "123456"
                }
                response = await client.post(
                    f"{base_url}/api/users/login",
                    json=login_data
                )
                print(f"   状态码: {response.status_code}")
                data = response.json()
                print(f"   响应: {data}\n")
                
                if response.status_code == 200:
                    print("✓ 登录成功!")
                    print(f"   Access Token: {data['access_token'][:50]}...")
                    print(f"   用户信息:")
                    print(f"     - 手机号: {data['user']['phone']}")
                    print(f"     - 用户名: {data['user']['username']}")
                    print(f"     - 姓名: {data['user'].get('name', '未填写')}")
                    print(f"     - 用户ID: {data['user']['user_id']}")
        
        except Exception as e:
            print(f"✗ 测试失败: {e}")
        
        # 4. 测试用户名重复
        print("\n4. 测试用户名重复注册")
        duplicate_data = {
            "phone": "13900139001",  # 不同手机号
            "username": "zhangsan",  # 相同用户名
            "name": "李四",
            "password": "654321",
            "sms_code": "123456"
        }
        response = await client.post(
            f"{base_url}/api/users/register",
            json=duplicate_data
        )
        print(f"   状态码: {response.status_code}")
        print(f"   响应: {response.json()}")
        if response.status_code == 400:
            print("✓ 正确拦截了重复的用户名!")


if __name__ == "__main__":
    print("=" * 60)
    print("  用户注册功能测试(用户名+密码)")
    print("=" * 60)
    print()
    asyncio.run(test_register())
    print("\n" + "=" * 60)
    print("  测试完成!")
    print("=" * 60)
