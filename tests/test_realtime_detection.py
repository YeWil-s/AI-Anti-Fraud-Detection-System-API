"""
实时检测功能测试脚本
"""
import asyncio
import websockets
import json
import base64
import httpx
import numpy as np


async def test_websocket():
    """测试WebSocket连接和实时检测（带鉴权）"""
    print("=== 测试WebSocket实时检测 ===")
    
    # 1. 先登录获取Token
    async with httpx.AsyncClient() as client:
        login_response = await client.post(
            "http://localhost:8000/api/users/login",
            json={
                "phone": "13800138000",  # 确保数据库中有此用户
                "password": "123456"
            }
        )
        
        if login_response.status_code != 200:
            print(f"✗ 登录失败: {login_response.text}")
            return
            
        data = login_response.json()
        token = data["access_token"]
        user_id = data["user"]["user_id"]
        print(f"✓ 登录成功，User ID: {user_id}")

    # 2. 拼接带Token的WebSocket URL
    uri = f"ws://localhost:8000/api/detection/ws/{user_id}?token={token}"
    
    try:
        # 添加额外headers以模拟真实环境
        extra_headers = {
            "Authorization": f"Bearer {token}"
        }
        
        async with websockets.connect(uri, extra_headers=extra_headers) as websocket:
            print(f"✓ WebSocket连接成功: {uri}")
            
            # 发送心跳测试
            await websocket.send(json.dumps({"type": "heartbeat"}))
            response = await websocket.recv()
            print(f"✓ 心跳响应: {response}")
            
            # ... 后续发送音频/视频数据的测试保持不变 ...
            
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"✗ 连接被拒绝: 状态码 {e.status_code}")
    except Exception as e:
        print(f"✗ WebSocket测试失败: {e}")


async def test_file_upload():
    """测试文件上传功能"""
    print("\n=== 测试文件上传 ===")
    
    # 先登录获取token
    async with httpx.AsyncClient() as client:
        # 登录
        login_response = await client.post(
            "http://localhost:8000/api/users/login",
            json={
                "phone": "13800138000",
                "password": "123456"
            }
        )
        
        if login_response.status_code != 200:
            print("✗ 需要先登录")
            return
        
        token = login_response.json()["access_token"]
        print(f"✓ 登录成功,获取token")
        
        # 创建模拟音频文件
        fake_audio = b"fake audio content for testing"
        
        # 测试音频上传
        files = {"file": ("test_audio.mp3", fake_audio, "audio/mpeg")}
        headers = {"Authorization": f"Bearer {token}"}
        
        try:
            upload_response = await client.post(
                "http://localhost:8000/api/detection/upload/audio",
                files=files,
                headers=headers
            )
            print(f"✓ 音频上传状态: {upload_response.status_code}")
            if upload_response.status_code == 200:
                print(f"  响应: {upload_response.json()}")
        except Exception as e:
            print(f"✗ 音频上传失败: {e}")


async def test_async_tasks():
    """测试异步任务"""
    print("\n=== 测试Celery异步任务 ===")
    
    async with httpx.AsyncClient() as client:
        # 登录
        login_response = await client.post(
            "http://localhost:8000/api/users/login",
            json={
                "phone": "13800138000",
                "password": "123456"
            }
        )
        
        if login_response.status_code != 200:
            print("✗ 需要先登录")
            return
        
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # 提交音频检测任务
        try:
            task_response = await client.post(
                "http://localhost:8000/api/tasks/audio/detect",
                json={
                    "audio_features": [[0.1, 0.2, 0.3] for _ in range(10)],
                    "call_id": 1
                },
                headers=headers
            )
            
            if task_response.status_code == 200:
                task_data = task_response.json()
                print(f"✓ 任务提交成功")
                print(f"  任务ID: {task_data['data']['task_id']}")
                
                # 查询任务状态
                task_id = task_data['data']['task_id']
                await asyncio.sleep(2)  # 等待任务执行
                
                status_response = await client.get(
                    f"http://localhost:8000/api/tasks/status/{task_id}"
                )
                
                if status_response.status_code == 200:
                    print(f"✓ 任务状态查询成功")
                    print(f"  状态: {status_response.json()}")
            else:
                print(f"✗ 任务提交失败: {task_response.text}")
                
        except Exception as e:
            print(f"⚠ 异步任务测试失败(可能Celery未启动): {e}")


async def test_api_docs():
    """测试API文档是否可访问"""
    print("\n=== 测试API文档 ===")
    
    async with httpx.AsyncClient() as client:
        # 测试Swagger UI
        docs_response = await client.get("http://localhost:8000/docs")
        if docs_response.status_code == 200:
            print(f"✓ Swagger UI可访问: http://localhost:8000/docs")
        else:
            print(f"✗ Swagger UI不可访问")
        
        # 测试ReDoc
        redoc_response = await client.get("http://localhost:8000/redoc")
        if redoc_response.status_code == 200:
            print(f"✓ ReDoc可访问: http://localhost:8000/redoc")
        else:
            print(f"✗ ReDoc不可访问")


async def main():
    """主测试函数"""
    print("=" * 60)
    print("  实时检测服务功能测试")
    print("=" * 60)
    
    # 测试API文档
    await test_api_docs()
    
    # 测试WebSocket
    await test_websocket()
    
    # 测试文件上传
    await test_file_upload()
    
    # 测试异步任务
    await test_async_tasks()
    
    print("\n" + "=" * 60)
    print("  测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
