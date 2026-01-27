import pytest
from fastapi.testclient import TestClient
from main import app
from app.core.security import create_access_token

client = TestClient(app)

def test_websocket_connection_success():
    """测试携带正确Token连接成功"""
    user_id = 1
    token = create_access_token(data={"sub": str(user_id)})
    
    # 连接WebSocket
    with client.websocket_connect(f"/api/detection/ws/{user_id}?token={token}") as websocket:
        # 发送心跳验证连接可用
        websocket.send_json({"type": "heartbeat"})
        data = websocket.receive_json()
        assert data["type"] == "heartbeat_ack"

def test_websocket_no_token():
    """测试不带Token连接失败"""
    user_id = 1
    # 缺少 token 参数
    with pytest.raises(Exception) as excinfo:
        with client.websocket_connect(f"/api/detection/ws/{user_id}") as websocket:
            pass
    
    # 获取异常信息的字符串形式
    error_msg = str(excinfo.value)
    
    # 修正：检查 WebSocket 关闭代码 1008 或具体的验证错误信息 "Field required"
    assert "1008" in error_msg or "Field required" in error_msg

def test_websocket_invalid_token():
    """测试无效Token连接失败"""
    user_id = 1
    invalid_token = "invalid_token_string"
    
    with pytest.raises(Exception) as excinfo:
        with client.websocket_connect(f"/api/detection/ws/{user_id}?token={invalid_token}") as websocket:
            pass
            
    error_msg = str(excinfo.value)
    # 鉴权失败通常也是关闭连接，状态码可能是 1008
    assert "1008" in error_msg or "403" in error_msg

def test_websocket_user_mismatch():
    """测试Token用户ID与路径不匹配"""
    token_user_id = 1
    target_user_id = 999  # 试图连接其他用户的频道
    
    token = create_access_token(data={"sub": str(token_user_id)})
    
    with pytest.raises(Exception) as excinfo:
        with client.websocket_connect(f"/api/detection/ws/{target_user_id}?token={token}") as websocket:
            pass
            
    error_msg = str(excinfo.value)
    assert "1008" in error_msg or "403" in error_msg