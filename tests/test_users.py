"""
用户模块单元测试
"""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import NullPool

from app.db.database import Base, get_db
from app.models.user import User
from main import app

# 创建测试数据库
SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///./test.db"
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=NullPool
)
TestingSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def override_get_db():
    """覆盖数据库依赖"""
    async with TestingSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# 覆盖依赖
app.dependency_overrides[get_db] = override_get_db


@pytest_asyncio.fixture(scope="function")
async def setup_database():
    """每个测试前创建表,测试后删除"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client():
    """创建异步测试客户端"""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_check(client):
    """测试健康检查接口"""
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_root(client):
    """测试根路径"""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_register_success(client, setup_database):
    """测试用户注册成功"""
    from unittest.mock import patch
    
    with patch("app.api.users.verify_sms_code", return_value=True):
        response = await client.post(
            "/api/users/register",
            json={
                "phone": "13800138000",
                "username": "testuser",
                "name": "测试用户",
                "password": "123456",
                "sms_code": "123456"
            }
        )
        assert response.status_code == 201
        data = response.json()
        assert data["code"] == 201
        assert data["message"] == "注册成功"
        assert "user_id" in data["data"]


@pytest.mark.asyncio
async def test_register_duplicate_phone(client, setup_database):
    """测试重复手机号注册"""
    from unittest.mock import patch
    
    with patch("app.api.users.verify_sms_code", return_value=True):
        await client.post(
            "/api/users/register",
            json={
                "phone": "13800138001",
                "username": "user1",
                "name": "用户1",
                "password": "123456",
                "sms_code": "123456"
            }
        )
        
        response = await client.post(
            "/api/users/register",
            json={
                "phone": "13800138001",
                "username": "user2",
                "name": "用户2",
                "password": "654321",
                "sms_code": "123456"
            }
        )
        assert response.status_code == 400
        assert "该手机号已注册" in response.json()["detail"]


@pytest.mark.asyncio
async def test_login_success(client, setup_database):
    """测试登录成功"""
    from unittest.mock import patch
    
    with patch("app.api.users.verify_sms_code", return_value=True):
        await client.post(
            "/api/users/register",
            json={
                "phone": "13800138002",
                "username": "logintest",
                "name": "登录测试",
                "password": "123456",
                "sms_code": "123456"
            }
        )
    
    response = await client.post(
        "/api/users/login",
        json={
            "phone": "13800138002",
            "password": "123456"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "user" in data
    assert data["user"]["phone"] == "13800138002"


@pytest.mark.asyncio
async def test_login_wrong_password(client, setup_database):
    """测试密码错误"""
    from unittest.mock import patch
    
    with patch("app.api.users.verify_sms_code", return_value=True):
        await client.post(
            "/api/users/register",
            json={
                "phone": "13800138003",
                "username": "testuser3",
                "name": "测试",
                "password": "123456",
                "sms_code": "123456"
            }
        )
    
    response = await client.post(
        "/api/users/login",
        json={
            "phone": "13800138003",
            "password": "wrong_password"
        }
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user(client, setup_database):
    """测试获取当前用户信息"""
    from unittest.mock import patch
    
    with patch("app.api.users.verify_sms_code", return_value=True):
        await client.post(
            "/api/users/register",
            json={
                "phone": "13800138004",
                "username": "testuser4",
                "name": "测试用户",
                "password": "123456",
                "sms_code": "123456"
            }
        )
    
    login_response = await client.post(
        "/api/users/login",
        json={
            "phone": "13800138004",
            "password": "123456"
        }
    )
    token = login_response.json()["access_token"]
    
    response = await client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["phone"] == "13800138004"
    assert data["username"] == "testuser4"
    assert data["name"] == "测试用户"


@pytest.mark.asyncio
async def test_bind_family(client, setup_database):
    """测试绑定家庭组"""
    from unittest.mock import patch
    
    with patch("app.api.users.verify_sms_code", return_value=True):
        await client.post(
            "/api/users/register",
            json={
                "phone": "13800138005",
                "username": "testuser5",
                "name": "测试",
                "password": "123456",
                "sms_code": "123456"
            }
        )
    
    login_response = await client.post(
        "/api/users/login",
        json={"phone": "13800138005", "password": "123456"}
    )
    token = login_response.json()["access_token"]
    
    response = await client.put(
        "/api/users/family/1",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["family_id"] == 1


@pytest.mark.asyncio
async def test_unbind_family(client, setup_database):
    """测试解绑家庭组"""
    from unittest.mock import patch
    
    with patch("app.api.users.verify_sms_code", return_value=True):
        await client.post(
            "/api/users/register",
            json={
                "phone": "13800138006",
                "username": "testuser6",
                "name": "测试",
                "password": "123456",
                "sms_code": "123456"
            }
        )
    
    login_response = await client.post(
        "/api/users/login",
        json={"phone": "13800138006", "password": "123456"}
    )
    token = login_response.json()["access_token"]
    
    await client.put(
        "/api/users/family/1",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    response = await client.delete(
        "/api/users/family",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "解绑成功"
