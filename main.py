"""
FastAPI主应用入口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.db.database import init_db
from app.core.redis import close_redis
from app.api import users_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    print("启动应用...")
    await init_db()
    print("数据库初始化完成")
    
    yield
    
    # 关闭时执行
    print("关闭应用...")
    await close_redis()
    print("Redis连接已关闭")


# 创建FastAPI应用实例
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI伪造检测与诈骗预警系统后端API",
    lifespan=lifespan
)

# CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应设置具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
# 用户路由
app.include_router(users_router)


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "AI Anti-Fraud Detection System API",
        "version": settings.APP_VERSION,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",          # 指定运行的应用：main.py文件中的app实例
        host="127.0.0.1",          # 监听所有网络
        port=8000,               # 服务端口（客户端通过8000端口访问）
        reload=settings.DEBUG    # 开发模式下热重载
    )
