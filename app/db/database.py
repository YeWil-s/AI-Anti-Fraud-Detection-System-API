"""
数据库连接配置 - 修复版
解决 Celery 异步任务中 "Event loop is closed" 问题
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool  # [新增] 导入 NullPool
from app.core.config import settings
from app.core.logger import get_logger

# 初始化模块级 logger
logger = get_logger(__name__)

# [修改] 创建异步引擎
# 关键修改：使用 poolclass=NullPool 禁用连接池
# 原因：Celery 每个任务会创建独立的 asyncio loop，连接池中的连接如果绑定了旧 loop 会导致 "Event loop is closed" 错误。
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    poolclass=NullPool,  # [核心修复] 禁用连接池，每次请求创建新连接
    # pool_pre_ping=True, # NullPool 不需要预检测
    # pool_size=10,       # NullPool 不使用这些参数
    # max_overflow=20
)

# 创建异步会话工厂
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

# 创建基类
Base = declarative_base()


async def get_db():
    """获取数据库会话"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            # 发生异常回滚前，记录具体的数据库错误堆栈
            logger.error(f"Database transaction failed, rolling back: {e}", exc_info=True)
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """初始化数据库"""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schema initialized successfully")
    except Exception as e:
        logger.critical(f"Database initialization failed: {e}", exc_info=True)
        raise