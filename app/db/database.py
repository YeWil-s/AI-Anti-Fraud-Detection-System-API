"""
数据库连接配置 
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool 
from app.core.config import settings
from app.core.logger import get_logger
import sys
# 初始化模块级 logger
logger = get_logger(__name__)

IS_CELERY = "celery" in sys.argv[0] or "celery" in sys.modules
# 针对不同环境应用不同的连接池策略
if IS_CELERY:
    logger.info("运行Celery: 使用 NullPool.")
    pool_kwargs = {"poolclass": NullPool}
else:
    logger.info("运行在 Web 环境: 使用标准连接池.")
    pool_kwargs = {"pool_pre_ping": True, "pool_size": 20, "max_overflow": 20}

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    **pool_kwargs
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False, autocommit=False, autoflush=False
)
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