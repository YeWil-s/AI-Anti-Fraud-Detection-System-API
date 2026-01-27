"""
数据库连接配置
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.core.config import settings
# [新增] 导入日志
from app.core.logger import get_logger

# [新增] 初始化模块级 logger
logger = get_logger(__name__)

# 创建异步引擎
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
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
            # [修改] 发生异常回滚前，记录具体的数据库错误堆栈
            # 这对于排查 SQL 语法错误、约束冲突非常关键
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
        # [新增] 记录数据库初始化成功
        logger.info("Database schema initialized successfully")
    except Exception as e:
        # [新增] 启动时连接数据库失败通常是致命的，使用 critical 级别
        logger.critical(f"Database initialization failed: {e}", exc_info=True)
        # 这里继续抛出异常，阻止应用在数据库未就绪的情况下启动
        raise