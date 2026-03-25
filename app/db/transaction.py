"""
统一的数据库事务上下文管理器

设计目标：
1. 提供原子性事务保证 - 要么全部成功，要么全部回滚
2. 与现有 AsyncSessionLocal 兼容
3. 支持嵌套使用（但不创建真正的嵌套事务）
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import AsyncSessionLocal
from app.core.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def managed_transaction() -> AsyncGenerator[AsyncSession, None]:
    """
    统一的异步事务上下文管理器
    
    使用方式：
        async with managed_transaction() as session:
            # 所有数据库操作都在这个事务中
            session.add(some_entity)
            session.add(another_entity)
            # 上下文结束时自动 commit
            # 如果发生异常，自动 rollback
    
    重要：外部服务调用（Redis发布、邮件发送）应在事务 commit 之后执行
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            logger.error(f"Transaction failed, rolling back: {e}", exc_info=True)
            await session.rollback()
            raise


@asynccontextmanager
async def managed_transaction_with_callbacks() -> AsyncGenerator[tuple[AsyncSession, list], None]:
    """
    带回调队列的事务上下文管理器
    
    使用方式：
        async with managed_transaction_with_callbacks() as (session, callbacks):
            # 数据库操作
            session.add(some_entity)
            
            # 注册外部服务调用（commit 后执行）
            callbacks.append(lambda: redis.publish(...))
            callbacks.append(lambda: send_email(...))
        
        # 事务已 commit，现在执行回调
        for callback in callbacks:
            try:
                await callback() if asyncio.iscoroutinefunction(callback) else callback()
            except Exception as e:
                logger.error(f"Post-commit callback failed: {e}")
    
    这种设计确保：
    - 数据库操作在同一事务中
    - 外部服务调用在事务 commit 后进行
    - 外部服务失败不会回滚数据库
    """
    callbacks = []
    async with AsyncSessionLocal() as session:
        try:
            yield session, callbacks
            await session.commit()
        except Exception as e:
            logger.error(f"Transaction failed, rolling back: {e}", exc_info=True)
            await session.rollback()
            raise


class TransactionContext:
    """
    事务上下文类 - 支持收集 commit 后执行的操作
    
    使用场景：
    - 数据库写入与外部服务（Redis/邮件）分离
    - 数据库部分先 commit，外部服务后执行
    - 外部服务失败仅记录日志，不影响数据库数据
    
    使用方式：
        ctx = TransactionContext()
        async with ctx.begin() as session:
            # 数据库操作
            session.add(entity)
            
            # 注册外部服务调用
            ctx.add_post_commit_task(
                lambda: redis.publish("channel", message)
            )
            ctx.add_post_commit_task(
                send_email, recipient, subject, body
            )
        
        # 事务已提交，执行外部服务
        await ctx.execute_post_commit_tasks()
    """
    
    def __init__(self):
        self._post_commit_tasks = []
    
    @asynccontextmanager
    async def begin(self) -> AsyncGenerator[AsyncSession, None]:
        """开始事务"""
        async with AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                logger.error(f"Transaction failed, rolling back: {e}", exc_info=True)
                await session.rollback()
                self._post_commit_tasks.clear()  # 事务失败，清空回调
                raise
    
    def add_post_commit_task(self, func, *args, **kwargs):
        """
        添加 commit 后执行的任务
        
        Args:
            func: 要执行的函数（同步或异步）
            *args, **kwargs: 传给函数的参数
        """
        self._post_commit_tasks.append((func, args, kwargs))
    
    async def execute_post_commit_tasks(self):
        """
        执行所有 commit 后的任务
        
        每个任务独立执行，单个失败不影响其他任务
        """
        import asyncio
        
        for func, args, kwargs in self._post_commit_tasks:
            try:
                if asyncio.iscoroutinefunction(func):
                    await func(*args, **kwargs)
                else:
                    func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Post-commit task failed: {func.__name__ if hasattr(func, '__name__') else func}, "
                    f"error: {e}",
                    exc_info=True
                )
        
        self._post_commit_tasks.clear()
