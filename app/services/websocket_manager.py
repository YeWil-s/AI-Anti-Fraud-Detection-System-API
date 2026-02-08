"""
WebSocket连接管理器
"""
from fastapi import WebSocket
from typing import Dict, List
import asyncio
from datetime import datetime
from app.core.redis import set_user_preference
# [新增] 导入日志工厂
from app.core.logger import get_logger

# [新增] 初始化模块级 logger
logger = get_logger(__name__)


class ConnectionManager:
    """管理所有WebSocket连接"""
    
    def __init__(self):
        # 存储活跃连接 {user_id: websocket}
        self.active_connections: Dict[int, WebSocket] = {}
        # 连接时间记录
        self.connection_times: Dict[int, datetime] = {}
        # 记录每个用户的当前防御等级 (默认 0)
        self.user_levels: Dict[int, int] = {}
    
    async def connect(self, websocket: WebSocket, user_id: int):
        """接受新连接"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.connection_times[user_id] = datetime.now()
        # 初始防御等级为 Level 0 (安全/待机)
        self.user_levels[user_id] = 0

        # 记录当前在线人数，这是非常关键的运维指标
        logger.info(f"User {user_id} connected. Total connections: {len(self.active_connections)}")

   
    def disconnect(self, user_id: int):
        """断开连接"""
        if user_id in self.user_levels:
            del self.user_levels[user_id]
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.connection_times:
            del self.connection_times[user_id]
            
        # [修改] print -> logger.info
        logger.info(f"User {user_id} disconnected. Total connections: {len(self.active_connections)}")

    # 设置防御等级并同步给前端   
    async def set_defense_level(self, user_id: int, level: int, config: dict = None):
        """
        供后端逻辑调用：变更防御等级 -> 下发控制指令 -> 改变前端采集策略
        """
        # 1. 更新服务端状态
        self.user_levels[user_id] = level
        
        # 2. 如果用户在线，下发指令
        if user_id in self.active_connections:
            # 构造同步消息
            message = {
                "type": "level_sync",
                "level": level,  # 0, 1, 2
                "config": config or {}, # 包含 fps, sensitive 等配置
                "timestamp": datetime.now().isoformat()
            }
            try:
                await self.send_personal_message(message, user_id)
                logger.info(f"🛡️ Defense Level Upgraded: User {user_id} -> Level {level}")
            except Exception as e:
                logger.error(f"Failed to sync level to user {user_id}: {e}")

    async def send_personal_message(self, message: dict, user_id: int):
        """发送个人消息"""
        if user_id in self.active_connections:
            websocket = self.active_connections[user_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send personal message to {user_id}: {e}", exc_info=True)
    
    async def broadcast(self, message: dict, exclude_user: int = None):
        """广播消息给所有连接(可排除某个用户)"""
        for user_id, websocket in self.active_connections.items():
            if exclude_user and user_id == exclude_user:
                continue
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send broadcast to user {user_id}: {e}", exc_info=True)
    
    async def send_to_family(self, message: dict, family_id: int, family_members: List[int]):
        """发送消息给家庭组成员"""
        for user_id in family_members:
            await self.send_personal_message(message, user_id)

    async def handle_command(self, user_id: int, command_data: dict):
        """处理控制指令"""
        action = command_data.get("action")
        
        if action == "set_config":
            # 例如: {"action": "set_config", "fps": 5, "sensitivity": 0.8}
            fps = command_data.get("fps")
            if fps:
                await set_user_preference(user_id, "fps", str(fps))
                logger.info(f"User {user_id} set FPS to {fps}")
                
            sensitivity = command_data.get("sensitivity")
            if sensitivity:
                await set_user_preference(user_id, "sensitivity", str(sensitivity))
                
            # 可以回执给前端
            await self.send_personal_message(
                {"type": "ack", "msg": "Config updated", "config": command_data},
                user_id
            )
            
        elif action == "pause_detection":
            # 暂停/恢复检测逻辑 (配合 redis 标记位)
            await set_user_preference(user_id, "status", "paused")

    def get_active_users(self) -> List[int]:
        """获取所有在线用户ID"""
        return list(self.active_connections.keys())
    
    def is_user_online(self, user_id: int) -> bool:
        """检查用户是否在线"""
        return user_id in self.active_connections
    
    async def heartbeat_check(self, interval: int = 30):
        """
        心跳检测
        定期检查连接状态并清理失效连接
        """
        logger.info(f"Starting heartbeat check (Interval: {interval}s)")
        while True:
            await asyncio.sleep(interval)
            disconnected_users = []
            
            # 使用 list() 复制 keys，避免在迭代时修改字典
            current_users = list(self.active_connections.items())
            
            for user_id, websocket in current_users:
                try:
                    # 发送心跳ping
                    await websocket.send_json({
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception:
                    # 连接已断开，加入清理列表
                    # [可选] 这里不需要 log error，因为心跳检测的目的就是发现断连
                    logger.debug(f"Heartbeat failed for user {user_id}, marking for cleanup")
                    disconnected_users.append(user_id)
            
            # 清理断开的连接
            if disconnected_users:
                logger.info(f"Heartbeat cleanup: Removing {len(disconnected_users)} dead connections")
                for user_id in disconnected_users:
                    self.disconnect(user_id)


# 全局连接管理器实例
connection_manager = ConnectionManager()