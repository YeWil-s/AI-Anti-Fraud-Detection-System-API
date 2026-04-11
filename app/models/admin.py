"""
管理员模型
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.sql import func
from app.db.database import Base


class Admin(Base):
    """管理员表"""
    __tablename__ = "admins"
    
    admin_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, index=True, nullable=False, comment="管理员用户名")
    password_hash = Column(String(255), nullable=False, comment="密码哈希")
    email = Column(String(100), unique=True, index=True, nullable=True, comment="邮箱")
    phone = Column(String(20), nullable=True, comment="手机号")
    name = Column(String(50), nullable=True, comment="姓名")
    role = Column(String(20), default="admin", comment="角色: admin/super_admin/operator")
    is_active = Column(Boolean, default=True, comment="账号是否激活")
    last_login = Column(DateTime(timezone=True), nullable=True, comment="最后登录时间")
    login_ip = Column(String(50), nullable=True, comment="最后登录IP")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")
    
    def __repr__(self):
        return f"<Admin(admin_id={self.admin_id}, username={self.username}, role={self.role})>"


class AdminLog(Base):
    """管理员操作日志表"""
    __tablename__ = "admin_logs"
    
    log_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    admin_id = Column(Integer, nullable=True, comment="管理员ID")
    action = Column(String(50), nullable=False, comment="操作类型")
    resource = Column(String(50), nullable=True, comment="操作对象")
    resource_id = Column(Integer, nullable=True, comment="对象ID")
    details = Column(Text, nullable=True, comment="操作详情")
    ip_address = Column(String(50), nullable=True, comment="IP地址")
    user_agent = Column(String(500), nullable=True, comment="用户代理")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="操作时间")
    
    def __repr__(self):
        return f"<AdminLog(log_id={self.log_id}, action={self.action}, admin_id={self.admin_id})>"


class SystemMonitor(Base):
    """系统监控数据表 - 用于存储历史监控指标"""
    __tablename__ = "system_monitor"
    
    monitor_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    metric_name = Column(String(50), nullable=False, comment="指标名称")
    metric_value = Column(String(100), nullable=False, comment="指标值")
    metric_type = Column(String(20), default="count", comment="指标类型: count/percentage/duration")
    category = Column(String(30), default="general", comment="分类: general/detection/user/performance")
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), comment="记录时间")
    
    def __repr__(self):
        return f"<SystemMonitor(metric={self.metric_name}, value={self.metric_value})>"
