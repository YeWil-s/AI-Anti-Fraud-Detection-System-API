"""
新增chat_messages表，规范化ai_detection_logs字段语义

Revision ID: e5f6g7h8i9j0
Revises: 617a43371be9
Create Date: 2026-03-19 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers
revision = 'e5f6g7h8i9j0'
down_revision = '617a43371be9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands ###
    
    # 1. 创建chat_messages表 - 存储对话历史
    op.create_table('chat_messages',
        sa.Column('message_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('call_id', sa.Integer(), sa.ForeignKey('call_records.call_id', ondelete='CASCADE'), nullable=False, comment='对应通话ID'),
        sa.Column('sequence', sa.Integer(), nullable=False, comment='消息序号'),
        sa.Column('speaker', sa.String(length=20), nullable=False, comment='说话人: user/other'),
        sa.Column('content', sa.Text(), nullable=False, comment='消息内容'),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True, comment='消息时间'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('message_id')
    )
    op.create_index(op.f('ix_chat_messages_call_id'), 'chat_messages', ['call_id'], unique=False)
    op.create_index(op.f('ix_chat_messages_sequence'), 'chat_messages', ['call_id', 'sequence'], unique=False)
    
    # 2. 为ai_detection_logs添加新字段
    op.add_column('ai_detection_logs', 
        sa.Column('detected_text', sa.Text(), nullable=True, comment='检测到的完整文本内容')
    )
    op.add_column('ai_detection_logs',
        sa.Column('match_script', sa.String(length=100), nullable=True, comment='匹配的诈骗剧本')
    )
    op.add_column('ai_detection_logs',
        sa.Column('intent', sa.String(length=100), nullable=True, comment='识别的用户意图')
    )
    
    # 3. 修改detected_keywords字段注释（保持数据，仅规范语义）
    # 使用alter_column修改注释
    op.alter_column('ai_detection_logs', 'detected_keywords',
        existing_type=sa.Text(),
        comment='检测到的敏感关键词，如: 转账,验证码,屏幕共享',
        existing_nullable=True
    )
    
    # 4. 修改algorithm_details字段注释
    op.alter_column('ai_detection_logs', 'algorithm_details',
        existing_type=sa.Text(),
        comment='技术细节JSON，如: {"face_swap": 0.9, "lip_sync": 0.8}',
        existing_nullable=True
    )
    
    # 5. 为call_records添加match_script字段（通话级别的剧本匹配）
    op.add_column('call_records',
        sa.Column('match_script', sa.String(length=100), nullable=True, comment='匹配的主要诈骗剧本')
    )
    
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands ###
    
    # 删除call_records的新字段
    op.drop_column('call_records', 'match_script')
    
    # 恢复ai_detection_logs字段注释（无法真正恢复，仅删除新字段）
    op.drop_column('ai_detection_logs', 'intent')
    op.drop_column('ai_detection_logs', 'match_script')
    op.drop_column('ai_detection_logs', 'detected_text')
    
    # 删除chat_messages表
    op.drop_index(op.f('ix_chat_messages_sequence'), table_name='chat_messages')
    op.drop_index(op.f('ix_chat_messages_call_id'), table_name='chat_messages')
    op.drop_table('chat_messages')
    
    # ### end Alembic commands ###
