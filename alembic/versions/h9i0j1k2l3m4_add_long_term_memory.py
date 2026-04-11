"""
添加长期记忆功能：user_memories表和users.memory_summary字段

Revision ID: h9i0j1k2l3m4
Revises: g8h9i0j1k2l3
Create Date: 2026-03-19 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

revision = 'h9i0j1k2l3m4'
down_revision = 'g8h9i0j1k2l3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. 添加 users.memory_summary 字段
    op.add_column('users',
        sa.Column('memory_summary', sa.Text(), nullable=True, comment='长期记忆摘要（由系统自动生成）')
    )
    
    # 2. 创建 user_memories 表
    op.create_table(
        'user_memories',
        sa.Column('memory_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('memory_type', sa.String(30), nullable=False, comment='记忆类型: fraud_experience/alert_response/preference/risk_pattern'),
        sa.Column('content', sa.Text(), nullable=False, comment='记忆内容'),
        sa.Column('importance', sa.Integer(), default=3, comment='重要性 1-5'),
        sa.Column('source_call_id', sa.Integer(), sa.ForeignKey('call_records.call_id', ondelete='SET NULL'), nullable=True, comment='来源通话ID'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now(), comment='更新时间'),
        sa.Index('ix_user_memories_user_type', 'user_id', 'memory_type')
    )
    
    # 3. 创建索引
    op.create_index('ix_user_memories_importance', 'user_memories', ['importance'])


def downgrade() -> None:
    # 1. 删除 user_memories 表
    op.drop_index('ix_user_memories_importance', table_name='user_memories')
    op.drop_index('ix_user_memories_user_type', table_name='user_memories')
    op.drop_table('user_memories')
    
    # 2. 删除 users.memory_summary 字段
    op.drop_column('users', 'memory_summary')
