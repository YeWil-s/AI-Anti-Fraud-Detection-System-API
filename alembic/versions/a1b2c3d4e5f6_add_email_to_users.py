"""
添加邮箱字段到用户表

Revision ID: a1b2c3d4e5f6
Revises: 6aa208783993
Create Date: 2026-03-17
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'a1b2c3d4e5f6'
down_revision = '6aa208783993'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 添加 email 列(允许NULL)
    op.add_column('users', sa.Column('email', sa.String(length=100), nullable=True, comment='邮箱'))
    
    # 创建唯一索引
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)


def downgrade() -> None:
    # 删除索引和列
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_column('users', 'email')
