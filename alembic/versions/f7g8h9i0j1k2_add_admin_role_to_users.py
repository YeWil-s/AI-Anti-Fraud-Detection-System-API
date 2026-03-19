"""
新增多级管理员机制：User表添加admin_role字段

Revision ID: f7g8h9i0j1k2
Revises: e5f6g7h8i9j0
Create Date: 2026-03-19 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'f7g8h9i0j1k2'
down_revision = 'e5f6g7h8i9j0'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. 添加admin_role字段到users表
    op.add_column('users',
        sa.Column('admin_role', sa.String(20), nullable=True, 
                  comment='管理员角色: None/secondary/primary')
    )
    
    # 2. 迁移现有数据：is_admin=True的设为primary
    op.execute("""
        UPDATE users SET admin_role = 'primary' WHERE is_admin = 1
    """)
    
    # 3. 添加索引
    op.create_index('ix_users_admin_role', 'users', ['admin_role'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_users_admin_role', table_name='users')
    op.drop_column('users', 'admin_role')
