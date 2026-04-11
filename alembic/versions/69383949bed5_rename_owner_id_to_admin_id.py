"""
Alembic迁移脚本模板
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = '69383949bed5'
down_revision = '64ce3837fdec'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        'family_groups',           # 表名
        'owner_id',                # 旧列名
        new_column_name='admin_id',# 新列名
        existing_type=sa.Integer(),
        existing_comment="创建者ID",
        existing_nullable=False
    )


def downgrade() -> None:
    # 回滚操作：将 admin_id 重新改回 owner_id
    op.alter_column(
        'family_groups', 
        'admin_id', 
        new_column_name='owner_id',
        existing_type=sa.Integer(),
        existing_comment="创建者ID",
        existing_nullable=False
    )
