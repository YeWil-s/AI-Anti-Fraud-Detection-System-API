"""
Alembic迁移脚本模板
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = '74f93f5c5832'
down_revision = '617a43371be9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. 创建 family_groups 表
    op.create_table(
        'family_groups',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('group_name', sa.String(length=100), nullable=False, comment="家庭组名称"),
        sa.Column('owner_id', sa.Integer(), nullable=False, comment="创建者ID"),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_family_groups_id'), 'family_groups', ['id'], unique=False)

    # 2. 为 users 表增加 is_admin 字段
    # 注意：为了兼容现有数据，我们将默认值设为 False
    op.add_column('users', sa.Column('is_admin', sa.Boolean(), server_default=sa.text('0'), nullable=True))
    
    # 3. 修改 users 表的 family_id 为外键关联
    # 如果你使用的是 SQLite，建议使用 batch_alter_table 以保证兼容性
    with op.batch_alter_table('users') as batch_op:
        batch_op.create_foreign_key(
            'fk_user_family', 
            'family_groups', 
            ['family_id'], 
            ['id']
        )


def downgrade() -> None:
    with op.batch_alter_table('users') as batch_op:
        batch_op.drop_constraint('fk_user_family', type_='foreignkey')
        batch_op.drop_column('is_admin')
    
    op.drop_index(op.f('ix_family_groups_id'), table_name='family_groups')
    op.drop_table('family_groups')
