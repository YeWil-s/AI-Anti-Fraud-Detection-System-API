"""
添加family_admins表，删除users表的admin_role和guardian_phone

Revision ID: g8h9i0j1k2l3
Revises: 6d1e0718d995
Create Date: 2026-03-19 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = 'g8h9i0j1k2l3'
down_revision = '6d1e0718d995'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 检查family_admins表是否已存在
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    # 1. 创建family_admins表（如果不存在）
    if 'family_admins' not in inspector.get_table_names():
        op.create_table(
            'family_admins',
            sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False),
            sa.Column('family_id', sa.Integer(), sa.ForeignKey('family_groups.id', ondelete='CASCADE'), nullable=False),
            sa.Column('admin_role', sa.String(20), nullable=False, comment='管理员角色: primary/secondary'),
            sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), comment='添加时间'),
            sa.UniqueConstraint('user_id', 'family_id', name='uq_family_admin')
        )
        
        # 2. 创建索引
        op.create_index('ix_family_admins_user_id', 'family_admins', ['user_id'])
        op.create_index('ix_family_admins_family_id', 'family_admins', ['family_id'])
    
    # 3. 迁移现有管理员数据到family_admins表
    try:
        op.execute("""
            INSERT IGNORE INTO family_admins (user_id, family_id, admin_role)
            SELECT fg.admin_id, fg.id, 'primary'
            FROM family_groups fg
            WHERE NOT EXISTS (
                SELECT 1 FROM family_admins fa 
                WHERE fa.user_id = fg.admin_id AND fa.family_id = fg.id
            )
        """)
    except Exception as e:
        print(f"数据迁移跳过（可能已存在）: {e}")
    
    # 4. 删除users表的admin_role字段（如果存在）
    columns = [c['name'] for c in inspector.get_columns('users')]
    if 'admin_role' in columns:
        try:
            op.drop_index('ix_users_admin_role', table_name='users')
        except:
            pass
        op.drop_column('users', 'admin_role')
    
    # 5. 删除users表的guardian_phone字段（如果存在）
    if 'guardian_phone' in columns:
        op.drop_column('users', 'guardian_phone')


def downgrade() -> None:
    # 1. 恢复users表字段
    op.add_column('users', sa.Column('admin_role', sa.String(20), nullable=True))
    op.add_column('users', sa.Column('guardian_phone', sa.String(20), nullable=True))
    op.create_index('ix_users_admin_role', 'users', ['admin_role'])
    
    # 2. 从family_admins表恢复数据到users表（仅主管理员）
    op.execute("""
        UPDATE users u
        SET admin_role = 'primary'
        WHERE EXISTS (
            SELECT 1 FROM family_admins fa 
            WHERE fa.user_id = u.user_id AND fa.admin_role = 'primary'
        )
    """)
    
    # 3. 删除family_admins表
    op.drop_index('ix_family_admins_family_id', table_name='family_admins')
    op.drop_index('ix_family_admins_user_id', table_name='family_admins')
    op.drop_table('family_admins')
