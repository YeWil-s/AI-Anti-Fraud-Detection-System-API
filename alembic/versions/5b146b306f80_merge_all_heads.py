"""
Alembic迁移脚本模板
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = '5b146b306f80'
down_revision = ('2e76a6d68123', 'a1b2c3d4e5f6', 'b2c3d4e5f6g7', 'c3d4e5f6g7h8')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
