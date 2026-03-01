"""
Alembic迁移脚本模板
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = '6aa208783993'
down_revision = ('5f0f690bd683', '74f93f5c5832')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
