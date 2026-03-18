"""add_image_ocr_fields_to_detection_log

Revision ID: b2c3d4e5f6g7
Revises: 6ae724be076f
Create Date: 2026-03-17 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6g7'
down_revision: Union[str, None] = '6ae724be076f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """添加图片OCR相关字段到 ai_detection_logs 表"""
    # 添加 detection_type 字段
    op.add_column(
        'ai_detection_logs',
        sa.Column('detection_type', sa.String(length=20), nullable=True, server_default='text', comment='检测类型: text/audio/video/image')
    )
    
    # 添加 image_ocr_text 字段
    op.add_column(
        'ai_detection_logs',
        sa.Column('image_ocr_text', sa.Text(), nullable=True, comment='图片OCR提取的完整文字')
    )
    
    # 添加 ocr_dialogue_hash 字段
    op.add_column(
        'ai_detection_logs',
        sa.Column('ocr_dialogue_hash', sa.String(length=64), nullable=True, comment='对话内容哈希，用于去重')
    )
    
    # 为 detection_type 添加索引
    op.create_index(
        'idx_detection_logs_type',
        'ai_detection_logs',
        ['detection_type']
    )
    
    # 为 ocr_dialogue_hash 添加索引
    op.create_index(
        'idx_detection_logs_hash',
        'ai_detection_logs',
        ['ocr_dialogue_hash']
    )
    
    # 为 call_id + detection_type 添加复合索引（用于查询最近OCR记录）
    op.create_index(
        'idx_detection_logs_call_type',
        'ai_detection_logs',
        ['call_id', 'detection_type', 'created_at']
    )


def downgrade() -> None:
    """回滚：删除添加的字段"""
    # 删除索引
    op.drop_index('idx_detection_logs_call_type', table_name='ai_detection_logs')
    op.drop_index('idx_detection_logs_hash', table_name='ai_detection_logs')
    op.drop_index('idx_detection_logs_type', table_name='ai_detection_logs')
    
    # 删除字段
    op.drop_column('ai_detection_logs', 'ocr_dialogue_hash')
    op.drop_column('ai_detection_logs', 'image_ocr_text')
    op.drop_column('ai_detection_logs', 'detection_type')
