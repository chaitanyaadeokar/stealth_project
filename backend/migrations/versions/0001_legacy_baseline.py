"""Adopt the three existing tables, or create them on a fresh database."""

from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # These tables were previously created by SQLAlchemy at startup.
    existing = sa.inspect(op.get_bind()).get_table_names()
    if "podcast_sessions" not in existing:
        op.create_table(
            "podcast_sessions",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )
    for name in ("transcript_segments", "reasoning_entries"):
        if name not in existing:
            op.create_table(
                name,
                sa.Column("id", sa.Integer, primary_key=True),
                sa.Column("mic_id", sa.Integer, nullable=False),
                sa.Column("speaker", sa.String(80), nullable=False),
                sa.Column("text", sa.Text, nullable=False),
                sa.Column(
                    "created_at",
                    sa.DateTime(timezone=True),
                    nullable=False,
                    server_default=sa.func.now(),
                ),
            )


def downgrade():
    # Never delete legacy conversation data, including adopted tables.
    pass
