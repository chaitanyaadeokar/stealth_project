"""Long-term evidence, decisions, and nullable ethics placeholders."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def common():
    return [
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("session_id", sa.Uuid, sa.ForeignKey("sessions.id"), nullable=False),
        sa.Column(
            "utterance_id", sa.Uuid, sa.ForeignKey("utterances.id"), nullable=False
        ),
        sa.Column(
            "source_result_id",
            sa.Uuid,
            sa.ForeignKey("pipeline_results.id"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    ]


def upgrade():
    op.create_table(
        "ltm_evidence",
        *common(),
        sa.Column("claim", sa.Text, nullable=False),
        sa.Column(
            "fact_status",
            sa.Enum("true", "false", "uncertain", name="fact_status"),
            nullable=False,
        ),
        sa.Column("sources", JSONB, nullable=False),
    )
    op.create_table(
        "ltm_decisions",
        *common(),
        sa.Column("agent", sa.String(80), nullable=False),
        sa.Column("decision", sa.Text, nullable=False),
        sa.Column("rationale", sa.Text, nullable=False),
    )
    op.create_table(
        "ltm_ethics",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column(
            "utterance_id", sa.Uuid, sa.ForeignKey("utterances.id"), nullable=False
        ),
        sa.Column("ethical_score", sa.Numeric, nullable=True),
        sa.Column("dimensions", JSONB(none_as_null=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    for table in ("ltm_evidence", "ltm_decisions"):
        for column in ("session_id", "utterance_id", "created_at"):
            op.create_index(f"ix_{table}_{column}", table, [column])
    op.create_index("ix_ltm_evidence_fact_status", "ltm_evidence", ["fact_status"])
    op.create_index(
        "ix_ltm_ethics_utterance_id", "ltm_ethics", ["utterance_id"], unique=True
    )
    op.create_index("ix_ltm_ethics_created_at", "ltm_ethics", ["created_at"])


def downgrade():
    for table in ("ltm_ethics", "ltm_decisions", "ltm_evidence"):
        op.drop_table(table)
    sa.Enum(name="fact_status").drop(op.get_bind())
