"""Sessions, utterances, classifications, results and append-only audit chain."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def uuid_pk():
    return sa.Column("id", sa.Uuid, primary_key=True)


def created():
    return sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    )


def upgrade():
    op.create_table(
        "sessions",
        uuid_pk(),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
        sa.Column("metadata", JSONB, nullable=False),
    )
    op.create_table(
        "participants",
        uuid_pk(),
        sa.Column("session_id", sa.Uuid, sa.ForeignKey("sessions.id"), nullable=False),
        sa.Column("mic_id", sa.Integer, nullable=False),
        sa.Column("display_name", sa.String(80), nullable=False),
        sa.UniqueConstraint("session_id", "mic_id"),
    )
    op.create_index("ix_participants_session_id", "participants", ["session_id"])
    op.create_table(
        "utterances",
        uuid_pk(),
        sa.Column("session_id", sa.Uuid, sa.ForeignKey("sessions.id"), nullable=False),
        sa.Column(
            "participant_id", sa.Uuid, sa.ForeignKey("participants.id"), nullable=False
        ),
        sa.Column("raw_text", sa.Text, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("stt_model", sa.String(120)),
        sa.Column("stt_confidence", sa.Float),
    )
    for field in ("session_id", "participant_id", "started_at"):
        op.create_index(f"ix_utterances_{field}", "utterances", [field])
    op.create_table(
        "intent_classifications",
        uuid_pk(),
        sa.Column(
            "utterance_id", sa.Uuid, sa.ForeignKey("utterances.id"), nullable=False
        ),
        sa.Column("intent", sa.String(40), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("model_version", sa.String(160), nullable=False),
        sa.Column("raw_output", JSONB, nullable=False),
        sa.Column("latency_ms", sa.Float, nullable=False),
        created(),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1"),
    )
    op.create_index(
        "ix_intent_classifications_utterance_id",
        "intent_classifications",
        ["utterance_id"],
        unique=True,
    )
    op.create_index(
        "ix_intent_classifications_created_at", "intent_classifications", ["created_at"]
    )
    op.create_table(
        "pipeline_results",
        uuid_pk(),
        sa.Column(
            "utterance_id", sa.Uuid, sa.ForeignKey("utterances.id"), nullable=False
        ),
        sa.Column(
            "pipeline",
            sa.Enum(
                "question_query", "fact_claim", "opinion_debate", name="pipeline_kind"
            ),
            nullable=False,
        ),
        sa.Column("output", JSONB, nullable=False),
        sa.Column("latency_ms", sa.Float, nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        created(),
    )
    for field in ("utterance_id", "created_at"):
        op.create_index(f"ix_pipeline_results_{field}", "pipeline_results", [field])
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("session_id", sa.Uuid, sa.ForeignKey("sessions.id"), nullable=False),
        sa.Column("event_type", sa.String(80), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("prev_hash", sa.String(64), nullable=False),
        sa.Column("hash", sa.String(64), nullable=False, unique=True),
        created(),
    )
    for field in ("session_id", "created_at"):
        op.create_index(f"ix_audit_log_{field}", "audit_log", [field])
    op.execute(
        """CREATE FUNCTION reject_audit_mutation() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'audit_log is append-only'; END $$"""
    )
    op.execute(
        "CREATE TRIGGER audit_immutable BEFORE UPDATE OR DELETE OR TRUNCATE ON audit_log FOR EACH STATEMENT EXECUTE FUNCTION reject_audit_mutation()"
    )


def downgrade():
    op.drop_table("audit_log")
    op.execute("DROP FUNCTION reject_audit_mutation()")
    for name in (
        "pipeline_results",
        "intent_classifications",
        "utterances",
        "participants",
        "sessions",
    ):
        op.drop_table(name)
    sa.Enum(name="pipeline_kind").drop(op.get_bind())
