"""PostgreSQL persistence (Section 30).

LangGraph state holds the *current workflow*; this module holds the
*authoritative record*. SQLite is the default so the repository is testable
without infrastructure; set ``DATABASE_URL`` to a Postgres DSN in deployment —
the schema is identical.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from pathora.config import get_settings


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str | None] = mapped_column(String(320), unique=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    students: Mapped[list[Student]] = relationship(back_populates="user")


class Student(Base):
    __tablename__ = "students"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    display_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    graduation_year: Mapped[int | None] = mapped_column(nullable=True)
    digital_twin: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    academics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    test_scores: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    activities: Mapped[list[Any]] = mapped_column(JSON, default=list)
    projects: Mapped[list[Any]] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
    user: Mapped[User] = relationship(back_populates="students")
    analyses: Mapped[list[Analysis]] = relationship(back_populates="student")


class Analysis(Base):
    __tablename__ = "analyses"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # thread_id
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"))
    workflow_status: Mapped[str] = mapped_column(String(64), default="started")
    college_list: Mapped[list[Any]] = mapped_column(JSON, default=list)
    admission_results: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    critic_results: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    next_actions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    roadmap: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
    student: Mapped[Student] = relationship(back_populates="analyses")
    evidence: Mapped[list[Evidence]] = relationship(back_populates="analysis")


class Evidence(Base):
    """Evidence is stored independently of generated prose (Section 25)."""

    __tablename__ = "evidence"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id"))
    evidence_id: Mapped[str] = mapped_column(String(200))
    university: Mapped[str] = mapped_column(String(200))
    source_url: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(String(64))
    snippet: Mapped[str] = mapped_column(Text, default="")
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    analysis: Mapped[Analysis] = relationship(back_populates="evidence")


class Repository:
    def __init__(self, url: str | None = None) -> None:
        self.engine = create_engine(url or get_settings().database_url, future=True)
        Base.metadata.create_all(self.engine)

    def save_analysis(self, state: dict[str, Any], *, thread_id: str) -> None:
        """Persist a workflow snapshot as the authoritative record."""
        with Session(self.engine) as session:
            user = session.get(User, state["user_id"]) or User(id=state["user_id"])
            session.merge(user)

            student = Student(
                id=state["student_id"],
                user_id=state["user_id"],
                digital_twin=state.get("student_twin", {}),
                academics=state.get("verified_academics", {}),
                test_scores=state.get("student_twin", {}).get("testing", {}),
                activities=state.get("student_twin", {}).get("activities", []),
                projects=state.get("student_twin", {}).get("projects", []),
                graduation_year=state.get("verified_academics", {}).get("graduation_year"),
            )
            session.merge(student)

            analysis = Analysis(
                id=thread_id,
                student_id=state["student_id"],
                workflow_status=state.get("workflow_status", "unknown"),
                college_list=state.get("college_candidates", []),
                admission_results=state.get("admission_results", {}),
                critic_results=state.get("critic_results", {}),
                next_actions=state.get("next_actions", []),
                roadmap=state.get("roadmap", {}),
            )
            session.merge(analysis)
            session.flush()

            session.query(Evidence).filter(Evidence.analysis_id == thread_id).delete()
            for research in state.get("college_research", {}).values():
                for record in research.get("evidence", []):
                    session.add(
                        Evidence(
                            analysis_id=thread_id,
                            evidence_id=record["evidence_id"],
                            university=record["university"],
                            source_url=record["source_url"],
                            source_type=record["source_type"],
                            snippet=record.get("snippet", ""),
                        )
                    )
            session.commit()

    def get_analysis(self, thread_id: str) -> Analysis | None:
        with Session(self.engine) as session:
            return session.scalar(select(Analysis).where(Analysis.id == thread_id))

    def evidence_for(self, thread_id: str) -> list[Evidence]:
        with Session(self.engine) as session:
            return list(session.scalars(select(Evidence).where(Evidence.analysis_id == thread_id)))
