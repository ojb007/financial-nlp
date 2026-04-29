from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Boolean, Float, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship, Session

from app.api.database import Base, get_db, SessionLocal


# ── SQLAlchemy 모델 ──────────────────────────────────────────────

class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    group_name = Column(String, nullable=False)
    dataset = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    result = relationship("Result", back_populates="experiment", uselist=False)


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False, unique=True)
    model = Column(String, nullable=True)
    prompt_strategy = Column(String, nullable=True)
    rag = Column(Boolean, nullable=True)
    accuracy = Column(Float, nullable=True)
    f1_macro = Column(Float, nullable=True)
    f1_micro = Column(Float, nullable=True)
    f1_weighted = Column(Float, nullable=True)
    avg_latency_ms = Column(Float, nullable=True)
    total_cost_usd = Column(Float, nullable=True)
    cost_per_item = Column(Float, nullable=True)
    llm_judge_score = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)

    experiment = relationship("Experiment", back_populates="result")


# ── Pydantic 스키마 ──────────────────────────────────────────────

class ExperimentRequest(BaseModel):
    group_name: str
    dataset: str


class ExperimentResponse(BaseModel):
    experiment_id: int
    group_name: str
    dataset: str


class ResultRequest(BaseModel):
    experiment_id: int
    model: Optional[str] = None
    prompt_strategy: Optional[str] = None
    rag: Optional[bool] = None
    accuracy: Optional[float] = None
    f1_macro: Optional[float] = None
    f1_micro: Optional[float] = None
    f1_weighted: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    total_cost_usd: Optional[float] = None
    cost_per_item: Optional[float] = None
    llm_judge_score: Optional[float] = None
    notes: Optional[str] = None


class ResultResponse(BaseModel):
    id: int
    group_name: str
    dataset: str
    model: Optional[str]
    prompt_strategy: Optional[str]
    rag: Optional[bool]
    accuracy: Optional[float]
    f1_macro: Optional[float]
    f1_micro: Optional[float]
    f1_weighted: Optional[float]
    avg_latency_ms: Optional[float]
    total_cost_usd: Optional[float]
    cost_per_item: Optional[float]
    llm_judge_score: Optional[float]
    notes: Optional[str]


# ── 라우터 ────────────────────────────────────────────────────────

router = APIRouter()


@router.post("/experiments", response_model=ExperimentResponse)
def create_experiment(req: ExperimentRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    from app.api.runner import run_inference, GROUP_STRATEGIES
    exp = Experiment(group_name=req.group_name, dataset=req.dataset)
    db.add(exp)
    db.commit()
    db.refresh(exp)

    if req.group_name in GROUP_STRATEGIES and req.dataset in ["FPB", "FiQA", "FinQA", "MMLU-KO"]:
        background_tasks.add_task(run_inference, exp.id, req.group_name, req.dataset, SessionLocal)

    return ExperimentResponse(
        experiment_id=exp.id,
        group_name=exp.group_name,
        dataset=exp.dataset,
    )


@router.post("/results", response_model=ResultResponse)
def create_result(req: ResultRequest, db: Session = Depends(get_db)):
    result = Result(
        experiment_id=req.experiment_id,
        model=req.model,
        prompt_strategy=req.prompt_strategy,
        rag=req.rag,
        accuracy=req.accuracy,
        f1_macro=req.f1_macro,
        f1_micro=req.f1_micro,
        f1_weighted=req.f1_weighted,
        avg_latency_ms=req.avg_latency_ms,
        total_cost_usd=req.total_cost_usd,
        cost_per_item=req.cost_per_item,
        llm_judge_score=req.llm_judge_score,
        notes=req.notes,
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    exp = db.query(Experiment).filter(Experiment.id == result.experiment_id).first()
    return _to_response(result, exp)


@router.get("/results", response_model=list[ResultResponse])
def get_results(
    group_name: Optional[str] = None,
    dataset: Optional[str] = None,
    db: Session = Depends(get_db),
):
    query = db.query(Result).join(Experiment)
    if group_name:
        query = query.filter(Experiment.group_name == group_name)
    if dataset:
        query = query.filter(Experiment.dataset == dataset)

    return [_to_response(r, r.experiment) for r in query.all()]


def _to_response(r: Result, exp: Experiment) -> ResultResponse:
    return ResultResponse(
        id=r.id,
        group_name=exp.group_name,
        dataset=exp.dataset,
        model=r.model,
        prompt_strategy=r.prompt_strategy,
        rag=r.rag,
        accuracy=r.accuracy,
        f1_macro=r.f1_macro,
        f1_micro=r.f1_micro,
        f1_weighted=r.f1_weighted,
        avg_latency_ms=r.avg_latency_ms,
        total_cost_usd=r.total_cost_usd,
        cost_per_item=r.cost_per_item,
        llm_judge_score=r.llm_judge_score,
        notes=r.notes,
    )
