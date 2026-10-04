import os
from io import BytesIO
from pathlib import Path
from typing import Literal, TypedDict

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate


class Settings(BaseSettings):
    anthropic_api_key: str
    anthropic_model: str = "claude-haiku-4-5-20251001"
    chroma_dir: str = "./data/chroma"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    full_score: int = 90
    partial_score: int = 60
    no_score: int = 20
    training_score_threshold: int = 70
    confidence_threshold: float = 0.80
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def origins(self):
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]


settings = Settings()
Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)


class Requirement(BaseModel):
    requirement_id: str
    requirement: str
    partner_response: str = ""


class Evidence(BaseModel):
    source: str
    content: str


class RequirementAnalysis(BaseModel):
    requirement_id: str
    coverage: Literal["full", "partial", "none"]
    confidence: float = Field(ge=0, le=1)
    explanation: str
    evidence: list[Evidence] = Field(default_factory=list)
    skill_gap: str = ""


class ScoredRequirement(RequirementAnalysis):
    compliance: Literal["YES", "PARTIAL", "NO"]
    score: int = Field(ge=0, le=100)
    training_required: bool


class TrainingRecommendation(BaseModel):
    course_name: str
    reason: str
    priority: Literal["HIGH", "MEDIUM", "LOW"]


class RecommendationList(BaseModel):
    recommendations: list[TrainingRecommendation]


class RFPResult(BaseModel):
    requirements: list[ScoredRequirement]
    training_recommendations: list[TrainingRecommendation]
    overall_score: float


class RFPState(TypedDict, total=False):
    requirements: list[Requirement]
    scored: list[ScoredRequirement]
    recommendations: list[TrainingRecommendation]
    overall_score: float


def find_column(columns, aliases):
    normalized = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    for c in columns:
        name = str(c).strip().lower()
        if any(alias in name for alias in aliases):
            return c
    return None


def read_rfp_excel(data: bytes) -> list[Requirement]:
    try:
        df = pd.read_excel(BytesIO(data))
    except Exception as exc:
        raise HTTPException(400, f"Could not read RFP Excel: {exc}")

    req_col = find_column(
        df.columns,
        ["requirement", "customer requirement", "rfp requirement", "requirements"],
    )
    response_col = find_column(
        df.columns,
        ["partner response", "cisco partner response", "draft response", "response", "answer"],
    )

    if not req_col:
        raise HTTPException(400, "RFP Excel must contain a Requirement column.")

    result = []
    for idx, row in df.iterrows():
        req = str(row.get(req_col, "") or "").strip()
        if not req or req.lower() == "nan":
            continue
        response = str(row.get(response_col, "") or "").strip() if response_col else ""
        result.append(
            Requirement(
                requirement_id=f"REQ-{idx + 1:03d}",
                requirement=req,
                partner_response="" if response.lower() == "nan" else response,
            )
        )

    if not result:
        raise HTTPException(400, "No valid requirements found.")
    return result


def read_training_excel(data: bytes) -> list[dict]:
    try:
        df = pd.read_excel(BytesIO(data))
    except Exception as exc:
        raise HTTPException(400, f"Could not read training Excel: {exc}")

    name_col = find_column(df.columns, ["training name", "course name", "course", "training"])
    desc_col = find_column(df.columns, ["training description", "course description", "description"])

    if not name_col or not desc_col:
        raise HTTPException(400, "Training Excel must contain Training Name and Training Description.")

    records = []
    for _, row in df.iterrows():
        name = str(row.get(name_col, "") or "").strip()
        desc = str(row.get(desc_col, "") or "").strip()
        if name and name.lower() != "nan":
            records.append({"name": name, "description": "" if desc.lower() == "nan" else desc})
    return records


embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

product_store = Chroma(
    collection_name="product_rfp_knowledge",
    embedding_function=embeddings,
    persist_directory=settings.chroma_dir,
)

training_store = Chroma(
    collection_name="training_catalog",
    embedding_function=embeddings,
    persist_directory=settings.chroma_dir,
)


def add_product_knowledge(text: str, source: str) -> int:
    chunks = [x.strip() for x in text.split("\n\n") if x.strip()]
    if not chunks:
        return 0
    base = product_store._collection.count()
    ids = [f"product-{base + i}" for i in range(len(chunks))]
    product_store.add_texts(
        texts=chunks,
        metadatas=[{"source": source} for _ in chunks],
        ids=ids,
    )
    return len(chunks)


def add_training_catalog(records: list[dict]) -> int:
    if not records:
        return 0
    base = training_store._collection.count()
    texts = [
        f"Training Name: {r['name']}\nTraining Description: {r['description']}"
        for r in records
    ]
    ids = [f"training-{base + i}" for i in range(len(texts))]
    training_store.add_texts(
        texts=texts,
        metadatas=[{"training_name": r["name"]} for r in records],
        ids=ids,
    )
    return len(records)


llm = ChatAnthropic(
    model=settings.anthropic_model,
    api_key=settings.anthropic_api_key,
    max_tokens=1200,
)

analysis_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You evaluate an RFP requirement against a partner draft response.
Use only the supplied evidence. Never invent product capabilities.
coverage:
- full = clearly satisfied
- partial = partly satisfied or evidence is incomplete
- none = not supported by the response/evidence
Return a skill_gap when the score could be improved through knowledge or training."""
    ),
    (
        "human",
        """Requirement ID: {id}
Requirement: {requirement}
Partner response: {response}

Product / previous RFP evidence:
{context}"""
    ),
])

recommendation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Recommend training only from the supplied training catalog.
Do not invent course names.
Choose the most relevant course(s) for the skill gap and current score."""
    ),
    (
        "human",
        """Skill gap: {gap}
Current score: {score}

Training catalog:
{courses}"""
    ),
])


def analyze_requirement(req: Requirement, context: str) -> RequirementAnalysis:
    structured = llm.with_structured_output(RequirementAnalysis)
    return (analysis_prompt | structured).invoke({
        "id": req.requirement_id,
        "requirement": req.requirement,
        "response": req.partner_response,
        "context": context or "No relevant evidence found.",
    })


def apply_rules(analysis: RequirementAnalysis) -> ScoredRequirement:
    if analysis.coverage == "full" and analysis.confidence >= settings.confidence_threshold:
        compliance, score = "YES", settings.full_score
    elif analysis.coverage == "partial":
        compliance, score = "PARTIAL", settings.partial_score
    else:
        compliance, score = "NO", settings.no_score

    training_required = score < settings.training_score_threshold and bool(analysis.skill_gap.strip())

    return ScoredRequirement(
        **analysis.model_dump(),
        compliance=compliance,
        score=score,
        training_required=training_required,
    )


def recommend_training(gap: str, score: int, courses: str) -> list[TrainingRecommendation]:
    structured = llm.with_structured_output(RecommendationList)
    result = (recommendation_prompt | structured).invoke({
        "gap": gap,
        "score": score,
        "courses": courses,
    })
    return result.recommendations


def analyze_and_score(state: RFPState):
    scored = []
    for req in state["requirements"]:
        hits = product_store.similarity_search_with_score(req.requirement, k=4)
        context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for doc, _ in hits
        )
        analysis = analyze_requirement(req, context)
        scored.append(apply_rules(analysis))

    overall = round(sum(x.score for x in scored) / len(scored), 2)
    return {"scored": scored, "overall_score": overall}


def training_recommendation(state: RFPState):
    recommendations = []
    seen = set()

    for item in state.get("scored", []):
        if not item.training_required:
            continue

        hits = training_store.similarity_search_with_score(item.skill_gap or item.explanation, k=5)
        courses = "\n\n".join(doc.page_content for doc, _ in hits)
        if not courses:
            continue

        for rec in recommend_training(item.skill_gap, item.score, courses):
            key = rec.course_name.lower().strip()
            if key and key not in seen:
                seen.add(key)
                recommendations.append(rec)

    return {"recommendations": recommendations}


def build_graph():
    graph = StateGraph(RFPState)
    graph.add_node("analyze_and_score", analyze_and_score)
    graph.add_node("training_recommendation", training_recommendation)
    graph.add_edge(START, "analyze_and_score")
    graph.add_edge("analyze_and_score", "training_recommendation")
    graph.add_edge("training_recommendation", END)
    return graph.compile()


workflow = build_graph()


def create_report(result: RFPResult) -> BytesIO:
    training_names = "; ".join(x.course_name for x in result.training_recommendations)
    rows = []

    for item in result.requirements:
        evidence = " | ".join(f"{e.source}: {e.content}" for e in item.evidence)
        rows.append({
            "Requirement ID": item.requirement_id,
            "Compliance": item.compliance,
            "Score": item.score,
            "Confidence": item.confidence,
            "Explanation": item.explanation,
            "Evidence": evidence,
            "Skill Gap": item.skill_gap,
            "Training Required": item.training_required,
            "Recommended Training": training_names if item.training_required else "",
        })

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="RFP Evaluation")
        pd.DataFrame([{"Overall Score": result.overall_score}]).to_excel(
            writer, index=False, sheet_name="Summary"
        )
        if result.training_recommendations:
            pd.DataFrame(
                [x.model_dump() for x in result.training_recommendations]
            ).to_excel(writer, index=False, sheet_name="Training Recommendations")

    output.seek(0)
    return output


app = FastAPI(
    title="Scoring RFP Agent",
    version="0.1.0",
    description="POC: Excel RFP scoring + evidence + training recommendation",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "app": "Scoring RFP Agent"}


@app.post("/api/v1/knowledge/product")
async def upload_product(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.lower().endswith((".txt", ".md")):
        raise HTTPException(400, "POC product knowledge accepts .txt or .md files.")
    data = await file.read()
    text = data.decode("utf-8", errors="ignore")
    return {"status": "ok", "chunks_added": add_product_knowledge(text, filename)}


@app.post("/api/v1/knowledge/training")
async def upload_training(file: UploadFile = File(...)):
    records = read_training_excel(await file.read())
    return {"status": "ok", "records_added": add_training_catalog(records)}


@app.post("/api/v1/score", response_model=RFPResult)
async def score_rfp(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith((".xlsx", ".xls")):
        raise HTTPException(400, "RFP input must be Excel.")

    requirements = read_rfp_excel(await file.read())
    state = workflow.invoke({"requirements": requirements})

    return RFPResult(
        requirements=state["scored"],
        training_recommendations=state.get("recommendations", []),
        overall_score=state["overall_score"],
    )


@app.post("/api/v1/score/report")
async def score_rfp_report(file: UploadFile = File(...)):
    result = await score_rfp(file)
    output = create_report(result)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=rfp_scoring_result.xlsx"},
    )
