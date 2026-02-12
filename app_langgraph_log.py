import os
import time
import operator
import logging
from typing import TypedDict, Annotated, List

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langsmith import traceable
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


# =========================================================
# 🔹 ENV SETUP
# =========================================================
load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "LangGraph Essay Evaluator Logging"

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")


# =========================================================
# 🔹 LOGGING SETUP (Console + File)
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Logs saved to file
        logging.StreamHandler(),  # Logs shown in terminal
    ],
)

logger = logging.getLogger("essay_evaluator")


# =========================================================
# 🔹 FASTAPI INIT
# =========================================================
app = FastAPI(title="LangGraph Essay Evaluator API")


# =========================================================
# 🔹 REQUEST MODEL
# =========================================================
class EssayRequest(BaseModel):
    essay: str = Field(..., min_length=5, description="Essay text to evaluate")


# =========================================================
# 🔹 MODEL SETUP
# =========================================================
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class EvaluationSchema(BaseModel):
    feedback: str
    score: int = Field(ge=0, le=10)


structured_model = model.with_structured_output(EvaluationSchema)


# =========================================================
# 🔹 LANGGRAPH STATE
# =========================================================
class UPSCState(TypedDict, total=False):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[List[int], operator.add]
    avg_score: float


# =========================================================
# 🔹 GRAPH NODES
# =========================================================
@traceable(name="evaluate_language", tags=["language"])
async def evaluate_language(state: UPSCState):
    prompt = f"Evaluate language quality and give score out of 10:\n\n{state['essay']}"
    out = await structured_model.ainvoke(prompt)
    return {"language_feedback": out.feedback, "individual_scores": [out.score]}


@traceable(name="evaluate_analysis", tags=["analysis"])
async def evaluate_analysis(state: UPSCState):
    prompt = f"Evaluate depth of analysis and give score out of 10:\n\n{state['essay']}"
    out = await structured_model.ainvoke(prompt)
    return {"analysis_feedback": out.feedback, "individual_scores": [out.score]}


@traceable(name="evaluate_clarity", tags=["clarity"])
async def evaluate_clarity(state: UPSCState):
    prompt = (
        f"Evaluate clarity of thought and give score out of 10:\n\n{state['essay']}"
    )
    out = await structured_model.ainvoke(prompt)
    return {"clarity_feedback": out.feedback, "individual_scores": [out.score]}


@traceable(name="final_evaluation", tags=["aggregate"])
async def final_evaluation(state: UPSCState):
    prompt = f"""
    Summarize overall feedback:

    Language: {state.get('language_feedback','')}
    Analysis: {state.get('analysis_feedback','')}
    Clarity: {state.get('clarity_feedback','')}
    """

    overall = (await model.ainvoke(prompt)).content
    scores = state.get("individual_scores", [])
    avg = sum(scores) / len(scores) if scores else 0.0

    return {"overall_feedback": overall, "avg_score": avg}


# =========================================================
# 🔹 BUILD GRAPH
# =========================================================
graph = StateGraph(UPSCState)

graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_clarity", evaluate_clarity)
graph.add_node("final_evaluation", final_evaluation)

graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_clarity")

graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_clarity", "final_evaluation")

graph.add_edge("final_evaluation", END)

workflow = graph.compile()


# =========================================================
# 🔹 MIDDLEWARE FOR REQUEST TIMING
# =========================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)

    logger.info(
        f"{request.method} {request.url.path} | "
        f"Status: {response.status_code} | "
        f"Duration: {duration}ms"
    )

    return response


# =========================================================
# 🔹 API ENDPOINT
# =========================================================
@app.post("/evaluate")
async def evaluate_essay(request: EssayRequest):
    try:
        logger.info("Received essay evaluation request")
        logger.info(f"Essay length: {len(request.essay)} characters")

        result = await workflow.ainvoke(
            {"essay": request.essay},
            config={
                "run_name": "evaluate_upsc_essay",
                "tags": ["essay", "langgraph", "evaluation"],
                "metadata": {
                    "essay_length": len(request.essay),
                    "model": "gpt-4o-mini",
                },
            },
        )

        logger.info(f"Evaluation completed | Avg Score: {result.get('avg_score')}")

        return result

    except Exception as e:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# =========================================================
# 🔹 HEALTH CHECK ENDPOINT
# =========================================================
@app.get("/health")
async def health_check():
    return {"status": "healthy 🚀"}
