from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import operator
import os
from typing import TypedDict, Annotated, List

from langsmith import traceable
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# ---------- Setup ----------
load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "LangGraph Demo"

app = FastAPI(title="LangGraph Essay Evaluator")

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------- Request Schema ----------
class EssayRequest(BaseModel):
    essay: str


# ---------- Structured Schema ----------
class EvaluationSchema(BaseModel):
    feedback: str
    score: int


structured_model = model.with_structured_output(EvaluationSchema)


# ---------- LangGraph State ----------
class UPSCState(TypedDict, total=False):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[List[int], operator.add]
    avg_score: float


# ---------- Node Functions (ASYNC VERSION) ----------
@traceable(name="evaluate_language")
async def evaluate_language(state: UPSCState):
    prompt = f"Evaluate language quality and give score out of 10:\n\n{state['essay']}"
    out = await structured_model.ainvoke(prompt)
    return {"language_feedback": out.feedback, "individual_scores": [out.score]}


@traceable(name="evaluate_analysis")
async def evaluate_analysis(state: UPSCState):
    prompt = f"Evaluate depth of analysis and give score out of 10:\n\n{state['essay']}"
    out = await structured_model.ainvoke(prompt)
    return {"analysis_feedback": out.feedback, "individual_scores": [out.score]}


@traceable(name="evaluate_thought")
async def evaluate_thought(state: UPSCState):
    prompt = (
        f"Evaluate clarity of thought and give score out of 10:\n\n{state['essay']}"
    )
    out = await structured_model.ainvoke(prompt)
    return {"clarity_feedback": out.feedback, "individual_scores": [out.score]}


@traceable(name="final_evaluation")
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


# ---------- Build Graph ----------
graph = StateGraph(UPSCState)

graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")
graph.add_edge("final_evaluation", END)

workflow = graph.compile()


# ---------- API Endpoint ----------
@app.post("/evaluate")
async def evaluate_essay(request: EssayRequest):
    try:
        result = await workflow.ainvoke({"essay": request.essay})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "LangGraph Essay Evaluator API running 🚀"}
