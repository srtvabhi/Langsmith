from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# Validate API key early
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

app = FastAPI(title="LLM API", version="1.0")


# Request schema
class QuestionRequest(BaseModel):
    question: str


# Initialize model ONCE (important)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  # or your preferred model

prompt = PromptTemplate.from_template("{question}")
parser = StrOutputParser()
chain = prompt | model | parser


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        result = await chain.ainvoke({"question": request.question})
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
