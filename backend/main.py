import os
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from agent import process_query

DATA_PATH = "../data/titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure data directory exists and download dataset if missing
    if not os.path.exists(DATA_PATH):
        os.makedirs("../data", exist_ok=True)
        df = pd.read_csv(DATA_URL)
        df.to_csv(DATA_PATH, index=False)
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    response = process_query(request.question)
    return response