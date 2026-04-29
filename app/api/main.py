from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.database import engine, Base
from app.api.experiment import router

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Capstone Financial NLP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    return {"message": "Capstone Financial NLP API"}
