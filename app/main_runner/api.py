import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from app.main_runner.config import Config
from app.main_runner.qa_service import QAService
from app.shared.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Compressed Q&A API", version="1.0.0")
config = Config()
service: QAService = None


class Question(BaseModel):
    context: str
    question: str


class Answer(BaseModel):
    question: str
    answer: str
    compression_ratio: float
    token_reduction_pct: float


@app.on_event("startup")
async def startup():
    global service
    logger.info("Loading model...")
    service = QAService(config)
    service.load()
    logger.info("Ready!")


@app.post("/ask", response_model=Answer)
async def ask(q: Question) -> Answer:
    result = service.ask(q.context, q.question)
    return Answer(**result)


@app.get("/health")
async def health():
    return {"status": "ok", "loaded": service._loaded if service else False}


if __name__ == "__main__":
    logger.info(f"Starting API at http://{config.api_host}:{config.api_port}")
    uvicorn.run(app, host=config.api_host, port=config.api_port)
