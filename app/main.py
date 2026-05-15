from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import router
from app.rag.v1.chain import get_chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.chain = get_chain()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router, prefix="/api/v1")
