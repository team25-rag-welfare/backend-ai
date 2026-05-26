from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v2.chat import router as router_v2
from app.rag.v2.chain import get_chain as get_chain_v2, get_regenerate_chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.chain_v2 = get_chain_v2()
    app.state.chain_v2_regenerate = get_regenerate_chain()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(router_v2, prefix="/api/v2")
