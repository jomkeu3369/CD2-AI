import sys
sys.dont_write_bytecode = True

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from G4dn.app.api import router
from log import setup_logging, handle_exception

logger = setup_logging()
sys.excepthook = handle_exception

app = FastAPI(
  title="PBL2 AI server",
  version="1.0",
  description="PBL2 AI 서버입니다.",
)

origins = [
    "*"
]   

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/version", tags=["root"])
async def get_version():
    return {"version": "1.0.0"}

app.include_router(router.router)