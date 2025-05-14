import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router
from app.log import setup_logging, handle_exception

sys.dont_write_bytecode = True

class PBL2AIServer:
    def __init__(self):
        self.logger = setup_logging()
        sys.excepthook = handle_exception

        self.app = FastAPI(
            title="우문현답 AI 서버",
            version="1.0",
            description="우문현답 AI 서버입니다.",
        )

        self._configure_cors()
        self._register_routes()

    def _configure_cors(self):
        origins = ["*"]

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _register_routes(self):

        @self.app.get("/version", tags=["root"])
        async def get_version():
            return {"version": "1.0.0"}

        self.app.include_router(router.router)

    def get_app(self) -> FastAPI:
        return self.app
    
server_instance = PBL2AIServer()
app = server_instance.get_app()