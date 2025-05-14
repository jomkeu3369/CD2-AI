import os
import httpx
import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status

from app.api.crud import status_manager
from app.api.schema import OllamaPromptRequest, OllamaGenerateResponse, BartClassifyRequest, BartClassifyResponse, OptimizeResponse, OptimizeRequest
from app.models.model import graph

from starlette.websockets import WebSocketState
from typing import Optional, Literal

router = APIRouter()

OLLAMA_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
client = httpx.AsyncClient()

@router.on_event("shutdown")
async def shutdown_event():
    await client.aclose()

# --- 라우터 정의 ---

@router.post("/generate/", response_model=OllamaGenerateResponse)
async def generate_text_ollama(request: OllamaPromptRequest):
    """ FastAPI 앱이 Ollama 서버의 /api/generate 엔드포인트에 비동기 요청을 보냅니다. """
    try:
        response = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": request.model,
                "prompt": request.prompt,
                "stream": False,
            },
            headers={'Content-Type': 'application/json'},
            timeout=120.0
        )
        response.raise_for_status()
        ollama_response = response.json()
        generated_text = ollama_response.get("response", "").strip()

        if ollama_response.get("done"):
             return OllamaGenerateResponse(generated_text=generated_text)
        else:
            return OllamaGenerateResponse(error="Ollama did not complete generation", detail=ollama_response)

    except httpx.RequestError as e:
        return OllamaGenerateResponse(error="Failed to call Ollama service", detail=f"Request error: {e}")
    except httpx.HTTPStatusError as e:
        detail = e.response.text
        try: detail = e.response.json()
        except: pass
        return OllamaGenerateResponse(error="Ollama service returned error", detail=detail)
    except Exception as e:
        return OllamaGenerateResponse(error="An unexpected error occurred", detail=str(e))

@router.post("/classify/", response_model=BartClassifyResponse)
async def classify_text_bart(request: BartClassifyRequest):
    """ 로드된 BART 모델을 사용하여 직접 분류를 수행합니다. """
    from app.models.classifiers import classifier

    if classifier is None:
        raise HTTPException(status_code=503, detail="BART model is not available (failed to load?)")
    try:
        result = classifier(
            request.sequence,
            request.candidate_labels,
            multi_label=request.multi_label
        )
        return BartClassifyResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@router.post("/optimize/", response_model=OptimizeResponse)
async def optimize_prompt(request: OptimizeRequest):
    """
    입력된 한국어 프롬프트를 받아 LangGraph 시스템을 실행하고,
    개선된 프롬프트와 관련 정보를 반환합니다.
    """

    try:
        # LangGraph 실행 함수 호출
        final_state = await graph.ainvoke(
            {
                "original_prompt": request.prompt,
                "evaluation_data": dict(),
                "weak_categories": [],
                "improvement_suggestions": {},
                "enhanced_prompt": "",
                "execution_log": [],
                "exit": False
            }
        )

        if "error" in final_state:
            raise HTTPException(status_code=500, detail=f"프롬프트 처리 중 오류 발생: {final_state.get('error_message', final_state['error'])}")

        eval_data = final_state.get('evaluation_data')
        eval_summary = {}
        if eval_data and isinstance(eval_data, dict):
            for cat, data in eval_data.items():
                 if isinstance(data, dict):
                      eval_summary[cat] = round(data.get('average_score', 0.0), 3)

        suggestions = final_state.get('improvement_suggestions')
        sugg_count = len(suggestions) if suggestions else 0

        response_data = OptimizeResponse(
            original_prompt=final_state.get('original_prompt', request.prompt),
            enhanced_prompt=final_state.get('enhanced_prompt'),
            english_prompt=final_state.get('english_prompt'),
            error_message=final_state.get('error_message'),
            weak_categories=final_state.get('weak_categories'),
            suggestions_count=sugg_count,
            evaluation_summary=eval_summary if eval_summary else None
        )
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류 발생: {type(e).__name__}")


@router.websocket("/ws")
async def websocket_optimize_endpoint(websocket: WebSocket, token: Optional[str] = None):
    # 실제 운영 환경에서는 토큰 검증 로직이 필요합니다.
    # 예: if not token or not await is_valid_token(token): # is_valid_token 함수는 별도 구현 필요
    #         await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
    #         return

    await websocket.accept()
    # LangGraphManager의 상태 초기화는 ConnectionManager 내부에서 관리하거나,
    # 연결 시점에 한 번만 호출되도록 하는 것이 좋습니다.
    # status_manager.lang_graph.init_workflow(websocket) # 필요시 crud.py의 connect 같은 메서드로 이동 고려

    # 토큰 검증 시스템
    # 백엔드에 토큰을 전송하여 검증

    try:
        while True:
            # 클라이언트로부터 원본 프롬프트를 텍스트 형태로 받습니다.
            original_prompt = await websocket.receive_text()
            
            # app.api.crud.ConnectionManager의 process_workflow를 호출하여 로직을 처리하고,
            # 해당 함수 내에서 웹소켓으로 스트리밍 응답을 보냅니다.
            await status_manager.process_workflow(websocket, original_prompt)
            
            # 현재 구현은 하나의 프롬프트 처리 후 다음 프롬프트를 기다립니다.
            # 만약 한 번의 최적화 후 연결을 종료하고 싶다면, 클라이언트 또는 서버에서
            # 특정 조건 하에 연결을 닫는 로직이 추가되어야 합니다.

    except WebSocketDisconnect:
        print(f"Client {websocket.client} disconnected from /ws/optimize/")
        status_manager.cleanup_connection(websocket) # 연결 종료 시 리소스 정리
    except Exception as e:
        error_message = f"Unexpected WebSocket error in /ws/optimize/: {str(e)}"
        print(error_message)
        # 클라이언트에게 에러 알림 (이미 연결이 끊겼을 수도 있음)
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.send_json({"type": "error", "message": error_message})
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            except Exception: # send_json이나 close 중 에러 발생 가능성
                pass 
        status_manager.cleanup_connection(websocket) # 리소스 정리
        
@router.get("/")
def read_root():
    return {"message": "FastAPI service running (Ollama Proxy & BART Classifier)"}

"""
    1. 웹소켓 연결

    2. 토큰 검증   
        [ 방법 ]
            1. 백엔드에 토큰을 전송하여 검증 후 결과를 받음 (True, False)
            2. 인공지능 서버에서 토큰을 똑같이 검증함

    3. 대화
        [ 쿼리 종류 ]
            1. 프롬프트 최적화 최초 프롬프트
            2. human in the loop

        [ 구별 방법 ]
            1. 웹 소켓 최초 연결 시 - 프롬프트 최적화 최초 프롬프트로 인식
            2. 이후 human in the loop 인터럽트 발생 시 - 내부 status를 변경하여 human in the loop 메시지를 받을 준비를 함
            3. 이후 들어오는 메시지는 human in the loop 메시지
            4. 만약 도중에 웹 소켓 연결이 끊길 시 - human in the loop 종료 및 프롬프팅 생성 종료 -> 재연결 시 1번 과정으로 이동

"""


"""
    T3 인스턴스의 모델과의 차이점

    1. 로컬 모델 구동 여부
        1. G4DN 인스턴스는 로컬 모델을 사용
        2. T3 인스턴스는 API만 사용

    2. 제로샷 모델 사용 여부
        1. G4DN 인스턴스는 facebook/bart 모델 사용
        2. T3 인스턴스에서는 LLM-as-a-judge 방식을 사용해 평가

        * 두 인스턴스의 평가 데이터는 다를 수 있음
        * 속도 측면에서는 T3가 압도적

        => 차라리 G4DN에서 모델 스왑을 하지 말고 LLM-as-a-judge 방식으로 통일하는 것은 어떤가?

"""