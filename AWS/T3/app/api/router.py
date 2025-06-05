import asyncio
import httpx
import json
import os
import sys
import traceback
import logging
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from starlette.websockets import WebSocketState
from dotenv import load_dotenv

from app.model.schema import criteria as model_criteria
from app.api.schema import model_list, ReinforceRequest
from app.model.model import graph
from app.api.crud import get_model_name_by_id, session_manager, process_reinforcement_learning as crud_process_rl
from app.log import setup_logging, handle_exception

load_dotenv()

logger: logging.Logger = setup_logging()
sys.excepthook = handle_exception

BACKEND_HOST = os.getenv("BACKEND_HOST")
if not BACKEND_HOST:
    logger.critical("BACKEND_HOST 환경 변수가 설정되지 않았습니다. 애플리케이션이 올바르게 작동하지 않을 수 있습니다.")

router = APIRouter(prefix="")

@router.get("/")
async def read_root():
    return {"message": "W.A.F.Q. AI SERVER is running!"}

@router.get("/model/list")
async def get_models_list():
    return model_list

@router.post("/feedback/{session_id}")
async def reinforce_feedback_endpoint(request: ReinforceRequest, session_id: str):
    try:
        success, message, updated_info = await crud_process_rl(
            token=request.token,
            message_id=request.message_id,
            recommend=request.recommand,
            session_id=session_id
        )
        if not success:
            logger.warning(f"강화 학습 처리 실패 (세션 ID: {session_id}): {message}")
            raise HTTPException(status_code=400, detail=message)
            
        logger.info(f"강화 학습 처리 성공 (세션 ID: {session_id}). 업데이트 정보: {updated_info}")
        return {"status": "success", "message": message, "updated_info": updated_info}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"강화 학습 엔드포인트 처리 중 예기치 않은 오류 발생 (세션 ID: {session_id}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during reinforcement learning processing.")

async def save_history_to_backend_api(messages: List[Dict[str, Any]], token: str) -> Optional[List[str]]:
    logger.info(f"대화 기록 저장을 시도합니다. 메시지 수: {len(messages)}")
    if not BACKEND_HOST:
        logger.error("BACKEND_HOST가 설정되지 않아 대화 기록을 저장할 수 없습니다.")
        return None

    url = f"{BACKEND_HOST}/api/v1/faiss/add"
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url=url, json=messages, headers=headers)
            response.raise_for_status()
            
            response_data = response.json()
            message_ids = response_data.get("added_ids")
            logger.info(f"대화 기록 저장 성공. 반환된 Message IDs: {message_ids}")
            return message_ids
            
    except httpx.HTTPStatusError as e:
        logger.error(f"대화 기록 저장 API 호출 실패 - 상태 코드 {e.response.status_code}: {e.response.text}", exc_info=True)
        return None
    except httpx.RequestError as e:
        logger.error(f"대화 기록 저장 API 요청 오류: {e}", exc_info=True)
        return None
    except ValueError as e: 
        logger.error(f"대화 기록 저장 API 응답 JSON 디코딩 오류: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"대화 기록 저장 API 호출 중 예기치 않은 오류 발생: {e}", exc_info=True)
        return None

async def run_graph_and_process_response(websocket: WebSocket, session_id: str, data: Dict[str, Any]):
    token = data.get("token")
    prompt = data.get("prompt")
    topic = data.get("topic")
    option: Dict[str, Any] = data.get("option", {})
    user_id: Optional[str] = None

    try:
        if not BACKEND_HOST:
            logger.error(f"[Session {session_id}] BACKEND_HOST가 설정되지 않아 토큰 검증을 진행할 수 없습니다.")
            await websocket.send_json({"type": "error", "text": "서버 설정 오류로 요청을 처리할 수 없습니다."})
            return

        if token != "token_0546":
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(f"{BACKEND_HOST}/api/v1/auth/internal/validate-token", json={"token": token})
            response.raise_for_status()
            response_data = response.json()
            if not response_data.get("is_valid", False):
                logger.warning(f"[Session {session_id}] 유효하지 않은 토큰으로 접근 시도.")
                await websocket.send_json({"type": "error", "text": "올바르지 않은 토큰 정보입니다."})
                return
            user_id = str(response_data.get("user_id"))
        else:
            user_id = "1" 

        if not user_id:
            logger.error(f"[Session {session_id}] 사용자 ID를 확인할 수 없습니다.")
            await websocket.send_json({"type": "error", "text": "사용자 ID를 확인할 수 없습니다."})
            return

        await websocket.send_json({"type": "result_start", "text": "답변 생성을 시작합니다."})
        
        model_id_from_option = option.get("model", 0)
        try:
            model_id_int = int(model_id_from_option)
        except ValueError:
            logger.warning(f"[Session {session_id}] 제공된 모델 ID '{model_id_from_option}'가 유효하지 않아 기본값 0을 사용합니다.")
            model_id_int = 0
        model_name = await get_model_name_by_id(model_id_int)

        initial_graph_state = {
            "user_id": user_id, "token": token, "initial_prompt": prompt, "model": model_name, "topic": topic,
            "translated_prompt": None, "optimized_prompt": None, "is_completed": False, "evaluation_data": None,
            "final_evaluation_data": None, "improvement_suggestions": None, "human_feedback": None,
            "error_message": None, "generate_detailed_report": option.get("optimize", False),
            "needs_web_search": option.get("web_search", False), "web_search_data": None,
            "has_file_upload": option.get("file", False), "uploaded_files": None, "report_data": None, 'human_feedback_ai': None
        }
        
        final_state: Optional[Dict[str, Any]] = None
        invocation_config = {"configurable": {"thread_id": session_id, "session_id": session_id, "websocket": websocket}}

        logger.info(f"[Session {session_id}] LangGraph 스트림 처리를 시작합니다. 초기 상태: { {k:v for k,v in initial_graph_state.items() if k not in ['token']} }")
        async for event in graph.astream_events(initial_graph_state, config=invocation_config, version="v2"):
            event_type = event.get("event")
            node_name = event.get("name")
            logger.debug(f"[Session {session_id}] | Event: {event_type:<15} | Node: {node_name}")
            if event_type == "on_chain_end" and node_name == "LangGraph":
                logger.info(f"[Session {session_id}] 'LangGraph' end event 수신. final_state를 설정합니다.")
                final_state = event.get("data", {}).get("output")
                logger.info(f"[Session {session_id}] final_state 설정 완료. State is None: {final_state is None}")
        
        logger.info(f"[Session {session_id}] LangGraph 스트림 처리를 종료했습니다. 최종 final_state is None: {final_state is None}")
        
        if not final_state:
            logger.error(f"[Session {session_id}] LangGraph 처리 후 final_state가 설정되지 않았습니다.")
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({"type": "error", "text": "서버 내부 처리 중 오류로 결과 생성에 실패했습니다."})
            return

        logger.info(f"[Session {session_id}] 최종 결과 전송 및 기록 저장 프로세스를 시작합니다.")
        evaluation_indices: List[int] = []
        if final_state.get("improvement_suggestions"):
            flat_english_criteria = [en_crit for cat_criteria in model_criteria.values() for _, en_crit in cat_criteria]
            
            improvement_keys = final_state.get("improvement_suggestions", {}).keys()
            for key in improvement_keys:
                try:
                    idx = flat_english_criteria.index(str(key)) + 1
                    evaluation_indices.append(idx)
                except ValueError:
                    logger.warning(f"[Session {session_id}] '{key}'에 해당하는 평가 기준 인덱스를 찾을 수 없습니다.")
        
        logger.info(f"[Session {session_id}] 추출된 평가 인덱스: {evaluation_indices}")

        messages_to_process = []
        messages_to_process.append(("user", final_state.get("initial_prompt")))
        messages_to_process.append(("hitl_ai", final_state.get("human_feedback_ai")))
        messages_to_process.append(("hitl_user", final_state.get("human_feedback")))
        messages_to_process.append(("optimize", final_state.get("optimized_prompt")))

        report_data = final_state.get("report_data")
        if report_data and isinstance(report_data, dict):
            messages_to_process.append(("report", report_data.get("detailed_markdown_report")))
        
        messages_to_save: List[Dict[str, Any]] = []
        roles_for_ids: List[str] = []
        for role, content in messages_to_process:
            if content:
                message_data = {
                    "page_content": str(content), 
                    "session_id": int(session_id), 
                    "user_id": int(user_id), 
                    "message_role": role
                }
                if role in ["optimize", "report"]:
                    message_data["evaluation_indices"] = sorted(list(set(evaluation_indices)))
                    message_data["recommendation_status"] = None
                messages_to_save.append(message_data)
                roles_for_ids.append(role)
        
        logger.info(f"[Session {session_id}] DB에 저장할 메시지: {len(messages_to_save)}개. (역할: {roles_for_ids})")
        
        saved_message_ids_map: Optional[Dict[str, Any]] = None
        if token != "token_0546" and messages_to_save:
            returned_ids = await save_history_to_backend_api(messages_to_save, token)
            if returned_ids and len(returned_ids) == len(roles_for_ids):
                saved_message_ids_map = dict(zip(roles_for_ids, returned_ids))
            elif returned_ids:
                logger.warning(f"[Session {session_id}] 저장된 ID 수({len(returned_ids)})와 역할 수({len(roles_for_ids)})가 일치하지 않습니다.")
                saved_message_ids_map = {"raw_ids": returned_ids}
            else:
                logger.error(f"[Session {session_id}] 메시지 저장에 실패했거나 ID가 반환되지 않았습니다.")
        
        if websocket.client_state == WebSocketState.CONNECTED:
            final_response_message = {"type": "result_end", "text": "요청 처리가 완료되었습니다."}
            if saved_message_ids_map:
                final_response_message["saved_message_ids"] = saved_message_ids_map
            
            logger.info(f"[Session {session_id}] 프론트엔드로 최종 result_end 메시지를 전송합니다:\n{json.dumps(final_response_message, ensure_ascii=False, indent=2)}")
            await websocket.send_json(final_response_message)

    except httpx.HTTPStatusError as e:
        logger.error(f"[Session {session_id}] 토큰 검증 API 요청 실패 - 상태 코드 {e.response.status_code}: {e.response.text}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "text": f"인증 서버 통신 오류: {e.response.status_code}"})
    
    except httpx.RequestError as e:
        logger.error(f"[Session {session_id}] 토큰 검증 중 API 요청 오류: {e}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "text": "인증 서버 연결 오류"})
    
    except Exception as e:
        logger.error(f"[Session {session_id}] 그래프 실행 또는 응답 처리 중 예기치 않은 예외 발생: {e}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "text": f"요청 처리 중 서버 내부 오류 발생: {str(e)}"})

@router.websocket("/ws/{session_id}")
async def websocket_optimize_prompt_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"[Session {session_id}] 클라이언트가 연결되었습니다: {websocket.client}")
    
    current_graph_task: Optional[asyncio.Task] = None

    try:
        while True:
            received_data: Dict[str, Any] = await websocket.receive_json()
            logger.debug(f"[Session {session_id}] 클라이언트로부터 메시지 수신: {received_data}")

            if "prompt" in received_data and "topic" in received_data:
                if current_graph_task and not current_graph_task.done():
                    logger.warning(f"[Session {session_id}] 이전 요청 처리 중 새 요청 수신됨. 새 요청 무시.")
                    await websocket.send_json({"type": "error", "text": "현재 다른 요청을 처리하고 있습니다. 완료 후 다시 시도해주세요."})
                    continue
                
                logger.info(f"[Session {session_id}] 새 그래프 처리 작업 생성 시작.")
                current_graph_task = asyncio.create_task(
                    run_graph_and_process_response(websocket, session_id, received_data)
                )
            elif "feedback" in received_data:
                feedback_content = received_data.get('feedback')
                logger.info(f"[Session {session_id}] HITL 피드백 수신: {feedback_content}")
                session_manager.set_feedback(session_id, feedback_content)
            else:
                logger.warning(f"[Session {session_id}] 알 수 없는 메시지 타입 수신: {received_data}")
                await websocket.send_json({"type": "warning", "text": "알 수 없는 요청 형식입니다."})

    except WebSocketDisconnect as e:
        logger.warning(f"[Session {session_id}] 클라이언트({websocket.client}) 연결 종료됨. 코드: {e.code}, 이유: {e.reason}")
        if current_graph_task and not current_graph_task.done():
            logger.info(f"[Session {session_id}] 진행 중인 그래프 작업 취소 시도.")
            current_graph_task.cancel()
            try:
                await current_graph_task
            except asyncio.CancelledError:
                logger.info(f"[Session {session_id}] 그래프 작업이 성공적으로 취소되었습니다.")
            except Exception as task_exc:
                logger.error(f"[Session {session_id}] 취소된 그래프 작업에서 예외 발생: {task_exc}", exc_info=True)
    
    except json.JSONDecodeError:
        logger.warning(f"[Session {session_id}] 잘못된 JSON 형식 수신. 연결 유지.")
        if websocket.client_state == WebSocketState.CONNECTED:
             await websocket.send_json({"type": "error", "text": "잘못된 JSON 형식입니다."})
    
    except Exception as e:
        logger.error(f"[Session {session_id}] 웹소켓 메인 루프에서 예기치 않은 예외 발생: {e}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json({"type": "error", "text": f"서버 내부 오류로 연결에 문제가 발생했습니다: {str(e)}"})
            except Exception as send_err:
                 logger.error(f"[Session {session_id}] 웹소켓 오류 메시지 전송 실패: {send_err}", exc_info=True)
    
    finally:
        logger.info(f"[Session {session_id}] 웹소켓 연결 정리 및 종료 시작.")
        session_manager.cleanup_session(session_id)
        if websocket.client_state != WebSocketState.DISCONNECTED:
            logger.info(f"[Session {session_id}] 웹소켓 연결을 명시적으로 닫습니다.")
            try:
                await websocket.close(code=1000)
            except RuntimeError as e:
                 logger.warning(f"[Session {session_id}] 웹소켓 닫기 중 런타임 오류 발생 (이미 닫혔을 수 있음): {e}")
            except Exception as e:
                 logger.error(f"[Session {session_id}] 웹소켓 닫기 중 예외 발생: {e}", exc_info=True)
        logger.info(f"[Session {session_id}] 클라이언트({websocket.client})와의 웹소켓 처리가 완전히 종료되었습니다.")

