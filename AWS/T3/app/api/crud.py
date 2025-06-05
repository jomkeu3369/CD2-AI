import asyncio
import httpx
import os
import logging
from typing import Dict, Any, Optional, List

from app.api.schema import model_list
from app.log import setup_logging
from dotenv import load_dotenv
load_dotenv()

logger: logging.Logger = setup_logging()

session_weights: Dict[str, Dict[str, float]] = {}

class SessionManager:
    _instance: Optional['SessionManager'] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> 'SessionManager':
        if cls._instance is None:
            logger.debug("새로운 SessionManager 인스턴스를 생성합니다.")
            cls._instance = super(SessionManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if SessionManager._initialized:
            return
        logger.debug("SessionManager 인스턴스를 초기화합니다.")
        self.sessions: Dict[str, Dict[str, Any]] = {}
        SessionManager._initialized = True

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """지정된 session_id에 대한 세션 데이터를 가져오거나 생성합니다."""
        if session_id not in self.sessions:
            logger.info(f"세션 ID '{session_id}'에 대한 새 세션 항목을 생성합니다.")
            self.sessions[session_id] = {
                "feedback_event": asyncio.Event(),
                "feedback_data": None,
            }
        else:
            logger.debug(f"세션 ID '{session_id}'에 대한 기존 세션에 접근합니다.")
        return self.sessions[session_id]

    def set_feedback(self, session_id: str, feedback: Optional[Any]):
        """세션에 피드백 데이터를 설정하고 이벤트를 트리거합니다."""
        session = self.get_session(session_id)
        session["feedback_data"] = feedback
        session["feedback_event"].set()
        logger.info(f"세션 ID '{session_id}'에 피드백이 설정되었습니다.")

    async def wait_for_feedback(self, session_id: str, timeout: float) -> Optional[Any]:
        """지정된 시간 동안 세션 피드백을 기다립니다."""
        session = self.get_session(session_id)
        logger.debug(f"세션 ID '{session_id}'의 피드백을 {timeout}초 동안 기다립니다.")
        try:
            await asyncio.wait_for(session["feedback_event"].wait(), timeout=timeout)
            logger.info(f"세션 ID '{session_id}'에 대한 피드백을 받았습니다.")
            return session["feedback_data"]
        except asyncio.TimeoutError:
            logger.warning(f"세션 ID '{session_id}'의 피드백 대기 시간 초과.")
            return None
    
    def cleanup_session(self, session_id: str):
        """세션 ID와 관련된 데이터를 정리합니다."""
        if session_id in self.sessions:
            logger.info(f"세션 ID '{session_id}'를 정리합니다.")
            del self.sessions[session_id]
        else:
            logger.warning(f"존재하지 않는 세션 ID '{session_id}'의 정리를 시도했습니다.")

session_manager = SessionManager()

async def get_model_name_by_id(model_id: int) -> str:
    for model_spec in model_list:
        try:
            if isinstance(model_spec, dict) and 'id' in model_spec and 'name' in model_spec:
                if int(model_spec['id']) == model_id:
                    logger.debug(f"모델 ID {model_id}를 찾았습니다: {model_spec['name']}")
                    return str(model_spec['name'])
            else:
                logger.warning(f"model_list에 잘못된 형식의 모델 항목이 있습니다: {model_spec}")
        except (KeyError, ValueError) as e:
            logger.error(f"model_list의 모델 항목 처리 중 오류 발생: {model_spec}. 오류: {e}")
            continue

    logger.warning(f"모델 ID {model_id}를 model_list에서 찾을 수 없습니다. 기본값 'gpt-4o-mini'를 사용합니다.")
    return "gpt-4o-mini"

async def save_history_to_api(messages_to_save: List[Dict[str, Any]], token: str):
    """
    대화 기록을 백엔드 API에 저장합니다.
    """
    api_url_base = os.getenv('BACKEND_HOST')
    if not api_url_base:
        logger.error("BACKEND_HOST 환경 변수가 설정되지 않았습니다. 기록을 저장할 수 없습니다.")
        return

    url = f"{api_url_base}/api/v1/faiss/add"
    headers = {"Authorization": f"Bearer {token}"}

    logger.debug(f"API에 기록 저장을 시도합니다. URL: {url}, 메시지 수: {len(messages_to_save)}")

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(url, headers=headers, json=messages_to_save)
            response.raise_for_status()
            
            response_data = response.json()
            api_message = response_data.get('message', '응답에 메시지가 없습니다.')
            logger.info(f"API 응답 (기록 저장): {api_message} (상태 코드: {response.status_code})")
        except httpx.HTTPStatusError as e:
            logger.error(f"API 호출 실패 (기록 저장) - 상태 코드 {e.response.status_code}: {e.response.text}", exc_info=True)
        except httpx.RequestError as e:
            logger.error(f"API 요청 오류 (기록 저장): {e}", exc_info=True)
        except ValueError as e:
            logger.error(f"API 응답 JSON 디코딩 오류 (기록 저장): {e}", exc_info=True)
        except Exception as e:
            logger.error(f"기록 저장 중 예기치 않은 오류 발생: {e}", exc_info=True)

async def get_user_id_from_token(token: str) -> Optional[int]:
    backend_host = os.getenv("BACKEND_HOST")
    if not backend_host:
        logger.error("BACKEND_HOST 환경 변수가 설정되지 않았습니다. 토큰을 검증할 수 없습니다.")
        return None

    url = f"{backend_host}/api/v1/auth/internal/validate-token"
    logger.debug(f"사용자 ID를 얻기 위해 토큰을 검증합니다. URL: {url}")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(url, json={"token": token})
        
        response.raise_for_status()
        response_data = response.json()

        if response_data.get("is_valid", False):
            user_id_str = response_data.get("user_id")
            if user_id_str is not None:
                try:
                    user_id = int(user_id_str)
                    logger.info(f"토큰이 성공적으로 검증되었습니다. 사용자 ID: {user_id}")
                    return user_id
                except ValueError:
                    logger.error(f"API에서 반환된 user_id를 정수로 변환할 수 없습니다: '{user_id_str}'")
                    return None
            else:
                logger.warning("토큰은 유효하지만 응답에 user_id가 없습니다.")
                return None
        else:
            logger.warning(f"토큰 검증 실패 (is_valid=false). 응답: {response_data}")
            return None
            
    except httpx.HTTPStatusError as e:
        logger.error(f"토큰 검증 API 요청 실패 - 상태 코드 {e.response.status_code}: {e.response.text}", exc_info=True)
        return None
    except httpx.RequestError as e:
        logger.error(f"토큰 검증 중 API 요청 오류: {e}", exc_info=True)
        return None
    except ValueError as e:
        logger.error(f"토큰 검증 응답 처리 중 오류: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"토큰 검증 중 예기치 않은 오류 발생: {e}", exc_info=True)
        return None

# 강화 학습
async def process_reinforcement_learning(token: str, message_id: str, recommend: bool, session_id: str) -> tuple[bool, str, Optional[Dict[str, float]]]:
    backend_host = os.getenv("BACKEND_HOST")
    if not backend_host:
        logger.error("BACKEND_HOST 환경 변수가 설정되지 않았습니다. 강화 학습을 처리할 수 없습니다.")
        return False, "내부 설정 오류: 백엔드 호스트가 설정되지 않았습니다.", None

    headers = {"Authorization": f"Bearer {token}"}

    logger.info(f"강화 학습 프로세스 시작 - 세션 ID: {session_id}, 메시지 ID: {message_id}, 추천 여부: {recommend}")

    user_id = await get_user_id_from_token(token)
    if user_id is None:
        logger.warning(f"토큰에 대한 user_id를 가져오지 못했습니다. 세션 '{session_id}'의 RL 프로세스를 중단합니다.")
        return False, "유효하지 않은 토큰이거나 사용자 ID를 확인할 수 없습니다.", None

    try:
        history_request_body = {"session_id": int(session_id), "user_id": int(user_id)}
        
        logger.debug(f"RL을 위한 기록 조회 중. URL: {backend_host}/api/v1/faiss/history, 본문: {history_request_body}")
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{backend_host}/api/v1/faiss/history",
                json=history_request_body,
                headers=headers
            )
            response.raise_for_status()
            history_data = response.json()
            logger.debug(f"세션 '{session_id}'에 대한 기록을 성공적으로 조회했습니다.")

    except httpx.HTTPStatusError as e:
        logger.error(f"RL 기록 조회 실패 (세션 {session_id}). 상태 코드 {e.response.status_code}: {e.response.text}", exc_info=True)
        return False, f"대화 기록 조회 실패: HTTP {e.response.status_code}", None
    except httpx.RequestError as e:
        logger.error(f"RL 기록 조회 중 요청 오류 (세션 {session_id}): {e}", exc_info=True)
        return False, "네트워크 문제로 대화 기록 조회에 실패했습니다.", None
    except ValueError:
        logger.error(f"세션 '{session_id}'의 기록 요청/응답 처리 중 오류 발생.", exc_info=True)
        return False, "기록 데이터 처리 중 오류가 발생했습니다.", None
    except Exception as e:
        logger.error(f"RL 기록 조회 중 예기치 않은 오류 발생 (세션 {session_id}): {e}", exc_info=True)
        return False, f"기록 조회 중 예기치 않은 오류 발생: {str(e)}", None

    message_data = None
    for message in history_data.get("messages", []):
        if isinstance(message, dict) and message.get("message_id") == message_id:
            message_data = message
            logger.debug(f"세션 '{session_id}'의 기록에서 메시지 ID '{message_id}'를 찾았습니다.")
            break
    
    if not message_data:
        logger.warning(f"세션 '{session_id}'의 기록에서 메시지 ID '{message_id}'를 찾을 수 없습니다.")
        return False, f"세션 기록에서 해당 메시지 ID({message_id})를 찾을 수 없습니다.", None

    if message_data.get("recommendation_status") is not None:
        logger.info(f"메시지 ID '{message_id}' (세션 '{session_id}')에 대해 이미 강화 학습이 진행되었습니다.")
        return False, "이미 강화학습이 진행된 메시지입니다.", None

    evaluation_indices = message_data.get("evaluation_indices", [])
    if not isinstance(evaluation_indices, list) or not evaluation_indices:
        logger.warning(f"메시지 ID '{message_id}' (세션 '{session_id}')에 평가 인덱스가 없습니다. RL을 적용할 수 없습니다.")
        return False, "강화학습에 필요한 평가 인덱스가 없습니다.", None

    current_session_id_str = str(session_id) 

    if current_session_id_str in session_weights:
        weights_data = session_weights[current_session_id_str]
        logger.info(f"세션 ID '{current_session_id_str}'에 대한 기존 가중치를 로드합니다.")
    else:
        weights_data = {str(i): 1.0 for i in range(1, 31)} 
        logger.info(f"세션 ID '{current_session_id_str}'에 대한 신규 가중치를 생성합니다.")

    learning_rate = 0.05
    logger.debug(f"RL 적용 중. 인덱스: {evaluation_indices}, 추천 여부: {recommend}, 학습률: {learning_rate}")

    updated_indices_info: Dict[str, float] = {}

    for index_val in evaluation_indices:
        key = str(index_val)
        if key in weights_data:
            original_weight = weights_data[key]
            if recommend:
                weights_data[key] = min(1.5, weights_data[key] + learning_rate)
            else:
                weights_data[key] = max(0.5, weights_data[key] - learning_rate)
            
            updated_indices_info[key] = weights_data[key]
            logger.debug(f"인덱스 {key}의 가중치가 {original_weight:.4f}에서 {weights_data[key]:.4f}로 업데이트되었습니다.")
        else:
            logger.warning(f"평가 인덱스 {key}가 세션 '{current_session_id_str}'의 가중치 데이터에 없습니다. 건너뜁니다.")

    session_weights[current_session_id_str] = weights_data
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("="*20 + " 강화 학습 결과 " + "="*20)
        logger.debug(f"세션 ID: {current_session_id_str}") 
        logger.debug(f"메시지 ID: {message_id}")
        logger.debug(f"피드백: {'좋아요' if recommend else '싫어요'}")
        logger.debug(f"평가된 인덱스: {evaluation_indices}")
        logger.debug(f"해당 인덱스에 대한 업데이트된 가중치: {updated_indices_info}")
        logger.debug("="*63)

    return True, "인메모리 강화학습 계산 완료", updated_indices_info