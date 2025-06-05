import httpx
import asyncio
import os
import json
import hmac
import hashlib
import base64
from io import BytesIO

# --- 설정 ---
os.environ['BACKEND_HOST'] = 'https://pbl.kro.kr'
SECRET_KEY = '0339e2395f6567e68d171c600f2f5c6a393e90bd852404d9d21f6d54df252cfe'

def generate_hmac(user_id: str):
    """HMAC-SHA256 서명을 생성하고 Base64로 인코딩합니다."""
    return base64.b64encode(hmac.new(SECRET_KEY.encode(), user_id.encode(), hashlib.sha256).digest()).decode()

async def test_upload_weights(weights_to_upload: dict, session_id: str, user_id: str):
    """가중치 파일 업로드 기능을 테스트합니다."""
    print("\n--- 1. 가중치 파일 업로드 테스트 시작 ---")
    if not weights_to_upload:
        print("업로드할 데이터가 없습니다.")
        return None

    backend_host = os.getenv("BACKEND_HOST")
    url = f"{backend_host}/api/v1/preference/ai_file_upload"
    
    # 업로드할 데이터 복사 후 수정 (테스트를 위해 1번 인덱스 가중치 변경)
    upload_payload = weights_to_upload.copy()
    key_to_modify = '1'
    original_value = upload_payload.get(key_to_modify, 1.0)
    upload_payload[key_to_modify] = round(original_value + 0.01, 3)
    print(f"데이터 수정: 인덱스 '{key_to_modify}'의 가중치를 {original_value} -> {upload_payload[key_to_modify]}로 변경하여 업로드합니다.")
    
    # Form-data 준비
    form_data = {'session_id': session_id}
    
    # 파일 준비
    updated_weights_bytes = json.dumps(upload_payload, indent=2).encode('utf-8')
    files = {'file': ('weights_upload_test.json', BytesIO(updated_weights_bytes), 'application/json')}
    
    headers = {"X-Signature-HMAC-SHA256": generate_hmac(user_id)}
    
    async with httpx.AsyncClient() as client:
        try:
            print(f"\n요청 URL: POST {url}")
            print(f"Form 데이터: {form_data}")
            response = await client.post(url, data=form_data, files=files, headers=headers, timeout=10)
            response.raise_for_status()
            
            print("\n[성공] 업로드 성공!")
            print(f"상태 코드: {response.status_code}")
            print(f"서버 응답: {response.text}")
            return upload_payload  # 검증을 위해 업로드된 데이터를 반환
            
        except httpx.HTTPStatusError as e:
            print(f"\n[실패] 업로드 실패: HTTP 상태 코드 {e.response.status_code}, 응답: {e.response.text}")
        except Exception as e:
            print(f"\n[실패] 업로드 중 오류 발생: {e}")
    return None


async def test_download_weights(session_id: str, user_id: str):
    """가중치 파일 다운로드 기능을 테스트합니다."""
    print("\n--- 2. 가중치 파일 다운로드 테스트 시작 ---")
    backend_host = os.getenv("BACKEND_HOST")
    url = f"{backend_host}/api/v1/preference/request_ai_file"
    params = {"session_id": int(session_id)}
    
    headers = {"X-Signature-HMAC-SHA256": generate_hmac(user_id)} 
    
    async with httpx.AsyncClient() as client:
        try:
            print(f"요청 URL: POST {url}")
            print(f"요청 파라미터: {params}")
            response = await client.post(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            print("\n[성공] 다운로드 성공!")
            print(f"상태 코드: {response.status_code}")
            downloaded_weights = response.json()
            print(f"받은 데이터 (첫 5개): {dict(list(downloaded_weights.items())[:5])}")
            return downloaded_weights
            
        except httpx.HTTPStatusError as e:
            print(f"\n[실패] 다운로드 실패: HTTP 상태 코드 {e.response.status_code}, 응답: {e.response.text}")
        except Exception as e:
            print(f"\n[실패] 다운로드 중 오류 발생: {e}")
            
    return None


async def main():
    """테스트를 '업로드 -> 다운로드 -> 검증' 순서로 실행합니다."""
    print("🚀 백엔드 파일 I/O 연동 테스트를 시작합니다. (순서: 업로드 -> 다운로드)")
    
    # 테스트에 사용할 고정 값
    test_session_id = "52"
    test_user_id = "10"
    
    # 1. 업로드를 위한 초기 데이터 생성
    initial_weights = {str(i): 1.0 for i in range(1, 31)}
    
    # 2. 업로드 테스트 실행
    uploaded_data = await test_upload_weights(
        weights_to_upload=initial_weights,
        session_id=test_session_id,
        user_id=test_user_id
    )

    # 3. 다운로드 테스트 실행 (업로드가 성공했을 경우)
    if uploaded_data:
        downloaded_data = await test_download_weights(
            session_id=test_session_id,
            user_id=test_user_id
        )

        # 4. 데이터 검증
        if downloaded_data:
            print("\n--- 3. 업로드/다운로드 데이터 검증 ---")
            key_to_check = '1'
            expected_value = uploaded_data.get(key_to_check)
            actual_value = downloaded_data.get(key_to_check)

            if expected_value == actual_value:
                print(f"[성공] 키 '{key_to_check}'의 값이 예상대로 일치합니다. (업로드값: {expected_value}, 다운로드값: {actual_value})")
            else:
                print(f"[실패] 키 '{key_to_check}'의 값이 일치하지 않습니다! (업로드값: {expected_value}, 다운로드값: {actual_value})")
    
    print("\n✅ 테스트가 모두 종료되었습니다.")

if __name__ == "__main__":
    asyncio.run(main())