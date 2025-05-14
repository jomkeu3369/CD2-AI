```python
import socket

# 서버 측 코드
def server_code(host, port):
    """
    TCP 서버를 시작하고 클라이언트 연결을 수락합니다.
    """
    try:
        # 소켓 생성
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # 주소 바인딩
        server_socket.bind((host, port))
        
        # 연결 대기 (백로그 5)
        server_socket.listen(5)
        print(f"서버 시작: {host}:{port} 에서 연결 대기 중...")
        
        while True:
            # 클라이언트 연결 수락
            client_socket, address = server_socket.accept()
            print(f"클라이언트 연결됨: {address}")
            try:
                # 클라이언트와 데이터 교환                  
                while True:
                    data = client_socket.recv(1024)  # 1024바이트씩 수신
                    if not data:
                        break  # 클라이언트가 연결 종료
                        
                        print(f"수신 데이터: {data.decode()}")
                        client_socket.sendall(data)  # 에코: 수신한 데이터를 다시 보냄
                        
                        except Exception as e:
                        print(f"클라이언트 통신 오류: {e}")
                        finally:                # 클라이언트 연결 닫기
                        client_socket.close()
                        print(f"클라이언트 연결 종료: {address}")
            except Exception as e:
                print(f"서버 시작 오류: {e}")
            finally:
                # 서버 소켓 닫기
                server_socket.close()
                print("서버 종료")
                
            # 클라이언트 측 코드
            def client_code(host, port, message):
                """
                    TCP 서버에 연결하고 데이터를 전송합니다.
                """
                try:        
                    # 소켓 생성
                    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    
                    # 서버에 연결
                        client_socket.connect((host, port))
                        print(f"서버에 연결됨: {host}:{port}")

                        # 데이터 전송
                        client_socket.sendall(message.encode())
                        print(f"데이터 전송: {message}")
                            
                        # 서버 응답 수신
                        data = client_socket.recv(1024)
                        print(f"수신 데이터: {data.decode()}")
                        
                except Exception as e:
                    print(f"클라이언트 오류: {e}")
                finally:
                    # 소켓 닫기
                    client_socket.close()
                    print("클라이언트 종료")
                
                
                if __name__ == "__main__":
                    HOST = '127.0.0.1'  # 로컬 호스트
                    PORT = 65432  # 사용할 포트 (1024보다 큰 값)
                    # 서버 시작 (별도의 스레드에서 실행하는 것이 좋습니다.)
                    
                    import threading
                    server_thread = threading.Thread(target=server_code, args=(HOST, PORT))
                    server_thread.daemon = True  # 프로그램 종료 시 스레드도 종료
                    server_thread.start()
                    
                    # 클라이언트 코드 실행
                    MESSAGE = "Hello, Server!"
                    client_code(HOST, PORT, MESSAGE)
                    
```
                    
**코드 설명:**

1.  **`server_code(host, port)`:**
    *   **소켓 생성:** `socket.socket(socket.AF_INET, socket.SOCK_STREAM)`은 IPv4 주소체계(`AF_INET`)와 TCP 소켓(`SOCK_STREAM`)를 사용하여 소켓을 생성합니다.
    *   **주소 바인딩:** `server_socket.bind((host, port))`은 소켓을 특정 IP 주소와 포트 번호에 연결합니다.
    *   **연결 대기:** `server_socket.listen(5)`는 서버가 클라이언트 연결을 수신할 준비가 되었음을 나타냅니다.  `5`는 대기열에 최대 5개의 연결 요청을 보관할 수 있음을 의미합니다.
    *   **클라이언트 연결 수락:** `server_socket.accept()`는 클라이언트 연결 요청을 기다리고 수락합니다.  이 함수는 클라이언트 소켓(`client_socket`)과 클라이언트의 주소(`address`)를 반환합니다.
    *   **데이터 교환:** `client_socket.recv(1024)`는 클라이언트로부터 1024바이트의 데이터를 수신합니다. `client_socket.sendall(data)`는 수신한 데이터를 다시 클라이언트로 전송합니다 (에코).
    *   **클라이언트 연결 종료:** `client_socket.close()`는 클라이언트와 연결을 닫습니다.
    *   **오류 처리:** `try...except...finally` 블록을 사용하여 오류를 처리하고 소켓을 안전하게 닫습니다.
    *   `server_thread.daemon = True`는 서버 스레드를 데몬 스레드로 설정하여 프로그램이 종료될 때 자동으로 종료되도록 합니다.

2.  **`client_code(host, port, message)`:**
    *   **소켓 생성:**  서버 코드와 동일한 방식으로 소켓을 생성합니다.
    *   **서버 연결:** `client_socket.connect((host, port))`는 지정된 IP 주소와 포트 번호의 서버에 연결을 시도합니다.
    *   **데이터 전송:** `client_socket.sendall(message.encode())`는 메시지를 바이트(`encode()`)로 변환하여 서버로 전송합니다. `sendall()` 함수는 모든 데이터를 전송할 때까지 계속해서 데이터를 전송하므로 `send()` 함수보다 더 안전합니다.
    *   **서버 응답 수신:** `client_socket.recv(1024)`는 서버로부터 1024바이트의 데이터를 수신합니다.
    *   **소켓 닫기:** `client_socket.close()`는 소켓을 닫습니다.
    *   **오류 처리:** `try...except...finally` 블록을 사용하여 오류를 처리하고 소켓을 안전하게 닫습니다.
    
**사용 방법:**

1.  **코드 저장:** 위의 코드를 `tcp_socket.py`와 같은 파일로 저장합니다.
2.  **서버 실행:** 터미널에서 `python tcp_socket.py`를 실행합니다.  (서버 코드가 실행됩니다.)
3.  **클라이언트 실행:**  새로운 터미널 창에서 `python tcp_socket.py`를 실행합니다. (클라이언트 코드가 실행됩니다.)
4.  **결과 확인:** 서버 터미널에서는 클라이언트의 연결을 수락하고 받은 메시지를 출력합니다. 클라이언트 터미널에서는 서버에 연결하고 메시지를 전송한 후 서버로부터 응답을 받습니다.

**참고:**

*   `HOST` 변수는 서버가 수신 대기할 IP 주소를 지정합니다.  `'127.0.0.1'`은 로컬 호스트를 의미합니다.
*   `PORT` 변수는 서버가 수신 대기할 포트 번호를 지정합니다.  1024보다 큰 포트 번호를 사용하는 것이 좋습니다.  이미 사용 중인 포트 번호는 사용할 수 없습니다.
*   `recv()` 함수의 인수는 서버가 한 번에 수신할 최대 바이트 수를 나타냅니다.  더 큰 데이터를 수신하려면 이 값을 늘려야 합니다.
*   실제 환경에서는 서버 코드를 별도의 프로세스 또는 스레드로 실행하는 것이 좋습니다.  이렇게 하면 클라이언트가 서버에 연결할 때 서버가 응답하지 않게 됩니다.
*   이 코드는 기본적인 TCP 소켓 통신 예제입니다.  더 복잡한 응용 프로그램을 만들려면 추가적인 기능을 구현해야 할 수 있습니다.  예를 들어, 오류 처리, 데이터 암호화, 연결 유지 등이 필요할 수 있습니다.

이 코드를 실행하면 서버는 클라이언트의 연결을 수락하고 "Hello, Server!" 메시지를 수신한 후 다시 클라이언트로 전송합니다. 클라이언트는 서버에 연결하고 메시지를 전송한 후 서버로부터 응답을 받습니다."