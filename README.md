# CD2-AI

2025년 종합전공PBL2 "우문현답" 팀의 AI 모델 레포지토리입니다.

## 전체 프로젝트 파이프라인
<img width="4787" height="2545" alt="프로젝트 종합 파이프라인 (최종) (1)" src="https://github.com/user-attachments/assets/f2846894-8f23-4a00-8423-b24aedeecb9c" />

## 프로젝트 개요

**우문현답**은 사용자의 프롬프트를 평가하고 최적화하여 LLM이 더 나은 결과를 생성하도록 돕는 AI 기반 프롬프트 엔지니어링 프로젝트입니다. 
이 프로젝트는 LangChain과 LLM(대규모 언어 모델)을 활용하여 다음과 같은 기능을 제공합니다.

-   **프롬프트 평가**: 사용자가 입력한 프롬프트를 명확성, 간결성, 목표 적합성 등 다양한 기준에 따라 평가합니다.
-   **프롬프트 최적화**: 평가 점수가 낮은 항목에 대해 개선 방안을 제안하고, 이를 바탕으로 최적화된 프롬프트를 생성합니다.
-   **Tree of Thoughts (ToT) 리포트**: 사용자가 원할 경우, 프롬프트 주제에 대한 심층 분석 리포트를 생성합니다.
-   **인간 피드백 기반 강화학습(RLHF)**: 사용자의 피드백(좋아요/싫어요)을 받아 모델의 가중치를 동적으로 조절하여 개인화된 사용자 경험을 제공합니다.

## 기술 스택

### 주요 기술

-   **언어**: Python 3.12
-   **프레임워크**: FastAPI
-   **AI/ML**:
    -   LangChain & LangGraph: AI 에이전트 및 그래프 기반 워크플로우 관리
    -   Ollama: 로컬 LLM(대규모 언어 모델) 실행 환경
    -   DeepL: 텍스트 번역
    -   Pytorch, Transformers: 딥러닝 모델 개발 및 활용
-   **컨테이너**: Docker, Docker Compose
-   **웹 서버**: Nginx (리버스 프록시 및 SSL/TLS 처리)

### 아키텍처

이 프로젝트는 마이크로서비스 아키텍처로 구성되어 있으며, 각 서비스는 Docker 컨테이너로 실행됩니다.

-   **FastAPI Backend (`backend`)**: API 엔드포인트 및 웹소켓 통신을 처리하는 메인 애플리케이션 서버입니다.
-   **Ollama (`ollama`)**: `gemma3:12b`, `llama3.1:8b`와 같은 로컬 LLM을 실행하고 관리하는 서비스입니다.
-   **Nginx (`nginx_proxy`)**: 리버스 프록시 역할을 수행하며, SSL/TLS 인증서 관리 및 로드 밸런싱을 담당합니다.
-   **Certbot (`certbot`)**: Let's Encrypt를 사용하여 SSL/TLS 인증서를 자동으로 발급하고 갱신합니다.

## 설치 및 실행 방법

### 사전 요구 사항

-   Docker
-   Docker Compose
-   NVIDIA 그래픽 카드 및 NVIDIA Container Toolkit (GPU 가속을 위해 필요)

### 실행 절차

1.  **레포지토리 클론**:
    ```bash
    git clone [https://github.com/jomkeu3369/cd2-ai.git](https://github.com/jomkeu3369/cd2-ai.git)
    cd cd2-ai/AWS/G4dn
    ```

2.  **환경 변수 설정**:
    `.env` 파일을 생성하고 필요한 환경 변수를 설정합니다.
    ```bash
    # .env 파일 예시
    DEEPL_API_KEY="your_deepl_api_key"
    TAVILY_API_KEY="your_tavily_api_key"
    OLLAMA_BASE_URL="http://ollama:11434"
    BACKEND_HOST="https://your_domain.com"
    ```

3.  **Docker Compose를 이용한 실행**:
    - **개발 환경**
        ```bash
        docker-compose -f docker-compose.yaml up --build
        ```
    - **배포 환경 (Nginx 포함)**
        ```bash
        # 1. 네트워크 생성
        docker network create app_network

        # 2. Ollama 및 FastAPI 실행
        docker-compose -f docker-compose.yaml up --build -d

        # 3. Nginx 및 Certbot 실행
        docker-compose -f docker-compose-nginx.yaml up --build -d
        ```

## 주요 기능 및 로직

### 1. 프롬프트 최적화 워크플로우 (LangGraph)

-   **`topic_evaluation_node`**: 입력된 프롬프트와 주제 간의 연관성을 평가하여 0.6 미만일 경우 처리를 중단합니다.
-   **`translation_prompt_node`**: 정확한 평가를 위해 프롬프트를 영어로 번역합니다.
-   **`evaluate_prompt_node`**: 번역된 프롬프트를 6가지 대분류와 30가지 소분류 기준에 따라 평가합니다.
-   **`improvement_prompt_node`**: 평가 점수가 낮은 항목에 대해 개선 방안을 제안합니다.
-   **`hitl_request_node`**: 모호하거나 부족한 정보가 있을 경우, 사용자에게 명확한 질문을 생성하여 피드백을 요청합니다 (Human-in-the-Loop).
-   **`enhance_prompt_node`**: 생성된 개선안과 사용자 피드백을 종합하여 최종적으로 최적화된 프롬프트를 한국어로 생성합니다.

### 2. Tree of Thoughts (ToT) 리포트 생성

-   **`dr_initializer_node`**: 상세 리포트 생성이 요청되면, Tavily API를 통해 웹 검색을 수행하여 초기 컨텍스트를 수집하고 ToT 프로세스를 초기화합니다.
-   **`dr_thought_generator_node`, `dr_evaluator_node`, `dr_tot_control_node`**: 생각을 확장하고 평가하는 과정을 반복하며 최적의 아이디어를 도출합니다.
-   **`dr_markdown_report_generator_node`**: 도출된 핵심 아이디어와 웹 검색 결과를 바탕으로 심층적인 마크다운 형식의 리포트를 생성합니다.

### 3. 강화학습 및 사용자 피드백

-   **`reinforce_feedback_endpoint`**: 사용자가 '좋아요' 또는 '싫어요' 피드백을 제출하면 이 엔드포인트가 호출됩니다.
-   **`process_reinforcement_learning`**:
    -   백엔드 서버에서 해당 메시지의 대화 기록과 평가 인덱스를 조회합니다.
    -   피드백(추천/비추천)에 따라 `session_weights` 딕셔너리에 저장된 평가 항목별 가중치를 동적으로 업데이트합니다. (좋아요: +0.05, 싫어요: -0.05)
    -   이렇게 업데이트된 가중치는 다음 프롬프트 평가 시에 적용되어 개인화된 최적화 결과를 제공합니다.
