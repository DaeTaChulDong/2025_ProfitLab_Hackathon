# 2025_ProfitLab_Hackathon

---

# 🎬 Think:it Pro - 데이터 기반 AI 유튜브 컨설팅 솔루션

**Think:it Pro**는 사용자가 업로드한 영상을 멀티모달(Vision & Audio) AI로 분석하고, 실제 유튜브 인기 급상승 동영상 데이터와 비교하여 조회수를 높이는 구체적인 솔루션(썸네일, 제목, 전략)을 제공하는 AI 웹 서비스입니다.

---

## 🛠 실행 환경 (Execution Environment)

이 프로젝트는 고성능 AI 모델(Whisper Large-v3 등)의 원활한 구동을 위해 다음 환경에 최적화되어 있습니다.

* **Platform:** Google Colab
* **GPU:** **NVIDIA A100** (권장) 또는 T4 이상
* *Whisper Large 모델과 영상 처리 로직의 병렬 처리를 위해 고성능 GPU가 필요합니다.*


* **External Access:** Cloudflare Tunnel (`cloudflared`) 사용

---

## 📦 필수 라이브러리 (Requirements)

프로젝트 구동을 위해 필요한 핵심 Python 라이브러리입니다.

```bash
pip install streamlit openai torch torchvision torchaudio transformers accelerate moviepy opencv-python pandas requests

```

* **Streamlit:** 웹 인터페이스(UI) 구현
* **OpenAI:** GPT-4o (분석/기획) 및 DALL-E 3 (썸네일 생성) API 연동
* **Torch & Transformers:** Whisper (STT) 모델 구동 및 가속화
* **MoviePy & OpenCV:** 영상 데이터 전처리 (오디오 분리 및 프레임 추출)

---

## 🧠 메인 백엔드 로직 (Main Backend Logic)

Think:it Pro는 단순한 API 래퍼가 아닌, **데이터 기반의 복합적인 파이프라인**을 통해 결과를 도출합니다.

### 1. 멀티모달 데이터 추출 (Multimodal Extraction)

* **Audio Pipeline:** `MoviePy`를 사용하여 영상에서 오디오 트랙을 분리한 후, OpenAI의 **Whisper Large-v3** 모델을 로컬 GPU에 로드하여 고정밀 STT(Speech-to-Text)를 수행합니다. 이를 통해 대본 추출 및 발화 속도(WPM)를 계산합니다.
* **Vision Pipeline:** `OpenCV`를 활용하여 영상의 타임라인을 분석, 시청자 이탈이 가장 많은 구간(Hook, Body, Climax)의 핵심 프레임을 이미지로 추출합니다.

### 2. 데이터 드리븐 벤치마킹 (Data-Driven Benchmarking)

* **Real-time Comparison:** 추출된 사용자 데이터(대본, 시각 정보)를 `youtube_top200_data.csv`에 저장된 '실제 인기 영상 메타데이터'와 실시간으로 비교합니다.
* **Scoring Algorithm:** 해당 카테고리 상위 1% 영상들의 평균 조회수, 키워드 패턴, 썸네일 특징을 기준점으로 삼아 사용자의 영상을 60~100점 척도로 평가합니다.

### 3. 생성형 AI 솔루션 (Generative AI Solution)

* **GPT-4o Reasoning:** 시각 데이터와 청각 데이터를 종합하여, 클릭률(CTR)을 높일 수 있는 제목 3종과 개선 전략을 도출합니다.
* **DALL-E 3 Thumbnail:** 분석된 데이터를 바탕으로 GPT-4o가 최적의 프롬프트를 작성하고, 이를 DALL-E 3에 전달하여 **클릭을 부르는 고화질 썸네일**을 즉시 생성 및 제공합니다.

---

## 📂 주요 파일 설명 (File Structure)

### 1. `app.py` (Main Application)

* Streamlit 기반의 메인 웹 애플리케이션입니다.
* UI 렌더링, 세션 상태 관리(`st.session_state`), 모델 Lazy Loading, 이메일 발송(SMTP) 등 서비스의 모든 핵심 기능이 통합되어 있습니다.

### 2. `VideoCollect.py` (Data Crawler)

* **역할:** 프로젝트의 핵심 자산인 **벤치마킹 데이터셋을 구축하는 수집기**입니다.
* **기능:** YouTube Data API를 활용하여 현재 '인기 급상승 동영상(Trending)' 및 주요 카테고리별 상위 랭킹 영상들의 메타데이터(제목, 조회수, 썸네일, 태그 등)를 수집합니다.
* **산출물:** 이 코드를 실행하여 생성된 결과물이 `youtube_top200_data.csv`이며, AI 분석의 기준 지표(Ground Truth)로 활용됩니다.

### 3. `API_KEY_Error.py` (Troubleshooter)

* **역할:** OpenAI API 키 또는 환경 변수 설정 오류를 진단하는 유틸리티입니다.
* **기능:** `secrets.toml` 파일 로드 실패나 API 인증 오류가 발생했을 때, 연결 상태를 테스트하고 디버깅 정보를 제공하여 신속한 문제 해결을 돕습니다.

### 4. `youtube_top200_data.csv` (Dataset)

* `VideoCollect.py`를 통해 수집된 실제 유튜브 인기 영상 데이터셋입니다. AI가 "잘된 영상"의 기준을 학습하고 비교하는 데 사용됩니다.

---

## 🚀 실행 방법 (How to Run)

1. **Google Colab**을 엽니다. (런타임 유형: T4 또는 A100 GPU 선택)
2. `app.py` 및 필요한 파일들을 업로드합니다.
3. 필수 라이브러리를 설치합니다.
4. 보안 비밀(Secrets)에 `OPENAI_API_KEY`, `EMAIL_SENDER`, `EMAIL_PASSWORD`를 설정합니다.
5. 아래 명령어로 서버를 실행합니다.

```bash
# Cloudflare Tunnel을 통한 외부 접속 실행
!wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64
!streamlit run app.py & ./cloudflared-linux-amd64 tunnel --url http://localhost:8501

```
