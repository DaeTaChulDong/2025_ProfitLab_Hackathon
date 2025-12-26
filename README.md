# 2025_ProfitLab_Hackathon

---

# 🎬 Think:it Pro - 데이터 기반 AI 유튜브 컨설팅 솔루션

**Think:it Pro**는 사용자가 업로드한 영상을 멀티모달(Vision & Audio) AI로 분석하고, 실제 유튜브 인기 급상승 동영상 데이터와 비교하여 조회수를 높이는 구체적인 솔루션(썸네일, 제목, 전략)을 제공하는 AI 웹 서비스입니다.

---

## [소개]
본 프로젝트는 AI 기반 유튜브 콘텐츠 분석 및 맞춤형 전략 컨설팅 서비스입니다. 급변하는 유튜브 시장에서 크리에이터가 '감'이 아닌 데이터 기반의 명확한 성공 전략을 수립할 수 있도록 돕습니다. 시장 트렌드와 사용자 콘텐츠의 고유한 매력을 결합하여, 조회수를 극대화할 수 있는 최적의 썸네일과 제목을 제안합니다.

## [대상]
- 유튜브 채널 성장을 원하는 초보 및 중급 크리에이터
- 시행착오와 시간 낭비를 줄이고 빠르게 채널을 성장시키고자 하는 사용자
- 어떤 썸네일과 제목이 효과적인지 객관적인 데이터가 필요한 사용자

## [주요 기능]
**1. 시장 트렌드 분석**: 특정 카테고리(예: 게임, 요리, 여행) 내 높은 조회수 영상 데이터를 AI가 학습하여 현재 시장의 성공 공식을 도출합니다.
- 썸네일 스타일 분석:
  - 인물/사물 배치, 구도, 배경 등의 시각적 요소 분석
  - 색감, 대비 등 주목도를 높이는 컬러 팔레트 분석
  - 폰트 스타일, 크기, 위치 등 가독성과 매력을 높이는 텍스트 요소 분석
- 제목 키워드 패턴 분석
  -  자주 사용되는 고효율 키워드(예: '역대급', '꿀팁', '충격') 추출
  -  문장 구조, 길이, 감정적 요소 포함 여부 등 제목 패턴 분석
    
**2. 내 콘텐츠 분석**: 사용자가 업로드할 영상을 AI가 분석하여 고유한 매력 포인트를 추출합니다.
- 가장 인상적인 장면 분석: 표정, 구도 등을 분석하여 썸네일로 활용하기 좋은 'Best 컷' 또는 '하이라이트 장면' 제시
- 핵심 키워드 추출: 영상의 내용과 스크립트를 기반으로 가장 중요한 주제와 키워드 자동 추출
- 하이라이트 구간 식별: 시청자의 몰입도가 높을 것으로 예상되는 구간을 분석

**3. 데이터 기반 맞춤형 전략 컨설팅**: '시장의 성공 공식'과 '내 영상의 매력 포인트'를 결합하여 가장 최적화된 결과물을 제안합니다.
- 썸네일 최적화 제안
  - "현재 IT리뷰 채널 성공 공식에 따르면, 썸네일에 인물과 제품이 함께 나오는 것이 효과적입니다."
  - "당신 영상의 3분 15초 표정이 가장 좋으니, 이 장면을 활용해 이런 스타일로 만드세요." (스타일 예시와 함께 구체적인 가이드라인 제공)
- 제목 최적화 제안
  - "인기 여행 영상들은 '역대급', '인생샷' 같은 키워드를 많이 사용합니다."
  - "당신 영상의 핵심 주제 **'제주도'**와 결합하여 '내 인생샷 건진 제주도 역대급 숨은 명소' 같은 제목을 추천합니다."

## [문제 정의]
- Key words: AI 컨설턴트, 멀티모달 AI, 트렌드 분석, 맞춤형 전략
- 누구를 위해: 유튜브 채널 성장에 어려움을 겪는 크리에이터, 시간 및 노동력 절약이 필요한 크리에이터
- 해결하는 문제: 크리에이터들이 직감에 의존하여 썸네일과 제목을 만들어 시간 낭비와 성장 정체를 겪는 문제. (시장의 성공 방식과 내 콘텐츠의 장점을 객관적으로 파악하기 어려움)
- 어떤 기술로: 유튜브 데이터 마이닝, 컴퓨터 비전, 자연어 처리 AI 기술
- 무엇을 만들려고 하는가: 시장 성공 공식과 사용자 콘텐츠의 매력 포인트를 결합한 데이터 기반의 맞춤형 썸네일/제목 전략 컨설팅 플랫폼

---

## 🛠 실행 환경 (Execution Environment)

* **Platform:** Google Colab
* **GPU:** **NVIDIA A100**
* *Whisper Large 모델과 영상 처리 로직의 병렬 처리를 위해 고성능 GPU 필요.*


* **External Access:** Cloudflare Tunnel (cloudflared 사용)

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

Think:it Pro는 **데이터 기반의 복합적인 파이프라인**을 통해 결과를 도출합니다.

### 1. 멀티모달 데이터 추출 (Multimodal Extraction)

* **Audio Pipeline:** `MoviePy`를 사용하여 영상에서 오디오 트랙을 분리한 후, OpenAI의 **Whisper Large-v3** 모델을 로컬 GPU에 로드하여 고정밀 STT(Speech-to-Text)를 수행합니다. 이를 통해 대본 추출 및 발화 속도(WPM)를 계산합니다.
* **Vision Pipeline:** `OpenCV`를 활용하여 영상의 타임라인을 분석, 시청자 이탈이 가장 많은 구간(Hook, Body, Climax)의 핵심 프레임을 이미지로 추출합니다.

### 2. 데이터 드리븐 벤치마킹 (Data-Driven Benchmarking)

* **Real-time Comparison:** 추출된 사용자 데이터(대본, 시각 정보)를 `youtube_top200_data.csv`에 저장된 '실제 인기 영상 메타데이터'와 실시간으로 비교합니다.
* **Scoring Algorithm:** 해당 카테고리 조회수 상위 영상들의 평균 조회수, 키워드 패턴, 썸네일 특징을 기준점으로 삼아 사용자의 영상을 60~100점 척도로 평가합니다.

### 3. 생성형 AI 솔루션 (Generative AI Solution)

Personalized Prompt Engineering: 사용자가 입력한 한국어 요청 사항("텍스트를 크게 넣어줘", "밝은 분위기로 해줘")을 GPT-4o가 영어로 번역하여 이미지 생성 프롬프트에 반영합니다.

Thumbnail Variations: DALL-E 3를 활용하여 서로 다른 3가지 전략의 썸네일을 생성합니다.

Style 1 (High CTR): 강렬한 색감과 큰 텍스트로 클릭을 유도하는 어그로형

Style 2 (Emotional): 감성적이고 스토리텔링이 강조된 스타일

Style 3 (Informative): 깔끔하고 정보 전달이 명확한 스타일

Text Rendering: 영상의 핵심 키워드를 추출하여 AI가 생성된 이미지 내에 직접 텍스트(타이포그래피)를 렌더링하도록 지시합니다.

* **GPT-4o Reasoning:** 시각 데이터와 청각 데이터를 종합하여, 클릭률(CTR)을 높일 수 있는 제목 3종과 개선 전략을 도출합니다.
* **DALL-E 3 Thumbnail:** 분석된 데이터를 바탕으로 GPT-4o가 최적의 프롬프트를 작성하고, 이를 DALL-E 3에 전달하여 **클릭을 부르는 고화질 썸네일**을 즉시 생성 및 제공합니다.

---

## 📂 주요 파일 설명 (File Structure)

### 1. `main.py` (Main Application)

Streamlit 기반의 메인 웹 애플리케이션입니다.

Session State: 분석 결과와 생성된 이미지를 캐싱하여, 이메일 발송 등 추가 작업 시 리로딩 없이 데이터를 유지합니다.

Strict Language Separation: 분석 리포트는 한국어로, 이미지 생성 프롬프트는 영어로 처리하도록 GPT-4o의 출력을 엄격하게 제어합니다.

Email Service: SMTP를 연동하여 분석된 전체 리포트를 사용자의 이메일로 즉시 발송합니다.

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

1. **Google Colab**을 엽니다. (런타임 유형: A100 GPU 선택)
2. `main.py` 및 필요한 파일들을 업로드합니다.
3. 필수 라이브러리를 설치합니다.
4. 보안 비밀(Secrets)에 `OPENAI_API_KEY`, `EMAIL_SENDER`, `EMAIL_PASSWORD`를 설정합니다.
5. 아래 명령어로 서버를 실행합니다.

```bash
# Cloudflare Tunnel을 통한 외부 접속 실행
!wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64
!streamlit run app.py & ./cloudflared-linux-amd64 tunnel --url http://localhost:8501

```
