%%writefile app.py
import streamlit as st
import os
import time
import base64
import json
import pandas as pd
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from google.colab import userdata

# [중요] 무거운 라이브러리는 Lazy Loading (속도 최적화)

# =========================================================
# 1. 페이지 설정 & 세션 상태 초기화
# =========================================================
st.set_page_config(page_title="Think:it Pro", page_icon="⚡", layout="wide")

# 세션 상태(Session State) 초기화
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'frames_data' not in st.session_state:
    st.session_state.frames_data = None
if 'dalle_variations' not in st.session_state:
    st.session_state.dalle_variations = []
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

# =========================================================
# 2. CSS 스타일
# =========================================================
st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem; color: #222; }
    .section-header {
        font-size: 1.4rem; font-weight: bold; margin-top: 30px; margin-bottom: 15px;
        color: #333; border-left: 5px solid #FF4B4B; padding-left: 12px;
    }
    .score-circle-container { display: flex; justify-content: center; align-items: center; height: 100%; }
    .score-circle {
        position: relative; width: 160px; height: 160px; border-radius: 50%;
        border: 8px solid #FF4B4B; display: flex; justify-content: center;
        align-items: center; flex-direction: column; background-color: #fff;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .score-num { font-size: 4rem; font-weight: 900; color: #FF4B4B; line-height: 1; }
    .score-max { font-size: 1.2rem; color: #999; font-weight: normal; }
    .score-comment { text-align: center; font-size: 1.2rem; font-weight: bold; color: #555; margin-top: 15px; }
    .info-card {
        background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 12px;
        padding: 20px; height: 100%; display: flex; flex-direction: column; justify-content: center;
    }
    .summary-card {
        background-color: #fff; border: 2px solid #eee; border-radius: 12px;
        padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px;
    }
    .rank-tag {
        display: block; width: 100%; text-align: center;
        padding: 8px 0; border-radius: 8px; font-weight: bold; color: white; margin-bottom: 10px;
    }
    .bg-1 { background-color: #FFD700; color: #333; }
    .bg-2 { background-color: #C0C0C0; color: #333; }
    .bg-3 { background-color: #CD7F32; color: white; }
    .dalle-card {
        border: 1px solid #ddd; background-color: #fff; border-radius: 12px; padding: 15px; text-align: center; height: 100%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stFileUploader { padding: 15px; border: 2px dashed #FF4B4B; border-radius: 15px; text-align: center;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. 기능 함수
# =========================================================

def send_email_smtp(to_email, subject, body):
    try:
        sender_email = userdata.get('EMAIL_SENDER')
        sender_password = userdata.get('EMAIL_PASSWORD')
    except:
        try:
            sender_email = st.secrets["EMAIL_SENDER"]
            sender_password = st.secrets["EMAIL_PASSWORD"]
        except:
            return False, "❌ 보안 비밀(EMAIL_SENDER, EMAIL_PASSWORD)을 찾을 수 없습니다."

    if not sender_email or not sender_password:
        return False, "❌ 이메일 설정이 비어있습니다. Colab 보안 비밀을 확인하세요."

    try:
        msg = MIMEMultipart()
        msg['From'] = str(Header(f"Think:it AI <{sender_email}>", 'utf-8'))
        msg['To'] = to_email
        msg['Subject'] = Header(subject, 'utf-8')
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, to_email, text)
        server.quit()
        return True, "✅ 이메일이 성공적으로 발송되었습니다!"
    except Exception as e:
        return False, f"❌ 발송 에러: {str(e)}"

def generate_dalle_image(client, prompt):
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        img_data = requests.get(image_url).content
        return image_url, img_data
    except Exception as e:
        return None, None

# =========================================================
# 4. UI 및 데이터 로드
# =========================================================
st.markdown('<div class="main-title">✨ Think:it Pro | AI 컨설팅</div>', unsafe_allow_html=True)

@st.cache_data
def load_benchmark_data():
    if os.path.exists("youtube_top200_data.csv"):
        return pd.read_csv("youtube_top200_data.csv")
    return pd.DataFrame()

df = load_benchmark_data()
cat_list = df['Category_Name'].unique() if not df.empty else ["General", "Vlog", "Gaming"]

# 사이드바
with st.sidebar:
    st.header("📊 설정")
    category = st.selectbox("카테고리 선택", cat_list)
    
    st.markdown("---")
    st.header("💬 AI 맞춤 요청 (Option)")
    st.caption("원하는 분위기나 필수 텍스트를 적어주세요.")
    user_custom_prompt = st.text_area(
        "추가 프롬프트 입력", 
        placeholder="예: 텍스트 'VLOG'를 크게 넣어줘. 전체적으로 밝은 톤으로.",
        height=100
    )
    
    st.markdown("---")
    st.info("💡 Cloudflare Tunnel로 연결되어 훨씬 빠릅니다.")

# 메인 업로더
with st.expander("📤 영상 파일 업로드 (MP4)", expanded=True):
    uploaded_file = st.file_uploader("여기에 파일을 드래그하거나 선택하세요", type=["mp4"])

# =========================================================
# 5. 분석 로직 (버튼 클릭 시)
# =========================================================
if uploaded_file:
    tfile = "temp_input.mp4"
    with open(tfile, "wb") as f:
        f.write(uploaded_file.read())
    
    if st.button("🚀 AI 데이터 분석 & 썸네일 생성 시작", type="primary", use_container_width=True):
        
        st.session_state.uploaded_file_name = uploaded_file.name
        
        with st.status("⚙️ AI 엔진 가동 중... (약 1~2분 소요)", expanded=True) as status:
            try:
                import torch
                import cv2
                from moviepy.editor import VideoFileClip
                from openai import OpenAI
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
                import numpy as np
                import re
                from collections import Counter
            except ImportError as e:
                st.error("라이브러리 로딩 실패")
                st.stop()

            API_KEY = None
            try: API_KEY = userdata.get('OPENAI_API_KEY')
            except: 
                try: API_KEY = st.secrets["OPENAI_API_KEY"]
                except: pass
            
            if not API_KEY:
                st.error("🚨 API 키가 없습니다.")
                st.stop()
                
            client = OpenAI(api_key=API_KEY)

            @st.cache_resource
            def load_models():
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                model_id = "openai/whisper-large-v3"
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
                )
                model.to(device)
                processor = AutoProcessor.from_pretrained(model_id)
                pipe = pipeline(
                    "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor, max_new_tokens=128,
                    chunk_length_s=30, batch_size=16, return_timestamps=True,
                    torch_dtype=torch_dtype, device=device,
                )
                return pipe

            st.write("🎙️ Whisper 모델 준비 중...")
            whisper_pipe = load_models()
            
            st.write("👀 영상/오디오 데이터 추출 중...")
            clip = VideoFileClip(tfile)
            audio_path = "temp_audio.mp3"
            clip.audio.write_audiofile(audio_path, logger=None)
            transcription = whisper_pipe(audio_path, generate_kwargs={"language": "korean"})
            text = transcription["text"]
            duration = clip.duration
            wpm = (len(text.split()) / duration) * 60
            
            cap = cv2.VideoCapture(tfile)
            temp_frames_data = []
            timestamps = [duration * 0.15, duration * 0.5, duration * 0.85]
            
            def encode_image(img):
                _, buffer = cv2.imencode('.jpg', img)
                return base64.b64encode(buffer).decode('utf-8')
            
            def convert_to_bytes(img):
                is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                return buffer.tobytes()

            for t in timestamps:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    temp_frames_data.append({
                        "img": frame_rgb, 
                        "time_str": time.strftime('%M:%S', time.gmtime(t)),
                        "time_sec": t,
                        "b64": encode_image(frame_bgr),
                        "bytes": convert_to_bytes(frame_rgb)
                    })
            cap.release()
            clip.close()
            if os.path.exists(audio_path): os.remove(audio_path)

            # GPT-4o 분석
            st.write("🧠 GPT-4o 심층 분석 및 카피라이팅 기획 중...")
            
            custom_instruction_text = ""
            if user_custom_prompt:
                custom_instruction_text = f"""
                [User's Custom Request]
                User Input: "{user_custom_prompt}"
                Action: 
                1. Translate this input into English.
                2. Apply the translated intent to ALL 'thumbnail_variations' prompts.
                """

            prompt = f"""
            당신은 한국의 유튜브 데이터 분석 및 썸네일 전문가입니다. 
            카테고리: '{category}'
            [데이터] 대본: {text[:1500]}..., WPM: {int(wpm)}
            
            {custom_instruction_text}

            [작업 지시사항]
            1. **분석 리포트 (한국어 필수)**: 점수(60~95), 코멘트, 전략 등을 한국어로 작성.
            
            2. **제목 추천 (한국어 필수)**:
               - 클릭률(CTR)을 높일 수 있는 **서로 다른 3가지 전략**의 제목을 제안하세요.
               - 전략 1: 호기심 자극형 (Curiosity Gap)
               - 전략 2: 문제 해결/이익 강조형 (Benefit/How-to)
               - 전략 3: 강력한 비교/어그로형 (Strong/Controversial)
            
            3. **썸네일 프롬프트 (영어 필수)**:
               - DALL-E 3를 위한 3가지 스타일 프롬프트 작성.
               - 영상의 핵심 키워드를 이미지 내에 텍스트로 포함하도록 지시.

            [Output JSON]
            {{
                "score": (int),
                "score_comment": (string, 한국어),
                "summary_points": ["string (한국어)", "string (한국어)"],
                "scene_reasons": ["string (한국어)", "string (한국어)", "string (한국어)"],
                "titles": [
                    {{"text": "제목 1 (호기심)", "why": "이 제목이 선택된 이유..."}},
                    {{"text": "제목 2 (문제해결)", "why": "이 제목이 선택된 이유..."}},
                    {{"text": "제목 3 (강력함)", "why": "이 제목이 선택된 이유..."}}
                ],
                "detail_analysis": (string, 한국어),
                "thumbnail_variations": [
                    {{"style_name": "강렬한 클릭 유도형", "prompt": "English prompt..."}},
                    {{"style_name": "감성 스토리형", "prompt": "English prompt..."}},
                    {{"style_name": "깔끔한 정보형", "prompt": "English prompt..."}}
                ]
            }}
            """
            
            content = [{"type": "text", "text": prompt}]
            for fd in temp_frames_data:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{fd['b64']}"}})
            
            response = client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": content}], response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)

            # DALL-E 3 이미지 생성 (Loop)
            st.write("🎨 AI 썸네일 3종 생성 중... (텍스트 렌더링 포함)")
            
            dalle_results = []
            for variation in result.get('thumbnail_variations', []):
                img_url, img_data = generate_dalle_image(client, variation['prompt'])
                if img_data:
                    dalle_results.append({
                        "style": variation['style_name'],
                        "prompt": variation['prompt'],
                        "image": img_data
                    })
            
            st.session_state.analysis_result = result
            st.session_state.frames_data = temp_frames_data
            st.session_state.dalle_variations = dalle_results
            
            status.update(label="✅ 분석 및 생성 완료!", state="complete", expanded=False)
            st.rerun()

# =========================================================
# 6. 결과 리포트 UI
# =========================================================
if st.session_state.analysis_result is not None:
    result = st.session_state.analysis_result
    frames_data = st.session_state.frames_data
    dalle_variations = st.session_state.dalle_variations
    file_name = st.session_state.uploaded_file_name

    st.divider()

    # [1] 상단
    col_top_L, col_top_R = st.columns([1, 1], gap="medium")
    with col_top_L:
        st.markdown('<div class="section-header" style="text-align:center;">🏆 종합 트렌드 적합도</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="score-circle-container">
            <div class="score-circle">
                <div class="score-num">{result['score']}</div>
                <div class="score-max">/ 100</div>
            </div>
        </div>
        <div class="score-comment">{result['score_comment']}</div>
        """, unsafe_allow_html=True)

    with col_top_R:
        st.markdown('<div class="section-header">📁 파일 정보</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-card">
            <div><span class="info-label">파일명</span><div class="info-value">{file_name}</div></div>
            <div style="margin-top:15px;"><span class="info-label">카테고리</span><div class="info-value">{category}</div></div>
            <div style="margin-top:15px;"><span class="info-label">분석 상태</span><div class="info-value">완료 ✅</div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # [2] 중단: 1, 2, 3순위 장면
    st.markdown('<div class="section-header">📸 영상 캡처 추천 (Best 3)</div>', unsafe_allow_html=True)
    thumb_c1, thumb_c2, thumb_c3 = st.columns(3, gap="medium")

    for i, col in enumerate([thumb_c1, thumb_c2, thumb_c3]):
        with col:
            st.markdown(f'<span class="rank-tag bg-{i+1}">{i+1}순위</span>', unsafe_allow_html=True)
            st.image(frames_data[i]['img'], use_container_width=True)
            st.download_button("📥 다운로드", frames_data[i]['bytes'], f"capture_{i+1}.jpg", "image/jpeg", use_container_width=True)
            st.caption(result['scene_reasons'][i])

    st.markdown("---")

    # [NEW] DALL-E 3 베리에이션 (텍스트 포함)
    st.markdown('<div class="section-header">🎨 AI 완성형 썸네일 (텍스트 포함)</div>', unsafe_allow_html=True)
    st.info("💡 Tip: AI가 영상 내용을 요약한 **핵심 키워드**를 이미지 안에 직접 써넣었습니다.")
    
    if dalle_variations:
        ai_cols = st.columns(3, gap="medium")
        for i, var in enumerate(dalle_variations):
            with ai_cols[i]:
                st.markdown(f"""
                <div style="background-color:#f3e5f5; border-radius:10px; padding:10px; text-align:center; margin-bottom:10px; font-weight:bold; color:#4a148c;">
                    {var['style']}
                </div>
                """, unsafe_allow_html=True)
                
                st.image(var['image'], use_container_width=True)
                st.download_button(
                    label=f"📥 {var['style']} 다운로드",
                    data=var['image'],
                    file_name=f"ai_thumb_{i+1}.png",
                    mime="image/png",
                    use_container_width=True
                )
                with st.expander("프롬프트 보기"):
                    st.caption(var['prompt'])
    else:
        st.warning("이미지가 생성되지 않았습니다.")

    st.markdown("---")

    # [3] 하단 정보
    st.markdown('<div class="section-header">📍 핵심 전략 요약</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="summary-card">
        <ul style="font-size: 1.1rem; line-height: 1.8;">
            <li>{result['summary_points'][0]}</li>
            <li>{result['summary_points'][1]}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top: 40px;">🏷️ 클릭을 부르는 제목 추천</div>', unsafe_allow_html=True)
    for i, t in enumerate(result['titles']):
        with st.expander(f"📝 추천 {i+1}: {t['text']}", expanded=True):
            st.info(f"**WHY?** {t['why']}")

    st.markdown('<div class="section-header" style="margin-top: 40px;">📊 AI 상세 분석 리포트</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown(f"""
        <div style="background-color:#fff; padding:20px; border-radius:10px; border:1px solid #ddd; line-height:1.6;">
            {result['detail_analysis']}
        </div>
        """, unsafe_allow_html=True)

    # 이메일 발송
    st.markdown("---")
    st.markdown('<div class="section-header">📧 결과 리포트 메일로 받기</div>', unsafe_allow_html=True)
    
    with st.container():
        col_email, col_btn = st.columns([3, 1])
        with col_email:
            user_email = st.text_input("받을 이메일 주소", placeholder="result@example.com", key="email_input")
        with col_btn:
            st.write("") 
            st.write("")
            if st.button("📩 리포트 발송", type="primary", use_container_width=True):
                if user_email:
                    with st.spinner("메일 서버 접속 중..."):
                        email_body = f"""
                        [Think:it AI 유튜브 컨설팅 리포트]
                        종합 점수: {result['score']}점 ({result['score_comment']})
                        
                        [핵심 전략]
                        1. {result['summary_points'][0]}
                        2. {result['summary_points'][1]}
                        
                        [AI 추천 제목]
                        1. {result['titles'][0]['text']}
                        2. {result['titles'][1]['text']}
                        3. {result['titles'][2]['text']}
                        
                        [상세 분석]
                        {result['detail_analysis']}
                        """
                        success, msg = send_email_smtp(user_email, f"[Think:it] {file_name} 분석 결과", email_body)
                        if success: st.success(msg)
                        else: st.error(msg)
                else:
                    st.warning("이메일 주소를 입력해주세요.")
