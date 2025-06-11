import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import openai
import os
from datetime import datetime
from gtts import gTTS
import base64
import tempfile
import av
import queue
import numpy as np
import soundfile as sf

st.set_page_config(
    page_title="정인의 음성 비서 프로그램",
    layout="wide"
)

# whisper와 연결하기 위한 STT 함수
def STT(filepath, apikey):
    with open(filepath, "rb") as audio_file:
        client = openai.OpenAI(api_key=apikey)
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return response.text

# GPT 질의 함수
def ask_gpt(prompt, model, apikey):
    client = openai.OpenAI(api_key=apikey)
    response = client.chat.completions.create(
        model=model,
        messages=prompt
    )
    return response.choices[0].message.content

# TTS 변환 및 출력 함수
def TTS(response):
    filename = 'output.mp3'
    tts = gTTS(text=response, lang='ko')
    tts.save(filename)

    with open(filename, 'rb') as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay='True'>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(md, unsafe_allow_html=True)
    os.remove(filename)

# 오디오 수집용 큐와 콜백 정의
audio_queue = queue.Queue()

def audio_callback(frame: av.AudioFrame):
    audio = frame.to_ndarray()
    audio_queue.put(audio)
    return av.AudioFrame.from_ndarray(audio, layout="mono")

# 메인 함수
def main():
    st.header("정인의 음성 비서 프로그램")
    st.markdown("---")

    with st.expander("음성비서 프로그램에 관하여", expanded=True):
        st.write("""
            - 음성 비서 프로그램의 UI는 Streamlit을 활용했습니다.
            - STT(Speech-To-Text)는 OpenAI Whisper AI를 사용합니다.
            - GPT는 OpenAI의 GPT-3.5 또는 GPT-4.1 모델을 사용합니다.
            - TTS는 Google의 gTTS를 사용합니다.
        """)

    if "chat" not in st.session_state:
        st.session_state['chat'] = []
    if "OPENAI_API" not in st.session_state:
        st.session_state["OPENAI_API"] = ""
    if "messages" not in st.session_state:
        st.session_state['messages'] = [{"role": "system", "content": "You are a thoughtful assistant. Respond to all input in 25 words and answer in korean."}]
    if "check_reset" not in st.session_state:
        st.session_state["check_reset"] = False

    with st.sidebar:
        st.session_state["OPENAI_API"] = st.text_input(
            label="OPENAI API 키",
            key="openai_key_input",
            placeholder="여기에 OPENAI API 키를 입력하세요",
            value="",
            type="password"
        )
        st.markdown("---")
        model = st.radio(label="GPT 모델", options=["gpt-4.1", "gpt-3.5-turbo"])
        st.markdown("---")

        if st.button("초기화"):
            st.session_state['chat'] = []
            st.session_state['messages'] = [{"role": "system", "content": "You are a thoughtful assistant. Respond to all input in 25 words and answer in korean."}]
            st.session_state["check_reset"] = True

    st.subheader("질문하기")
    webrtc_ctx = webrtc_streamer(
        key="speech",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        media_stream_constraints={"audio": True, "video": False},
        audio_frame_callback=audio_callback
    )

    # webrtc_context가 활성화되었을 때만 처리
    if webrtc_ctx.state.playing:
        audio_data = []
        while not audio_queue.empty():
            audio_data.append(audio_queue.get())

        if audio_data:
            audio_np = np.concatenate(audio_data, axis=0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                sf.write(f.name, audio_np, 16000)
                temp_path = f.name

            # 음성 → 텍스트
            question = STT(temp_path, st.session_state["OPENAI_API"])
            now = datetime.now().strftime("%H:%M")
            st.session_state['chat'].append(("user", now, question))
            st.session_state['messages'].append({"role": "user", "content": question})

            # GPT 응답
            response = ask_gpt(st.session_state['messages'], model, st.session_state["OPENAI_API"])
            now = datetime.now().strftime("%H:%M")
            st.session_state['chat'].append(("bot", now, response))
            st.session_state['messages'].append({"role": "system", "content": response})

            st.subheader("질문/답변")
            for sender, time, message in st.session_state['chat']:
                if sender == 'user':
                    st.markdown(f"""
                        <div style="display:flex;align-items:center;">
                            <div style="background-color:#007AFF;color:white;border-radius:12px;padding:8px 12px;margin-right:8px;">
                                {message}
                            </div>
                            <div style="font-size:0.8rem;color:gray;">{time}</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style="display:flex;align-items:center;justify-content:flex-end;">
                            <div style="background-color:lightgray;border-radius:12px;padding:8px 12px;margin-left:8px;">
                                {message}
                            </div>
                            <div style="font-size:0.8rem;color:gray;">{time}</div>
                        </div>
                    """, unsafe_allow_html=True)

            TTS(response)
            os.remove(temp_path)

if __name__ == "__main__":
    main()
