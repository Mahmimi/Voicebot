#App code from https://blog.futuresmart.ai/building-a-conversational-voice-chatbot-integrating-openais-speech-to-text-text-to-speech
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *
import time
import networkx as nx
import os
import speech_recognition as sr
import asyncio
import edge_tts
import base64
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

float_init()

model_names = [
    'wangchanberta-base-att-spm-uncased',
]

tokenizers = {
    'wangchanberta-base-att-spm-uncased': AutoTokenizer,
}
public_models = ['xlm-roberta-base', 'bert-base-multilingual-cased']
#Choose Pretrained Model
model_name = "wangchanberta-base-att-spm-uncased" 

#create tokenizer
tokenizer = tokenizers[model_name].from_pretrained(
                f'airesearch/{model_name}' if model_name not in public_models else f'{model_name}',
                revision='main',
                model_max_length=416,)

#pipeline
zero_classify = pipeline(task='zero-shot-classification',
         tokenizer=tokenizer,
         model=AutoModelForSequenceClassification.from_pretrained(
             f'airesearch/{model_name}' if model_name not in public_models else f'airesearch/{model_name}-finetuned',
             revision='finetuned@xnli_th')
         )

def intent_classifier(text_input, candidate_labels, zero_classify=zero_classify):
    output_label = zero_classify(text_input, candidate_labels=candidate_labels)
    return output_label['labels'][0]

customer_name = "จิรานุวัฒน์"
bot_identity = 'female'
bot_name = 'ท้องฟ้า'
pronoun = 'ดิฉัน' if bot_identity == 'female' else 'กระผม'
sentence_ending = ['ค่ะ','คะ'] if bot_identity == 'female' else ['ครับ','ครับ']
comany_name = 'แมวเหมียว'

# Create a directed graph
A = nx.DiGraph(section='A')

# Add nodes and edges
A.add_node("START A", response=f"สวัสดี{sentence_ending[0]} ขอเรียนสายคุณ {customer_name} {sentence_ending[0]}")
A.add_node("A1", response=f"{pronoun} ต้องกราบขอประทานโทษเป็นอย่างสูงที่โทรมารบกวนนะ{sentence_ending[1]} {pronoun} ชื่อ {bot_name} ใบอนุญาตนายหน้าประกันวินาศภัยเลขที่ XXXXXXXXXX ติดต่อจากบริษัท {comany_name} จำกัด โทรมาเพื่อขออนุญาตนำเสนอสิทธิประโยชน์สำหรับลูกค้าของธนาคาร{comany_name} ไม่ทราบว่าจะสะดวกหรือไม่{sentence_ending[1]}", intent_classify= lambda x :intent_classifier(x,["ได้","ไม่ได้ ไม่ตกลง ยังไม่ตกลง ยังไม่ได้"]))
A.add_node("A2", response=f"{pronoun} ขออนุญาตติดต่อกลับคุณ{customer_name} อีกครั้งในวันที่....ไม่ทราบว่า คุณ{customer_name} สะดวกไหม{sentence_ending[1]} ")
A.add_node("END", response=f"ต้องกราบขอประทานโทษเป็นอย่างสูงที่โทรมารบกวนนะ{sentence_ending[1]} {pronoun} หวังเป็นอย่างยิ่งว่าทางบริษัท {comany_name} จะได้ให้บริการคุณ{customer_name} ในโอกาสถัดไปนะ{sentence_ending[1]} หากคุณ{customer_name} ไม่ประสงค์ที่จะให้บริษัท {comany_name} ติดต่อเพื่อนำเสนอบริการของ บริษัท {comany_name} สามารถแจ้งผ่าน Call Center โทร 02-123-4567 ได้{sentence_ending[0]} ขอขอบพระคุณ ที่สละเวลาในการฟังข้อมูลของ บริษัท {comany_name} ขออนุญาตวางสาย{sentence_ending[0]} สวัสดี{sentence_ending[0]}")
A.add_node("A3", response=f"ขอบพระคุณ{sentence_ending[0]} และเพื่อเป็นการปรับปรุงคุณภาพในการให้บริการ ขออนุญาตบันทึกเสียงการสนทนาในครั้งนี้ด้วยนะ{sentence_ending[1]}", intent_classify= lambda x :intent_classifier(x,["ได้","ไม่ได้ ไม่ตกลง ยังไม่ตกลง ยังไม่ได้"]))
A.add_node("END A1", response=f"ขอบพระคุณ{sentence_ending[0]} ดิฉันจะไม่บันทึกเสียงการสนทนาในครั้งนี้{sentence_ending[0]}")
A.add_node("END A2", response=f"ขอบพระคุณ{sentence_ending[0]} ขณะนี้ได้เริ่มบันทึกการสนทนาแล้วนะ{sentence_ending[1]}")

A.add_edges_from((("START A","A1"),("A1","A2"),("A2","END"),("A1","A3"),("A3","END A1"),("A3","END A2")))

# Create a directed graph
B = nx.DiGraph(section='B')

# Add nodes and edges
B.add_node("START B", response=f"เนื่องในโอกาสที่ ธนาคาร{comany_name} ได้จัดตั้งบริษัท {comany_name} จำกัด เข้าเป็นบริษัทในกลุ่มธุรกิจการเงินของธนาคาร โดยมีวัตถุประสงค์ประกอบกิจการเป็นนายหน้าประกันวินาศภัย {pronoun} {bot_name} จึงติดต่อมาเพื่อขออนุญาตนำเสนอแผนประกันภัยรถยนต์แบบพิเศษเฉพาะลูกค้าของธนาคาร{comany_name}เท่านั้น {pronoun}ขอชี้แจงรายละเอียดนะ{sentence_ending[1]} ")
B.add_node("B1", response=f"เพื่อให้ท่านสมาชิกได้รับประโยชน์สูงสุด จึงขออนุญาตสอบถามข้อมูลรถยนต์ของคุณ{customer_name} นะ{sentence_ending[1]}")
B.add_node("B2", response=f"รถยนต์มีประกันประเภทใด (1,2,3,2+,3+) รับประกันภัยโดยบริษัทฯใด สิ้นสุดความคุ้มครองเมื่อใด")
B.add_node("END B", response=f"{comany_name}ได้คัดสรรค์แบบประกัน เพื่อเป็นทางเลือกที่คุ้มค่าไว้บริการสำหรับลูกค้าของธนาคาร{comany_name} ดังนี้")

B.add_edges_from((("START B","B1"),("B1","B2"),("B2","END B")))

Bot_dialog = nx.compose(A, B)
Bot_dialog.add_edges_from((("END A1","START B"),("END A2","START B")))

# Initialize session state
if "Bot_dialog" not in st.session_state:
    st.session_state.Bot_dialog = Bot_dialog
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": st.session_state.Bot_dialog.nodes["START A"]["response"]}
        ]
if "current_node" not in st.session_state:
    st.session_state.current_node = "START A"

def speech_to_text(audiofile_path):
    recognizer = sr.Recognizer()

    try:
        with sr.WavFile(audiofile_path) as source:
            audio = recognizer.record(source)
        transcription = recognizer.recognize_google(audio,language = "th-TH")
        return transcription
    except:
        return "Could not understand audio"

def get_answer(prompt):
    next_nodes = list(st.session_state.Bot_dialog.successors(st.session_state.current_node))
    if next_nodes:
        if "intent_classify"  in st.session_state.Bot_dialog.nodes[st.session_state.current_node]:
            intent = st.session_state.Bot_dialog.nodes[st.session_state.current_node]["intent_classify"](prompt)

        if len(next_nodes) == 1:
            st.session_state.current_node = next_nodes[0]
        else:
            if intent == "ไม่ได้ ไม่ตกลง ยังไม่ตกลง ยังไม่ได้":
                st.session_state.current_node = next_nodes[0]
            else:
                st.session_state.current_node = next_nodes[1]
    
    return st.session_state.Bot_dialog.nodes[st.session_state.current_node]["response"]

async def text_to_speech(input_text: str, filename: str = "tts_temp.wav"):
    communicate = edge_tts.Communicate(input_text, "th-TH-PremwadeeNeural")

    with open(filename, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                pass
    
    return filename

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

async def main():
    st.title("Voicebot's Chatbot Demo")

    # Create footer container for the microphone
    footer_container = st.container()
    with footer_container:
        audio_bytes = audio_recorder()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if audio_bytes:
        # Write the audio bytes to a file
        with st.spinner("Transcribing..."):
            webm_file_path = "temp_audio.wav"
            with open(webm_file_path, "wb") as f:
                f.write(audio_bytes)

            transcript = speech_to_text(webm_file_path)
            if transcript:
                st.session_state.messages.append({"role": "user", "content": transcript})
                with st.chat_message("user"):
                    st.write(transcript)
                os.remove(webm_file_path)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking🤔..."):
                final_response = get_answer(transcript)

                # Simulate stream of response with milliseconds delay
                message_placeholder = st.empty()
                full_response = ""
                for chunk in final_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            with st.spinner("Generating audio response..."):    
                audio_file = await text_to_speech(final_response)
                autoplay_audio(audio_file)
                
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            os.remove(audio_file)
    # Float the footer container and provide CSS to target it with
    footer_container.float("bottom: 0rem;")

if __name__ == "__main__":
    asyncio.run(main())

