import streamlit as st
import time
import networkx as nx

#transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

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
    st.session_state.messages = []
if "current_node" not in st.session_state:
    st.session_state.current_node = "START A"

def main():
    st.title("Voicebot's Chatbot Demo")

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = st.session_state.Bot_dialog.nodes["START A"]["response"]

        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Accept user input
    if prompt := st.chat_input("Enter your message."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

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

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = st.session_state.Bot_dialog.nodes[st.session_state.current_node]["response"]
            if st.session_state.current_node == "END B" or st.session_state.current_node == "END":
                st.warning("Conversation Ended. Please refresh this page to start a new conversation.")

            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()