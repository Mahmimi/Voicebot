import gradio as gr
import time
import networkx as nx
import speech_recognition as sr
import edge_tts
import asyncio
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
A.add_node("START A", response=f"สวัสดี{sentence_ending[0]} ขอเรียนสายคุณ {customer_name}{sentence_ending[0]}")
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

current_node = "START A"

def addfile(filepath):
    return filepath
def submit_file(filepath):
    return filepath
def clear_audio():
    return None
def speech_to_text(audiofile_path):
    recognizer = sr.Recognizer()

    try:
        with sr.WavFile(audiofile_path) as source:
            audio = recognizer.record(source)
        transcription = recognizer.recognize_google(audio,language = "th-TH")
        return transcription
    except:
        return "Could not understand audio"
    
#bot response
current_node = "START A"
def user(user_input, history):
        global current_node, Bot_dialog
        
        next_nodes = list(Bot_dialog.successors(current_node))
    
        if next_nodes != []:
            if "intent_classify"  in Bot_dialog.nodes[current_node]:
                intent = Bot_dialog.nodes[current_node]["intent_classify"](user_input)

            if len(next_nodes) == 1:
                current_node = next_nodes[0]
            else:
                if intent == "ไม่ได้ ไม่ตกลง ยังไม่ตกลง ยังไม่ได้":
                    current_node = next_nodes[0]
                else:
                    current_node = next_nodes[1]

        return user_input, history + [[user_input, None]]
        
def bot(history):
    global current_node, Bot_dialog
    history[-1][1] = ""
    bot_message = Bot_dialog.nodes[current_node]["response"]

    if current_node == "END B" or current_node == "END":
        bot_message += """<span style="color:red"> \n**Conversation Endded** </span>"""

    for character in bot_message:
        history[-1][1] += character
        time.sleep(0.01)
        yield history

async def text_to_speech(inputtext_history: list[str], filename: str = "tts_temp.wav"):
    input_text = inputtext_history[-1][1]
    communicate = edge_tts.Communicate(input_text, "th-TH-PremwadeeNeural")

    with open(filename, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                pass
    
    return filename

async def restart(history):
    global current_node, Bot_dialog
    current_node = "START A"
    await text_to_speech([[None, Bot_dialog.nodes["START A"]["response"]]])
    return [history[0]], '', 'tts_temp.wav'

async def main():
    # init first audio greeting
    await text_to_speech([[None, Bot_dialog.nodes["START A"]["response"]]])

    # Create a Gradio interface
    with gr.Blocks(title='Chatbot Demo') as demo:
        gr.Markdown(
                        """
                        <div style="text-align: center;">
                            <h1 style="font-weight: bold; font-size: 30px;">Insurance Voicebot Demo</h1>
                        </div>
                        <div style="text-align: left;">
                            <p style="font-weight: bold; font-size: 20px;"><strong>To try this demo follow these steps.</strong></p> 
                        </div>                  
                        """
        )
        gr.Markdown("""
                        - From **Audio** block, click **Record** button and speak something to the bot.
                        - When finished recording, click **Stop** button.
                        - Click **Submit Voice✔️** button.
                        - Wait until bot generate text and audio response.  
                        - From **Generated audio response** block, click play button to play bot's audio response.
                        - Continue these steps until you want to restart.          
                    """
        )

        chatbot = gr.Chatbot(
            [[None, Bot_dialog.nodes["START A"]["response"]]],
            elem_id="chatbot",
            bubble_full_width=False,
            avatar_images=('https://aui.atlassian.com/aui/9.3/docs/images/avatar-person.svg',
                            'https://avatars.githubusercontent.com/u/51063788?s=200&v=4')
        )

        with gr.Row():
            txt = gr.Textbox(
                scale=3,
                show_label=False,
                placeholder="transcription is Here.",
                container=False,
                interactive=True,
            )
            voice_btn = gr.Audio(sources="microphone",type="filepath", scale=4)
            voicesubmit_btn = gr.Button(value="Submit Voice✔️", scale=1,variant='primary')

        with gr.Row():
            sentence = gr.Textbox(visible=False)
            audio = gr.Audio(
                value="tts_temp.wav",
                label="Generated audio response",
                streaming=True,
                autoplay=False,
                interactive=False,
                show_label=True,
            )

        restart_btn = gr.Button(value='Restart🔄')

        voice_btn.stop_recording(addfile, inputs=voice_btn, outputs=txt)
        voicesubmit_btn.click(submit_file, inputs=voice_btn, outputs=txt).then(speech_to_text, inputs=voice_btn, outputs=txt).then(user, [txt, chatbot], [txt, chatbot], queue=False).then(bot, chatbot, chatbot).then(clear_audio, outputs=voice_btn).then(text_to_speech, inputs=chatbot, outputs=audio)
        restart_btn.click(restart, chatbot, [chatbot,txt,audio])

    demo.launch(debug=False)

if __name__ == "__main__":
    loop = asyncio.get_event_loop_policy().get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()