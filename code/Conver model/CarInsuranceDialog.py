from ChatbotConfig import ChatbotConfig
from DialogModelConfig import DialogModelConfig
import networkx as nx

class CarInsuranceDialog():
    def __init__(self, customer_name:str = 'ผู้ใช้งาน', model:DialogModelConfig = DialogModelConfig(),
                 chatbot_config:ChatbotConfig = ChatbotConfig()) -> None:
        
        """
    	Initialize the CarInsuranceDialog class with the given customer name, dialog model configuration, and chatbot configuration.
    	use .dialog_graph to return dialog graph networkx
    	:param customer_name: str, default='ผู้ใช้งาน'
    	:param model: DialogModelConfig, default=DialogModelConfig()
    	:param chatbot_config: ChatbotConfig, default=ChatbotConfig()
    	:return: None
    	"""

        self.customer_name = customer_name
        self.model = model
        self.zero_classify = model.zero_classify

        #chatbot config part
        self.chatbot_config = chatbot_config

        #build dialog
        self.dialog_graph = self.__build_dialog_graph()

    def __build_dialog_graph(self):
        """
    	Builds a dialog graph for the chatbot. It includes nodes and edges for different sections of the conversation. 
    	"""

        def intent_classifier(self, text_input:str, candidate_labels:list):
            """
            This function classifies the input text using the zero-shot classification method.
            
            Args:
                text_input (str): The input text to be classified.
                candidate_labels (list): The list of candidate labels for classification.
            
            Returns:
                str: The output label assigned to the input text.
            """
            output_label = self.zero_classify(text_input, candidate_labels=candidate_labels)
            return output_label['labels'][0]
        
        customer_name = self.customer_name
        comany_name = self.chatbot_config.comany_name
        bot_name = self.chatbot_config.bot_name
        pronoun = self.chatbot_config.pronoun
        sentence_ending = self.chatbot_config.sentence_ending

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

        return Bot_dialog