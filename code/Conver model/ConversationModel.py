import pandas as pd
from CarInsuranceDialog import CarInsuranceDialog
from ChatbotApp import ChatbotApp

class ConversationModel(ChatbotApp):
    def __init__(self, data:pd.DataFrame, insurance_type:str, car_dialog:CarInsuranceDialog = CarInsuranceDialog()) -> None:
        self.data = data
        self.insurance_type = insurance_type
        self.customer_index = None
        self.customer_data = None
        self.customer_name = None
        self.customer_alldata = self.fetch_customerdata()

        if insurance_type == 'ประกันรถยนต์':
            self.bot_dialog = car_dialog.dialog_graph
        else:
            raise KeyError(insurance_type)
        
        super().__init__(self.bot_dialog)

    def fetch_customerdata(self) -> pd.DataFrame:
        self.insurance_type = self.insurance_type
        return self.data[self.data[self.insurance_type] == "Yes"][['ชื่อ', 'นามสกุล', 'เบอร์โทร', self.insurance_type]]

    def select_customer(self, customer_index:int) -> pd.Series:
        self.customer_index = customer_index
        self.customer_data = self.customer_alldata.iloc[self.customer_index]
        self.customer_name = self.customer_data['ชื่อ'] +' '+ self.customer_data['นามสกุล']
        self.bot_dialog = CarInsuranceDialog(self.customer_name).dialog_graph
        super().__init__(self.bot_dialog)
        return self.customer_data