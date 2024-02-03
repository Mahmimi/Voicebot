class ChatbotConfig():
    def __init__(self, bot_identity:str = 'female', bot_name:str = 'ท้องฟ้า', comany_name:str = 'แมวเหมียว') -> None:
        """
        Initializes the bot with the specified identity, name, and company name.

        Args:
            bot_identity (str): The identity of the bot, defaults to 'female'.
            bot_name (str): The name of the bot, defaults to 'ท้องฟ้า'.
            comany_name (str): The name of the company, defaults to 'แมวเหมียว'.

        Returns:
            None
        """
        self.bot_identity = bot_identity
        self.bot_name = bot_name
        self.pronoun = 'ดิฉัน' if bot_identity == 'female' else 'กระผม'
        self.sentence_ending = ['ค่ะ','คะ'] if bot_identity == 'female' else ['ครับ','ครับ']
        self.comany_name = comany_name