import gradio as gr
import time
import networkx as nx

class ChatbotApp():
    def __init__(self, bot_dialog:nx.DiGraph):
        self.current_node = "START A"
        self.bot_dialog = bot_dialog

    def user(self, user_input, history):
        next_nodes = list(self.bot_dialog.successors(self.current_node))

        if next_nodes != []:
            if "intent_classify" in self.bot_dialog.nodes[self.current_node]:
                intent = self.bot_dialog.nodes[self.current_node]["intent_classify"](user_input)

            if len(next_nodes) == 1:
                self.current_node = next_nodes[0]
            else:
                if intent == "ไม่ได้ ไม่ตกลง ยังไม่ตกลง ยังไม่ได้":
                    self.current_node = next_nodes[0]
                else:
                    self.current_node = next_nodes[1]

        return "", history + [[user_input, None]]

    def bot(self, history):
        history[-1][1] = ""
        bot_message = self.bot_dialog.nodes[self.current_node]["response"]

        if self.current_node == "END B" or self.current_node == "END":
            bot_message += """<span style="color:red"> \n**Conversation Ended** </span>"""

        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.01)
            yield history

    def restart(self, history):
        self.current_node = "START A"
        return [history[0]]

    def run(self):
        with gr.Blocks(title="Voicebot Chatbot Demo") as demo:
            with gr.Column(variant="panel"):
                gr.Markdown("")
                chatbot = gr.Chatbot(
                    [[None, self.bot_dialog.nodes["START A"]["response"]]],
                    elem_id="chatbot",
                    bubble_full_width=False,
                    avatar_images=('https://aui.atlassian.com/aui/9.3/docs/images/avatar-person.svg',
                                'https://avatars.githubusercontent.com/u/51063788?s=200&v=4')
                )
                with gr.Group():
                    with gr.Row():
                        textbox = gr.Textbox(
                            container=False,
                            show_label=False,
                            label="Message",
                            placeholder="Enter text and press enter...",
                            scale=7,
                            autofocus=True,
                        )
                with gr.Row():
                    restart_btn = gr.Button("🔄  Restart", variant="secondary")

            textbox.submit(self.user, [textbox, chatbot], [textbox, chatbot], queue=False).then(self.bot, chatbot, chatbot)
            restart_btn.click(self.restart, chatbot, chatbot)

        demo.launch(debug=False)