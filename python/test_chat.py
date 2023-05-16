
#### Usage of SystemMessage and HumanMessage

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
import yaml
from langchain import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
import json
import time
import os
import sys

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)

class Natasha:
    def __init__(self, cfg, session_dict={}):
        with open(cfg["chat_params"]["template"], 'r') as file:
            self.template =  file.read()
        if "generated" not in session_dict:
            session_dict["generated"] = []
        if "past" not in session_dict:
            session_dict["past"] = []
        if "input" not in session_dict:
            session_dict["input"] = ""
        if "stored_session" not in session_dict:
            session_dict["stored_session"] = []
        # Create a ConversationEntityMemory object if not already created
        if 'entity_memory' not in session_dict:
                session_dict["entity_memory"] = ConversationBufferWindowMemory(k=cfg["langchain_params"]["k"], return_messages=True)
        self.session_dict = session_dict

        self.create_chat_prompt()
        self.get_model(cfg)
        self.get_conversation_chain()

    def create_chat_prompt(self):
        self.chat_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(self.template),
                                                MessagesPlaceholder(variable_name="history"),
                                                HumanMessagePromptTemplate.from_template("{input}")])

    def get_text(self):
        """
        Get the user input text.
        Returns:
            (str): The text entered by the user
        """
        input_text = input("Enter Input: ")
        return input_text

    def new_chat(self):
        """
        Clears session state and starts a new chat.
        """
        save = []
        for i in range(len(self.session_dict['generated'])-1, -1, -1):
            save.append("User:" + self.session_dict["past"][i])
            save.append("Bot:" + self.session_dict["generated"][i])        
        self.session_dict["stored_session"].append(save)
        self.session_dict["generated"] = []
        self.session_dict["past"] = []
        self.session_dict["input"] = ""
        self.session_dict.entity_memory.store = {}
        self.session_dict.entity_memory.buffer.clear()

    def get_model(self, cfg):
        self.llm = ChatOpenAI(temperature=0,
                openai_api_key=cfg["openai_params"]["openai_api_key"], 
                model_name='gpt-3.5-turbo', 
                verbose=False) 

    def get_conversation_chain(self):
        self.conversation = ConversationChain(
            llm=self.llm, 
            prompt=self.chat_prompt,
            memory=self.session_dict["entity_memory"]
        )  

    def run(self, cfg):

        user_input = ''
        tokens_counter = 0
        total_cost_running = 0
        num_interactions = 0
        temp_dict={'conversation':[]}

        while user_input != 'exit':
            user_input = self.get_text()
            if user_input == 'exit':
                break
            with get_openai_callback() as cb:
                output = self.conversation.run(input=user_input)  
                num_interactions+=1
                tokens_counter += cb.total_tokens
                total_cost_running += cb.total_cost
            print(user_input)
            print(output)
            print(f'Tokens info: {cb.total_tokens}')
            temp_dict['conversation'].append({'input': user_input, 'output': output, 'total_tokens': cb.total_tokens, 'total_tokens_running': tokens_counter, 'cost ($)': round(cb.total_cost, 4), 'cost_running ($)': round(total_cost_running, 4)})
            # print()
            # print(self.session_dict["entity_memory"])

        temp_dict['conversation_info'] = {"Number of interactions":num_interactions, "Total Tokens":tokens_counter, "Total Cost ($)": round(total_cost_running, 4)}

        f_name = os.path.basename(sys.argv[0])
        f_name = f_name.split('.')[0]
        k = cfg["langchain_params"]["k"]
        if cfg["chat_params"]["save_conversation_info"]:
            with open(f"conversation_info_{f_name}_k={k}.json", "w") as outfile:
                json.dump(temp_dict, outfile, indent=4)



if __name__ == "__main__":
    with open("configs/chat_config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    natasha_chat = Natasha(cfg)
    natasha_chat.run(cfg)


        
    
        