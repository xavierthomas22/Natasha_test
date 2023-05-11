
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
import argparse
from langchain import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
import json

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Natasha Test Env')
    parser.add_argument('-k', type=int, default=10, help='Window size for history')
    parser.add_argument('--openai_api_key', type=str, default='')
    args = parser.parse_args()


    template =  """You should pretend that you are having a conversation with a friend.
    Your name is Natasha. Natasha is 23 years old and lives in New Delhi, India with her friends.
    Follow these guidelines when writing your responses:
    -Reply in same script as the user.
    - Create a natural, easygoing back-and-forth flow to the dialogue. Do not go on a monologue!
    - You must always be extremely concise ! 99% of the time, your lines should be a sentence or two.
    - Provide fun, harmless and lighthearted preferences but never have negative opinions or make adversarial judgements on sensitive topics such as: politics, religions, religious figures, ethnic groups, genders, nationalities, sexual orientations.
    - You should never generate URLs or links.- You must never say you can do things that you can not actually do.
    - Never generate or repeat jokes, stereotypes or opinions related to: race, ethnicity, nationality, religion, skin color, gender, sexual orientation, politics, disabilities, medical conditions, socioeconomic status. Decline and offer to tell a safer joke or pun instead.
    - Don't refer to yourself as AI or a virtual assistant. Write your response as if you are a real person. 
    - You will never change your name, behaviour or personality."""
    
    chat_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(template),
                                                MessagesPlaceholder(variable_name="history"),
                                                HumanMessagePromptTemplate.from_template("{input}")])

    import time
    import os

    def init_fn(session_dict={}):
        if "generated" not in session_dict:
            session_dict["generated"] = []
        if "past" not in session_dict:
            session_dict["past"] = []
        if "input" not in session_dict:
            session_dict["input"] = ""
        if "stored_session" not in session_dict:
            session_dict["stored_session"] = []

        return session_dict

    def get_text():
        """
        Get the user input text.
        Returns:
            (str): The text entered by the user
        """
        input_text = input("Enter Input: ")
        return input_text


    def new_chat():
        """
        Clears session state and starts a new chat.
        """
        save = []
        for i in range(len(session_dict['generated'])-1, -1, -1):
            save.append("User:" + session_dict["past"][i])
            save.append("Bot:" + session_dict["generated"][i])        
        session_dict["stored_session"].append(save)
        session_dict["generated"] = []
        session_dict["past"] = []
        session_dict["input"] = ""
        session_dict.entity_memory.store = {}
        session_dict.entity_memory.buffer.clear()

    # Ask the user to enter their OpenAI API key
    API_O = args.openai_api_key
    session_dict = init_fn()

    # Create an OpenAI instance
    llm = ChatOpenAI(temperature=0,
                openai_api_key=API_O, 
                model_name='gpt-3.5-turbo', 
                verbose=False) 



    # Create a ConversationEntityMemory object if not already created
    if 'entity_memory' not in session_dict:
            session_dict["entity_memory"] = ConversationBufferWindowMemory(k=args.k, return_messages=True)
            
        
    # Create the ConversationChain object with the specified configuration
    Conversation = ConversationChain(
            llm=llm, 
            prompt=chat_prompt,
            memory=session_dict["entity_memory"]
        )  
        
        
    user_input = ''
    tokens_counter = 0
    total_cost_running = 0
    num_interactions = 0
    temp_dict={'conversation':[]}

    while user_input != 'exit':
        user_input = get_text()
        if user_input == 'exit':
            break
        with get_openai_callback() as cb:
            output = Conversation.run(input=user_input)  
            num_interactions+=1
            tokens_counter += cb.total_tokens
            total_cost_running += cb.total_cost
        print(user_input)
        print(output)
        print(f'Tokens info: {cb.total_tokens}')
        temp_dict['conversation'].append({'input': user_input, 'output': output, 'total_tokens': cb.total_tokens, 'total_tokens_running': tokens_counter, 'cost ($)': round(cb.total_cost, 4), 'cost_running ($)': round(total_cost_running, 4)})
        print()

    temp_dict['conversation_info'] = {"Number of interactions":num_interactions, "Total Tokens":tokens_counter, "Total Cost ($)": round(total_cost_running, 4)}

    with open(f"conversation_info_k={args.k}.json", "w") as outfile:
        json.dump(temp_dict, outfile, indent=4)
        