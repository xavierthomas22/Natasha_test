from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp, OpenAI
from langchain.embeddings import LlamaCppEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA

from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
import glob
import os
import sys
import getpass
import argparse
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.chains import ConversationalRetrievalChain, RetrievalQA

from langchain.llms import LlamaCpp
# from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager

import numpy as np

class QA:
    def __init__(self, args):

        template = """You are an AI that generates a sentence based on a Question and an Answer. Do not include dates in the output. You will not make up or add any additional information. You should always start your response with "The answer to your query is {answer}", in the following format:
        Question: Who is the best football player?
        Answer: Mesut Ozil
        AI: The answer to your query is Mesut Ozil. He is widely known as a top class creative footballer.

        Question: {question}
        Answer:{answer}
        AI: """

        self.prompt = PromptTemplate(template=template, input_variables=["question", "answer"])

        self.get_embeddings(args)
        self.get_model(args)
        
        self.get_embeddings(args)
        self.get_model(args)
        self.get_qa_texts(args)

    def get_embeddings(self, args):
        if args.model=='llama':
            self.embeddings = LlamaCppEmbeddings(model_path="llama_models/7B/ggml-model-q4_0.bin")
        elif args.model=='openai':
            self.embeddings = OpenAIEmbeddings(openai_api_key=args.openai_api_key)

    def get_model(self, args):
        if args.model=='llama':
            # Use llama-cpp as the LLM for langchain
            self.llm = LlamaCpp(
                model_path="llama_models/7B/ggml-model-q4_0.bin",
                n_ctx= 2048,
                verbose=False,
                use_mlock=True
            )
            self.llm.client.verbose = False
        elif args.model=='openai':
            self.llm = OpenAI(openai_api_key=args.openai_api_key) 

        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)


    def cosine_similarity(self, u, v):
        distance = 0.0
        dot = np.dot(u,v)
        norm_u = np.sqrt(np.sum(np.square(u)))
        norm_v = np.sqrt(np.sum(np.square(v)))
        cosine_similarity = dot / (norm_u * norm_v)
        return cosine_similarity


    def get_qa_texts(self, args):
        text_list = []
        files_list = glob.glob("sample_data/*.txt")
        num_files = len(files_list)
        for idx, docs in enumerate(files_list):
            with open(docs) as f:
                for line in f:
                    text_list.append(line.rstrip())
        self.questions_list = []
        self.answers_list = []
        for i in text_list:
            self.questions_list.append(i.split("Answer:")[0].strip())
            self.answers_list.append(i.split("Answer:")[1].strip())

        self.doc_result = self.embeddings.embed_documents(self.questions_list)

    def semantic_search(self, args):
        query_embedding = self.embeddings.embed_query(args.question)
        embedding_ids = [i for i in range(len(self.doc_result))]
        similarities = []
        for emb in self.doc_result:
            similarity = self.cosine_similarity(query_embedding, emb)
            similarities.append(similarity)
        sorted_tups = sorted(
            zip(similarities, embedding_ids), key=lambda x: x[0], reverse=True
        )
        similarity_cutoff = args.similarity_cutoff
        similarity_top_k = 1
        if similarity_cutoff is not None:
            sorted_tups = [tup for tup in sorted_tups if tup[0] > similarity_cutoff]
        similarity_top_k = similarity_top_k or len(sorted_tups)
        self.result_tups = sorted_tups[:similarity_top_k]
        result_similarities = [s for s, _ in self.result_tups]
        self.result_ids = [n for _, n in self.result_tups]

        if len(self.result_ids)>0:
            print(self.answers_list[self.result_ids[0]], self.result_tups)
            return self.result_ids[0]
        else:
            return None


    def run(self, args):
        closest_id = self.semantic_search(args)
        if closest_id is not None:
            answer =  self.llm_chain.run(question = args.question, answer=self.answers_list[closest_id])
        else:
            answer = "I don't know"
        print(answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Natasha Test QA Custom')
    parser.add_argument('--model', type=str, default='openai', help='[llama, openai]')
    parser.add_argument('--question', type=str, default='How are you?', help='[llama, openai]')
    parser.add_argument('--similarity_cutoff', type=int, default=0.8, help='[llama, openai]')
    parser.add_argument('--openai_api_key', type=str, default='')
    args = parser.parse_args()

    natasha_qa = QA(args)
    natasha_qa.run(args)