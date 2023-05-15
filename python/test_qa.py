from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp, OpenAI
from langchain.embeddings import LlamaCppEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
import glob
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
import argparse
import os
import getpass
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.chains import ConversationalRetrievalChain, RetrievalQA

from llama_index import StorageContext, load_index_from_storage, SimpleDirectoryReader, LangchainEmbedding, GPTListIndex, GPTVectorStoreIndex, PromptHelper, LLMPredictor, ServiceContext
from langchain.llms.base import LLM

class QA:
    def __init__(self, args):
        self.get_embeddings(args)
        self.get_model(args)
        
        self.get_embeddings(args)
        self.get_model(args)
        self.get_docs(args)

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

    def get_docs(self, args):

        directory_path = "sample_data/"

        max_input_size = 4096
        num_outputs = 2000
        max_chunk_overlap = 20
        chunk_size_limit = 600
        prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
        self.documents = SimpleDirectoryReader(directory_path).load_data()

    def run(self, args):

        embed_model = LangchainEmbedding(self.embeddings)
        llm_predictor = LLMPredictor(llm=self.llm)

        service_context = ServiceContext.from_defaults(embed_model=embed_model, llm_predictor=llm_predictor)
        index = GPTVectorStoreIndex.from_documents(self.documents,service_context=service_context)
        index.storage_context.persist()

        storage_context = StorageContext.from_defaults(persist_dir='./storage')
        index = load_index_from_storage(storage_context, service_context=service_context)

        query_engine = index.as_query_engine()

        response = query_engine.query(f"{args.question}")
        print(f"Question: {args.question}, Answer: {response}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Natasha Test QA')
    parser.add_argument('--model', type=str, default='openai', help='[llama, openai]')
    parser.add_argument('--question', type=str, default='How are you?', help='[llama, openai]')
    parser.add_argument('--openai_api_key', type=str, default='')
    args = parser.parse_args()

    natasha_qa = QA(args)
    natasha_qa.run(args)


# yes, the best matched text from best matched nodes are chosen and sent to model API for synthesis.