from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq  # ← Changed
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_huggingface import HuggingFaceEmbeddings  # ← Changed
from langchain_community.vectorstores import FAISS  # ← Changed
from langchain_classic.chains import RetrievalQA  # ← Changed
import os
from dotenv import load_dotenv
from src.prompt import prompt_template, refine_template

# Groq authentication
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def file_processing(file_path):
    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()
    
    question_gen = ''
    
    for page in data:
        question_gen += page.page_content
    
    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=200
    )
    
    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)
    
    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]
    
    splitter_ans_gen = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    document_answer_gen = splitter_ans_gen.split_documents(
        document_ques_gen
    )
    
    return document_ques_gen, document_answer_gen


def llm_pipeline(file_path):
    document_ques_gen, document_answer_gen = file_processing(file_path)
    
    llm_ques_gen_pipeline = ChatGroq(
        temperature=0.3,
        model="llama-3.1-8b-instant"  # or "llama-3.3-70b-versatile"
    )
    
    PROMPT_QUESTIONS = PromptTemplate(
        template=prompt_template, 
        input_variables=["text"]
    )
    
    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template
    )
    
    ques_gen_chain = load_summarize_chain(
        llm=llm_ques_gen_pipeline,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS
    )
    
    ques = ques_gen_chain.run(document_ques_gen)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)
    
    llm_answer_gen = ChatGroq(
        temperature=0.1, 
        model="llama-3.1-8b-instant"  # or "llama-3.3-70b-versatile"
    )
    
    ques_list = ques.split("\n")
    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]
    
    answer_generation_chain = RetrievalQA.from_chain_type(
        llm=llm_answer_gen,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    
    return answer_generation_chain, filtered_ques_list

