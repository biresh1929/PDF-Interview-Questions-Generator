from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
import os
import re
from dotenv import load_dotenv
from src.prompt import prompt_template, refine_template

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

def file_processing(file_path):
    """Process PDF and split into chunks for question and answer generation"""
    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()
    
    question_gen = ''
    
    for page in data:
        question_gen += page.page_content
    
    # Larger chunks for question generation
    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=200
    )
    
    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)
    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]
    
    # Smaller chunks for answer generation
    splitter_ans_gen = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)
    
    return document_ques_gen, document_answer_gen


def llm_pipeline(file_path):
    """Main pipeline for question generation and answer retrieval"""
    
    # Process the PDF
    document_ques_gen, document_answer_gen = file_processing(file_path)
    
    # Initialize LLM for question generation
    llm_ques_gen_pipeline = ChatGroq(
        temperature=0.3,
        model="llama-3.3-70b-versatile"
    )
    
    # Setup prompts
    PROMPT_QUESTIONS = PromptTemplate(
        template=prompt_template, 
        input_variables=["text"]
    )
    
    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template
    )
    
    # Create question generation chain
    ques_gen_chain = load_summarize_chain(
        llm=llm_ques_gen_pipeline,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS
    )
    
    # Generate questions
    print("🔄 Generating questions...")
    ques = ques_gen_chain.run(document_ques_gen)
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Connect to Pinecone vector store
    print("🔗 Connecting to Pinecone...")
    vector_store = PineconeVectorStore(
        index_name="interview-qa-bot",
        embedding=embeddings
    )
    
    # Initialize LLM for answer generation
    llm_answer_gen = ChatGroq(
        temperature=0.1, 
        model="llama-3.3-70b-versatile"
    )
    
    # Filter and clean questions
    print("🧹 Filtering questions...")
    ques_list = ques.split("\n")
    filtered_ques_list = []
    
    for element in ques_list:
        element = element.strip()
        
        # Skip empty, short lines, or lines with answers
        if not element or len(element) < 15 or "Answer:" in element or "answer:" in element:
            continue
        
        # Remove various numbering patterns
        element = re.sub(r'^\d+\.\s*\d+\.\s*', '', element)  # "1. 1."
        element = re.sub(r'^\d+\.\s*[a-z]\)\s*', '', element)  # "1. a)"
        element = re.sub(r'^\d+\.\s*\*\*.*?\*\*:?\s*', '', element)  # "1. **Target**:"
        element = re.sub(r'^\d+\.\s*', '', element)  # "1."
        element = re.sub(r'^[a-z]\)\s*', '', element)  # "a)"
        element = re.sub(r'^-\s*\*\*.*?\*\*:?\s*', '', element)  # "- **SDG**:"
        element = element.strip()
        
        # Only keep actual questions
        if element.endswith('?') and len(element) > 20:
            # Avoid duplicates
            if element not in filtered_ques_list:
                filtered_ques_list.append(element)
    
    # Limit to reasonable number
    filtered_ques_list = filtered_ques_list[:15]
    
    print(f"\n✅ Generated {len(filtered_ques_list)} clean questions\n")
    
    # Create answer generation chain with Pinecone retriever
    answer_generation_chain = RetrievalQA.from_chain_type(
        llm=llm_answer_gen,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3})  # Return top 3 matches
    )
    
    return answer_generation_chain, filtered_ques_list