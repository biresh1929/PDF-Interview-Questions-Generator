from dotenv import load_dotenv
import os
from src.helper import file_processing
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Process your PDF file
pdf_path = "static/docs/your_file.pdf"  # Replace with your PDF path
document_ques_gen, document_answer_gen = file_processing(pdf_path)

# Initialize embeddings (same as in helper.py)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize Pinecone
pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

# Create index name (customize this)
index_name = "interview-qa-bot"  # Change to your preference

# Create index if it doesn't exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # all-MiniLM-L6-v2 uses 384 dimensions
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"✅ Created new Pinecone index: {index_name}")
else:
    print(f"✅ Using existing Pinecone index: {index_name}")

# Get the index
index = pc.Index(index_name)

# Store documents in Pinecone
print("📤 Uploading vectors to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=document_answer_gen,  # Use answer generation documents
    index_name=index_name,
    embedding=embeddings
)

print("✅ Vectors successfully stored in Pinecone!")
print(f"📊 Total documents stored: {len(document_answer_gen)}")