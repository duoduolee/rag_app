

# rag_local.py

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# 1. Load Documents
loader = TextLoader("data.txt")  # Simple text file
documents = loader.load()

# 2. Split Documents
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 3. Load Sentence-Transformer Embeddings (e.g., all-MiniLM)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)

# 4. Setup Retriever
retriever = vectorstore.as_retriever()

# 5. Load Local Language Model (e.g., Mistral)
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
local_llm = HuggingFacePipeline(pipeline=pipe)

# 6. Setup RAG Chain
rag_chain = RetrievalQA.from_chain_type(
    llm=local_llm,
    retriever=retriever,
    return_source_documents=True
)

# 7. Ask a Question
query = "What is your name"
result = rag_chain({"query": query})

print("\nAnswer:\n", result["result"])
print("\nSources:\n", result["source_documents"])


print('done processed')
