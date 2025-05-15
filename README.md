## Laptop Recommendation Chatbot using LangChain

This project is an AI-powered chatbot assistant that helps users find suitable laptops based on their requirements. It uses a local CSV dataset of laptops, processes the data into searchable chunks, and uses natural language processing (NLP) to answer queries intelligently.

---

## Project Structure

- **Dataset**: A CSV file with laptop details (features, price, brand, etc.)
- **LangChain**: For chaining LLM tools and managing retrieval-based question answering.
- **FAISS**: Vector store to store and search document embeddings.
- **HuggingFace Embeddings**: For converting text into numerical vectors.
- **Ollama (LLaMA3)**: Local LLM for understanding and refining queries.

---
## library Versions 

faiss-cpu                 1.11.0 
huggingface-hub           0.31.1 
langchain                 0.3.25 
langchain-community       0.3.24 
langchain-core            0.3.59 
langchain-text-splitters  0.3.8 
pandas                    2.2.3  
prompt_toolkit            3.0.50 


##How It Works

### 1. **Load CSV File**
We load the dataset using `CSVLoader` from LangChain:

```python
loader = CSVLoader(file_path="path/to/your/csv")
docs = loader.load()


2. Split Text into Chunks
To make the text searchable, we split it into smaller pieces:

python
Copy
Edit
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_documents(docs)
3. Generate Embeddings
Use HuggingFace's all-mpnet-base-v2 model to convert text into embeddings:

python
Copy
Edit
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)
4. Semantic Search
Allow users to ask questions, and find relevant chunks from the dataset:

python
Copy
Edit
result = vectorstore.similarity_search("i want a budget laptop for remote work with good battery life and i5")
5. Refine User Query with LLM
We use an LLM (LLaMA3 via Ollama) to rewrite user queries into structured filter requirements:

python
Copy
Edit
template = """You are a laptop shopping assistant..."""
query_chain = prompt | llm
refined_query = query_chain.run("I want a budget laptop for remote work")
6. Answer Questions with RAG (Retrieval-Augmented Generation)
Using LangChain's RetrievalQA, we retrieve data and generate answers:

python
Copy
Edit
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
response = rag_chain.run(refined_query)
7. Build an Intelligent Agent
Use LangChain's Agent to handle conversational flows with memory and tools:

python
Copy
Edit
agent_executor = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

8. Example Interaction
python
Copy
Edit
response = agent_executor.run("Show me Dell laptops under ₹50000")
followup = agent_executor.run("Only i5 models please")

Requirements
Python 3.8+
Libraries:
pandas
langchain
faiss-cpu
sentence-transformers
ollama with a local model like llama3
CSV dataset with laptop information

Use Case
This bot is ideal for:
Laptop shopping guidance
Customer support on e-commerce sites
Tech assistants in stores or apps

Author
admin (replace with your contact info or GitHub link)
