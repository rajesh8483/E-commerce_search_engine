# Laptop Recommendation Chatbot using LangChain

This project is an AI-powered chatbot assistant that helps users find suitable laptops based on their requirements. It uses a local CSV dataset of laptops, processes the data into searchable chunks, and uses natural language processing (NLP) to answer queries intelligently.

---

## Project Structure

- *Dataset*: A CSV file with laptop details (features, price, brand, etc.)
- *LangChain*: For chaining LLM tools and managing retrieval-based question answering.
- *FAISS*: Vector store to store and search document embeddings.
- *HuggingFace Embeddings*: For converting text into numerical vectors.
- *Ollama (LLaMA3)*: Local LLM for understanding and refining queries.

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


##Project Flow

1. *Load CSV File*
We load the dataset using CSVLoader from LangChain:

2. Split Text into Chunks
To make the text searchable, we split it into smaller pieces:

3. Generate Embeddings
Use HuggingFace's all-mpnet-base-v2 model to convert text into embeddings:

4. Semantic Search
Allow users to ask questions, and find relevant chunks from the dataset:

5. Refine User Query with LLM
We use an LLM (LLaMA3 via Ollama) to rewrite user queries into structured filter requirements:

6. Answer Questions with RAG (Retrieval-Augmented Generation)
Using LangChain's RetrievalQA, we retrieve data and generate answers:

7. Build an Intelligent Agent
Use LangChain's Agent to handle conversational flows with memory and tools:

8. Example Interaction
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
