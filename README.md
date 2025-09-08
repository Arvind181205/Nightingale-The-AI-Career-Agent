🚀 AI Career Agent
An autonomous, multi-agent system designed to automate the initial phases of a job hunt. This agent finds relevant job postings, analyzes them against a user's resume, and crafts tailored application materials like resume suggestions and cover letters.
![Uploading Screenshot (242).png…]()


## How It Works
This project is built on a multi-agent graph architecture using LangGraph. A master orchestrator delegates tasks to specialized agents, creating a robust and scalable workflow.

🔎 The Job Scout Agent: The user provides a job title and location. This agent autonomously searches the web to find the top 3 most relevant job descriptions.

🧠 The Analyst Agent (RAG):

The user's PDF resume is processed and converted into a searchable vector store using FAISS and Hugging Face embeddings.

This agent then performs a Retrieval-Augmented Generation (RAG) task, comparing each job description against the vectorized resume.

It generates a detailed analysis report, including an alignment score, strengths, and gaps.

✍️ The Content Crafter Agent:

This agent takes the analysis report as input.

It generates specific, actionable suggestions to tailor the user's resume.

It drafts a complete, professional cover letter based on the identified strengths.

## Tech Stack
Orchestration: LangChain & LangGraph

LLMs: Google Gemini 1.5 Pro & Flash

Embeddings: Hugging Face instructor-large (Local)

Vector Store: FAISS

Web Framework: Streamlit

Core Libraries: PyPDF, python-dotenv
