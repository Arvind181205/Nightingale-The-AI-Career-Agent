import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import time
import json
import requests
from streamlit_lottie import st_lottie

# --- Load Lottie Animations ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_resume = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")
lottie_search = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_sSF6EG.json")
lottie_success = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_rycdh53q.json")

# --- LangGraph State Definition ---
class AgentState(TypedDict):
    job_search_query: str
    jobs: List[str]
    job_description: str
    analysis_report: str
    resume_suggestions: str
    cover_letter: str
    vector_store: object

# --- Cached Function for Vector Store Creation ---
@st.cache_resource
def create_vector_store_from_pdf(pdf_file):
    if pdf_file:
        with open("temp_resume.pdf", "wb") as f:
            f.write(pdf_file.getbuffer())
        
        loader = PyPDFLoader("temp_resume.pdf")
        pages = loader.load_and_split()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(pages)
        
        embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
        vector_store = FAISS.from_documents(chunks, embedding=embeddings)
        return vector_store
    return None

# --- LangGraph Nodes ---
def scout_node(state):
    job_search_query = state.get('job_search_query')
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    
    prompt_template = """
    Find 3 relevant job postings for the query:
    Query: {query}
    Return ONLY raw job descriptions, separated by '--- NEXT JOB ---'.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    search_agent = prompt | llm
    
    response = search_agent.invoke({"query": job_search_query})
    jobs = response.content.split("--- NEXT JOB ---")
    cleaned_jobs = [job.strip() for job in jobs if job.strip()]
    return {"jobs": cleaned_jobs}

def analyst_node(state):
    job_description = state.get('job_description')
    vector_store = state.get('vector_store')

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    retriever = vector_store.as_retriever()
    retriever_tool = Tool(
        name="resume_search",
        description="Searches the resume for relevant skills and experiences.",
        func=retriever.invoke
    )
    
    prompt_template = """
    Compare this job description with the resume:
    {job_description}
    Provide:
    - Alignment Score
    - Strengths
    - Gaps
    - Suggested Keywords
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, [retriever_tool], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True)
    
    response = agent_executor.invoke({
        "job_description": job_description,
        "input": "Analyze and generate report."
    })
    return {"analysis_report": response.get("output")}

def crafter_node(state):
    analysis_report = state.get('analysis_report')
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    
    prompt_template = """
    Based on this analysis report:
    ---
    {analysis_report}
    ---
    1. Resume Suggestions
    2. A professional Cover Letter
    Use '---' as a separator.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    response = chain.invoke({"analysis_report": analysis_report})
    
    parts = response.content.split("---")
    return {"resume_suggestions": parts[0].strip(), "cover_letter": parts[1].strip() if len(parts) > 1 else ""}

# --- Main Streamlit App ---
def main():
    load_dotenv()
    st.set_page_config(page_title="AI Career Agent", page_icon="📄", layout="wide")

    # Title animation
    col1, col2 = st.columns([1, 3])
    with col1:
        if lottie_resume:
            st_lottie(lottie_resume, key="resume", height=120)
    with col2:
        st.title("📄 AI Career Agent")
        st.caption("✨ Your AI-powered job search companion")

    st.sidebar.title("⚙️ Settings")
    delay_time = st.sidebar.slider("⏱️ Delay between jobs", 5, 30, 15)

    st.header("📤 Upload Your Resume")
    resume_file = st.file_uploader("Upload resume (PDF)", type=["pdf"])
    if resume_file:
        with st.spinner("🔍 Processing resume..."):
            vector_store = create_vector_store_from_pdf(resume_file)
            st.session_state.vector_store = vector_store
            st.success("✅ Resume processed successfully!")
            st.toast("Resume uploaded!", icon="📂")

    st.header("🔎 Job Search")
    job_title = st.text_input("Job Title", placeholder="e.g., Data Scientist")
    location = st.text_input("Location", placeholder="e.g., London, UK")

    if st.button("🚀 Find Jobs"):
        if 'vector_store' not in st.session_state:
            st.warning("⚠️ Please upload your resume first.")
            return
        if not job_title or not location:
            st.warning("⚠️ Enter both Job Title and Location.")
            return

        query = f"'{job_title}' job in {location}"

        with st.spinner(f"🌐 Searching jobs for: {query}..."):
            if lottie_search:
                st_lottie(lottie_search, key="search", height=150)
            scout_result = scout_node({"job_search_query": query})
            jobs_list = scout_result.get("jobs", [])

        if not jobs_list:
            st.error("❌ No jobs found. Try refining your query.")
            return

        st.success(f"🎯 Found {len(jobs_list)} jobs!")

        workflow = StateGraph(AgentState)
        workflow.add_node("analyst", analyst_node)
        workflow.add_node("crafter", crafter_node)
        workflow.set_entry_point("analyst")
        workflow.add_edge("analyst", "crafter")
        workflow.add_edge("crafter", END)
        app = workflow.compile()

        progress_bar = st.progress(0)
        total_jobs = len(jobs_list)

        for i, job_desc in enumerate(jobs_list, start=1):
            with st.expander(f"📌 Application #{i}"):
                with st.spinner(f"Analyzing job #{i}..."):
                    state = {"job_description": job_desc, "vector_store": st.session_state.vector_store}
                    final_state = app.invoke(state)

                    st.subheader("📊 Analysis Report")
                    st.markdown(final_state.get("analysis_report"))

                    st.subheader("📝 Resume Suggestions")
                    st.markdown(final_state.get("resume_suggestions"))

                    st.subheader("📄 Cover Letter")
                    st.markdown(final_state.get("cover_letter"))

            progress_bar.progress(i / total_jobs)
            if i < total_jobs:
                time.sleep(delay_time)

        st.toast("✅ All jobs processed!", icon="🎉")
        st.balloons()
        if lottie_success:
            st_lottie(lottie_success, key="done", height=180)

if __name__ == '__main__':
    main()
