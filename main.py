import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import time

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
    """Reads a PDF, splits it, and creates a FAISS vector store."""
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
    """Searches for job postings online."""
    print("--- Executing Scout Node ---")
    job_search_query = state.get('job_search_query')

    # Use the faster, more efficient model for this simple task
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    
    prompt_template = """
    You are an expert job researcher. Your mission is to find 3 relevant job postings based on the following query.
    Query: {query}
    You must return ONLY the raw text content of the top 3 job descriptions you find. Do not add any commentary or introductory text.
    Separate each job description clearly with the separator '--- NEXT JOB ---'.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    search_agent = prompt | llm
    
    response = search_agent.invoke({"query": job_search_query})
    
    jobs = response.content.split("--- NEXT JOB ---")
    cleaned_jobs = [job.strip() for job in jobs if job.strip()]
    
    print(f"--- Scout Node Found {len(cleaned_jobs)} Jobs ---")
    
    return {"jobs": cleaned_jobs}

def analyst_node(state):
    """Analyzes the resume against the job description."""
    print("--- Executing Analyst Node ---")
    job_description = state.get('job_description')
    vector_store = state.get('vector_store')

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    retriever = vector_store.as_retriever()
    retriever_tool = Tool(
        name="resume_search",
        description="Searches the resume to find relevant skills and experiences for a given query.",
        func=retriever.invoke
    )
    tools = [retriever_tool]
    
    prompt_template = """
    You are an expert career coach and resume analyst. Your task is to analyze a job description and compare it against a candidate's resume.
    Here is the job description:
    {job_description}
    Use your `resume_search` tool to find relevant experiences, skills, and qualifications from the candidate's resume based on the job requirements.
    Based on your analysis, provide a concise, professional report in markdown format with the following sections:
    - **Alignment Score:** A percentage score of how well the resume matches the job description (e.g., 85%).
    - **Strengths:** 3-4 bullet points highlighting the candidate's key skills and experiences that are a strong match for the role.
    - **Gaps/Areas for Improvement:** 2-3 bullet points identifying key skills or experiences mentioned in the job description that are missing or underrepresented in the resume.
    - **Suggested Resume Keywords:** A list of 5-7 important keywords from the job description that should be included in the resume.
    Begin your analysis now.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    response = agent_executor.invoke({
        "job_description": job_description,
        "input": "Analyze the provided job description against the resume and generate the report."
    })
    
    analysis_result = response.get("output")
    print("--- Analyst Node Finished ---")
    
    return {"analysis_report": analysis_result}

def crafter_node(state):
    """Generates resume suggestions and a cover letter based on the analysis."""
    print("--- Executing Crafter Node ---")
    analysis_report = state.get('analysis_report')

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    
    prompt_template = """
    You are a professional resume writer and career coach. You have been provided with an analysis report that compares a candidate's resume to a job description.
    Here is the analysis report:
    ---
    {analysis_report}
    ---
    Based *only* on the information in this report, perform the following two tasks:
    1.  **Resume Suggestions:** Write 2-3 specific, actionable bullet points for the candidate's resume. These suggestions should directly address the "Gaps/Areas for Improvement" and incorporate the "Suggested Resume Keywords" from the report. Frame them as if you are suggesting edits. For example: "Consider adding a bullet point under your Project X experience that highlights your work with Python and data analysis to better match the job's requirements."
    2.  **Cover Letter:** Draft a concise, professional, and enthusiastic three-paragraph cover letter for the candidate. The letter should:
        - Start by expressing excitement for the role.
        - Use the "Strengths" from the report to highlight 2-3 key qualifications.
        - End with a strong call to action.
    Structure your entire response with the resume suggestions first, followed by a clear separator, and then the full cover letter. Use '---' as the separator between the two sections.
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    response = chain.invoke({"analysis_report": analysis_report})
    
    parts = response.content.split("---")
    resume_suggestions = parts[0].strip()
    cover_letter = parts[1].strip() if len(parts) > 1 else ""
    
    print("--- Crafter Node Finished ---")
    
    return {
        "resume_suggestions": resume_suggestions,
        "cover_letter": cover_letter
    }

# --- Main Streamlit App ---
def main():
    load_dotenv()
    st.set_page_config(page_title="AI Career Agent")
    st.title("📄 AI Career Agent")
    
    st.header("Upload Your Resume")
    resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

    if resume_file:
        with st.spinner("Processing resume..."):
            vector_store = create_vector_store_from_pdf(resume_file)
        st.session_state.vector_store = vector_store
        st.success("Resume processed and ready!")
    
    st.header("Autonomous Job Application")
    job_title = st.text_input("Enter the Job Title you're looking for")
    location = st.text_input("Enter the Location (e.g., London, UK)")

    if st.button("Find and Prepare Applications"):
        if 'vector_store' not in st.session_state:
            st.warning("Please upload your resume first.")
            return
        if not job_title or not location:
            st.warning("Please provide a Job Title and Location.")
            return

        full_query = f"'{job_title}' job in {location}"

        with st.spinner(f"Searching for jobs matching: {full_query}..."):
            scout_result = scout_node({"job_search_query": full_query})
            jobs_list = scout_result.get("jobs", [])
        
        if not jobs_list:
            st.error("Could not find any jobs matching your query. Please try again.")
            return

        st.success(f"Found {len(jobs_list)} jobs. Now analyzing and preparing materials...")

        workflow = StateGraph(AgentState)
        workflow.add_node("analyst", analyst_node)
        workflow.add_node("crafter", crafter_node)
        workflow.set_entry_point("analyst")
        workflow.add_edge("analyst", "crafter")
        workflow.add_edge("crafter", END)
        app = workflow.compile()

        for i, job_desc in enumerate(jobs_list):
            with st.expander(f"Application Materials for Job #{i+1}"):
                with st.spinner(f"Processing job #{i+1}..."):
                    initial_state = {
                        "job_description": job_desc,
                        "vector_store": st.session_state.vector_store
                    }
                    final_state = app.invoke(initial_state)

                    st.subheader("Analysis Report")
                    st.markdown(final_state.get("analysis_report"))
                    
                    st.subheader("Resume Suggestions")
                    st.markdown(final_state.get("resume_suggestions"))
                    
                    st.subheader("Drafted Cover Letter")
                    st.markdown(final_state.get("cover_letter"))

                # Add a delay to avoid hitting the rate limit
                if i < len(jobs_list) - 1:
                    time.sleep(20)

if __name__ == '__main__':
    main()