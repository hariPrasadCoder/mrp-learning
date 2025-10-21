# simple_app.py - A simplified LangGraph + Streamlit app for learning
import os
import re
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv

import streamlit as st

# Load environment variables
load_dotenv()

# --- Core imports ---
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from pinecone import Pinecone

# --------------- Configuration ---------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
INDEX_NAME = "rag-demo-gemini"  # Your existing Pinecone index
EMBED_MODEL = "text-embedding-004"
CHAT_MODEL = "gemini-2.5-flash"

# Check for required API keys
if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY in environment variables")
    st.stop()
if not PINECONE_API_KEY:
    st.error("Missing PINECONE_API_KEY in environment variables")
    st.stop()

# --------------- Initialize AI Models ---------------
llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
pindex = pc.Index(INDEX_NAME)

# Initialize DuckDuckGo search with better parameters
ddg_search = DuckDuckGoSearchRun(max_results=5)

# --------------- Tools (The "Actions" our agent can take) ---------------

def rag_tool(question: str, k: int = 2) -> str:
    """
    Tool 1: RAG (Retrieval Augmented Generation)
    Searches our vector database for relevant resume tips
    """
    try:
        # Convert question to vector
        query_vector = embeddings.embed_query(question)
        
        # Search Pinecone
        results = pindex.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True
        )
        
        # Format results
        tips = []
        for match in results.matches or []:
            text = match.metadata.get("text", "")
            source = match.metadata.get("source", "unknown")
            tips.append(f"💡 {text} (from {source})")
        
        return "\n".join(tips) if tips else "No relevant tips found in database"
        
    except Exception as e:
        return f"RAG search failed: {str(e)}"

def web_search_tool(query: str) -> str:
    """
    Simple Web Search - searches for current information
    """
    try:
        results = ddg_search.invoke(query)
        return f"🌐 Current Web Info: {results}" if results else "No web results found"
    except Exception as e:
        return f"Web search failed: {str(e)}"

# --------------- LangGraph State & Nodes ---------------

class AgentState(TypedDict):
    user_question: str
    rag_results: str
    web_results: str
    final_answer: str
    use_web: bool
    use_rag: bool

def router_node(state: AgentState) -> AgentState:
    """
    Smart Router: Uses LLM to decide what tools to use
    """
    question = state["user_question"]
    
    # Let the LLM decide what tools to use
    decision_prompt = f"""
    User asked: "{question}"
    
    Should I use web search for current/trending info? (yes/no)
    Should I use RAG for resume tips? (yes/no)
    
    Respond with just: web:yes/no rag:yes/no
    """
    
    decision = llm.invoke(decision_prompt).content.lower()
    
    # Parse the decision
    state["use_web"] = "web:yes" in decision
    state["use_rag"] = "rag:yes" in decision
    
    return state

def rag_node(state: AgentState) -> AgentState:
    """Get resume tips from database"""
    state["rag_results"] = rag_tool(state["user_question"]) if state.get("use_rag", True) else "Skipped"
    return state

def web_search_node(state: AgentState) -> AgentState:
    """Get current web information"""
    state["web_results"] = web_search_tool(state["user_question"]) if state.get("use_web", False) else "Skipped"
    return state

def llm_node(state: AgentState) -> AgentState:
    """Generate smart response using all available information"""
    
    # Build dynamic context
    context_parts = []
    if state.get("rag_results") and state["rag_results"] != "Skipped":
        context_parts.append(f"Database Tips:\n{state['rag_results']}")
    
    if state.get("web_results") and state["web_results"] != "Skipped":
        context_parts.append(f"Current Web Info:\n{state['web_results']}")
    
    context = "\n\n".join(context_parts) if context_parts else "No additional context available"
    
    prompt = f"""You are a smart resume advisor. Answer this question using any relevant information:

Question: {state["user_question"]}

{context}

Give helpful, practical advice. Be concise and encouraging."""
    
    state["final_answer"] = llm.invoke(prompt).content
    return state

# --------------- Build the Graph ---------------

# Create the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("router", router_node)
graph.add_node("rag", rag_node)
graph.add_node("web_search", web_search_node)
graph.add_node("llm", llm_node)

# Set entry point
graph.set_entry_point("router")

# Add edges (the flow)
graph.add_edge("router", "rag")
graph.add_edge("rag", "web_search")
graph.add_edge("web_search", "llm")
graph.add_edge("llm", END)

# Compile the graph
app = graph.compile()

# --------------- Streamlit UI ---------------

st.set_page_config(
    page_title="Simple Resume Advisor", 
    page_icon="📝",
    layout="wide"
)

st.title("📝 Simple Resume Advisor")
st.caption("Learn LangGraph with RAG + Web Search tools")

# Simple sidebar
with st.sidebar:
    st.header("🤖 Smart Resume Advisor")
    st.write("**AI-powered tools:**")
    st.write("• 🧠 Smart routing")
    st.write("• 📚 Resume database")
    st.write("• 🌐 Current web info")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Dynamic chat input
if user_input := st.chat_input("Ask anything about resumes..."):
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Prepare state for LangGraph
    initial_state = {
        "user_question": user_input,
        "rag_results": "",
        "web_results": "",
        "final_answer": "",
        "use_web": False,
        "use_rag": False
    }
    
    # Run the graph
    with st.spinner("Thinking..."):
        result = app.invoke(initial_state)
    
    # Display the response
    response = result["final_answer"]
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Show what tools were used (simplified)
    with st.expander("🔍 What I used to help you"):
        tools_used = []
        if result.get("rag_results") and result["rag_results"] != "Skipped":
            tools_used.append("📚 Resume database")
        if result.get("web_results") and result["web_results"] != "Skipped":
            tools_used.append("🌐 Current web info")
        
        if tools_used:
            st.write("**Tools used:** " + " • ".join(tools_used))
        else:
            st.write("**Used:** Just my knowledge")

# Dynamic footer
st.markdown("---")
st.caption("💡 **Try asking:** 'How to write a summary?' • 'Current resume trends' • 'What skills to highlight?'")
