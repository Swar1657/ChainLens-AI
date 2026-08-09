import os
import time
import json
import streamlit as st
import pandas as pd
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

from src.agents.supervisor.agent import supervisor_graph, SupervisorState
from src.agents.contracts import StatusEnum

st.set_page_config(
    page_title="ChainLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 10px;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25em 0.4em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
    }
    .status-success { background-color: #28a745; color: white; }
    .status-warning { background-color: #ffc107; color: black; }
    .status-danger { background-color: #dc3545; color: white; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("🔍 ChainLens AI")
    st.markdown("Supply Chain Decision Intelligence")
    
    st.header("Configuration")
    
    # Check LLM Availability
    has_gemini = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    if has_gemini:
        st.success("✅ LLM Active (Gemini API Key present)")
    else:
        st.warning("⚠️ LLM unavailable — heuristic fallback active")
        st.caption("Routing and interpretation will use deterministic heuristics. The core Python analytics engine is still active.")
        
    has_opik = bool(os.getenv("OPIK_API_KEY") or os.getenv("OPIK_WORKSPACE"))
    if has_opik:
        st.success("✅ Opik Observability Active")
    else:
        st.info("ℹ️ Opik Tracing Offline (Graceful Degradation)")
        
    st.markdown("---")
    st.header("Example Queries")
    examples = [
        "Give me a complete health check of the supply chain.",
        "How many orders were placed in Q2 2016?",
        "What products have the highest delivery risk right now?",
        "Compare last month's revenue to this month's.",
        "What happens if a shipment is delayed by 3 days according to our SLA?",
    ]
    
    selected_example = st.selectbox("Select an example query:", ["-- Custom --"] + examples)

# --- Main UI ---
st.title("Supply Chain Control Tower")
st.markdown("Ask natural language questions about your supply chain data, risks, and financials.")

# Query Input
query_input = st.text_area("Enter your query:", value="" if selected_example == "-- Custom --" else selected_example)

def render_agent_selection(selection):
    st.markdown("### 🚦 Supervisor Routing Decision")
    cols = st.columns(3)
    
    def get_icon(is_selected):
        return "✅" if is_selected else "❌"
        
    with cols[0]:
        st.markdown(f"**Data Analyst**: {get_icon(selection.use_data_analyst)}")
    with cols[1]:
        st.markdown(f"**Risk**: {get_icon(selection.use_risk)}")
    with cols[2]:
        st.markdown(f"**Finance**: {get_icon(selection.use_finance)}")

def render_rag_evidence(rag_context: str, source_name: str):
    """Renders RAG context, clearly distinguishing it as synthetic project guidance."""
    if rag_context and rag_context.strip() != "No context provided.":
        with st.expander(f"📚 View Grounded Operational Guidance ({source_name})"):
            st.warning("This operational guidance is drawn from project-authored synthetic policies, not real corporate data.")
            st.markdown(rag_context)

def render_executive_report(report):
    st.markdown("### 📝 Executive Synthesis")
    status_str = report.overall_status.name if hasattr(report.overall_status, 'name') else str(report.overall_status)
    if status_str == "FAILED":
        st.error(f"Status: {status_str}")
    elif status_str == "PARTIAL_SUCCESS":
        st.warning(f"Status: {status_str}")
    else:
        st.success(f"Status: {status_str}")
        
    st.markdown(report.summary)
    
    # Also render insights if present
    if getattr(report, "data_insights", None):
        st.markdown(f"**Data Analyst Insights:** {report.data_insights}")
    if getattr(report, "risk_insights", None):
        st.markdown(f"**Risk Insights:** {report.risk_insights}")
    if getattr(report, "finance_insights", None):
        st.markdown(f"**Finance Insights:** {report.finance_insights}")
    
    if report.failures:
        st.markdown("#### Failures encountered:")
        for fail in report.failures:
            st.error(fail)

if st.button("Run Analysis", type="primary"):
    if not query_input.strip():
        st.warning("Please enter a query.")
    else:
        # Get Supervisor Graph
        supervisor = supervisor_graph
        
        # Initialize State
        state = SupervisorState(
            query=query_input,
            selection=None,
            sub_results=[],
            report=None
        )
        
        st.markdown("---")
        
        with st.status("Executing Multi-Agent Orchestration...", expanded=True) as status_container:
            # We will stream the execution to show exactly what's happening
            start_time = time.time()
            
            # The LangGraph stream outputs the state after each node completes
            for event in supervisor.stream(state):
                for node_name, node_state in event.items():
                    if node_name == "router_node":
                        st.write("✅ Supervisor routing completed.")
                        selection = node_state.get("selection")
                    elif node_name == "fan_in_synthesis":
                        st.write("✅ Executive synthesis completed.")
                    else:
                        st.write(f"⚙️ Agent **{node_name}** finished execution.")
            
            end_time = time.time()
            latency = end_time - start_time
            status_container.update(label=f"Execution completed in {latency:.2f}s", state="complete", expanded=False)
        
        # After execution, fetch the final state from the graph
        # Wait, LangGraph stream returns the updates, not the full state. 
        # We can just invoke it fully to get the final state easily, but we already streamed it.
        # Actually, invoking is cleaner for extracting final outputs. Let's just run invoke.
        
        # Let's run invoke to get the guaranteed final state, but we already advanced the generator? No, stream executes it.
        # Let's re-execute via invoke for simplicity, or just use the last event's state if we accumulate it.
        # It's cleaner to just invoke.
        
        with st.spinner("Compiling Final Results..."):
            final_state = supervisor.invoke(SupervisorState(query=query_input, selection=None, sub_results=[], report=None))
            
        selection = final_state.get("selection")
        if selection:
            render_agent_selection(selection)
            
        st.markdown("---")
        st.markdown("### 🔬 Agent Execution Results")
        
        sub_results = final_state.get("sub_results", [])
        
        if not sub_results:
            st.info("No agents were triggered for this query.")
        else:
            tabs = st.tabs([res.agent_name for res in sub_results])
            
            for tab, res in zip(tabs, sub_results):
                with tab:
                    status_str = res.status.name if hasattr(res.status, 'name') else str(res.status)
                    if status_str == "FAILED":
                        st.error(f"Agent execution failed: {res.error_message}")
                    else:
                        st.success("Agent executed successfully.")
                        data = res.result_data
                        if not data:
                            st.write("No detailed data available.")
                            continue
                            
                        # Agent specific rendering
                        if res.agent_name == "Data Analyst":
                            st.markdown(f"**SQL Generated:**\n```sql\n{data.get('sql_executed', 'N/A')}\n```")
                            st.markdown(f"**Interpretation:** {data.get('interpretation', '')}")
                            
                        elif res.agent_name == "Risk":
                            if "assessment" in data and data["assessment"]:
                                assess = data["assessment"]
                                st.markdown(f"**Overall Score:** {assess.overall_score}")
                                st.markdown(f"**Severity:** {assess.overall_severity}")
                                
                                st.markdown("**Risk Components:**")
                                for comp in assess.risk_components:
                                    st.write(f"- {comp.name}: {comp.score} ({comp.severity})")
                                    
                            render_rag_evidence(data.get("rag_context", ""), "Risk Policies")
                            
                        elif res.agent_name == "Finance":
                            if "report" in data and data["report"]:
                                report = data["report"]
                                st.markdown(f"**Interpretation:** {report.interpretation}")
                                st.markdown("**Metrics:**")
                                for m in report.metrics:
                                    st.write(f"- {m.name}: {m.value} (Prev: {m.previous_value}, Change: {m.pop_change_pct}%)")
                                    
                            render_rag_evidence(data.get("rag_context", ""), "Financial Guidelines")

        st.markdown("---")
        report = final_state.get("report")
        if report:
            render_executive_report(report)
        else:
            st.error("Executive Synthesis failed to produce a report.")
