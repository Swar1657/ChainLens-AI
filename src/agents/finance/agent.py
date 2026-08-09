from typing import TypedDict, Optional
import os
from datetime import date
import pandas as pd
from sqlalchemy import text
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from src.storage.sql.database import DatabaseClient
from src.analytics.finance import calculate_period_metrics, calculate_period_over_period
from src.agents.contracts import FinancialReport, FinancialMetric, StatusEnum
from src.rag.vector_store import retrieve_context, format_retrieved_context
from src.observability.tracer import get_opik_callbacks, safe_track

# --- Agent State ---

class FinanceState(TypedDict):
    query: str
    current_start: date
    current_end: date
    previous_start: Optional[date]
    previous_end: Optional[date]
    
    # Deterministic Data
    current_metrics: dict
    previous_metrics: dict
    trends: dict
    
    # RAG Context
    rag_context: str
    
    # Final Output
    report: Optional[FinancialReport]
    error: Optional[str]

# --- Nodes ---

@safe_track
def fetch_and_calculate_metrics(state: FinanceState) -> FinanceState:
    """Queries the database and performs deterministic math."""
    try:
        client = DatabaseClient()
        query = text("""
            SELECT date, transaction_type, amount 
            FROM financial_transactions 
            WHERE date >= :start_date AND date <= :end_date
        """)
        
        with client.get_session() as session:
            # Current Period
            curr_res = session.execute(query, {"start_date": state["current_start"], "end_date": state["current_end"]}).fetchall()
            df_curr = pd.DataFrame(curr_res, columns=['date', 'transaction_type', 'amount'])
            curr_metrics = calculate_period_metrics(df_curr)
            
            # Previous Period (if requested)
            prev_metrics = {}
            trends = {}
            if state.get("previous_start") and state.get("previous_end"):
                prev_res = session.execute(query, {"start_date": state["previous_start"], "end_date": state["previous_end"]}).fetchall()
                df_prev = pd.DataFrame(prev_res, columns=['date', 'transaction_type', 'amount'])
                prev_metrics = calculate_period_metrics(df_prev)
                trends = calculate_period_over_period(curr_metrics, prev_metrics)
                
        docs = retrieve_context(state['query'], k=2)
        rag_context = format_retrieved_context(docs)
                
        return {
            **state,
            "current_metrics": curr_metrics,
            "previous_metrics": prev_metrics,
            "trends": trends,
            "rag_context": rag_context
        }
    except Exception as e:
        return {**state, "error": f"Database or calculation error: {str(e)}"}

@safe_track
def generate_interpretation(state: FinanceState) -> FinanceState:
    """Uses LLM strictly to interpret the deterministic math."""
    if state.get("error"):
        report = FinancialReport(
            metrics=[], 
            trends=[], 
            interpretation="Failed to generate report due to data errors.",
            status=StatusEnum.FAILED,
            error_message=state["error"]
        )
        return {**state, "report": report}
        
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(FinancialReport)
    
    # Provide the exact numbers to the LLM
    prompt = f"""You are the ChainLens Finance Agent.
Your task is to interpret the following exact financial metrics and generate a structured FinancialReport.
DO NOT invent any numbers. Rely strictly on the provided data.

Original Query: {state['query']}

Current Period ({state['current_start']} to {state['current_end']}):
{state['current_metrics']}

Previous Period ({state.get('previous_start')} to {state.get('previous_end')}):
{state['previous_metrics']}

Trends (Growth %):
{state['trends']}

Operational Guidelines (RAG Context):
{state.get('rag_context', 'No specific guidelines retrieved.')}

Provide a brief, executive-level interpretation of the financial health based on these numbers, referencing any official guidelines if margins fall below benchmarks.
"""
    try:
        report = structured_llm.invoke(
            [
                SystemMessage(content="You are a strict financial analyst. You do not calculate, you only interpret provided data into structured output."),
                HumanMessage(content=prompt)
            ],
            config={"callbacks": get_opik_callbacks()}
        )
        
        # We manually map the metrics array to ensure strict fidelity to the DB
        metrics_list = []
        for key, val in state['current_metrics'].items():
            metrics_list.append(FinancialMetric(
                name=key.replace('_', ' ').title(),
                value=val,
                previous_value=state['previous_metrics'].get(key),
                pop_change_pct=state['trends'].get(f"{key}_growth_pct")
            ))
            
        # Overwrite LLM's metrics with our deterministic ones to guarantee no hallucinations
        report.metrics = metrics_list
        report.status = StatusEnum.SUCCESS
        return {**state, "report": report}
    except Exception as e:
        return {**state, "error": f"LLM Generation error: {str(e)}"}

# --- Graph Construction ---

def create_finance_agent() -> StateGraph:
    workflow = StateGraph(FinanceState)
    
    workflow.add_node("calculate", fetch_and_calculate_metrics)
    workflow.add_node("interpret", generate_interpretation)
    
    workflow.set_entry_point("calculate")
    workflow.add_edge("calculate", "interpret")
    workflow.add_edge("interpret", END)
    
    return workflow.compile()
