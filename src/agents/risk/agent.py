from typing import TypedDict, Optional, List
import pandas as pd
from sqlalchemy import text
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from src.storage.sql.database import DatabaseClient
from src.analytics.risk import summarize_portfolio_risk
from src.agents.contracts import RiskAssessment, RiskComponent, StatusEnum, SeverityEnum
from src.rag.vector_store import retrieve_context, format_retrieved_context

# --- Agent State ---

class RiskState(TypedDict):
    query: str
    
    # Deterministic Data
    portfolio_risk: dict
    
    # RAG Context
    rag_context: str
    
    # Final Output
    assessment: Optional[RiskAssessment]
    error: Optional[str]

# --- Nodes ---

def fetch_and_calculate_risk(state: RiskState) -> RiskState:
    """Queries the database and computes deterministic risk metrics."""
    try:
        client = DatabaseClient()
        
        # We fetch the most recent data (e.g. the last 30 days of available data) 
        # to calculate current risk, as this is a snapshot agent.
        shipments_query = text("""
            SELECT days_for_shipping_real, days_for_shipping_scheduled 
            FROM shipments 
            ORDER BY shipping_date DESC LIMIT 5000
        """)
        
        inventory_query = text("""
            SELECT stock_level, reorder_point 
            FROM inventory 
            ORDER BY date DESC LIMIT 5000
        """)
        
        with client.get_session() as session:
            shipments_res = session.execute(shipments_query).fetchall()
            df_shipments = pd.DataFrame(shipments_res, columns=['days_for_shipping_real', 'days_for_shipping_scheduled'])
            
            inventory_res = session.execute(inventory_query).fetchall()
            df_inventory = pd.DataFrame(inventory_res, columns=['stock_level', 'reorder_point'])
            
            portfolio_risk = summarize_portfolio_risk(df_shipments, df_inventory)
            
        # Also fetch RAG context for operational definitions
        docs = retrieve_context(state['query'], k=2)
        rag_context = format_retrieved_context(docs)
            
        return {
            **state,
            "portfolio_risk": portfolio_risk,
            "rag_context": rag_context
        }
    except Exception as e:
        return {**state, "error": f"Database or calculation error: {str(e)}"}

def generate_interpretation(state: RiskState) -> RiskState:
    """Uses LLM strictly to interpret the deterministic math and write recommendations."""
    if state.get("error"):
        assessment = RiskAssessment(
            overall_score=0.0,
            overall_severity=SeverityEnum.SAFE,
            risk_components=[],
            recommended_actions=[],
            confidence="LOW",
            data_quality="Failed to retrieve risk data.",
            status=StatusEnum.FAILED,
            error_message=state["error"]
        )
        return {**state, "assessment": assessment}
        
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(RiskAssessment)
    
    prompt = f"""You are the ChainLens Risk Agent.
Your task is to interpret the following exact deterministic risk metrics and generate a structured RiskAssessment.
DO NOT invent any underlying risk numbers. Rely strictly on the provided data.

Original Query: {state['query']}

Current Portfolio Risk Metrics:
{state['portfolio_risk']}

Operational Context (RAG):
{state.get('rag_context', 'No context provided.')}

Instructions:
1. Map these metrics into appropriate RiskComponents (e.g. 'Delivery Risk' based on late_shipment_rate).
2. Assign an overall severity based on these metrics AND the operational policies in the context.
3. Provide actionable mitigation recommendations referencing the official policies if applicable.
4. Calculate an overall score (0.0 to 1.0) mathematically based on the component metrics.
"""
    try:
        assessment = structured_llm.invoke([
            SystemMessage(content="You are a strict supply chain risk analyst. You interpret hard data and recommend actions. You do not fabricate metrics."),
            HumanMessage(content=prompt)
        ])
        
        assessment.status = StatusEnum.SUCCESS
        return {**state, "assessment": assessment}
    except Exception as e:
        return {**state, "error": f"LLM Generation error: {str(e)}"}

# --- Graph Construction ---

def create_risk_agent() -> StateGraph:
    workflow = StateGraph(RiskState)
    
    workflow.add_node("calculate", fetch_and_calculate_risk)
    workflow.add_node("interpret", generate_interpretation)
    
    workflow.set_entry_point("calculate")
    workflow.add_edge("calculate", "interpret")
    workflow.add_edge("interpret", END)
    
    return workflow.compile()
