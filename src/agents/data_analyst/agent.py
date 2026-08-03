import re
from typing import TypedDict, Optional, List, Dict, Any
from sqlalchemy import text
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from src.storage.sql.database import DatabaseClient
from src.agents.contracts import DataAnalystResult, SQLQuery, StatusEnum
from src.observability.tracer import get_opik_callbacks, safe_track

# --- Agent State ---

class DataAnalystState(TypedDict):
    query_intent: str
    sql_generated: Optional[str]
    sql_explanation: Optional[str]
    sql_validated: bool
    structured_data: Optional[List[Dict[str, Any]]]
    result: Optional[DataAnalystResult]
    error: Optional[str]

# --- Schema Definition for LLM ---

SCHEMA_PROMPT = """
The database is PostgreSQL.
Allowed tables:
- orders (id, source_order_id, order_item_id, customer_id, customer_segment, customer_city, customer_state, customer_country, order_date, order_status, product_category, product_name, product_id, quantity, product_price, discount, discount_rate, sales, order_item_total, order_profit, profit_ratio)
- shipments (id, order_item_id, shipping_date, shipping_mode, days_for_shipping_real, days_for_shipping_scheduled, delivery_status, late_delivery_risk)
- inventory (id, date, product_id, product_category, product_name, stock_level, reorder_point, is_stockout)
- financial_transactions (id, date, transaction_type, category, amount)
"""

# --- Nodes ---

@safe_track
def generate_sql(state: DataAnalystState) -> DataAnalystState:
    """Uses LLM to convert query intent into SQL."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(SQLQuery)
    
    prompt = f"""You are an expert Data Analyst and PostgreSQL developer.
Write a SQL query to answer the user's intent.
Always use explicit JOINs if querying multiple tables.
Always apply a LIMIT of 100 if the query could return many rows, unless aggregating.
Do not use destructive operations (DROP, DELETE, UPDATE, INSERT). Only SELECT is allowed.

Schema Info:
{SCHEMA_PROMPT}

User Intent: {state['query_intent']}
"""
    try:
        sql_query: SQLQuery = structured_llm.invoke(
            [
                SystemMessage(content="You only output valid SELECT statements according to the schema."),
                HumanMessage(content=prompt)
            ],
            config={"callbacks": get_opik_callbacks()}
        )
        return {
            **state,
            "sql_generated": sql_query.query,
            "sql_explanation": sql_query.explanation
        }
    except Exception as e:
        return {**state, "error": f"Failed to generate SQL: {str(e)}"}

@safe_track
def validate_sql(state: DataAnalystState) -> DataAnalystState:
    """Validates the generated SQL for safety (Read-Only)."""
    if state.get("error"):
        return state
        
    sql = state["sql_generated"]
    if not sql:
        return {**state, "error": "No SQL was generated."}
        
    sql_upper = sql.upper()
    destructive_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE", "ALTER", "CREATE", "GRANT", "REVOKE"]
    
    for word in destructive_keywords:
        if re.search(r'\b' + word + r'\b', sql_upper):
            return {**state, "error": f"Destructive SQL keyword blocked: {word}", "sql_validated": False}
            
    if not re.search(r'^\s*SELECT\b', sql_upper):
        return {**state, "error": "Query must start with SELECT.", "sql_validated": False}
        
    return {**state, "sql_validated": True}

@safe_track
def execute_sql(state: DataAnalystState) -> DataAnalystState:
    """Executes the validated SQL query against PostgreSQL."""
    if state.get("error") or not state.get("sql_validated"):
        return state
        
    try:
        client = DatabaseClient()
        query = text(state["sql_generated"])
        
        with client.get_session() as session:
            result = session.execute(query)
            keys = result.keys()
            rows = result.fetchall()
            
            # Format to list of dicts
            structured_data = [dict(zip(keys, row)) for row in rows]
            
        return {**state, "structured_data": structured_data}
    except Exception as e:
        return {**state, "error": f"Database execution error: {str(e)}"}

@safe_track
def interpret_results(state: DataAnalystState) -> DataAnalystState:
    """Interprets the data returned from the database into natural language."""
    if state.get("error"):
        result = DataAnalystResult(
            query_intent=state['query_intent'],
            sql_executed=state.get('sql_generated', ''),
            structured_data=[],
            interpretation="Failed to process query.",
            status=StatusEnum.FAILED,
            error_message=state["error"]
        )
        return {**state, "result": result}
        
    structured_data = state["structured_data"]
    
    if not structured_data:
        interpretation = "The query returned no results."
    else:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # We don't use structured output here, we just ask for a string interpretation
        prompt = f"""You are a Data Analyst explaining SQL results to a user.
User Intent: {state['query_intent']}
SQL Executed: {state['sql_generated']}

Returned Data:
{structured_data}

Provide a concise, professional explanation of these results. Do not invent numbers. Just summarize what the data says.
"""
        try:
            msg = llm.invoke(
                [
                    SystemMessage(content="You interpret SQL results clearly."),
                    HumanMessage(content=prompt)
                ],
                config={"callbacks": get_opik_callbacks()}
            )
            interpretation = msg.content
        except Exception as e:
            interpretation = f"Failed to interpret results: {str(e)}"
    
    result = DataAnalystResult(
        query_intent=state['query_intent'],
        sql_executed=state["sql_generated"],
        structured_data=structured_data,
        interpretation=interpretation,
        status=StatusEnum.SUCCESS
    )
    
    return {**state, "result": result}

# --- Graph Construction ---

def create_data_analyst_agent() -> StateGraph:
    workflow = StateGraph(DataAnalystState)
    
    workflow.add_node("generate", generate_sql)
    workflow.add_node("validate", validate_sql)
    workflow.add_node("execute", execute_sql)
    workflow.add_node("interpret", interpret_results)
    
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "validate")
    workflow.add_edge("validate", "execute")
    workflow.add_edge("execute", "interpret")
    workflow.add_edge("interpret", END)
    
    return workflow.compile()
