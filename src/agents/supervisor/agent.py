import os
import json
import operator
from typing import TypedDict, List, Optional, Any, Annotated
from datetime import date
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.constants import Send

from src.agents.contracts import (
    AgentSelection, 
    ExecutiveReport, 
    AgentSubResult, 
    StatusEnum, 
    RiskAssessment, 
    FinancialReport, 
    DataAnalystResult
)

# Import the pre-compiled agent graphs
from src.agents.data_analyst.agent import create_data_analyst_agent
from src.agents.risk.agent import create_risk_agent
from src.agents.finance.agent import create_finance_agent
from src.observability.tracer import get_opik_callbacks, safe_track

data_analyst_graph = create_data_analyst_agent()
risk_graph = create_risk_agent()
finance_graph = create_finance_agent()

class SupervisorState(TypedDict):
    query: str
    
    # Selected Agents
    selection: Optional[AgentSelection]
    
    # Results from Agents (Using list to aggregate parallel Send results)
    sub_results: Annotated[list[AgentSubResult], operator.add]
    
    # Final Output
    report: Optional[ExecutiveReport]

# --- Nodes ---

@safe_track
def router_node(state: SupervisorState) -> dict:
    """Uses LLM to classify the query and select required agents. Falls back to heuristic if no API key."""
    query_lower = state['query'].lower()
    
    # Try LLM first
    if os.getenv("OPENAI_API_KEY"):
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            structured_llm = llm.with_structured_output(AgentSelection)
            
            prompt = f"""You are the Supervisor for a Supply Chain Decision Intelligence system.
Given the user's query, determine which specialized agents are required to answer it.

Data Analyst: For exploratory data analysis, counting records, identifying trends via raw SQL.
Risk: For identifying supply chain risks, late deliveries, inventory stockouts, or quality severity.
Finance: For revenue, profit, margins, expenses, or financial diagnostics.

You can select multiple agents if the query spans multiple domains.
If the query is general (e.g., "Give me a complete health check"), select all three.

User Query: {state['query']}
"""
            selection = structured_llm.invoke(
                prompt,
                config={"callbacks": get_opik_callbacks()}
            )
            return {"selection": selection}
        except Exception as e:
            pass # Fall back to heuristic
            
    # Heuristic Fallback Router
    sel = AgentSelection(use_data_analyst=False, use_risk=False, use_finance=False)
    
    if any(w in query_lower for w in ["risk", "delivery", "late", "inventory", "stock", "warning", "sla"]):
        sel.use_risk = True
        
    if any(w in query_lower for w in ["finance", "margin", "profit", "revenue", "expense", "cogs"]):
        sel.use_finance = True
        
    if any(w in query_lower for w in ["how many", "show me", "average", "top", "raw", "count"]):
        sel.use_data_analyst = True
        
    # Health check triggers all
    if "health check" in query_lower or "complete" in query_lower:
        sel.use_data_analyst = True
        sel.use_risk = True
        sel.use_finance = True
        
    # Default to data analyst if nothing matched
    if not sel.use_data_analyst and not sel.use_risk and not sel.use_finance:
        sel.use_data_analyst = True
        
    return {"selection": sel}

# --- Parallel Conditional Edge Functions ---

def route_to_agents(state: SupervisorState):
    """Returns a list of Send objects to run nodes in parallel."""
    sends = []
    selection = state.get("selection")
    if not selection:
        return sends
        
    if selection.use_data_analyst:
        sends.append(Send("data_analyst_node", {"query": state["query"]}))
        
    if selection.use_risk:
        sends.append(Send("risk_node", {"query": state["query"]}))
        
    if selection.use_finance:
        sends.append(Send("finance_node", {"query": state["query"]}))
        
    # If no agents were selected for some reason, jump straight to synthesize
    if not sends:
        sends.append(Send("synthesize_results", state))
        
    return sends

# --- Wrapper Nodes for Agents ---

def data_analyst_node(state: dict) -> dict:
    """Wrapper to call Data Analyst graph and map its state back to SupervisorState."""
    try:
        result = data_analyst_graph.invoke({"query": state["query"]})
        da_res: DataAnalystResult = result.get("result")
        
        status = da_res.status if da_res else StatusEnum.FAILED
        err = da_res.error_message if da_res else result.get("error")
        data = da_res.model_dump() if da_res else None
        
        sub_res = AgentSubResult(
            agent_name="Data Analyst",
            status=status,
            result_data=data,
            error_message=err
        )
        return {"sub_results": [sub_res]}
    except Exception as e:
        return {"sub_results": [AgentSubResult(agent_name="Data Analyst", status=StatusEnum.FAILED, error_message=str(e), result_data=None)]}

def risk_node(state: dict) -> dict:
    """Wrapper to call Risk graph and map its state back to SupervisorState."""
    try:
        result = risk_graph.invoke({"query": state["query"]})
        risk_res: RiskAssessment = result.get("assessment")
        
        status = risk_res.status if risk_res else StatusEnum.FAILED
        err = risk_res.error_message if risk_res else result.get("error")
        data = risk_res.model_dump() if risk_res else None
        
        sub_res = AgentSubResult(
            agent_name="Risk",
            status=status,
            result_data=data,
            error_message=err
        )
        return {"sub_results": [sub_res]}
    except Exception as e:
        return {"sub_results": [AgentSubResult(agent_name="Risk", status=StatusEnum.FAILED, error_message=str(e), result_data=None)]}

def finance_node(state: dict) -> dict:
    """Wrapper to call Finance graph and map its state back to SupervisorState."""
    try:
        # Default to a stable period before the dataset collapse in Q4 2017
        finance_state = {
            "query": state["query"],
            "current_start": date(2017, 9, 1),
            "current_end": date(2017, 9, 30),
            "previous_start": date(2017, 8, 1),
            "previous_end": date(2017, 8, 31)
        }
        
        result = finance_graph.invoke(finance_state)
        fin_res: FinancialReport = result.get("report")
        
        status = fin_res.status if fin_res else StatusEnum.FAILED
        err = fin_res.error_message if fin_res else result.get("error")
        data = fin_res.model_dump() if fin_res else None
        
        sub_res = AgentSubResult(
            agent_name="Finance",
            status=status,
            result_data=data,
            error_message=err
        )
        return {"sub_results": [sub_res]}
    except Exception as e:
        return {"sub_results": [AgentSubResult(agent_name="Finance", status=StatusEnum.FAILED, error_message=str(e), result_data=None)]}


@safe_track
def synthesize_results(state: SupervisorState) -> dict:
    """Fan-in node to collect sub_results and write the executive report."""
    sub_results = state.get("sub_results", [])
    
    # Format insights to pass to LLM
    context = ""
    failures = []
    
    for res in sub_results:
        if res.status != StatusEnum.SUCCESS:
            failures.append(f"{res.agent_name} failed: {res.error_message}")
            context += f"\n--- {res.agent_name} Agent (FAILED) ---\nError: {res.error_message}\n"
        else:
            context += f"\n--- {res.agent_name} Agent ---\n{json.dumps(res.result_data, indent=2)}\n"
            
    if os.getenv("OPENAI_API_KEY"):
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            structured_llm = llm.with_structured_output(ExecutiveReport)
            
            prompt = f"""You are the Executive Supervisor for a Supply Chain Decision Intelligence system.
You have routed the user's query to specialized agents, and they have returned structured data and interpretations.

User Query: {state['query']}

Agent Results:
{context}

Synthesize a coherent ExecutiveReport. Do NOT invent data. Rely strictly on the agent results provided.
If an agent failed, note it in the failures list.
"""
            report = structured_llm.invoke(
                prompt,
                config={"callbacks": get_opik_callbacks()}
            )
            # Ensure failures are captured
            if failures:
                report.overall_status = StatusEnum.PARTIAL_SUCCESS
                report.failures.extend(failures)
                
            return {"report": report}
        except Exception as e:
            pass
            
    # Fallback Synthesis if no LLM
    overall_status = StatusEnum.PARTIAL_SUCCESS if failures else StatusEnum.SUCCESS
    if len(failures) == len(sub_results) and sub_results:
        overall_status = StatusEnum.FAILED
        
    report = ExecutiveReport(
        summary=f"Automated fallback synthesis for: '{state['query']}'. Processed {len(sub_results)} agent responses.",
        data_insights="Data Analyst output returned." if any(r.agent_name == "Data Analyst" and r.status == StatusEnum.SUCCESS for r in sub_results) else None,
        risk_insights="Risk assessment completed." if any(r.agent_name == "Risk" and r.status == StatusEnum.SUCCESS for r in sub_results) else None,
        finance_insights="Financial metrics calculated." if any(r.agent_name == "Finance" and r.status == StatusEnum.SUCCESS for r in sub_results) else None,
        overall_status=overall_status,
        failures=failures
    )
    return {"report": report}

# --- Graph Definition ---

builder = StateGraph(SupervisorState)

builder.add_node("router", router_node)
builder.add_node("data_analyst_node", data_analyst_node)
builder.add_node("risk_node", risk_node)
builder.add_node("finance_node", finance_node)
builder.add_node("synthesize_results", synthesize_results)

builder.set_entry_point("router")

# Fan-out to agents based on the selection
builder.add_conditional_edges(
    "router",
    route_to_agents,
    ["data_analyst_node", "risk_node", "finance_node", "synthesize_results"]
)

# Fan-in from agents to the synthesizer
builder.add_edge("data_analyst_node", "synthesize_results")
builder.add_edge("risk_node", "synthesize_results")
builder.add_edge("finance_node", "synthesize_results")
builder.add_edge("synthesize_results", END)

supervisor_graph = builder.compile()
