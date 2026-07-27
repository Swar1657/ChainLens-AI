from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

# --- Shared Base Definitions ---

class StatusEnum(str):
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    FAILED = "FAILED"

class SeverityEnum(str):
    SAFE = "SAFE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    STOCKOUT = "STOCKOUT"

# --- Data Analyst Agent Contracts ---

class SQLQuery(BaseModel):
    query: str = Field(description="The generated PostgreSQL query.")
    explanation: str = Field(description="Why this query was generated.")

class SQLResult(BaseModel):
    query: str
    result_data: List[Dict[str, Any]] = Field(description="The raw data rows returned from the database.")
    row_count: int
    status: str = Field(default=StatusEnum.SUCCESS)
    error_message: Optional[str] = None

class DataAnalystResult(BaseModel):
    query_intent: str = Field(description="The user's original query intent.")
    sql_executed: str = Field(description="The actual SQL executed.")
    structured_data: List[Dict[str, Any]] = Field(description="The data returned.")
    interpretation: str = Field(description="A natural language interpretation of the data.")
    status: str = Field(default=StatusEnum.SUCCESS)
    error_message: Optional[str] = None

# --- Risk Agent Contracts ---

class RiskComponent(BaseModel):
    name: str = Field(description="Name of the risk component, e.g., 'Delivery Risk' or 'Inventory Risk'.")
    score: float = Field(description="Quantitative risk score, normalized 0.0 to 1.0 if possible.")
    severity: str = Field(description="Categorical severity (SAFE, WARNING, CRITICAL, STOCKOUT).")
    contributing_factors: List[str] = Field(description="Factors driving this risk.")

class RiskAssessment(BaseModel):
    overall_score: float = Field(description="Aggregate portfolio-level risk score.")
    overall_severity: str = Field(description="Overall categorical severity.")
    risk_components: List[RiskComponent] = Field(description="Detailed component risks.")
    recommended_actions: List[str] = Field(description="Actionable mitigation recommendations.")
    confidence: str = Field(description="Confidence level in the assessment (HIGH, MEDIUM, LOW).")
    data_quality: str = Field(description="Any data quality issues impacting the assessment.")
    status: str = Field(default=StatusEnum.SUCCESS)
    error_message: Optional[str] = None

# --- Finance Agent Contracts ---

class FinancialMetric(BaseModel):
    name: str = Field(description="Name of the metric (e.g., 'Gross Profit', 'Revenue').")
    value: float = Field(description="Current period value.")
    previous_value: Optional[float] = Field(description="Previous period value for comparison.")
    pop_change_pct: Optional[float] = Field(description="Period-over-period change percentage.")

class ForecastResult(BaseModel):
    target: str = Field(description="What is being forecasted (e.g., 'Weekly Revenue').")
    future_values: List[float] = Field(description="Forecasted values for the next periods.")
    horizon: int = Field(description="Number of periods forecasted.")
    confidence_interval: Optional[Dict[str, List[float]]] = Field(description="Upper and lower bounds if available.", default=None)

class FinancialReport(BaseModel):
    metrics: List[FinancialMetric] = Field(description="Key financial metrics.")
    forecast_summary: Optional[ForecastResult] = Field(description="Forecast results if requested/applicable.", default=None)
    trends: List[str] = Field(description="Identified trends in the financial data.")
    interpretation: str = Field(description="Executive interpretation of the financial health.")
    status: str = Field(default=StatusEnum.SUCCESS)
    error_message: Optional[str] = None

# --- Supervisor Agent Contracts ---

class AgentSelection(BaseModel):
    use_data_analyst: bool = Field(description="Whether the user query requires fetching basic/raw quantitative data or exploratory SQL analytics.")
    use_risk: bool = Field(description="Whether the user query requires identifying portfolio risks, late deliveries, or inventory severity.")
    use_finance: bool = Field(description="Whether the user query asks for revenue, profit, margins, or financial diagnostics.")

class AgentSubResult(BaseModel):
    agent_name: str
    status: str
    result_data: Any
    error_message: Optional[str] = None

class ExecutiveReport(BaseModel):
    summary: str = Field(description="High-level executive summary across all requested domains.")
    data_insights: Optional[str] = Field(description="Key insights from the Data Analyst agent, if called.", default=None)
    risk_insights: Optional[str] = Field(description="Key insights from the Risk agent, if called.", default=None)
    finance_insights: Optional[str] = Field(description="Key insights from the Finance agent, if called.", default=None)
    overall_status: str = Field(default=StatusEnum.SUCCESS)
    failures: List[str] = Field(description="List of agents that failed to execute properly.", default_factory=list)
