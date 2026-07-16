import pytest
from src.agents.data_analyst.agent import validate_sql

def test_sql_validation_safe():
    state = {
        "query_intent": "get top products",
        "sql_generated": "SELECT product_name, SUM(quantity) FROM orders GROUP BY product_name LIMIT 10;",
        "sql_explanation": "Test",
        "sql_validated": False,
        "structured_data": None,
        "result": None,
        "error": None
    }
    
    new_state = validate_sql(state)
    assert new_state["sql_validated"] is True
    assert new_state.get("error") is None

def test_sql_validation_destructive():
    state = {
        "query_intent": "delete bad records",
        "sql_generated": "DELETE FROM orders WHERE quantity < 0;",
        "sql_explanation": "Test",
        "sql_validated": False,
        "structured_data": None,
        "result": None,
        "error": None
    }
    
    new_state = validate_sql(state)
    assert new_state["sql_validated"] is False
    assert "Destructive SQL keyword blocked: DELETE" in new_state["error"]

def test_sql_validation_no_select():
    state = {
        "query_intent": "show tables",
        "sql_generated": "SHOW TABLES;",
        "sql_explanation": "Test",
        "sql_validated": False,
        "structured_data": None,
        "result": None,
        "error": None
    }
    
    new_state = validate_sql(state)
    assert new_state["sql_validated"] is False
    assert "Query must start with SELECT" in new_state["error"]
