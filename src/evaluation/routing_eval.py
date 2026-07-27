import time
import statistics
from typing import List, Tuple
from src.agents.supervisor.agent import supervisor_graph, router_node, SupervisorState
from src.agents.contracts import AgentSelection

# --- Benchmark Dataset ---
# Format: (query, expected_data_analyst, expected_risk, expected_finance)
ROUTING_DATASET = [
    # General exploratory Data Analyst queries
    ("How many orders were placed in Q2 2016?", True, False, False),
    ("Show me the top 5 product categories by volume.", True, False, False),
    ("What is the average order quantity across all regions?", True, False, False),
    
    # Pure Risk queries
    ("What products have the highest delivery risk right now?", False, True, False),
    ("Are any inventory items in a warning state?", False, True, False),
    ("What happens if a shipment is delayed by 3 days according to our SLA?", False, True, False),
    
    # Pure Finance queries
    ("What was our gross margin last month?", False, False, True),
    ("Has there been any abnormal spike in operating expenses recently?", False, False, True),
    ("What's the revenue forecast for the next 4 weeks?", False, False, True),
    
    # Multi-Agent queries
    ("Compare last month's revenue to this month's, and tell me if it poses a risk.", False, True, True),
    ("Are late shipments impacting our profitability?", False, True, True),
    ("Show me the raw order counts from last month and give me a full risk and finance breakdown.", True, True, True),
    ("What is causing margin deterioration, is it related to delivery penalties?", False, True, True),
    ("Give me a complete health check of the supply chain.", True, True, True),
    ("Query the raw inventory table and tell me our financial COGS.", True, False, True)
]

def evaluate_routing():
    print("--- Supervisor Routing Benchmark ---")
    print(f"Total test queries: {len(ROUTING_DATASET)}")
    
    correct_routing = 0
    total_unnecessary = 0
    total_agents_called = 0
    failed = 0
    
    for query, exp_da, exp_risk, exp_fin in ROUTING_DATASET:
        state = SupervisorState(query=query, sub_results=[], report=None, selection=None)
        
        try:
            result_state = router_node(state)
            sel: AgentSelection = result_state["selection"]
            
            # Check correctness (exact match)
            if (sel.use_data_analyst == exp_da and 
                sel.use_risk == exp_risk and 
                sel.use_finance == exp_fin):
                correct_routing += 1
                
            # Count unnecessary agents (false positives)
            unnecessary = 0
            if sel.use_data_analyst and not exp_da: unnecessary += 1
            if sel.use_risk and not exp_risk: unnecessary += 1
            if sel.use_finance and not exp_fin: unnecessary += 1
            
            total_unnecessary += unnecessary
            total_agents_called += sum([sel.use_data_analyst, sel.use_risk, sel.use_finance])
            
        except Exception as e:
            print(f"Routing failed for '{query}': {e}")
            failed += 1
            
    accuracy = correct_routing / len(ROUTING_DATASET)
    unnecessary_rate = total_unnecessary / max(1, total_agents_called)
    
    print(f"\nRouting Accuracy: {accuracy * 100:.1f}%")
    print(f"Unnecessary-Agent Rate: {unnecessary_rate * 100:.1f}%")
    print(f"Failed Routings: {failed}")
    
    return accuracy, unnecessary_rate

def test_parallel_latency():
    """Measures latency of executing a complex query via the parallel supervisor graph."""
    print("\n--- Parallel Execution Latency Test ---")
    query = "Give me a complete health check of the supply chain."
    print(f"Test Query: '{query}' (Triggers all 3 agents)")
    
    # We will run it 3 times to get an average
    latencies = []
    
    for i in range(3):
        start = time.time()
        # In a real async environment, LangGraph runs Sends in parallel automatically
        # LangGraph Python uses threadpools for Send objects by default
        state = {"query": query}
        result = supervisor_graph.invoke(state)
        end = time.time()
        
        latencies.append(end - start)
        print(f"Run {i+1}: {latencies[-1]:.2f}s")
        
    avg_latency = statistics.mean(latencies)
    print(f"Average Parallel Latency: {avg_latency:.2f}s")
    
    # We estimate sequential execution by inspecting the graph or knowing that 
    # sequentially it would be the sum of 3 LLM calls/DB fetches + 1 synthesis call.
    # Parallel reduces it to max(DA, Risk, Finance) + synthesis.
    # The actual latency output demonstrates this efficiency.
    
if __name__ == "__main__":
    evaluate_routing()
    test_parallel_latency()
