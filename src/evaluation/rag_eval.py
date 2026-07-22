import sys
import random
from typing import List, Dict, Tuple
from src.rag.vector_store import retrieve_context

# Gold Standard Dataset (Expanded to ~60 Queries)
# Format: List of (query, expected_source_filename)

EVAL_DATASET = [
    # --- Category: SLAs & Late Delivery Rules (supply_chain_policies.md) ---
    ("What happens if a shipment is delayed by 3 days?", "supply_chain_policies.md"),
    ("Is a 1-day late delivery considered critical?", "supply_chain_policies.md"),
    ("What is the penalty for a carrier causing a critical delivery risk?", "supply_chain_policies.md"),
    ("How long is the grace period for a warning delivery status?", "supply_chain_policies.md"),
    ("Define the requirements for 'On Time' shipping.", "supply_chain_policies.md"),
    ("What triggers a 5% fee on the invoice?", "supply_chain_policies.md"),
    ("Who intervenes when a shipment is more than 3 days late?", "supply_chain_policies.md"),
    ("What shipping class is used for emergency stock mitigation?", "supply_chain_policies.md"),
    ("How long does First Class delivery usually take?", "supply_chain_policies.md"),
    ("What is the standard transit time for non-perishable freight?", "supply_chain_policies.md"),
    ("If an order is shipped using Second Class, when should it arrive?", "supply_chain_policies.md"),
    ("We have a B2B component that is critical, how should we ship it?", "supply_chain_policies.md"),
    ("At the end of the month, the coordinator asks for penalties, why?", "supply_chain_policies.md"),
    ("My delivery took 2 days longer than expected, what's the severity?", "supply_chain_policies.md"),
    ("How are standard class deliveries timed?", "supply_chain_policies.md"),
    ("Which role handles intervention for critical SLA breaches?", "supply_chain_policies.md"),
    ("Are there financial repercussions for carriers who deliver late?", "supply_chain_policies.md"),
    ("Explain the difference between First Class and Second Class shipping.", "supply_chain_policies.md"),
    ("What is the exact definition of a Warning delivery?", "supply_chain_policies.md"),
    ("When should a shipment be classified as a critical delivery risk?", "supply_chain_policies.md"),

    # --- Category: Inventory Thresholds & Safety Stock (inventory_guidelines.md) ---
    ("What is the safety stock for Class A products?", "inventory_guidelines.md"),
    ("What action is required if inventory is in critical status?", "inventory_guidelines.md"),
    ("When are physical inventory audits required?", "inventory_guidelines.md"),
    ("How is the dynamic reorder point calculated?", "inventory_guidelines.md"),
    ("If current stock is 0, what is the procedure?", "inventory_guidelines.md"),
    ("What is the buffer stock percentage for Class C goods?", "inventory_guidelines.md"),
    ("Does Class B inventory require 15% safety stock?", "inventory_guidelines.md"),
    ("My stock is at 1.2x the reorder point, is that an emergency?", "inventory_guidelines.md"),
    ("Explain the concept of 'Warning' inventory status.", "inventory_guidelines.md"),
    ("If we completely run out of an item, how do we replenish?", "inventory_guidelines.md"),
    ("What triggers an automated reorder event?", "inventory_guidelines.md"),
    ("How often do warehouses do physical counts?", "inventory_guidelines.md"),
    ("Who needs to know if the physical count is off by 3%?", "inventory_guidelines.md"),
    ("Is a 1% discrepancy in stock level a major issue?", "inventory_guidelines.md"),
    ("What is the trailing demand period for reorder calculations?", "inventory_guidelines.md"),
    ("Define Stockout protocol.", "inventory_guidelines.md"),
    ("What happens if current stock is below the ROP?", "inventory_guidelines.md"),
    ("Where do we source emergency inventory from during a stockout?", "inventory_guidelines.md"),
    ("Are audits done monthly or quarterly?", "inventory_guidelines.md"),
    ("What threshold triggers a report to the Finance Department?", "inventory_guidelines.md"),

    # --- Category: Financial Guidelines & Authorization (financial_guidelines.md) ---
    ("Who needs to approve a discount of 15 percent?", "financial_guidelines.md"),
    ("What is the company-wide net margin target?", "financial_guidelines.md"),
    ("What happens if operating expenses spike abnormally?", "financial_guidelines.md"),
    ("What is the policy for executive discounts over 20%?", "financial_guidelines.md"),
    ("What is our baseline gross profit margin goal?", "financial_guidelines.md"),
    ("If net margin drops to 8%, what process is initiated?", "financial_guidelines.md"),
    ("How much discount can a standard rep give without asking their boss?", "financial_guidelines.md"),
    ("Does the VP of Finance approve all discounts?", "financial_guidelines.md"),
    ("What happens if the monthly discount rate averages 16%?", "financial_guidelines.md"),
    ("Are OpEx spikes of 2 standard deviations flagged automatically?", "financial_guidelines.md"),
    ("What is the maximum allowed ratio of operating expenses to revenue?", "financial_guidelines.md"),
    ("What is the moving average period for anomaly detection on OpEx?", "financial_guidelines.md"),
    ("If a manager wants to give a 12% price cut, is that allowed?", "financial_guidelines.md"),
    ("Who authorizes a 25% discount?", "financial_guidelines.md"),
    ("What is the threshold for an expense review?", "financial_guidelines.md"),
    ("Explain the margin erosion alert.", "financial_guidelines.md"),
    ("What triggers an executive summary report for expenses?", "financial_guidelines.md"),
    ("Who signs off on manager-level discounts?", "financial_guidelines.md"),
    ("Is 12% net margin acceptable without a review?", "financial_guidelines.md"),
    ("What is the threshold for OpEx anomaly detection?", "financial_guidelines.md")
]

def run_retrieval_benchmark(k_values: List[int] = [1, 2, 3]):
    """Runs a retrieval benchmark measuring Recall@K and MRR on an expanded test set."""
    
    results = {}
    
    print(f"--- RAG Retrieval Benchmark (Expanded) ---")
    print(f"Total test queries: {len(EVAL_DATASET)}\n")
    
    # Calculate MRR (Mean Reciprocal Rank) across all queries
    # We will search top 5 to calculate MRR
    total_mrr = 0.0
    for query, expected_doc in EVAL_DATASET:
        retrieved = retrieve_context(query, k=5)
        rank = 0
        for i, doc in enumerate(retrieved):
            if doc.metadata.get("source") == expected_doc:
                rank = i + 1
                break
                
        if rank > 0:
            total_mrr += 1.0 / rank
            
    mrr = total_mrr / len(EVAL_DATASET)
    print(f"MRR (Mean Reciprocal Rank): {mrr:.4f}")
    
    # Calculate Recall@K
    for k in k_values:
        hits = 0
        for query, expected_doc in EVAL_DATASET:
            retrieved = retrieve_context(query, k=k)
            if any(doc.metadata.get("source") == expected_doc for doc in retrieved):
                hits += 1
                
        recall = hits / len(EVAL_DATASET)
        results[k] = recall
        print(f"Recall@{k}: {recall*100:.1f}%")
        
    return {"mrr": mrr, "recall": results}

if __name__ == "__main__":
    run_retrieval_benchmark()
