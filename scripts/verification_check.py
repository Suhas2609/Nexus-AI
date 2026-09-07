# verification_check.py
from rag.vectorstore.store import get_or_create_store
from rag.vectorstore.retriever import get_mmr_retriever
import json

with open('evaluation/golden_dataset.json', encoding='utf-8') as f:
    golden = json.load(f)

store = get_or_create_store()
retriever = get_mmr_retriever(store)

mismatches = 0

for i, entry in enumerate(golden):
    question = entry["question"]
    stored_contexts = entry["reference_contexts"]
    
    # Retrieve what the live system returns
    docs = retriever.invoke(question)
    live_contexts = [doc.page_content for doc in docs]
    
    # Check if they match (order and content)
    if stored_contexts != live_contexts:
        mismatches += 1
        print(f"Q{i+1}: MISMATCH")
        print(f"  Question: {question[:60]}...")
        print(f"  Stored: {len(stored_contexts)} contexts")
        print(f"  Live:   {len(live_contexts)} contexts")
        if stored_contexts and live_contexts:
            if stored_contexts[0][:50] != live_contexts[0][:50]:
                print(f"  First chunk differs: stored starts with '{stored_contexts[0][:50]}...'")
                print(f"                      live starts with   '{live_contexts[0][:50]}...'")

print(f"\n{len(golden) - mismatches}/{len(golden)} questions match live retriever output.")
if mismatches == 0:
    print("✓ Golden dataset is consistent with production retriever. Ready to use.")
else:
    print(f"✗ {mismatches} questions have outdated contexts. Regenerate before evaluation.")