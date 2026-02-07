from dataset import generate_research_dataset

# Cell 4: Generate dataset (this will take 15-30 minutes for 15k samples)
counts, methods = generate_research_dataset(
    output_path="local_dataset.jsonl",
    target_per_class=25000,  # 15k total samples
    use_llm=True,
    llm_ratio=0.8,  # 80% LLM-generated
    hard_neg_ratio=0.1  # 10% hard negatives
)