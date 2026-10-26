# Machine Learning Research Labs & Benchmarks

A quantitative laboratory suite spanning **LLMOps Evaluation Frameworks**, **Edge Vision Quantization**, **Explainable AI (XAI)**, and **Data Contract Engineering**.

---

## Contained Research Laboratories

```
Research-Labs-and-Benchmarks/
├── RAG-and-LLM-Evaluation-Lab/
│   ├── rag_pipeline_evaluator.py       # Context precision, recall, faithfulness, and relevance
│   ├── benchmark_dataset.json          # Multi-turn evaluation ground truth dataset
│   └── README.md
│
├── Edge-Vision-Quantization-Lab/
│   ├── onnx_quantizer_benchmark.py     # FP32 to FP16 and INT8 PTQ latency benchmark
│   └── README.md
│
├── Explainable-AI-Interpretability-Lab/
│   ├── xai_tabular_interpreter.py      # Shapley values (SHAP) & counterfactual recourse
│   └── README.md
│
├── Data-Validation-and-ETL-Pipeline-Lab/
│   ├── etl_data_contract_pipeline.py   # Schema contract enforcement & DLQ routing
│   └── README.md
│
└── README.md                           # This overview
```
