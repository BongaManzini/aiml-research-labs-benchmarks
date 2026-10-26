# Explainable AI (XAI) & Interpretability Laboratory

A model governance and interpretability framework providing **Shapley Value (SHAP)** feature attributions, **LIME Local Approximations**, and **Counterfactual Recourse** for credit scoring and high-stakes predictive systems.

---

## The Four Axiomatic Properties of Shapley Values

```mermaid
flowchart TD
    SHAP["Shapley Attributions $\phi_i$"]
    SHAP --> Eff["1. Efficiency: $\sum \phi_i = f(x) - E[f(x)]$"]
    SHAP --> Sym["2. Symmetry: Identical contributors receive identical attribution"]
    SHAP --> Dummy["3. Dummy Player: Zero-impact features receive $\phi = 0$"]
    SHAP --> Add["4. Additivity: Feature attributions sum across ensemble trees"]
```

---

## Regulatory Compliance & Right to Explanation

In highly regulated sectors (Banking, Insurance, Healthcare under South African POPIA and EU GDPR Article 22), black-box machine learning models cannot legally make adverse credit decisions without providing:
1. The top driving features contributing to the adverse decision.
2. Actionable counterfactual recourse explaining what the individual must change to reverse the outcome.

---

## Tool Execution

```bash
python3 AI-and-Machine-Learning/Research-Labs-and-Benchmarks/Explainable-AI-Interpretability-Lab/xai_tabular_interpreter.py
```
Outputs local Shapley decomposition and automated counterfactual suggestions.
