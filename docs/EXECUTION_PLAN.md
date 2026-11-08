# VentureGuard execution record

Scope: financial ML research, cash-flow planning, API, UI, Azure deployment and a reproducible GitHub release. RAG is deferred; no retrieval, embeddings, chat or LLM calls are part of v1.

1. Freeze source data, target, contracts and evaluation protocol.
2. Fit train-only transformations; use the same grouped folds across matched experiments.
3. Compare logistic regression, histogram gradient boosting and a PyTorch residual MLP against training-prevalence baselines.
4. Evaluate 13/64 features, missingness dependence, class weights and sigmoid calibration. Report every candidate family, not only the winner.
5. Export versioned research artifacts, held-out demo cases and explanations.
6. Implement independently tested monthly cash-flow accounting and scenarios.
7. Build validated FastAPI endpoints, then a responsive React UI.
8. Run API, accounting, model export and browser checks before deployment.
9. Deploy CPU inference on Azure Consumption with scale-to-zero; verify the public endpoint.
10. Publish source, metrics, model/data cards, deployment evidence and reproducible commands.

Existing research files and unrelated working changes are preserved. Data already inspected is never described as a new untouched test set. Exact-vector grouping does not identify companies. Source ratios do not reconstruct company cash balances. The app does not claim validated Egyptian startup bankruptcy probabilities.
