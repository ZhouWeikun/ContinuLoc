# Source audit

Recovered from local `master` commit `cfb8d0724bf68bcab2c888aff790bc9c126711a9`.

Sources: `scripts_retrieval_then_matching/` main pipeline and recall exporter.
The legacy split helper is copied from `trainers/util_stage1_retrieval_evaluator.py`.
Adaptations: current continuloc imports, lazy runtime dependencies, explicit query paths,
optional current Stage1 bundle export, unique results directories, input validation.
GIM/RANSAC, inlier reranking and Top-1 pose correction retain the source algorithm.
