# Results

Numbers reproduced from the runs in this repository. Formatted versions and full
discussion are in the accompanying paper. Project-wise split unless noted; mean over seeds.

## Android — main comparison (enriched, project-wise, 7,067 test issues)

| Model | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR | FixSim@1 | CHit@1 | CJac@1 |
|---|---|---|---|---|---|---|---|---|
| TF–IDF | 0.116 | 0.296 | 0.388 | 0.485 | 0.199 | 0.045 | 0.177 | 0.205 |
| BM25 | 0.076 | 0.180 | 0.254 | 0.344 | 0.128 | 0.039 | 0.176 | 0.202 |
| Hybrid (TF–IDF + Mapping) | 0.134 | 0.315 | 0.412 | 0.510 | 0.219 | 0.052 | 0.188 | 0.214 |
| **Dual-encoder (fine-tuned)** | **0.207** | **0.402** | **0.503** | **0.595** | **0.299** | **0.061** | **0.194** | **0.214** |
| Oracle (upper bound, FixSim@1) | – | – | – | – | – | 0.288 | – | – |

## Semantic change agreement (GraphCodeBERT diff embeddings, mean-centered, oracle-normalized)

6,571 covered test issues; 25,128 candidate fixes with diffs. Centering collapses
the random-pair similarity to 0.005 (oracle ceiling 0.718), making the judge
discriminative. nSemSim is 0 for a random fix, 1 for the oracle-closest pool fix.

| Model | nSemSim@1 | nSemSim@10 | SemHit@1 |
|---|---|---|---|
| BM25 | 0.125 | 0.577 | 0.273 |
| TF–IDF | 0.129 | 0.595 | 0.283 |
| Hybrid | 0.141 | 0.599 | 0.291 |
| **Dual-encoder** | **0.163** | **0.631** | **0.313** |

The ordering (Dual > Hybrid > TF–IDF > BM25) is **identical** across rank-based,
fix-text, change-type, and semantic-code metrics — four independent lenses agree.

## Coverage vs. conditional precision (Android, Hybrid)

"Answerable" = the pool contains a fix textually close to the true fix (Oracle
similarity ≥ τ), decided independently of any model.

| Subset | Coverage | Hit@1 | Hit@10 | MRR |
|---|---|---|---|---|
| Full set | 1.00 | 0.134 | 0.412 | 0.219 |
| Drop lowest 25% | 0.75 | 0.145 | 0.424 | 0.231 |
| Drop lowest 50% | 0.50 | 0.155 | 0.422 | 0.238 |
| Keep top 25% | 0.25 | 0.174 | 0.411 | 0.249 |

Gains are small and monotone: genuinely-new issues are a secondary, not dominant,
limit on accuracy.

## Statistical significance (paired Wilcoxon vs. Hybrid, Cliff's δ)

All differences significant at p < 10⁻³ on both Android and Python. Effect size
over plain TF–IDF is small (δ ≈ 0.05); over BM25/LSA/Mapping larger.

## Generalization — Python (784 projects, 34,697 issue–PR pairs)

The Hybrid is the robust default across both domains. The dual-encoder's Android
advantage shrinks to parity on Python; a training-data-efficiency sweep shows its
fine-tuning saturates at ~10% of pairs, isolating the remaining gap as a property
of the domain, not supervision volume. See the accompanying paper for the full tables.
