# Retrieval-Based Fix Recommendation for Software Issues Using Issue–Pull-Request Links

Code, data pointers, and the manuscript for our study on recommending fixes for a
**newly reported issue in a previously unseen project** by retrieving the pull
requests (PRs) that resolved similar past issues. Built on the
[PI-Link](https://www.kaggle.com/datasets/zakareaalshara/android-closed-issues-20110101-20210101-clean)
ground-truth issue–PR link dataset, enriched with changed files and diff hunks.

## What's here

```
code/        retrieval models, evaluation, enrichment crawler, analyses
notebooks/   Colab notebooks (enrichment + model runs)
results/     metric tables reproduced from the runs
data/        pointers + instructions to obtain/regenerate the datasets
```

## The task

Given a new issue (natural-language text) in a project not seen during training,
retrieve, from a pool of historical PRs that fixed other projects' issues, the
fixes most likely to help. Evaluated under a strict **project-wise split** (test
projects disjoint from training projects) and a complementary **temporal split**
(train on past issues, test on future ones).

## Models (`code/pilink_fix_model.py`)

Random · TF–IDF · BM25 · LSA · trained issue→fix Mapping (ridge) ·
Hybrid (TF–IDF + Mapping) · zero-shot Sentence-BERT (MiniLM/MPNet) ·
contrastive two-tower · fine-tuned dual-encoder · cross-encoder re-ranker.

## Metrics

- **Rank**: Hit@k, MRR
- **Fix text**: FixSim@k / nFixSim@k (oracle-normalized TF–IDF similarity to the true fix)
- **Change type**: CHit@k, CExact@k, CJac@1 (Jaccard over change-type signatures from the enriched file lists)
- **Semantic code change**: nSemSim@k, SemHit@k (mean-centered GraphCodeBERT diff embeddings vs. the true fix, oracle-normalized)
- **Significance**: paired Wilcoxon signed-rank + Cliff's delta (`sig_test.py`)
- **Coverage vs. conditional precision**: answerable-subset analysis (`answerable.py`)

## Quick start

```bash
pip install -r requirements.txt

# main model comparison (project-wise split) on the enriched Android data
python code/pilink_fix_model.py \
    --data data/android_closed_issues_2011-01-01_2021-01-01_all_clean_issues.json \
    --files data/pr_files.jsonl \
    --models random,tfidf,bm25,lsa,mapping,hybrid

# semantic-diff agreement (needs GPU + diff hunks; see data/README.md)
SEMDIFF=1 SEM_CACHE=./sem_cache python code/pilink_fix_model.py \
    --data data/android_...issues.json --files data/pr_files.jsonl \
    --patches data/pr_patches_full.jsonl --models tfidf,bm25,hybrid,dual
```

See `data/README.md` for how to obtain the base dataset (Kaggle) and regenerate
the enrichment (`fetch_pr_files.py`), and `results/README.md` for the numbers.

## Reproducing the enrichment

```bash
# changed-file lists (fast, GraphQL)
GITHUB_TOKENS=tok1,tok2 python code/fetch_pr_files.py \
    --data data/android_...issues.json --mode files --out data/pr_files.jsonl

# diff hunks for the semantic metric (REST, slower; resumable)
GITHUB_TOKENS=tok1,tok2 python code/fetch_pr_files.py \
    --data data/android_...issues.json --mode patches --out data/pr_patches_full.jsonl
```

## Citation

If you use this code or the enriched data, please cite the accompanying paper
and the original PI-Link dataset.
