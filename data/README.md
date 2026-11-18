# Data

The datasets are too large for the git repository (the Android base file alone is
~484 MB, above GitHub's 100 MB per-file limit). Obtain them as follows.

## 1. Base dataset (PI-Link, Android partition)

Published on Kaggle:
https://www.kaggle.com/datasets/zakareaalshara/android-closed-issues-20110101-20210101-clean

Download `android_closed_issues_2011-01-01_2021-01-01_all_clean_issues.json`
(~484 MB) into this `data/` folder.

## 2. Changed-file enrichment (`pr_files.jsonl`, ~56 MB)

Per-PR changed files (path, additions, deletions, change type) for ~96.9% of the
linked PRs. Regenerate with:

```bash
GITHUB_TOKENS=tok1,tok2 python ../code/fetch_pr_files.py \
    --data android_closed_issues_2011-01-01_2021-01-01_all_clean_issues.json \
    --mode files --out pr_files.jsonl
```

## 3. Diff-hunk enrichment (`pr_patches_full.jsonl`, ~100–200 MB)

Unified diffs per file (capped), used by the semantic-change metric; ~96.1%
coverage. Slower REST crawl (~7 h on one token; resumable — re-run to continue):

```bash
GITHUB_TOKENS=tok1,tok2 python ../code/fetch_pr_files.py \
    --data android_closed_issues_2011-01-01_2021-01-01_all_clean_issues.json \
    --mode patches --out pr_patches_full.jsonl
```

## 4. Second-domain dataset (Python, 784 projects)

`python_pilink.json` (~78 MB) + `python_pilink_files.jsonl` collected with
`../code/collect_issue_pr_dataset.py` (GitHub GraphQL, merged PRs + closing
issues). Regenerate with:

```bash
GITHUB_TOKENS=tok1,tok2 python ../code/collect_issue_pr_dataset.py \
    --language python --min_stars 100 --max_repos 784 --out python_pilink.json
```

## Hosting the large files

For permanent, citable hosting of the enriched files, upload them as a **GitHub
Release asset** (up to 2 GB/file) or to **Zenodo** (gives a DOI, preferred for
papers). Do **not** commit them to the git history.
