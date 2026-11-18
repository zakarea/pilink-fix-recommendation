# Ground-truth issue–PR links

Compact CSVs of the ground-truth links used in this study: for each closed
issue, the pull request(s) that resolved it. These are small enough to live in
the repository, so reviewers can inspect and verify the links directly; the full
issue/PR text and changed files can be re-fetched from the URLs with the scripts
in `../../code/` (`fetch_pr_files.py`, `collect_issue_pr_dataset.py`).

| File | Pairs | Issues | Projects |
|---|---|---|---|
| `android_issue_pr_links.csv` | 34,433 | 27,098 | 5,741 |
| `python_issue_pr_links.csv`  | (generate — see below) | | |

Columns: `issue_repo, issue_url, pr_url` (one row per issue–PR pair; an issue with
several linked PRs spans several rows).

Regenerate from a PI-Link-format JSON:

```bash
python ../../code/export_links.py --data <dataset>.json --out <name>_issue_pr_links.csv
```

The Android links are generated from the published PI-Link Android partition
(Kaggle); the Python links from the 784-project corpus collected with
`collect_issue_pr_dataset.py`.
