"""
Export the ground-truth issue -> pull-request links from a PI-Link-format JSON
as a compact CSV (one row per issue-PR pair). This small file lets reviewers
inspect and verify the links without downloading the full multi-hundred-MB
dataset; the issue/PR text and changed files can be re-fetched from the URLs.

  python export_links.py --data python_pilink.json --out python_issue_pr_links.csv

Columns: issue_repo, issue_url, pr_url
"""
import argparse, csv, json, re

PR_RE = re.compile(r"github\.com/([^/]+/[^/]+)/pull/\d+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="PI-Link-format JSON")
    ap.add_argument("--out", required=True, help="output CSV")
    args = ap.parse_args()

    raw = json.load(open(args.data, encoding="utf-8"))
    rows, seen, seen_issues = [], set(), set()
    for it in raw:
        repo = (it.get("issue_repository") or {}).get("nameWithOwner", "")
        iurl = it.get("issue_url", "")
        for n in (it.get("timelineItems") or {}).get("nodes", []):
            u = (n.get("pull_info") or {}).get("pull_url")
            if not u or (iurl, u) in seen:          # dedup identical issue-PR pairs
                continue
            seen.add((iurl, u))
            seen_issues.add(iurl)
            rows.append((repo, iurl, u))
    issues, pairs = len(seen_issues), len(rows)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["issue_repo", "issue_url", "pr_url"])
        w.writerows(rows)
    repos = len({r for r, _, _ in rows})
    print(f"Wrote {pairs} issue-PR pairs from {issues} issues across {repos} "
          f"projects to {args.out}")


if __name__ == "__main__":
    main()
