"""
Collect a PI-Link-format issue->PR dataset for any language, from GitHub
========================================================================
Crawls merged pull requests and the issues they close across many repositories,
emitting the SAME format pilink_fix_model.py consumes -- with full PR text
(title, body, commit messages) AND the changed files in one pass, so no separate
enrichment is needed. This gives a clean, apples-to-apples second dataset for the
generalization study (unlike BeetleBox, which lacks PR prose and has few projects).

Outputs (resumable, appended as it goes):
  <out>.json          PI-Link-format issue list  (built at the end from the .jsonl)
  <out>.issues.jsonl  one issue per line (the resumable store)
  <out>_files.jsonl   pull_url -> changed files (the pr_files.jsonl the model reads)
  <out>.repos_done    repositories already crawled (skip on re-run)

Reuses the token pool / rate-limit handling from fetch_pr_files.py (keep both files
together). Needs a GitHub token:  export GITHUB_TOKEN=ghp_...   (classic, public_repo).

Usage (Colab, internet on):
  pip install requests
  # discover the top Python repos by stars and crawl them:
  python collect_issue_pr_dataset.py --language Python --min_stars 500 \
      --max_repos 300 --max_prs_per_repo 400 --out python_pilink
  # or crawl a fixed repo list (one owner/name per line):
  python collect_issue_pr_dataset.py --repos repos.txt --out python_pilink

Then evaluate exactly like Android (neutral taxonomy for cross-domain):
  python pilink_fix_model.py --data python_pilink.json \
      --files python_pilink_files.jsonl --change_taxonomy neutral \
      --models random,tfidf,hybrid,dual --seeds 42,7,123
"""

import argparse, json, os, sys, time

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

# reuse the proven token pool + GraphQL caller
try:
    from fetch_pr_files import Tokens, gql, PR_RE
except Exception as e:
    sys.exit(f"Need fetch_pr_files.py next to this script ({e})")


def q_search_repos(after, language, min_stars, batch):
    cur = f', after: "{after}"' if after else ""
    return f'''query {{
      search(query: "language:{language} stars:>{min_stars} sort:stars-desc", type: REPOSITORY, first: {batch}{cur}) {{
        pageInfo {{ hasNextPage endCursor }}
        nodes {{ ... on Repository {{ nameWithOwner isArchived isFork }} }}
      }}
      rateLimit {{ cost remaining resetAt }}
    }}'''


def q_repo_prs(owner, name, after, batch):
    cur = f', after: "{after}"' if after else ""
    return f'''query {{
      repository(owner: {json.dumps(owner)}, name: {json.dumps(name)}) {{
        pullRequests(states: MERGED, first: {batch}{cur}, orderBy: {{field: UPDATED_AT, direction: DESC}}) {{
          pageInfo {{ hasNextPage endCursor }}
          nodes {{
            url title body
            commits(first: 20) {{ nodes {{ commit {{ message }} }} }}
            files(first: 100) {{ totalCount nodes {{ path additions deletions changeType }} }}
            closingIssuesReferences(first: 5) {{
              nodes {{ url title body number createdAt
                       labels(first: 20) {{ nodes {{ name }} }} }}
            }}
          }}
        }}
      }}
      rateLimit {{ cost remaining resetAt }}
    }}'''


def discover_repos(session, tokens, language, min_stars, max_repos, batch=50):
    repos, after = [], None
    while len(repos) < max_repos:
        body = gql(session, tokens, q_search_repos(after, language, min_stars, batch))
        s = (body.get("data") or {}).get("search") or {}
        for n in s.get("nodes") or []:
            if n and n.get("nameWithOwner") and not n.get("isFork"):
                repos.append(n["nameWithOwner"])
        pi = s.get("pageInfo") or {}
        after = pi.get("endCursor")
        print(f"  discovered {len(repos)} repos (rate-limit {tokens.current()['remaining']})", flush=True)
        if not pi.get("hasNextPage"):
            break
    return repos[:max_repos]


def crawl_repo(session, tokens, repo, max_prs, issues_out, files_out, seen_issue, seen_pull):
    owner, name = repo.split("/", 1)
    after, got, pairs = None, 0, 0
    while got < max_prs:
        body = gql(session, tokens, q_repo_prs(owner, name, after, 25))
        r = ((body.get("data") or {}).get("repository")) or {}
        prs = r.get("pullRequests") or {}
        for pr in prs.get("nodes") or []:
            got += 1
            closing = (pr.get("closingIssuesReferences") or {}).get("nodes") or []
            if not closing:
                continue
            files = (pr.get("files") or {}).get("nodes") or []
            url = pr["url"]
            if url not in seen_pull and files:
                seen_pull.add(url)
                files_out.write(json.dumps({
                    "pull_url": url, "status": "ok", "merged": True,
                    "files_total": (pr.get("files") or {}).get("totalCount"),
                    "files": [{"path": f["path"], "additions": f.get("additions"),
                               "deletions": f.get("deletions"), "change_type": f.get("changeType")}
                              for f in files]}) + "\n")
            commits = [c.get("commit", {}).get("message", "")
                       for c in (pr.get("commits") or {}).get("nodes") or []]
            for iss in closing:
                iu = iss.get("url")
                if not iu or iu in seen_issue:
                    continue
                seen_issue.add(iu)
                labels = [{"label": l["name"]} for l in (iss.get("labels") or {}).get("nodes") or []]
                issues_out.write(json.dumps({
                    "issue_repository": {"nameWithOwner": repo},
                    "issue_url": iu,
                    "issue_title": iss.get("title") or "",
                    "issue_body": iss.get("body") or "",
                    "issue_labels": {"edges": [{"node": l} for l in labels]},
                    "created_at": iss.get("createdAt"),
                    "timelineItems": {"nodes": [{"pull_info": {
                        "pull_url": url,
                        "pull_title": pr.get("title") or "",
                        "pull_body": pr.get("body") or "",
                        "commits": {"nodes": [{"commit": {"commit_message": m}} for m in commits[:20]]},
                    }}]}}) + "\n")
                pairs += 1
        pi = prs.get("pageInfo") or {}
        after = pi.get("endCursor")
        if not pi.get("hasNextPage"):
            break
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="python_pilink")
    ap.add_argument("--language", default="Python", help="GitHub language facet (e.g. Python, JavaScript)")
    ap.add_argument("--min_stars", type=int, default=500)
    ap.add_argument("--max_repos", type=int, default=300)
    ap.add_argument("--max_prs_per_repo", type=int, default=400)
    ap.add_argument("--repos", default=None, help="file with explicit owner/name list (skips discovery)")
    args = ap.parse_args()

    toks = [t.strip() for t in (os.environ.get("GITHUB_TOKENS") or
                                os.environ.get("GITHUB_TOKEN") or "").split(",") if t.strip()]
    tokens = Tokens(toks)
    session = requests.Session()

    done_path = args.out + ".repos_done"
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    seen_issue = set(); seen_pull = set()
    # rebuild seen_* from existing output so a resumed run doesn't duplicate
    if os.path.exists(args.out + ".issues.jsonl"):
        for line in open(args.out + ".issues.jsonl", encoding="utf-8"):
            try: seen_issue.add(json.loads(line)["issue_url"])
            except Exception: pass
    if os.path.exists(args.out + "_files.jsonl"):
        for line in open(args.out + "_files.jsonl", encoding="utf-8"):
            try: seen_pull.add(json.loads(line)["pull_url"])
            except Exception: pass

    if args.repos:
        repos = [r.strip() for r in open(args.repos) if r.strip() and "/" in r]
    else:
        print("Discovering repositories...", flush=True)
        repos = discover_repos(session, tokens, args.language, args.min_stars, args.max_repos)
    repos = [r for r in repos if r not in done]
    print(f"{len(repos)} repositories to crawl ({len(done)} already done)", flush=True)

    t0 = time.time(); total = 0
    with open(args.out + ".issues.jsonl", "a", encoding="utf-8") as iout, \
         open(args.out + "_files.jsonl", "a", encoding="utf-8") as fout, \
         open(done_path, "a", encoding="utf-8") as dout:
        for k, repo in enumerate(repos, 1):
            try:
                n = crawl_repo(session, tokens, repo, args.max_prs_per_repo,
                               iout, fout, seen_issue, seen_pull)
            except Exception as e:
                print(f"  [skip {repo}] {str(e)[:120]}", flush=True)
                continue
            total += n
            iout.flush(); fout.flush()
            dout.write(repo + "\n"); dout.flush()
            print(f"  [{k}/{len(repos)}] {repo}: +{n} issue-PR pairs "
                  f"(total {total}, rate-limit {tokens.current()['remaining']})", flush=True)

    # assemble the JSON array the model loads, from the resumable .jsonl store
    issues = [json.loads(l) for l in open(args.out + ".issues.jsonl", encoding="utf-8")]
    json.dump(issues, open(args.out + ".json", "w", encoding="utf-8"))
    repos_n = len({i["issue_repository"]["nameWithOwner"] for i in issues})
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. "
          f"{len(issues)} issues / {len(seen_pull)} PRs / {repos_n} projects.")
    print(f"Wrote {args.out}.json and {args.out}_files.jsonl")


if __name__ == "__main__":
    main()
