"""
PI-Link enrichment: fetch the changed files (and optionally diffs) of every PR
=============================================================================
Reads the PI-Link JSON, collects every unique pull-request URL, and fetches from GitHub:

  mode "files"   (GraphQL, batched, fast)  -> per PR: merged state, PR labels,
                                              changed files with path, additions,
                                              deletions, changeType
  mode "patches" (REST, one call per PR)   -> the same plus the unified diff (patch)
                                              of each file, truncated to --max_patch chars

Output is JSON Lines (one PR per line), appended as it goes, so the run is RESUMABLE:
re-running the same command skips PRs that are already in the output file.

Rate limits: GitHub allows 5,000 GraphQL points/hour and 5,000 REST requests/hour per
token. The script reads the remaining budget from every response and sleeps until the
reset time when it runs low. Several tokens can be given to rotate between them.

Rough time for ~32.5k PRs with ONE token:
  files   : ~650 GraphQL queries -> about 1-3 hours (cost depends on file counts)
  patches : ~33k+ REST requests  -> about 7 hours

Usage
  export GITHUB_TOKEN=ghp_xxx               # or GITHUB_TOKENS=tok1,tok2,tok3
  pip install requests
  python fetch_pr_files.py --data android_closed_issues_...json --out pr_files.jsonl
  python fetch_pr_files.py --data ... --out pr_patches.jsonl --mode patches
  python fetch_pr_files.py --data ... --out pr_files.jsonl --limit 200      # quick trial

Output line example
  {"pull_url": "https://github.com/o/r/pull/19", "status": "ok", "merged": true,
   "labels": ["bug"], "files_total": 3,
   "files": [{"path": "app/src/main/AndroidManifest.xml", "additions": 2,
              "deletions": 0, "change_type": "MODIFIED"}, ...]}
  status is "ok", "not_found" (repo/PR deleted or private) or "error".
"""

import argparse, json, os, re, sys, time
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

API = "https://api.github.com"
GQL = API + "/graphql"
PR_RE = re.compile(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)")
FILES_PAGE = 100          # GitHub max per page
MAX_FILES = 3000          # GitHub never returns more than 3000 files per PR


# ---------------------------------------------------------------------------
# Token pool with rate-limit awareness
# ---------------------------------------------------------------------------
class Tokens:
    def __init__(self, tokens, min_remaining=50):
        if not tokens:
            sys.exit("No token. Set GITHUB_TOKEN or GITHUB_TOKENS (comma separated).")
        self.t = [{"tok": t, "remaining": None, "reset": 0} for t in tokens]
        self.i = 0
        self.min_remaining = min_remaining

    def current(self):
        return self.t[self.i]

    def update(self, remaining, reset):
        cur = self.current()
        if remaining is not None:
            cur["remaining"] = int(remaining)
        if reset:
            cur["reset"] = float(reset)

    def ensure_budget(self):
        """Rotate to a token with budget, or sleep until the earliest reset."""
        for _ in range(len(self.t)):
            cur = self.current()
            if cur["remaining"] is None or cur["remaining"] > self.min_remaining \
                    or time.time() > cur["reset"]:
                return
            self.i = (self.i + 1) % len(self.t)
        wait = max(5, min(t["reset"] for t in self.t) - time.time() + 5)
        print(f"  rate limit low on all tokens - sleeping {wait/60:.1f} min", flush=True)
        time.sleep(wait)
        for t in self.t:
            t["remaining"] = None

    def headers(self):
        return {"Authorization": f"bearer {self.current()['tok']}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "PI-Link-enrichment"}


def http(session, tokens, method, url, max_retries=6, **kw):
    """Request with retries for network errors, 5xx, secondary rate limits."""
    for attempt in range(max_retries):
        tokens.ensure_budget()
        try:
            r = session.request(method, url, headers=tokens.headers(), timeout=60, **kw)
        except requests.RequestException as e:
            wait = 2 ** attempt * 5
            print(f"  network error ({e.__class__.__name__}), retry in {wait}s", flush=True)
            time.sleep(wait)
            continue
        tokens.update(r.headers.get("X-RateLimit-Remaining"), r.headers.get("X-RateLimit-Reset"))
        if r.status_code in (403, 429):
            retry_after = r.headers.get("Retry-After")
            if retry_after:                                   # secondary rate limit
                time.sleep(int(retry_after) + 1)
                continue
            if r.headers.get("X-RateLimit-Remaining") == "0":  # primary limit
                tokens.update(0, r.headers.get("X-RateLimit-Reset"))
                continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt * 5)
            continue
        return r
    raise RuntimeError(f"giving up on {url}")


# ---------------------------------------------------------------------------
# Mode 1: GraphQL batched file lists
# ---------------------------------------------------------------------------
PR_FIELDS = """
  merged
  labels(first: 20) { nodes { name } }
  files(first: %d%s) {
    totalCount
    pageInfo { hasNextPage endCursor }
    nodes { path additions deletions changeType }
  }
"""


def gql_literal(s):
    return json.dumps(s)


def build_batch_query(prs):
    parts = []
    for k, (owner, repo, num) in enumerate(prs):
        parts.append(f'q{k}: repository(owner: {gql_literal(owner)}, name: {gql_literal(repo)}) '
                     f'{{ pullRequest(number: {num}) {{ {PR_FIELDS % (FILES_PAGE, "")} }} }}')
    return "query {\n" + "\n".join(parts) + "\n rateLimit { cost remaining resetAt } }"


def gql(session, tokens, query, json_retries=3):
    """POST a GraphQL query. GitHub occasionally returns a 200 with an empty or
    non-JSON body (transient edge error, or a batch too heavy to serve); retry a
    few times, then raise RuntimeError so the caller can split the batch."""
    last = ""
    for attempt in range(json_retries):
        r = http(session, tokens, "POST", GQL, max_retries=3, json={"query": query})
        if r.status_code != 200:
            raise RuntimeError(f"GraphQL HTTP {r.status_code}: {r.text[:300]}")
        try:
            body = r.json()
        except ValueError:
            last = f"non-JSON/empty body (len={len(r.text)})"
            time.sleep(2 ** attempt * 3)
            continue
        rl = (body.get("data") or {}).get("rateLimit")
        if rl:
            reset = datetime.fromisoformat(rl["resetAt"].replace("Z", "+00:00")).timestamp()
            tokens.update(rl["remaining"], reset)
        return body
    raise RuntimeError(f"GraphQL {last} after {json_retries} tries")


def more_files(session, tokens, owner, repo, num, cursor):
    """Follow pagination for PRs with more than 100 files."""
    files = []
    while cursor and len(files) < MAX_FILES:
        q = (f'query {{ repository(owner: {gql_literal(owner)}, name: {gql_literal(repo)}) '
             f'{{ pullRequest(number: {num}) {{ {PR_FIELDS % (FILES_PAGE, ", after: " + gql_literal(cursor))} }} }}'
             f' rateLimit {{ cost remaining resetAt }} }}')
        body = gql(session, tokens, q)
        pr = (((body.get("data") or {}).get("repository") or {}).get("pullRequest") or {})
        f = pr.get("files") or {}
        files += f.get("nodes") or []
        pi = f.get("pageInfo") or {}
        cursor = pi.get("endCursor") if pi.get("hasNextPage") else None
    return files


def to_record(url, pr):
    f = pr.get("files") or {}
    return {"pull_url": url, "status": "ok", "merged": pr.get("merged"),
            "labels": [n["name"] for n in (pr.get("labels") or {}).get("nodes") or []],
            "files_total": f.get("totalCount"),
            "files": [{"path": n["path"], "additions": n.get("additions"),
                       "deletions": n.get("deletions"), "change_type": n.get("changeType")}
                      for n in f.get("nodes") or []]}


def fetch_files_mode(session, tokens, todo, out, batch):
    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        try:
            body = gql(session, tokens, build_batch_query([p[1:] for p in chunk]))
            if not body.get("data") and body.get("errors"):
                raise RuntimeError(json.dumps(body["errors"])[:300])
        except RuntimeError as e:
            if len(chunk) > 1:
                # whole query failed (timeout / too heavy): retry in smaller pieces
                print(f"  batch failed ({str(e)[:80]}), splitting", flush=True)
                fetch_files_mode(session, tokens, chunk, out, max(1, len(chunk) // 4))
                continue
            for url, *_ in chunk:
                out.write(json.dumps({"pull_url": url, "status": "error", "error": str(e)[:200]}) + "\n")
            out.flush()
            continue
        data = body.get("data") or {}
        for k, (url, owner, repo, num) in enumerate(chunk):
            pr = (data.get(f"q{k}") or {}).get("pullRequest")
            if not pr:
                out.write(json.dumps({"pull_url": url, "status": "not_found"}) + "\n")
                continue
            rec = to_record(url, pr)
            pi = (pr.get("files") or {}).get("pageInfo") or {}
            if pi.get("hasNextPage"):
                try:
                    extra = more_files(session, tokens, owner, repo, num, pi.get("endCursor"))
                    rec["files"] += [{"path": n["path"], "additions": n.get("additions"),
                                      "deletions": n.get("deletions"), "change_type": n.get("changeType")}
                                     for n in extra]
                except RuntimeError as e:
                    rec["files_partial"] = True            # keep first 100 files, flag truncation
                    rec["error"] = f"pagination: {str(e)[:120]}"
            out.write(json.dumps(rec) + "\n")
        out.flush()
        rem = tokens.current()["remaining"]
        print(f"  {s + len(chunk)}/{len(todo)} PRs  (rate-limit remaining: {rem})", flush=True)


# ---------------------------------------------------------------------------
# Mode 2: REST, includes patches
# ---------------------------------------------------------------------------
def fetch_patches_mode(session, tokens, todo, out, max_patch):
    for n, (url, owner, repo, num) in enumerate(todo, 1):
        files, page, status = [], 1, "ok"
        while len(files) < MAX_FILES:
            r = http(session, tokens, "GET",
                     f"{API}/repos/{owner}/{repo}/pulls/{num}/files",
                     params={"per_page": FILES_PAGE, "page": page})
            if r.status_code in (404, 410, 451):
                status = "not_found"; break
            if r.status_code != 200:
                status = "error"; break
            items = r.json()
            for it in items:
                patch = it.get("patch")
                files.append({"path": it["filename"], "additions": it.get("additions"),
                              "deletions": it.get("deletions"),
                              "change_type": (it.get("status") or "").upper(),
                              "patch": patch[:max_patch] if patch else None,
                              "patch_truncated": bool(patch and len(patch) > max_patch)})
            if len(items) < FILES_PAGE:
                break
            page += 1
        rec = {"pull_url": url, "status": status}
        if status == "ok":
            rec.update(files_total=len(files), files=files)
        out.write(json.dumps(rec) + "\n")
        if n % 50 == 0:
            out.flush()
            print(f"  {n}/{len(todo)} PRs  (rate-limit remaining: {tokens.current()['remaining']})",
                  flush=True)
    out.flush()


# ---------------------------------------------------------------------------
def collect_prs(path):
    data = json.load(open(path, encoding="utf-8"))
    seen = {}
    for it in data:
        for n in (it.get("timelineItems") or {}).get("nodes", []):
            p = n.get("pull_info")
            if p and p.get("pull_url") and p["pull_url"] not in seen:
                m = PR_RE.search(p["pull_url"])
                if m:
                    seen[p["pull_url"]] = (m.group(1), m.group(2), int(m.group(3)))
    return seen


def already_done(path, retry_errors):
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") != "error" or not retry_errors:
                done.add(rec["pull_url"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="PI-Link JSON file")
    ap.add_argument("--out", default="pr_files.jsonl")
    ap.add_argument("--mode", choices=["files", "patches"], default="files")
    ap.add_argument("--batch", type=int, default=50, help="PRs per GraphQL query")
    ap.add_argument("--max_patch", type=int, default=20000, help="max chars kept per file patch")
    ap.add_argument("--limit", type=int, default=None, help="only first N PRs (trial run)")
    ap.add_argument("--retry_errors", action="store_true", help="re-fetch PRs that failed before")
    args = ap.parse_args()

    toks = [t.strip() for t in (os.environ.get("GITHUB_TOKENS") or
                                os.environ.get("GITHUB_TOKEN") or "").split(",") if t.strip()]
    tokens = Tokens(toks)

    prs = collect_prs(args.data)
    done = already_done(args.out, args.retry_errors)
    todo = [(u, *v) for u, v in prs.items() if u not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(prs)} unique PRs in dataset, {len(done)} already fetched, {len(todo)} to fetch "
          f"(mode={args.mode})", flush=True)
    if not todo:
        return

    t0 = time.time()
    with requests.Session() as session, open(args.out, "a", encoding="utf-8") as out:
        if args.mode == "files":
            fetch_files_mode(session, tokens, todo, out, args.batch)
        else:
            fetch_patches_mode(session, tokens, todo, out, args.max_patch)
    print(f"Finished in {(time.time() - t0)/60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
