"""
Sample a representative subset of a PI-Link JSON so the (slow) patches crawl
covers many projects instead of the first N issues in file order.

  python sample_pilink.py --data android_...issues.json --out sample.json \
      --n 1500 --seed 42

Then fetch diff hunks for just that subset:
  python fetch_pr_files.py --data sample.json --mode patches --out pr_patches.jsonl

The resulting pr_patches.jsonl is keyed by PR URL, so pilink_fix_model.py picks
up whatever patches exist via --patches; the semantic-diff metric then reports
on the covered fraction of the full test set.
"""
import argparse, json, random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="sample.json")
    ap.add_argument("--n", type=int, default=1500, help="number of issues to keep")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = json.load(open(args.data, encoding="utf-8"))
    by_repo = defaultdict(list)
    for it in raw:
        repo = (it.get("issue_repository") or {}).get("nameWithOwner", "?")
        by_repo[repo].append(it)

    rng = random.Random(args.seed)
    repos = list(by_repo)
    rng.shuffle(repos)
    # round-robin across repos so the sample spans as many projects as possible
    picked, pools = [], {r: iter(rng.sample(v, len(v))) for r, v in by_repo.items()}
    while len(picked) < min(args.n, len(raw)):
        progressed = False
        for r in repos:
            try:
                picked.append(next(pools[r]))
                progressed = True
                if len(picked) >= args.n:
                    break
            except StopIteration:
                continue
        if not progressed:
            break

    json.dump(picked, open(args.out, "w", encoding="utf-8"))
    nrepo = len({(it.get("issue_repository") or {}).get("nameWithOwner") for it in picked})
    print(f"Wrote {len(picked)} issues from {nrepo} projects to {args.out}")


if __name__ == "__main__":
    main()
