"""
Coverage vs. conditional precision for cross-project fix recommendation.

A new issue is "answerable" only if the pool actually contains a fix that is
textually close to its true fix. We measure that with a MODEL-INDEPENDENT
Oracle signal: the similarity between the issue's own ground-truth fix text and
its nearest neighbour among the training-pool fixes (computed once, from the
data, before any ranker runs). Thresholding that Oracle similarity at tau
splits the test issues into:
  - answerable  (Oracle sim >= tau): a good precedent exists in the pool
  - genuinely new (Oracle sim <  tau): no close precedent -> no system could
    retrieve one, so these cap every model's ceiling.

We report, for each tau:
  - coverage        = fraction of test issues that are answerable
  - Hit@k / MRR on the full test set
  - Hit@k / MRR on the answerable subset (conditional performance)

The criterion never looks at any ranker's output, so the subset is not chosen
to flatter a model (no circularity / survivorship bias): the SAME issues are
answerable for every model.

Usage mirrors pilink_fix_model.py:
  python answerable.py --data data/...json --files data/...jsonl \
      --models tfidf,bm25,hybrid,dual --android_only
"""
import argparse, numpy as np
import pilink_fix_model as M


def hit_mrr(ranks, rel, mask, kset=(1, 5, 10, 20)):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return {k: float("nan") for k in kset}, float("nan"), 0
    hits = {k: 0 for k in kset}
    rr = 0.0
    for i in idx:
        r = ranks[i]
        pos = next((p + 1 for p, c in enumerate(r) if c in rel[i]), None)
        rr += 1.0 / pos if pos else 0.0
        for k in kset:
            if pos and pos <= k:
                hits[k] += 1
    n = len(idx)
    return {k: hits[k] / n for k in kset}, rr / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--files", default=None)
    ap.add_argument("--android_only", action="store_true")
    ap.add_argument("--change_taxonomy", default="android", choices=["android", "neutral"])
    ap.add_argument("--models", default="tfidf,bm25,hybrid,dual")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quantiles", default="0.0,0.25,0.5,0.75",
                    help="fraction of issues to DROP as 'genuinely new' (tau = that quantile of Oracle sim)")
    args = ap.parse_args()

    issues, prs = M.load(args.data, args.files, args.android_only, args.change_taxonomy)
    _, ctx = M.run(issues, prs, args.models.split(","), seed=args.seed)
    rankings, rel = ctx["rankings"], ctx["rel"]
    osim = ctx["oracle_sim"]
    nq = len(rel)

    print(f"\nTest issues: {nq}")
    print(f"Oracle best-fix similarity  mean={osim.mean():.3f}  median={np.median(osim):.3f}  "
          f"p10={np.quantile(osim,0.1):.3f}  p90={np.quantile(osim,0.9):.3f}")

    qs = [float(x) for x in args.quantiles.split(",")]
    # tau at quantile q of the Oracle distribution -> drop the lowest-q share as "genuinely new"
    taus = [(q, float(np.quantile(osim, q))) for q in qs]

    kset = (1, 5, 10, 20)
    for q, tau in taus:
        mask = osim >= tau
        cov = mask.mean()
        tag = "FULL set" if q == 0.0 else f"answerable (drop lowest {int(q*100)}%)"
        print(f"\n=== {tag}   tau={tau:.3f}   coverage={cov:.3f}  (n={int(mask.sum())}/{nq}) ===")
        print(f"{'model':30s} " + " ".join(f"H@{k:<4d}" for k in kset) + "   MRR")
        for name in rankings:
            h, mrr, _ = hit_mrr(np.asarray(rankings[name]), rel, mask, kset)
            print(f"{name:30s} " + " ".join(f"{h[k]:.3f}" for k in kset) + f"   {mrr:.3f}")


if __name__ == "__main__":
    main()
