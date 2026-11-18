"""
Statistical significance of ranker differences.
For one project-wise split, compute each model's per-query reciprocal rank, then
run paired Wilcoxon signed-rank tests of the Hybrid against every other model,
with Cliff's delta effect size. Usage mirrors pilink_fix_model.py.
"""
import argparse, numpy as np
from scipy.stats import wilcoxon
import pilink_fix_model as M


def per_query_rr(ranks, rel):
    rr = np.zeros(len(ranks))
    for i in range(len(ranks)):
        pos = next((p + 1 for p, c in enumerate(ranks[i]) if c in rel[i]), None)
        rr[i] = 1.0 / pos if pos else 0.0
    return rr


def cliffs_delta(a, b):
    # P(a>b) - P(a<b), estimated on a subsample for speed
    n = min(len(a), 4000)
    ia = np.random.default_rng(0).choice(len(a), n, replace=False)
    aa, bb = a[ia], b[ia]
    gt = np.mean(aa[:, None] > bb[None, :])
    lt = np.mean(aa[:, None] < bb[None, :])
    return gt - lt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--files", default=None)
    ap.add_argument("--android_only", action="store_true")
    ap.add_argument("--change_taxonomy", default="android", choices=["android", "neutral"])
    ap.add_argument("--models", default="random,bm25,lsa,tfidf,mapping,hybrid")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ref", default="Hybrid (TF-IDF + Mapping)")
    args = ap.parse_args()

    issues, prs = M.load(args.data, args.files, args.android_only, args.change_taxonomy)
    _, ctx = M.run(issues, prs, args.models.split(","), seed=args.seed)
    rankings, rel = ctx["rankings"], ctx["rel"]
    rr = {name: per_query_rr(r, rel) for name, r in rankings.items()}

    ref = args.ref if args.ref in rr else max(rr, key=lambda n: rr[n].mean())
    print(f"\nReference model: {ref}  (MRR={rr[ref].mean():.3f}, n={len(rel)} queries)")
    print(f"{'model':32s} {'MRR':>7} {'vs ref p':>12} {'Cliff d':>9}  sig")
    for name in sorted(rr, key=lambda n: rr[n].mean()):
        if name == ref:
            print(f"{name:32s} {rr[name].mean():7.3f} {'(reference)':>12}")
            continue
        try:
            stat, p = wilcoxon(rr[ref], rr[name], zero_method="wilcox")
        except ValueError:
            p = 1.0
        d = cliffs_delta(rr[ref], rr[name])
        mark = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
        print(f"{name:32s} {rr[name].mean():7.3f} {p:12.2e} {d:9.3f}  {mark}")


if __name__ == "__main__":
    main()
