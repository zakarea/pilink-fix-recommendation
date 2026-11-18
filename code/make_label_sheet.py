"""
Build a human-validation sheet for the relevance/semantic-diff judges.

Samples N test issues from one project-wise split, and for the chosen model
writes the issue and its TOP-1 recommended fix side by side with the issue's
TRUE fix, plus blank columns for a human rater to mark whether the
recommendation addresses the same problem with the same kind of fix. Agreement
between these human labels and the automated metrics (FixSim, CJac, SemSim)
validates that the automated judges track real relevance.

  python make_label_sheet.py --data ...json --files ...jsonl \
      --model "Hybrid (TF-IDF + Mapping)" --n 100 --out label_sheet.csv

Rater fills: same_problem (1/0), same_fix_kind (1/0), notes.
"""
import argparse, csv, numpy as np
import pilink_fix_model as M


def clip(s, n=600):
    s = " ".join(str(s).split())
    return s[:n] + (" …" if len(s) > n else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--files", default=None)
    ap.add_argument("--android_only", action="store_true")
    ap.add_argument("--change_taxonomy", default="android", choices=["android", "neutral"])
    ap.add_argument("--model", default="Hybrid (TF-IDF + Mapping)")
    ap.add_argument("--models", default="tfidf,hybrid",
                    help="models to build (must include --model)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="label_sheet.csv")
    args = ap.parse_args()

    issues, prs = M.load(args.data, args.files, args.android_only, args.change_taxonomy)
    _, ctx = M.run(issues, prs, args.models.split(","), seed=args.seed)
    iss_te, prs_tr, rankings = ctx["iss_te"], ctx["prs_tr"], ctx["rankings"]
    if args.model not in rankings:
        raise SystemExit(f"--model {args.model!r} not in {list(rankings)}")
    ranks = np.asarray(rankings[args.model])
    osim = ctx["oracle_sim"]

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(iss_te), size=min(args.n, len(iss_te)), replace=False)
    iss_te = iss_te.reset_index(drop=True)

    cols = ["id", "oracle_sim", "issue_repo", "issue_text",
            "recommended_fix_repo", "recommended_fix_text",
            "true_fix_text", "same_problem", "same_fix_kind", "notes"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for rank_i, i in enumerate(sorted(idx)):
            top = int(ranks[i, 0])
            true_ids = iss_te.iloc[i]["pr_ids"]
            w.writerow([
                rank_i + 1,
                f"{osim[i]:.3f}",
                iss_te.iloc[i]["repo"],
                clip(iss_te.iloc[i]["text"]),
                prs_tr.iloc[top]["repo"],
                clip(prs_tr.iloc[top]["text"]),
                clip(" || ".join(prs.loc[true_ids, "text"])),
                "", "", "",
            ])
    print(f"Wrote {min(args.n, len(iss_te))} rows to {args.out} "
          f"(model={args.model}). Fill same_problem / same_fix_kind (1/0).")


if __name__ == "__main__":
    main()
