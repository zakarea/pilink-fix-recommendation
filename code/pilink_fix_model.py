"""
PI-Link Fix Recommender
=======================
Given a NEW issue from an Android project, recommend the solution (pull request)
of the most similar OLD issues in the PI-Link dataset.

Data: PI-Link JSON (Alshara et al., IEEE Access 2023)
      android_closed_issues_2011-01-01_2021-01-01_all_clean_issues.json

Evaluation setting: project-wise split -> test projects are never seen in training
(simulates "a new issue in a new Android project").

Because PI-Link stores commit URLs + messages (not file diffs), a recommended fix is
judged by how close its text (PR title + body + commit messages) is to the TRUE fix
of the test issue:
  - FixSim@k : best TF-IDF cosine between the true fix and the top-k recommended fixes
  - nFixSim@k: FixSim normalised between Random (0) and Oracle (1)
               Oracle = best fix that exists anywhere in the training pool
  - Hit@k / MRR: a recommendation is "relevant" if it is among the TOP_REL training
                 fixes most similar to the true fix (threshold-free, rank based)

Models
  random   : random ranking (lower bound)
  tfidf    : issue -> similar past issues (TF-IDF cosine), return their PRs
  bm25     : issue -> similar past issues (Okapi BM25), return their PRs
  lsa      : same but in a 300-d latent semantic space
  mapping  : TRAINED model - learns a linear map from issue space to fix space
             (ridge regression on issue/PR pairs), then retrieves the closest fixes
  hybrid   : z-score fusion of tfidf + mapping
  embed    : (needs internet for HuggingFace) pretrained sentence-transformer embeddings;
             set EMBED_MODEL to try a stronger or code-aware encoder, e.g.
             all-mpnet-base-v2, microsoft/unixcoder-base, microsoft/graphcodebert-base
  dual     : (needs a GPU) a sentence-transformer FINE-TUNED on this dataset's issue->PR
             pairs with a contrastive loss; set DUAL_BASE / DUAL_EPOCHS to configure.
             For a size ablation, run 'dual' several times with different DUAL_BASE, e.g.
             all-MiniLM-L6-v2, all-mpnet-base-v2, microsoft/unixcoder-base
  rerank   : (needs internet; GPU recommended) two-stage retrieve-then-rerank: TF-IDF
             shortlist of RERANK_K, re-scored by a cross-encoder (RERANK_MODEL)

Usage
  python pilink_fix_model.py --data android_closed_issues_...json
  python pilink_fix_model.py --data ... --models tfidf,mapping,hybrid,embed
  python pilink_fix_model.py --data ... --query "App crashes when rotating the screen"
"""

import argparse, json, os, re, sys, time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import normalize

TOP_K = [1, 5, 10, 20]
TOP_REL = 50           # relevant set size per test issue (top-50 closest fixes, ~0.1% of pool)
CHUNK = 1000


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def clean(t):
    if not t:
        return ""
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S)                # issue-template comments
    t = re.sub(r"```.*?```", " ", t, flags=re.S)                 # code blocks / logs
    t = re.sub(r"`[^`]*`", lambda m: m.group(0).strip("`"), t)   # keep inline code words
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"#\d+", " ", t)                                  # issue/PR references
    t = re.sub(r"[#*>_\-=|]{2,}", " ", t)                        # markdown noise
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------------------
# Changed files (from fetch_pr_files.py) -> change-type signature
#   'android' : the Android-specific taxonomy used in the main experiments.
#   'neutral' : a language-agnostic taxonomy (source/test/config/build/deps/...)
#               for cross-domain datasets (Python, JS, ...). Retrieval metrics
#               (Hit@k, MRR, FixSim) are taxonomy-independent; only CHit/CExact
#               depend on this choice, so the headline results are unaffected.
# ---------------------------------------------------------------------------
CHANGE_TYPES_ANDROID = [
    ("manifest",   r"androidmanifest\.xml$"),
    ("gradle",     r"(\.gradle|\.gradle\.kts|gradle\.properties|libs\.versions\.toml)$"),
    ("layout",     r"/layout[^/]*/[^/]*\.xml$"),
    ("values_res", r"/values[^/]*/[^/]*\.xml$"),
    ("drawable",   r"/(drawable|mipmap)[^/]*/"),
    ("activity",   r"activity[^/]*\.(java|kt)$"),
    ("fragment",   r"fragment[^/]*\.(java|kt)$"),
    ("service",    r"(service|receiver|worker)[^/]*\.(java|kt)$"),
    ("adapter",    r"adapter[^/]*\.(java|kt)$"),
    ("viewmodel",  r"(viewmodel|repository|dao|database)[^/]*\.(java|kt)$"),
    ("ui_compose", r"(screen|composable|view)[^/]*\.kt$"),
    ("test",       r"(/test/|/androidtest/|test[^/]*\.(java|kt)$)"),
    ("kotlin",     r"\.kt$"),
    ("java",       r"\.java$"),
    ("js_ts",      r"\.(js|jsx|ts|tsx)$"),
    ("dart",       r"\.dart$"),
    ("native",     r"\.(c|cc|cpp|h|hpp)$|cmakelists\.txt$|\.mk$"),
    ("ci_config",  r"(\.github/|\.travis\.yml|\.gitlab-ci|circleci)"),
    ("docs",       r"\.(md|txt|rst)$"),
    ("i18n",       r"/values-[a-z]{2}(-r[a-z]{2})?/strings\.xml$|\.po$|/locales?/"),
]

CHANGE_TYPES_NEUTRAL = [
    ("test",       r"(^|/)(tests?|__tests__|spec)/|(_test|test_|\.test\.|\.spec\.)"),
    ("docs",       r"\.(md|rst|txt|adoc)$|(^|/)docs?/"),
    ("ci",         r"(^|/)\.github/workflows/|\.travis\.yml$|\.gitlab-ci|/\.circleci/|azure-pipelines"),
    ("deps",       r"(requirements[^/]*\.txt|poetry\.lock|pipfile(\.lock)?|package-lock\.json|"
                   r"yarn\.lock|go\.(mod|sum)|cargo\.(toml|lock)|gemfile(\.lock)?|libs\.versions\.toml)$"),
    ("build",      r"(setup\.py|setup\.cfg|pyproject\.toml|package\.json|makefile|cmakelists\.txt|"
                   r"pom\.xml|build\.gradle(\.kts)?|\.gradle$|webpack|rollup|vite\.config|tox\.ini)"),
    ("config",     r"\.(ya?ml|toml|ini|cfg|conf|env|properties)$|(^|/)\.[a-z]+rc$|dockerfile"),
    ("ui_markup",  r"\.(html?|css|scss|less|vue|svelte|jsx|tsx)$"),
    ("i18n",       r"(^|/)(locales?|i18n|lang)/|\.(po|pot)$|/values-[a-z]{2}[^/]*/strings\.xml$"),
    ("data",       r"\.(csv|tsv|sql|parquet)$|(^|/)(fixtures|migrations)/"),
    ("source_py",  r"\.py$"),
    ("source_js",  r"\.(m|c)?jsx?$|\.tsx?$"),
    ("source_jvm", r"\.(java|kt|kts|scala|groovy)$"),
    ("source_sys", r"\.(c|cc|cpp|h|hpp|m|mm|swift|go|rs|rb|php|cs)$"),
]

_CHANGE_RE = {
    "android": [(n, re.compile(p)) for n, p in CHANGE_TYPES_ANDROID],
    "neutral": [(n, re.compile(p)) for n, p in CHANGE_TYPES_NEUTRAL],
}


def change_signature(paths, taxonomy="android"):
    rules = _CHANGE_RE[taxonomy]
    sig = set()
    for f in paths:
        f = "/" + f.lower()
        for name, rx in rules:
            if rx.search(f):
                sig.add(name)
    return frozenset(sig)


def load_files(path):
    """pull_url -> list of changed paths. Later 'ok' lines win (re-runs with --retry_errors)."""
    out = {}
    for line in open(path, encoding="utf-8"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("status") == "ok":
            out[rec["pull_url"]] = [f["path"] for f in rec.get("files") or []]
    print(f"Loaded changed files for {len(out)} PRs from {path}")
    return out


def load_patches(path, max_chars=4000):
    """pull_url -> a single diff string (concatenated per-file patches, capped).
    Only '+'/'-' content lines are kept so the signal is the actual code change,
    not surrounding context. Later 'ok' lines win."""
    out = {}
    for line in open(path, encoding="utf-8"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("status") != "ok":
            continue
        chunks = []
        for f in rec.get("files") or []:
            patch = f.get("patch")
            if not patch:
                continue
            kept = [ln for ln in patch.splitlines()
                    if ln[:1] in "+-" and not ln[:3] in ("+++", "---")]
            if kept:
                chunks.append(f.get("path", "") + "\n" + "\n".join(kept))
        diff = "\n".join(chunks).strip()
        if diff:
            out[rec["pull_url"]] = diff[:max_chars]
    print(f"Loaded diff hunks for {len(out)} PRs from {path}")
    return out


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def edges(obj, key):
    obj = obj or {}
    return [e.get("node", {}) for e in obj.get("edges", [])] if key == "edges" else obj.get("nodes", [])


# ---------------------------------------------------------------------------
# Load PI-Link JSON -> issues table + unique PRs table
# ---------------------------------------------------------------------------
def restrict_to_repos(issues, prs, kept_repos):
    """Filter issues+PRs to kept_repos and rebuild the positional cross-references."""
    ikeep = issues.index[issues["repo"].isin(kept_repos)]
    pkeep = prs.index[prs["repo"].isin(kept_repos)]
    inew = {o: n for n, o in enumerate(ikeep)}
    pnew = {o: n for n, o in enumerate(pkeep)}
    issues = issues.loc[ikeep].reset_index(drop=True)
    prs = prs.loc[pkeep].reset_index(drop=True)
    issues["pr_ids"] = [[pnew[p] for p in ids if p in pnew] for ids in issues["pr_ids"]]
    prs["issue_ids"] = [[inew[i] for i in ids if i in inew] for ids in prs["issue_ids"]]
    # drop issues that lost all their PRs (their fix is no longer in the pool)
    good = issues["pr_ids"].map(len) > 0
    if not good.all():
        issues2 = issues[good].reset_index(drop=True)
        inew2 = {o: n for n, o in enumerate(issues.index[good])}
        prs["issue_ids"] = [[inew2[i] for i in ids if i in inew2] for ids in prs["issue_ids"]]
        issues = issues2
    return issues, prs


def load(path, files_path=None, android_only=False, taxonomy="android", patches_path=None):
    raw = json.load(open(path, encoding="utf-8"))
    issues, prs, pr_index = [], [], {}
    for it in raw:
        nodes = (it.get("timelineItems") or {}).get("nodes", [])
        pr_ids = []
        for n in nodes:
            p = n.get("pull_info")
            if not p:
                continue
            url = p["pull_url"]
            if url not in pr_index:
                commits = [c.get("commit", {}).get("commit_message", "")
                           for c in (p.get("commits") or {}).get("nodes", [])]
                pr_index[url] = len(prs)
                prs.append(dict(url=url, title=p.get("pull_title") or "",
                                text=clean(" ".join([p.get("pull_title") or "",
                                                     p.get("pull_body") or ""] + commits[:30])),
                                issue_ids=[]))
            pr_ids.append(pr_index[url])
        if not pr_ids:
            continue
        iid = len(issues)
        for pid in pr_ids:
            prs[pid]["issue_ids"].append(iid)
        labels = [e.get("label", "").lower() for e in edges(it.get("issue_labels"), "edges")]
        issues.append(dict(
            repo=it["issue_repository"]["nameWithOwner"],
            url=it["issue_url"], title=it.get("issue_title") or "",
            text=clean((it.get("issue_title") or "") + ". " + (it.get("issue_title") or "") +
                       ". " + (it.get("issue_body") or "")),       # title counted twice
            labels=labels, pr_ids=pr_ids, created=it.get("created_at")))
    issues, prs = pd.DataFrame(issues), pd.DataFrame(prs)
    prs["repo"] = prs["issue_ids"].map(lambda ids: issues.at[ids[0], "repo"])
    print(f"Loaded {len(issues)} issues, {len(prs)} unique PRs, {issues.repo.nunique()} projects")
    if files_path:
        fmap = load_files(files_path)
        prs["has_files"] = prs["url"].isin(fmap)
        prs["files"] = prs["url"].map(lambda u: fmap.get(u, []))
        prs["sig"] = prs["files"].map(lambda f: change_signature(f, taxonomy))
        print(f"  PRs with file info: {prs.has_files.mean():.1%}")
        if android_only:
            if taxonomy != "android":
                sys.exit("--android_only requires --change_taxonomy android")
            strong = {"manifest", "gradle"}
            android_repos = {r for r, grp in prs.groupby("repo")
                             if any(strong & s for s in grp["sig"])}
            before = len(issues)
            issues, prs = restrict_to_repos(issues, prs, android_repos)
            print(f"  [android_only] kept {len(issues)}/{before} issues, "
                  f"{issues.repo.nunique()} real-Android projects, {len(prs)} PRs")
    elif android_only:
        sys.exit("--android_only needs --files")
    if patches_path:
        pmap = load_patches(patches_path)
        prs["patch_text"] = prs["url"].map(lambda u: pmap.get(u, ""))
        prs["has_patch"] = prs["patch_text"].str.len() > 0
        print(f"  PRs with diff hunks: {prs.has_patch.mean():.1%}")
    return issues, prs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tfidf(**kw):
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.3, sublinear_tf=True,
                           stop_words="english", max_features=300_000,
                           token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]{1,}\b", **kw)


def chunked_scores(Q, M, fn):
    """Yield (start, dense score block) for Q x M^T in chunks."""
    for s in range(0, Q.shape[0], CHUNK):
        block = Q[s:s + CHUNK] @ M.T
        block = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
        yield s, fn(block)


def pr_side_issue_text(prs_tr, issues):
    """For issue->issue retrieval: each candidate PR is represented by its linked issues' text."""
    return [" ".join(issues.loc[ids, "text"]) for ids in prs_tr["issue_ids"]]


# ---------------------------------------------------------------------------
# Models: each returns a function score(query_texts) -> (n_query x n_train_PR) matrix
# ---------------------------------------------------------------------------
def build_models(names, iss_tr, prs_tr, issues, seed):
    cand_issue_text = pr_side_issue_text(prs_tr, issues)
    models = {}

    if "random" in names:
        rng = np.random.default_rng(seed)
        models["Random"] = lambda Q: rng.random((len(Q), len(prs_tr)))

    need_issue_vec = any(n in names for n in ("tfidf", "lsa", "mapping", "hybrid"))
    if need_issue_vec:
        vi = tfidf().fit(list(iss_tr["text"]) + cand_issue_text)
        C = vi.transform(cand_issue_text)

    if "tfidf" in names or "hybrid" in names:
        tf_fn = lambda Q: (vi.transform(Q) @ C.T).toarray()
        if "tfidf" in names:
            models["TF-IDF (issue->issue)"] = tf_fn

    if "bm25" in names:
        # Okapi BM25, vectorised: the document-side weight of a term is independent
        # of the query, so scores = (query term-presence) @ (doc BM25 weights)^T.
        k1, b = 1.5, 0.75
        cv = CountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.3, stop_words="english",
                             token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]{1,}\b")
        Xc = cv.fit_transform(cand_issue_text)                 # candidates x terms (counts)
        dl = np.asarray(Xc.sum(1)).ravel(); avgdl = dl.mean() or 1.0
        df = np.asarray((Xc > 0).sum(0)).ravel()
        idf = np.log(1 + (Xc.shape[0] - df + 0.5) / (df + 0.5))
        coo = Xc.tocoo()
        denom = coo.data + k1 * (1 - b + b * dl[coo.row] / avgdl)
        w = idf[coo.col] * (coo.data * (k1 + 1)) / denom
        W = csr_matrix((w, (coo.row, coo.col)), shape=Xc.shape)  # candidate BM25 weights
        def bm25_fn(Q):
            q = cv.transform(Q); q.data[:] = 1.0                 # term presence
            return (q @ W.T).toarray()
        models["BM25 (issue->issue)"] = bm25_fn

    if "lsa" in names or "mapping" in names or "hybrid" in names:
        svd_i = TruncatedSVD(300, random_state=seed).fit(vi.transform(iss_tr["text"]))
        Ci = normalize(svd_i.transform(C))
        if "lsa" in names:
            models["LSA (issue->issue)"] = lambda Q: normalize(svd_i.transform(vi.transform(Q))) @ Ci.T

    if "mapping" in names or "hybrid" in names:
        # Learn issue-space -> fix-space mapping on training issue/PR pairs
        vp = tfidf().fit(prs_tr["text"])
        P = vp.transform(prs_tr["text"])
        svd_p = TruncatedSVD(300, random_state=seed).fit(P)
        Pz = normalize(svd_p.transform(P))
        pos = {pid: k for k, pid in enumerate(prs_tr.index)}
        X, Y = [], []
        Xi = normalize(svd_i.transform(vi.transform(iss_tr["text"])))
        for row, ids in enumerate(iss_tr["pr_ids"]):
            for pid in ids:
                if pid in pos:
                    X.append(Xi[row]); Y.append(Pz[pos[pid]])
        X, Y = np.array(X), np.array(Y)
        frac = float(os.environ.get("TRAIN_FRAC", "1.0"))   # data-efficiency sweep
        if frac < 1.0 and len(X) > 10:
            idx = np.random.default_rng(seed).choice(len(X), max(2, int(frac * len(X))), replace=False)
            X, Y = X[idx], Y[idx]
        reg = Ridge(alpha=1.0).fit(X, Y)
        map_fn = lambda Q: normalize(reg.predict(normalize(svd_i.transform(vi.transform(Q))))) @ Pz.T
        if "mapping" in names:
            models["Mapping (trained issue->fix)"] = map_fn

    if "hybrid" in names:
        def z(S):
            return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
        models["Hybrid (TF-IDF + Mapping)"] = lambda Q: z(tf_fn(Q)) + z(map_fn(Q))

    if "embed" in names:
        try:
            from sentence_transformers import SentenceTransformer
            # EMBED_MODEL can be a general encoder (all-MiniLM-L6-v2, all-mpnet-base-v2)
            # or a code-aware one (microsoft/unixcoder-base, microsoft/graphcodebert-base,
            # Salesforce/codet5p-110m-embedding). Code models may need trust_remote_code,
            # and raw HF models get a mean-pooling head from sentence-transformers.
            name = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            try:
                st = SentenceTransformer(name, trust_remote_code=True)
            except TypeError:
                st = SentenceTransformer(name)
            label = f"Embeddings ({name.split('/')[-1]})"
            E = st.encode(cand_issue_text, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
            models[label] = lambda Q: st.encode(list(Q), batch_size=128,
                                                 normalize_embeddings=True) @ E.T
        except Exception as e:
            print(f"[skip embed] {e}")

    if "twotower" in names:
        # Learned two-tower retriever WITHOUT a pretrained language model: two small
        # MLP towers over TF-IDF/LSA features, trained contrastively (InfoNCE, in-batch
        # negatives) on issue->PR pairs. CPU-only; isolates the effect of the contrastive
        # objective from the pretrained-LM knowledge the 'dual' model also brings.
        try:
            import torch
            import torch.nn as nn
            torch.manual_seed(seed)
            epochs = int(os.environ.get("TWOTOWER_EPOCHS", "3"))
            vt = tfidf().fit(list(iss_tr["text"]) + list(prs_tr["text"]))
            allX = vt.transform(list(iss_tr["text"]) + list(prs_tr["text"]))
            svd = TruncatedSVD(256, random_state=seed).fit(allX)
            def feat(texts):
                return normalize(svd.transform(vt.transform(list(texts)))).astype("float32")
            pid2local = {pid: k for k, pid in enumerate(prs_tr.index)}
            It = torch.tensor(feat(iss_tr["text"]))
            Pt = torch.tensor(feat(prs_tr["text"]))     # candidates, in prs_tr order
            pos_i, pos_p = [], []
            for row, ids in enumerate(iss_tr["pr_ids"]):
                for pid in ids:
                    if pid in pid2local:
                        pos_i.append(row); pos_p.append(pid2local[pid])
            pi = torch.tensor(pos_i); pp = torch.tensor(pos_p)

            class Tower(nn.Module):
                def __init__(self, d=256, h=256):
                    super().__init__()
                    self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h))
                def forward(self, x):
                    return nn.functional.normalize(self.net(x), dim=1)

            qt, pt = Tower(), Tower()
            opt = torch.optim.Adam(list(qt.parameters()) + list(pt.parameters()), lr=1e-3)
            temp = 0.05
            print(f"  [twotower] training on {len(pi)} issue->PR pairs, {epochs} epochs (CPU)")
            for ep in range(epochs):
                perm = torch.randperm(len(pi))
                for s in range(0, len(pi), 256):
                    idx = perm[s:s + 256]
                    if len(idx) < 2:
                        continue
                    a = qt(It[pi[idx]]); bb = pt(Pt[pp[idx]])
                    logits = a @ bb.T / temp
                    loss = nn.functional.cross_entropy(logits, torch.arange(len(idx)))
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                Ec = qt and pt(Pt).numpy()
            def tt_fn(Q):
                with torch.no_grad():
                    q = qt(torch.tensor(feat(Q))).numpy()
                return q @ Ec.T
            models["Two-tower (contrastive)"] = tt_fn
        except Exception as e:
            print(f"[skip twotower] {e}")

    if "dual" in names:
        # Fine-tuned dual-encoder: learn issue<->fix matching directly with a
        # contrastive objective (each issue and its linked PR is a positive pair;
        # other PRs in the batch are negatives). Needs a GPU; run in Colab.
        try:
            from sentence_transformers import SentenceTransformer, InputExample, losses
            from torch.utils.data import DataLoader
            base = os.environ.get("DUAL_BASE", "sentence-transformers/all-MiniLM-L6-v2")
            epochs = int(os.environ.get("DUAL_EPOCHS", "1"))
            model = SentenceTransformer(base)
            pid2text = dict(zip(prs_tr.index, prs_tr["text"]))   # training PRs only
            examples = []
            for row, ids in enumerate(iss_tr["pr_ids"]):
                q = iss_tr.iloc[row]["text"]
                for pid in ids:                                 # pid indexes global prs
                    ptxt = pid2text.get(pid)
                    if q and ptxt:
                        examples.append(InputExample(texts=[q, ptxt]))
            frac = float(os.environ.get("TRAIN_FRAC", "1.0"))   # data-efficiency sweep
            if frac < 1.0 and len(examples) > 10:
                rng = np.random.default_rng(seed)
                keep = rng.choice(len(examples), max(2, int(frac * len(examples))), replace=False)
                examples = [examples[i] for i in keep]
            print(f"  [dual] training on {len(examples)} issue->PR pairs, base={base}, epochs={epochs}")
            loader = DataLoader(examples, shuffle=True, batch_size=64, drop_last=True)
            loss = losses.MultipleNegativesRankingLoss(model)
            model.fit(train_objectives=[(loader, loss)], epochs=epochs,
                      warmup_steps=min(1000, len(loader) // 10), show_progress_bar=True)
            # candidates are the fix texts themselves (model was trained issue<->PR)
            Ef = model.encode(list(prs_tr["text"]), batch_size=128,
                              normalize_embeddings=True, show_progress_bar=True)
            models["Dual-encoder (fine-tuned)"] = lambda Q: model.encode(
                list(Q), batch_size=128, normalize_embeddings=True) @ Ef.T
        except Exception as e:
            print(f"[skip dual] {e}")

    if "rerank" in names:
        # Two-stage retrieve-then-rerank: a first-stage retriever proposes a shortlist
        # of RERANK_K candidates per issue, then a CROSS-ENCODER re-scores each
        # (issue, fix) pair jointly and reorders them. First stage is TF-IDF by default
        # (cheap, no GPU); the cross-encoder is a pretrained MS-MARCO re-ranker unless
        # RERANK_MODEL is set. Lifts precision at low k (Hit@1, MRR).
        try:
            import numpy as _np
            from sentence_transformers import CrossEncoder
            K = int(os.environ.get("RERANK_K", "50"))
            ce_name = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            # first stage: TF-IDF over candidate issue text
            vi2 = tfidf().fit(list(iss_tr["text"]) + cand_issue_text)
            C2 = vi2.transform(cand_issue_text)
            first_fn = lambda Q: (vi2.transform(Q) @ C2.T).toarray()
            cand_pr_text = list(prs_tr["text"])         # cross-encoder reads the fix text
            ce = CrossEncoder(ce_name, max_length=256)
            print(f"  [rerank] first-stage=TF-IDF, cross-encoder={ce_name}, K={K}")

            def rr_fn(Q):
                Q = list(Q)
                base = first_fn(Q)
                n = min(K, base.shape[1])
                topN = _np.argpartition(-base, n - 1, axis=1)[:, :n]
                pairs, loc = [], []
                for i in range(len(Q)):
                    for c in topN[i]:
                        pairs.append([Q[i], cand_pr_text[c]]); loc.append((i, c))
                cs = ce.predict(pairs, batch_size=256, show_progress_bar=False)
                out = _np.full(base.shape, -1e9, dtype="float32")
                for (i, c), s in zip(loc, cs):
                    out[i, c] = s
                return out
            models[f"Rerank (TF-IDF + {ce_name.split('/')[-1]})"] = rr_fn
        except Exception as e:
            print(f"[skip rerank] {e}")
    return models


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def true_fix_text(issues_te, prs):
    return [" ".join(prs.loc[ids, "text"]) for ids in issues_te["pr_ids"]]


def true_fix_diff(issues_te, prs):
    """Concatenated diff hunks of the PR(s) that actually closed each test issue."""
    return ["\n".join(prs.loc[ids, "patch_text"]).strip() for ids in issues_te["pr_ids"]]


def embed_code(texts, batch=32):
    """Mean-pooled, L2-normalised code embeddings from a HF encoder (default
    GraphCodeBERT). Lazy-imports torch/transformers; needs a GPU to be practical,
    so this path runs on Colab, not the local sandbox. Model picked via
    SEMDIFF_MODEL. Returns (n, d) float32."""
    import numpy as _np, torch
    from transformers import AutoTokenizer, AutoModel
    name = os.environ.get("SEMDIFF_MODEL", "microsoft/graphcodebert-base")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name).to(dev).eval()
    vecs = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            chunk = [t if t else " " for t in texts[s:s + batch]]
            enc = tok(chunk, padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to(dev)
            out = mdl(**enc).last_hidden_state                       # (b, L, d)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean pool
            emb = torch.nn.functional.normalize(emb, dim=1)
            vecs.append(emb.cpu().numpy().astype("float32"))
    return _np.vstack(vecs)


def run(issues, prs, model_names, seed=42, test_size=0.2, max_test=None,
        split=os.environ.get("SPLIT", "project")):
    if split == "temporal":
        # train on the earliest issues, test on the most recent (future-fix recovery)
        order = np.argsort(issues["created"].fillna("").to_numpy(), kind="stable")
        cut = int(len(order) * (1 - test_size))
        tr, te = order[:cut], order[cut:]
        iss_tr, iss_te = issues.iloc[tr], issues.iloc[te]
        keep_pr = set()                               # candidate pool = PRs of training issues
        for ids in iss_tr["pr_ids"]:
            keep_pr.update(ids)
        prs_tr = prs.iloc[sorted(keep_pr)]
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(issues, groups=issues["repo"]))
        iss_tr, iss_te = issues.iloc[tr], issues.iloc[te]
        train_repos = set(iss_tr["repo"])
        prs_tr = prs[prs["repo"].isin(train_repos)]
    if max_test and len(iss_te) > max_test:
        iss_te = iss_te.sample(max_test, random_state=seed)
    print(f"[seed {seed}, {split}] train: {len(iss_tr)} issues / {len(prs_tr)} PRs"
          f" | test: {len(iss_te)} issues / {iss_te.repo.nunique()} projects")

    # Fix-similarity space (judge), fitted on training PR texts only
    judge = tfidf().fit(prs_tr["text"])
    J_pool = judge.transform(prs_tr["text"])
    J_true = judge.transform(true_fix_text(iss_te, prs))

    # Oracle: relevant sets + best achievable similarity
    rel_sets, oracle_best = [], {k: [] for k in TOP_K}
    oracle_sim = []                       # per-query best pool-fix similarity (model-independent)
    for s, block in chunked_scores(J_true, J_pool, lambda b: b):
        top = np.argpartition(-block, TOP_REL, axis=1)[:, :TOP_REL]
        for r in range(block.shape[0]):
            rel_sets.append(set(top[r]))
            srt = np.sort(block[r])[::-1]
            oracle_sim.append(float(srt[0]))
            for k in TOP_K:
                oracle_best[k].append(srt[0])

    def fixsim(rank_idx):
        """Best judge similarity between true fix and top-k retrieved fixes."""
        out = {k: [] for k in TOP_K}
        for s in range(0, len(rank_idx), CHUNK):
            blk = (J_true[s:s + CHUNK] @ J_pool.T)
            for r in range(blk.shape[0]):
                row = blk.getrow(r).toarray().ravel()
                vals = row[rank_idx[s + r]]
                for k in TOP_K:
                    out[k].append(vals[:k].max())
        return out

    # File-based ground truth (only when changed files were fetched)
    use_files = "sig" in prs.columns
    if use_files:
        cand_sig = list(prs_tr["sig"])
        true_sig = [frozenset().union(*[prs.at[p, "sig"] for p in ids if prs.at[p, "has_files"]])
                    if any(prs.at[p, "has_files"] for p in ids) else None
                    for ids in iss_te["pr_ids"]]
        eval_rows = [i for i, s in enumerate(true_sig) if s]
        print(f"  file-based metrics on {len(eval_rows)}/{len(iss_te)} test issues "
              f"(true fix has non-empty change signature)")

    # Semantic-diff ground truth (only with --patches and SEMDIFF=1; GraphCodeBERT, Colab)
    use_sem = ("patch_text" in prs.columns) and os.environ.get("SEMDIFF", "0") == "1"
    if use_sem:
        sem_tau = float(os.environ.get("SEM_TAU", "0.25"))   # threshold in CENTERED cosine space
        cand_ok = prs_tr["has_patch"].to_numpy()
        tdiff = true_fix_diff(iss_te, prs)
        sem_rows = [i for i, d in enumerate(tdiff) if d]
        print(f"  semantic-diff metrics on {len(sem_rows)}/{len(iss_te)} test issues "
              f"(true fix has a diff), {int(cand_ok.sum())}/{len(prs_tr)} candidates have diffs")
        if sem_rows and cand_ok.any():
            import hashlib
            cache_dir = os.environ.get("SEM_CACHE")      # set to a Drive folder to reuse embeddings

            def embed_cached(texts, tag):
                if not cache_dir:
                    return embed_code(texts)
                os.makedirs(cache_dir, exist_ok=True)
                key = hashlib.md5("||".join(texts).encode("utf-8", "ignore")).hexdigest()[:16]
                fp = os.path.join(cache_dir, f"sememb_{tag}_{len(texts)}_{key}.npy")
                if os.path.exists(fp):
                    print(f"  [sem cache] load {tag} ({len(texts)})")
                    return np.load(fp)
                E = embed_code(texts)
                np.save(fp, E)
                print(f"  [sem cache] saved {tag} ({len(texts)})")
                return E

            cand_pos = np.where(cand_ok)[0]
            Ec = embed_cached(list(prs_tr["patch_text"].iloc[cand_pos]), "cand")
            Et = embed_cached([tdiff[i] for i in sem_rows], "true")
            # De-anisotropy: raw transformer embeddings sit in a narrow cone, so every
            # diff-pair has cosine ~0.8 and the judge cannot discriminate. Subtract the
            # common direction (mean of the candidate-fix embeddings) and re-normalise,
            # so cosine reflects *relative* code similarity, not the shared offset.
            mu = Ec.mean(axis=0, keepdims=True)

            def _center(X):
                Y = X - mu
                Y /= np.clip(np.linalg.norm(Y, axis=1, keepdims=True), 1e-9, None)
                return Y.astype("float32")

            Ec, Et = _center(Ec), _center(Et)
            E_cand = np.zeros((len(prs_tr), Ec.shape[1]), dtype="float32")
            E_cand[cand_pos] = Ec
            sem_true = {i: Et[j] for j, i in enumerate(sem_rows)}
            # Per-issue baselines, independent of any ranker:
            #   sem_oracle[i] = closest patched pool fix to the true fix (ceiling)
            #   sem_rand[i]   = a random patched pool fix (floor), mean over draws
            Ep = E_cand[cand_pos]                       # (n_patched, d)
            rng_sem = np.random.default_rng(seed)
            sem_oracle, sem_rand = {}, {}
            for j, i in enumerate(sem_rows):
                sims = Ep @ Et[j]
                sem_oracle[i] = float(sims.max())
                sem_rand[i] = float(sims[rng_sem.integers(0, len(sims), size=min(20, len(sims)))].mean())
            sem_oracle_mean = float(np.mean(list(sem_oracle.values())))
            sem_rand_mean = float(np.mean(list(sem_rand.values())))
            print(f"  semantic-diff baselines: random={sem_rand_mean:.3f}  "
                  f"oracle(ceiling)={sem_oracle_mean:.3f}")
        else:
            use_sem = False

    models = build_models(model_names, iss_tr, prs_tr, issues, seed)
    Q = list(iss_te["text"])
    kmax = max(TOP_K)
    results, rankings = [], {}
    for name, fn in models.items():
        t0 = time.time()
        ranks = []
        for s in range(0, len(Q), CHUNK):
            S = fn(Q[s:s + CHUNK])
            ranks.append(np.argsort(-S, axis=1)[:, :kmax])
        ranks = np.vstack(ranks)
        rankings[name] = ranks
        fs = fixsim(ranks)
        row = {"model": name}
        for k in TOP_K:
            hits = [len(set(ranks[i, :k]) & rel_sets[i]) > 0 for i in range(len(ranks))]
            row[f"Hit@{k}"] = np.mean(hits)
        rr = []
        for i in range(len(ranks)):
            pos = next((p + 1 for p, c in enumerate(ranks[i]) if c in rel_sets[i]), None)
            rr.append(1 / pos if pos else 0)
        row["MRR@20"] = np.mean(rr)
        for k in (1, 5, 10):
            row[f"FixSim@{k}"] = np.mean(fs[k])
        row["_fs"] = fs
        if use_files and eval_rows:
            # CHit@k : a top-k fix changes the same kinds of Android code (Jaccard >= 0.5)
            # CExact@k: a top-k fix has exactly the same change-type signature
            # CJac@1 : mean Jaccard of the top-1 fix
            for k in (1, 5, 10):
                jac = [[jaccard(true_sig[i], cand_sig[c]) for c in ranks[i, :k]] for i in eval_rows]
                row[f"CHit@{k}"] = np.mean([max(j) >= 0.5 for j in jac])
                row[f"CExact@{k}"] = np.mean([any(cand_sig[c] == true_sig[i] for c in ranks[i, :k])
                                              for i in eval_rows])
                if k == 1:
                    row["CJac@1"] = np.mean([j[0] for j in jac])
        if use_sem:
            # SemSim@k : best GraphCodeBERT cosine between the true fix's diff and
            #            the top-k recommended diffs that have patch content.
            # SemHit@k : that cosine >= sem_tau (same semantic change).
            # nSemSim@k: SemSim normalised between the random-patched floor and the
            #            oracle (closest patched pool fix) ceiling -> coverage-robust.
            # SemCov@k : fraction of covered issues with a patched candidate in top-k.
            # Averaged ONLY over issues that actually have a patched candidate in the
            # top-k window (no zero-fill), so the score is not diluted by coverage.
            sem = {k: [] for k in (1, 5, 10)}
            for i in sem_rows:
                t = sem_true[i]
                for k in (1, 5, 10):
                    cands = [c for c in ranks[i, :k] if cand_ok[c]]
                    if cands:
                        sem[k].append(float(max(E_cand[c] @ t for c in cands)))
            span = max(sem_oracle_mean - sem_rand_mean, 1e-9)
            for k in (1, 5, 10):
                row[f"SemCov@{k}"] = len(sem[k]) / len(sem_rows)
                if sem[k]:
                    m = float(np.mean(sem[k]))
                    row[f"SemSim@{k}"] = m
                    row[f"nSemSim@{k}"] = (m - sem_rand_mean) / span
                    row[f"SemHit@{k}"] = float(np.mean([v >= sem_tau for v in sem[k]]))
                else:
                    row[f"SemSim@{k}"] = row[f"nSemSim@{k}"] = row[f"SemHit@{k}"] = float("nan")
        row["time_s"] = round(time.time() - t0, 1)
        results.append(row)
        print(f"  {name:32s} done in {row['time_s']}s")

    # Normalise FixSim between Random and Oracle
    rnd = next((r for r in results if r["model"] == "Random"), None)
    if rnd:
        bounds = {k: (np.mean(rnd["_fs"][k]), np.mean(oracle_best[k])) for k in (1, 5, 10)}
        for r in results:
            for k, (lo, hi) in bounds.items():
                r[f"nFixSim@{k}"] = (r[f"FixSim@{k}"] - lo) / (hi - lo)
    for r in results:
        r.pop("_fs", None)
    oracle = {"model": "Oracle (upper bound)", "FixSim@1": np.mean(oracle_best[1])}
    table = pd.DataFrame(results + [oracle]).set_index("model")
    return table, dict(iss_tr=iss_tr, iss_te=iss_te, prs_tr=prs_tr, rankings=rankings,
                       rel=rel_sets, oracle_sim=np.asarray(oracle_sim))


def show_examples(ctx, issues, prs, model, n=3, seed=1):
    iss_te, prs_tr, ranks = ctx["iss_te"], ctx["prs_tr"], ctx["rankings"][model]
    rng = np.random.default_rng(seed)
    for i in rng.choice(len(iss_te), n, replace=False):
        it = iss_te.iloc[i]
        print("\n" + "=" * 90)
        print(f"NEW ISSUE  [{it.repo}] {it.title}")
        print(f"TRUE FIX   {prs.loc[it.pr_ids[0], 'title']}")
        for r, c in enumerate(ranks[i, :3], 1):
            p = prs_tr.iloc[c]
            src = issues.loc[p.issue_ids[0]]
            mark = "*" if c in ctx["rel"][i] else " "
            print(f" {mark}#{r} similar issue: {src.title[:70]}")
            print(f"      its fix:      {p.title[:70]}   ({p.url})")
            if "files" in p and p.files:
                print(f"      changes:      {sorted(p.sig)}  e.g. {p.files[:3]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--files", default=None, help="pr_files.jsonl from fetch_pr_files.py")
    ap.add_argument("--patches", default=None,
                    help="pr_patches.jsonl (fetch_pr_files.py --mode patches); enables "
                         "the semantic-diff metric when SEMDIFF=1 (GraphCodeBERT, GPU/Colab)")
    ap.add_argument("--change_taxonomy", choices=["android", "neutral"], default="android",
                    help="code-change taxonomy for CHit/CExact: 'android' (default) or "
                         "'neutral' for cross-domain datasets (Python, JS, ...)")
    ap.add_argument("--android_only", action="store_true",
                    help="keep only issues whose repo has a real Android signal "
                         "(a PR touching AndroidManifest.xml or a Gradle file)")
    ap.add_argument("--models", default="random,tfidf,lsa,mapping,hybrid")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--max_test", type=int, default=None)
    ap.add_argument("--examples", type=int, default=0)
    ap.add_argument("--query", default=None)
    args = ap.parse_args()

    issues, prs = load(args.data, args.files, args.android_only, args.change_taxonomy,
                       patches_path=args.patches)
    names = args.models.split(",")
    tables = []
    for seed in map(int, args.seeds.split(",")):
        t, ctx = run(issues, prs, names, seed=seed, max_test=args.max_test)
        tables.append(t)
    full = pd.concat(tables).groupby(level=0, sort=False)
    mean, std = full.mean(), full.std()
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    print("\n=== RESULTS (project-wise split; mean over seeds) ===")
    print(mean.drop(columns=["time_s"], errors="ignore").to_string())
    if len(tables) > 1:
        print("\n(std over seeds)")
        print(std.drop(columns=["time_s"], errors="ignore").to_string())
    mean.to_csv("results_mean.csv"); std.to_csv("results_std.csv")

    if args.examples:
        best = [m for m in ctx["rankings"] if m != "Random"][-1]
        show_examples(ctx, issues, prs, best, n=args.examples)

    if args.query:
        all_prs = prs
        models = build_models(["hybrid"], issues, all_prs, issues, 42)
        S = models["Hybrid (TF-IDF + Mapping)"]([clean(args.query + ". " + args.query)])[0]
        print(f"\nNEW ISSUE: {args.query}")
        for r, c in enumerate(np.argsort(-S)[:5], 1):
            p = all_prs.iloc[c]
            print(f"#{r} similar issue: {issues.loc[p.issue_ids[0], 'title'][:70]}")
            print(f"    suggested fix: {p.title[:70]}  ({p.url})")


if __name__ == "__main__":
    main()
