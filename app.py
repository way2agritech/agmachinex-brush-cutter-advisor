"""
Brush Cutter Advisor — Streamlit prototype
------------------------------------------
Everything (questions, answer options, features, ideal / acceptable values, weights)
is read from the two data files. Nothing about specific features is hardcoded.

Run:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Brush Cutter Advisor", page_icon="🌾", layout="wide")

BASE = Path(__file__).parent
MACHINE_FILES = ["machines_csv.xlsx", "machines.csv"]
RULE_FILES = ["rules_ideal_acceptable_csv.xlsx", "rules_ideal_acceptable.csv"]

# ---- Optional display polish only (fallbacks are derived from the data itself) ----
QUESTION_TEXT = {
    "land_extent": ("📐 How big is your land?", "Approximate area to be cleared (acres)."),
    "soil_feature": ("⛰️ What is the ground like?", "Pick the one that best describes your field."),
    "labour": ("👷 How is labour availability?", "Can you get enough workers when you need them?"),
    "weed_height": ("🌿 What are you cutting?", "The toughest vegetation you regularly deal with."),
}
OPTION_TEXT = {
    "<2": "Under 2", "3_to_5": "3 to 5", ">5": "Over 5",
    "Harsh": "Harsh / rough", "Even_Tilled": "Even / tilled", "Slopy": "Sloping",
    "Scarce": "Scarce", "Sufficient": "Sufficient",
    "Soft_Grass": "Soft grass", "Tall_Dense": "Tall & dense weeds", "Thick_Stubble": "Thick stubble",
}
ICON = {"ideal": "✅ Ideal", "acceptable": "🟡 OK", "miss": "❌ Poor fit"}
MEDALS = ["🥇", "🥈", "🥉"]


def pretty(s: str) -> str:
    return str(s).replace("_", " ").strip().capitalize()


# --------------------------------------------------------------------------- data
def read_raw(src) -> pd.DataFrame:
    name = getattr(src, "name", str(src)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(src, header=None, dtype=str)
    return pd.read_csv(src, header=None, dtype=str)


def tidy(raw: pd.DataFrame, key: str) -> pd.DataFrame:
    """Find the real header row (the one containing `key`), drop blank columns/rows."""
    hdr = None
    for i, row in raw.iterrows():
        if key in [str(c).strip() for c in row.values]:
            hdr = i
            break
    if hdr is None:
        raise ValueError(f"Could not find a header row containing '{key}'.")
    cols = [str(c).strip() if pd.notna(c) else "" for c in raw.iloc[hdr]]
    df = raw.iloc[hdr + 1:].copy()
    df.columns = cols
    df = df.loc[:, [c != "" for c in cols]].dropna(how="all").reset_index(drop=True)
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


def first_existing(names):
    for n in names:
        if (BASE / n).exists():
            return BASE / n
    return None


@st.cache_data(show_spinner=False)
def load_data(machine_src, rule_src):
    machines = tidy(read_raw(machine_src), "model")
    rules = tidy(read_raw(rule_src), "feature_type")
    feats = [c[: -len("_ideal")] for c in rules.columns if c.endswith("_ideal")]
    if not feats:
        raise ValueError("No '<feature>_ideal' columns found in the rules file.")
    for f in feats:
        for suffix in ("_acceptable", "_weight"):
            if f + suffix not in rules.columns:
                raise ValueError(f"Rules file is missing column '{f}{suffix}'.")
        if f not in machines.columns:
            raise ValueError(f"Machines file has no column '{f}' (needed by the rules).")
        rules[f + "_weight"] = pd.to_numeric(rules[f + "_weight"], errors="coerce").fillna(0)
    return machines, rules, feats


# ------------------------------------------------------------------------ scoring
def as_set(v) -> set:
    return {x.strip() for x in str(v).split(",") if x.strip()} if pd.notna(v) else set()


def evaluate(machine: pd.Series, active_rules: pd.DataFrame, feats, acc_credit: float) -> pd.DataFrame:
    """One row per (farmer answer x feature): ideal = full credit, acceptable = partial, else 0."""
    rows = []
    for _, rule in active_rules.iterrows():
        for f in feats:
            w = float(rule[f + "_weight"])
            ideal, acc = as_set(rule[f + "_ideal"]), as_set(rule[f + "_acceptable"])
            v = machine[f]
            if pd.notna(v) and v in ideal:
                status, credit = "ideal", 1.0
            elif pd.notna(v) and v in acc:
                status, credit = "acceptable", acc_credit
            else:
                status, credit = "miss", 0.0
            rows.append(dict(question=rule["feature_type"], answer=rule["option"], feature=f,
                             value=v if pd.notna(v) else "—", weight=w, status=status,
                             credit=credit, points=w * credit, wanted=rule[f + "_ideal"]))
    return pd.DataFrame(rows)


def rank_machines(machines, active_rules, feats, acc_credit):
    out = []
    for idx, m in machines.iterrows():
        ev = evaluate(m, active_rules, feats, acc_credit)
        out.append(dict(idx=idx, score=ev.points.sum() / ev.weight.sum() if ev.weight.sum() else 0.0,
                        n_ideal=int((ev.status == "ideal").sum()), ev=ev))
    # ties: more ideal matches first, then longer warranty (if the column exists)
    warranty = pd.to_numeric(machines.get("warranty_mo"), errors="coerce").fillna(0) \
        if "warranty_mo" in machines else pd.Series(0, index=machines.index)
    out.sort(key=lambda r: (-round(r["score"], 6), -r["n_ideal"], -warranty[r["idx"]]))
    return out


# ------------------------------------------------------------------------- explain
def explain(ev: pd.DataFrame, label_of) -> tuple[list[str], list[str]]:
    """Plain-language strengths and gaps for one machine."""
    good, gaps = [], []
    by_answer = ev.groupby("answer", sort=False).apply(
        lambda d: d.points.sum() / d.weight.sum(), include_groups=False)
    for ans, s in by_answer.sort_values(ascending=False).items():
        if s >= 0.85:
            good.append(f"Very good match for **{label_of(ans)}** ({s:.0%}).")
    # features where it loses the most points
    lost = ev.assign(lost=ev.weight * (1 - ev.credit)).groupby("feature")["lost"].sum().sort_values(ascending=False)
    full = [pretty(f) for f in ev.groupby("feature")["credit"].min().pipe(lambda s: s[s == 1.0]).index]
    if full:
        good.append("Ideal on every one of your answers for: " + ", ".join(full) + ".")
    for f, l in lost[lost > 0].head(2).items():
        sub = ev[ev.feature == f]
        val = sub.value.iloc[0]
        weakest = sub.assign(lost=sub.weight * (1 - sub.credit)).sort_values("lost", ascending=False).iloc[0]
        wanted = weakest.wanted
        gaps.append(f"**{pretty(f)}** is *{val}*, but your answer "
                    f"(**{label_of(weakest.answer)}**) calls for *{wanted}*.")
    return good, gaps


# ---------------------------------------------------------------------------- UI
st.title("🌾 Brush Cutter Advisor")
st.caption("Answer four quick questions about your farm — we'll suggest the 3 best-fitting models and show exactly why.")

# Sidebar: data + settings
with st.sidebar:
    st.header("⚙️ Settings")
    acc_credit = st.slider("Credit for an 'acceptable' (not ideal) match", 0.0, 1.0, 0.5, 0.1,
                           help="Ideal match = 100% of a feature's weight. Acceptable = this share. Otherwise 0.")
    st.divider()
    st.subheader("Data files")
    up_m = st.file_uploader("Machines file", type=["xlsx", "csv"])
    up_r = st.file_uploader("Rules file", type=["xlsx", "csv"])
    st.caption("Leave empty to use the bundled files.")

m_src = up_m or first_existing(MACHINE_FILES)
r_src = up_r or first_existing(RULE_FILES)
if m_src is None or r_src is None:
    st.error("Data files not found. Put the machines and rules files next to app.py, or upload them in the sidebar.")
    st.stop()
try:
    machines, rules, FEATS = load_data(m_src, r_src)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read the data files: {e}")
    st.stop()

# Questions come straight from the rules file
questions = list(dict.fromkeys(rules["feature_type"]))
label_of = lambda opt: OPTION_TEXT.get(opt, pretty(opt))  # noqa: E731

st.subheader("Tell us about your farm")
answers = {}
cols = st.columns(2)
for i, q in enumerate(questions):
    title, help_txt = QUESTION_TEXT.get(q, (pretty(q), None))
    opts = list(rules.loc[rules.feature_type == q, "option"])
    with cols[i % 2].container(border=True):
        answers[q] = st.radio(title, opts, index=None, format_func=label_of, help=help_txt, key=f"q_{q}")

if any(v is None for v in answers.values()):
    st.info("👆 Please answer all questions to see your recommendations.")
    st.stop()

# Active rules = one row per answered question
active = pd.concat([rules[(rules.feature_type == q) & (rules.option == a)] for q, a in answers.items()])
ranked = rank_machines(machines, active, FEATS, acc_credit)

st.divider()
st.subheader("🏆 Your top 3 machines")
st.caption("Your match score = weighted share of checks passed. "
           "Each of your 4 answers is checked against every feature; important features count more.")

for rank, r in enumerate(ranked[:3]):
    m = machines.loc[r["idx"]]
    ev = r["ev"]
    with st.container(border=True):
        h1, h2 = st.columns([4, 1])
        h1.markdown(f"### {MEDALS[rank]} #{rank + 1} · {m['model']}")
        h1.caption(f"Brand: {m.get('brand', '—')}")
        h2.metric("Match", f"{r['score']:.0%}")
        st.progress(min(max(r["score"], 0.0), 1.0))

        good, gaps = explain(ev, label_of)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**👍 Why it fits**")
            for g in good or ["No standout strengths for this combination."]:
                st.markdown(f"- {g}")
        with c2:
            st.markdown("**⚠️ Where it falls short**")
            for g in gaps or ["Nothing significant — it meets or nearly meets every requirement."]:
                st.markdown(f"- {g}")

        # score per farmer answer
        pa = ev.groupby("answer", sort=False).apply(lambda d: d.points.sum() / d.weight.sum(), include_groups=False)
        mc = st.columns(len(pa))
        for col, (ans, s) in zip(mc, pa.items()):
            col.metric(label_of(ans), f"{s:.0%}")

        with st.expander("See the full feature-by-feature check"):
            grid = ev.assign(cell=ev.status.map(ICON)).pivot(index="feature", columns="answer", values="cell")
            grid = grid[list(dict.fromkeys(ev.answer))].loc[FEATS]
            grid.columns = [label_of(c) for c in grid.columns]
            grid.insert(0, "This machine has", [ev[ev.feature == f].value.iloc[0] for f in FEATS])
            grid.index = [pretty(f) for f in FEATS]
            grid.index.name = "Feature"
            st.table(grid)
            counts = ev.status.value_counts()
            st.caption(f"{counts.get('ideal', 0)} ideal · {counts.get('acceptable', 0)} acceptable · "
                       f"{counts.get('miss', 0)} poor, out of {len(ev)} checks.")
            extra = [c for c in machines.columns if c not in FEATS + ["model", "brand"] and pd.notna(m[c])]
            if extra:
                st.caption("Other specs (not scored): " + " · ".join(f"{pretty(c)}: {m[c]}" for c in extra))

with st.expander("📋 See all machines ranked"):
    tbl = pd.DataFrame({"Rank": range(1, len(ranked) + 1),
                        "Model": [machines.loc[r["idx"], "model"] for r in ranked],
                        "Brand": [machines.loc[r["idx"], "brand"] for r in ranked],
                        "Match %": [round(r["score"] * 100, 1) for r in ranked]})
    st.dataframe(tbl, hide_index=True)

with st.expander("ℹ️ How the score works"):
    st.markdown(
        "- Each answer you give selects one row of rules (ideal + acceptable values and a weight per feature).\n"
        "- For every feature, a machine gets **full weight** if its spec is *ideal*, **partial weight** "
        "(set in the sidebar) if *acceptable*, and **zero** otherwise.\n"
        "- Match % = points earned ÷ total possible points across all four answers.\n"
        "- Ties are broken by number of ideal matches, then warranty length."
    )
