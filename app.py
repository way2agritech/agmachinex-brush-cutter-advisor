"""
Brush Cutter Advisor — Streamlit prototype
"""

from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Brush Cutter Advisor",
    page_icon="🌾",
    layout="wide",
)

BASE = Path(__file__).parent

MACHINE_FILES = ["machines_csv.xlsx", "machines.csv"]

RULE_FILES = [
    "rules_ideal_acceptable_csv.xlsx",
    "rules_ideal_acceptable.csv",
]

ACC_CREDIT = 0.5


QUESTION_TEXT = {
    "land_extent": (
        "📐 How big is your land?",
        "Approximate area to be cleared (acres).",
    ),
    "soil_feature": (
        "⛰️ What is the ground like?",
        "Pick the one that best describes your field.",
    ),
    "labour": (
        "👷 How is labour availability?",
        "Can you get enough workers when you need them?",
    ),
    "weed_height": (
        "🌿 What are you cutting?",
        "The toughest vegetation you regularly deal with.",
    ),
}

OPTION_TEXT = {
    "<2": "Under 2",
    "3_to_5": "3 to 5",
    ">5": "Over 5",
    "Harsh": "Harsh / rough",
    "Even_Tilled": "Even / tilled",
    "Slopy": "Sloping",
    "Scarce": "Scarce",
    "Sufficient": "Sufficient",
    "Soft_Grass": "Soft grass",
    "Tall_Dense": "Tall & dense weeds",
    "Thick_Stubble": "Thick stubble",
}

ICON = {
    "ideal": "✅ Ideal",
    "acceptable": "🟡 OK",
    "miss": "❌ Poor fit",
}

MEDALS = ["🥇", "🥈", "🥉"]


def pretty(s: str) -> str:
    return str(s).replace("_", " ").strip().capitalize()


def read_raw(src) -> pd.DataFrame:

    name = getattr(src, "name", str(src)).lower()

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(
            src,
            header=None,
            dtype=str,
        )

    return pd.read_csv(
        src,
        header=None,
        dtype=str,
    )


def tidy(raw: pd.DataFrame, key: str) -> pd.DataFrame:

    hdr = None

    for i, row in raw.iterrows():

        if key in [
            str(c).strip()
            for c in row.values
        ]:
            hdr = i
            break

    if hdr is None:
        raise ValueError(
            f"Could not find a header row containing '{key}'."
        )

    cols = [
        str(c).strip()
        if pd.notna(c)
        else ""
        for c in raw.iloc[hdr]
    ]

    df = raw.iloc[hdr + 1:].copy()

    df.columns = cols

    df = (
        df.loc[:, [c != "" for c in cols]]
        .dropna(how="all")
        .reset_index(drop=True)
    )

    for c in df.columns:
        df[c] = df[c].str.strip()

    return df


def first_existing(names):

    for name in names:

        path = BASE / name

        if path.exists():
            return path

    return None


@st.cache_data(show_spinner=False)
def load_data(
    machine_src,
    rule_src,
):

    machines = tidy(
        read_raw(machine_src),
        "model",
    )

    rules = tidy(
        read_raw(rule_src),
        "feature_type",
    )

    feats = [
        c[:-len("_ideal")]
        for c in rules.columns
        if c.endswith("_ideal")
    ]

    if not feats:
        raise ValueError(
            "No '<feature>_ideal' columns found in the rules file."
        )

    for f in feats:

        for suffix in (
            "_acceptable",
            "_weight",
        ):

            if f + suffix not in rules.columns:

                raise ValueError(
                    f"Rules file is missing column "
                    f"'{f}{suffix}'."
                )

        if f not in machines.columns:

            raise ValueError(
                f"Machines file has no column "
                f"'{f}' (needed by the rules)."
            )

        rules[f + "_weight"] = pd.to_numeric(
            rules[f + "_weight"],
            errors="coerce",
        ).fillna(0)

    return machines, rules, feats


def as_set(v) -> set:

    if pd.isna(v):
        return set()

    return {
        x.strip()
        for x in str(v).split(",")
        if x.strip()
    }


def evaluate(
    machine: pd.Series,
    active_rules: pd.DataFrame,
    feats,
    acc_credit: float,
) -> pd.DataFrame:

    rows = []

    for _, rule in active_rules.iterrows():

        for f in feats:

            weight = float(
                rule[f + "_weight"]
            )

            ideal = as_set(
                rule[f + "_ideal"]
            )

            acceptable = as_set(
                rule[f + "_acceptable"]
            )

            machine_values = as_set(
                machine[f]
            )

            if machine_values & ideal:

                status = "ideal"
                credit = 1.0

            elif machine_values & acceptable:

                status = "acceptable"
                credit = acc_credit

            else:

                status = "miss"
                credit = 0.0

            if pd.notna(machine[f]):

                display_value = str(
                    machine[f]
                ).strip()

            else:

                display_value = "—"

            rows.append(
                {
                    "question": rule["feature_type"],
                    "answer": rule["option"],
                    "feature": f,
                    "value": display_value,
                    "weight": weight,
                    "status": status,
                    "credit": credit,
                    "points": weight * credit,
                    "wanted": rule[f + "_ideal"],
                }
            )

    return pd.DataFrame(rows)


def rank_machines(
    machines,
    active_rules,
    feats,
    acc_credit,
):

    ranked = []

    for idx, machine in machines.iterrows():

        evaluation = evaluate(
            machine,
            active_rules,
            feats,
            acc_credit,
        )

        total_weight = (
            evaluation.weight.sum()
        )

        score = (
            evaluation.points.sum()
            / total_weight
            if total_weight
            else 0.0
        )

        ranked.append(
            {
                "idx": idx,
                "score": score,
                "n_ideal": int(
                    (
                        evaluation.status
                        == "ideal"
                    ).sum()
                ),
                "ev": evaluation,
            }
        )

    if "warranty_mo" in machines.columns:

        warranty = pd.to_numeric(
            machines["warranty_mo"],
            errors="coerce",
        ).fillna(0)

    else:

        warranty = pd.Series(
            0,
            index=machines.index,
        )

    ranked.sort(
        key=lambda r: (
            -round(r["score"], 6),
            -r["n_ideal"],
            -warranty[r["idx"]],
        )
    )

    return ranked


def explain(
    evaluation: pd.DataFrame,
    label_of,
):

    good = []
    gaps = []

    by_answer = (
        evaluation
        .groupby(
            "answer",
            sort=False,
        )
        .apply(
            lambda d:
            d.points.sum()
            / d.weight.sum(),
            include_groups=False,
        )
    )

    for answer, score in (
        by_answer
        .sort_values(
            ascending=False
        )
        .items()
    ):

        if score >= 0.85:

            good.append(
                f"Very good match for "
                f"**{label_of(answer)}** "
                f"({score:.0%})."
            )

    lost = (
        evaluation.assign(
            lost=evaluation.weight
            * (1 - evaluation.credit)
        )
        .groupby("feature")["lost"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    full = [
        pretty(feature)
        for feature in (
            evaluation
            .groupby("feature")["credit"]
            .min()
            .pipe(
                lambda s:
                s[s == 1.0]
            )
            .index
        )
    ]

    if full:

        good.append(
            "Ideal on every one of your answers for: "
            + ", ".join(full)
            + "."
        )

    for feature, lost_points in (
        lost[lost > 0]
        .head(2)
        .items()
    ):

        sub = evaluation[
            evaluation.feature == feature
        ]

        value = sub.value.iloc[0]

        weakest = (
            sub.assign(
                lost=sub.weight
                * (1 - sub.credit)
            )
            .sort_values(
                "lost",
                ascending=False,
            )
            .iloc[0]
        )

        wanted = weakest.wanted

        gaps.append(
            f"**{pretty(feature)}** is "
            f"*{value}*, but your answer "
            f"(**{label_of(weakest.answer)}**) "
            f"calls for *{wanted}*."
        )

    return good, gaps


st.title("🌾 Brush Cutter Advisor")

st.caption(
    "Answer four quick questions about your farm — "
    "we'll show how well each machine fits your requirements."
)


machine_src = first_existing(
    MACHINE_FILES
)

rule_src = first_existing(
    RULE_FILES
)

if (
    machine_src is None
    or rule_src is None
):

    st.error(
        "Data files not found. "
        "Put the machines and rules files "
        "next to app.py."
    )

    st.stop()


try:

    machines, rules, FEATS = load_data(
        machine_src,
        rule_src,
    )

except Exception as e:

    st.error(
        f"Could not read the data files: {e}"
    )

    st.stop()


questions = list(
    dict.fromkeys(
        rules["feature_type"]
    )
)

label_of = lambda opt: OPTION_TEXT.get(
    opt,
    pretty(opt),
)


st.subheader(
    "Tell us about your farm"
)

answers = {}

cols = st.columns(2)

for i, question in enumerate(
    questions
):

    title, help_text = QUESTION_TEXT.get(
        question,
        (
            pretty(question),
            None,
        ),
    )

    options = list(
        rules.loc[
            rules.feature_type == question,
            "option",
        ]
    )

    with cols[i % 2].container(
        border=True
    ):

        answers[question] = st.radio(
            title,
            options,
            index=None,
            format_func=label_of,
            help=help_text,
            key=f"q_{question}",
        )


if any(
    value is None
    for value in answers.values()
):

    st.info(
        "👆 Please answer all questions "
        "to see the machine matches."
    )

    st.stop()


active = pd.concat(
    [
        rules[
            (rules.feature_type == question)
            & (rules.option == answer)
        ]
        for question, answer
        in answers.items()
    ]
)


ranked = rank_machines(
    machines,
    active,
    FEATS,
    ACC_CREDIT,
)


st.divider()

st.subheader(
    f"🔎 Machine matches ({len(ranked)})"
)

st.caption(
    "Machines are ordered by their match with "
    "your requirements. The complete "
    "feature-by-feature check is shown "
    "for every machine."
)


for rank, result in enumerate(
    ranked
):

    machine = machines.loc[
        result["idx"]
    ]

    evaluation = result["ev"]

    with st.container(
        border=True
    ):

        h1, h2 = st.columns(
            [4, 1]
        )

        if rank < len(MEDALS):

            rank_label = (
                f"{MEDALS[rank]} "
                f"#{rank + 1}"
            )

        else:

            rank_label = (
                f"#{rank + 1}"
            )

        h1.markdown(
            f"### {rank_label} · "
            f"{machine['model']}"
        )

        h1.caption(
            f"Brand: "
            f"{machine.get('brand', '—')}"
        )

        h2.metric(
            "Match",
            f"{result['score']:.0%}",
        )

        st.progress(
            min(
                max(
                    result["score"],
                    0.0,
                ),
                1.0,
            )
        )

        good, gaps = explain(
            evaluation,
            label_of,
        )

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                "**👍 Why it fits**"
            )

            for item in good or [
                "No standout strengths "
                "for this combination."
            ]:

                st.markdown(
                    f"- {item}"
                )

        with c2:

            st.markdown(
                "**⚠️ Where it falls short**"
            )

            for item in gaps or [
                "Nothing significant — "
                "it meets or nearly meets "
                "every requirement."
            ]:

                st.markdown(
                    f"- {item}"
                )

        per_answer = (
            evaluation
            .groupby(
                "answer",
                sort=False,
            )
            .apply(
                lambda d:
                d.points.sum()
                / d.weight.sum(),
                include_groups=False,
            )
        )

        metric_columns = st.columns(
            len(per_answer)
        )

        for column, (
            answer,
            score,
        ) in zip(
            metric_columns,
            per_answer.items(),
        ):

            column.metric(
                label_of(answer),
                f"{score:.0%}",
            )

        st.markdown(
            "#### Full feature-by-feature check"
        )

        grid = (
            evaluation
            .assign(
                cell=evaluation.status.map(
                    ICON
                )
            )
            .pivot(
                index="feature",
                columns="answer",
                values="cell",
            )
        )

        grid = (
            grid[
                list(
                    dict.fromkeys(
                        evaluation.answer
                    )
                )
            ]
            .loc[FEATS]
        )

        grid.columns = [
            label_of(column)
            for column in grid.columns
        ]

        grid.insert(
            0,
            "This machine has",
            [
                evaluation[
                    evaluation.feature == feature
                ].value.iloc[0]
                for feature in FEATS
            ],
        )

        grid.index = [
            pretty(feature)
            for feature in FEATS
        ]

        grid.index.name = "Feature"

        st.table(grid)

        counts = (
            evaluation.status
            .value_counts()
        )

        st.caption(
            f"{counts.get('ideal', 0)} ideal · "
            f"{counts.get('acceptable', 0)} acceptable · "
            f"{counts.get('miss', 0)} poor, "
            f"out of {len(evaluation)} checks."
        )

        extra = [
            column
            for column in machines.columns
            if column not in (
                FEATS
                + ["model", "brand"]
            )
            and pd.notna(
                machine[column]
            )
        ]

        if extra:

            st.caption(
                "Other specs (not scored): "
                + " · ".join(
                    f"{pretty(column)}: "
                    f"{machine[column]}"
                    for column in extra
                )
            )
