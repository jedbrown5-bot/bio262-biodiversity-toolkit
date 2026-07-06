"""
toolkit.py
==========
Page-render functions and shared helpers for the BIO262 Biodiversity Toolkit.
Kept separate from app.py so each page can be imported and tested in isolation.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import rarefaction as R
import diversity as DIV
import anova as AN

GREEN = "#2e8b57"
RED = "#d1495b"
BLUE = "#3a7ca5"


# --------------------------------------------------------------------------- #
#  Shared parsing helpers
# --------------------------------------------------------------------------- #
def parse_vector(text: str) -> list[float]:
    text = text.replace(",", " ").replace("\t", " ").replace("\n", " ")
    return [float(v) for v in text.split(" ") if v.strip() != ""]


def parse_matrix(text: str) -> np.ndarray:
    rows = []
    for line in text.strip().splitlines():
        line = line.replace(",", " ").replace("\t", " ").strip()
        if not line:
            continue
        rows.append([float(v) for v in line.split()])
    if not rows:
        raise ValueError("No data rows found.")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("All rows must have the same number of columns.")
    return np.array(rows, dtype=float)


def read_uploaded_df(file, has_header: bool) -> pd.DataFrame:
    raw = file.getvalue().decode("utf-8", errors="replace")
    first = raw.splitlines()[0] if raw.strip() else ""
    sep = "\t" if "\t" in first else ","
    return pd.read_csv(io.StringIO(raw), sep=sep, header=0 if has_header else None)


# --------------------------------------------------------------------------- #
#  HOME
# --------------------------------------------------------------------------- #
def render_home(rare_pg=None, div_pg=None, anova_pg=None):
    st.title("🌿 BIO262 Biodiversity Toolkit")
    st.markdown(
        "A small suite of calculators for biodiversity practicals. "
        "Pick a tool below or from the sidebar."
    )
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🌿 Rarefaction")
        st.write(
            "Species-accumulation curves with confidence intervals and "
            "extrapolation, reproducing the core methods of EstimateS "
            "(Colwell). Compare richness at equal sampling effort."
        )
        if rare_pg is not None:
            st.page_link(rare_pg, label="Open Rarefaction")
    with c2:
        st.subheader("📊 Diversity")
        st.write(
            "Shannon and Simpson diversity indices from either individual "
            "counts or percent-cover data, for one sample or many sites "
            "side-by-side."
        )
        if div_pg is not None:
            st.page_link(div_pg, label="Open Diversity")
    with c3:
        st.subheader("🧪 One-way ANOVA")
        st.write(
            "Compare a measurement across groups: F-test, summary table, "
            "Tukey HSD, assumption checks, and a non-parametric "
            "Kruskal-Wallis fallback."
        )
        if anova_pg is not None:
            st.page_link(anova_pg, label="Open ANOVA")
    st.divider()
    st.caption(
        "Built for teaching. Methods are standard and cited within each tool. "
        "Rarefaction reproduces Colwell et al. (2004, 2012); it is not "
        "affiliated with the EstimateS source code."
    )


# --------------------------------------------------------------------------- #
#  RAREFACTION
# --------------------------------------------------------------------------- #
EX_ABUND = "40 20 15 12 10 8 6 5 4 3 3 2 2 2 1 1 1 1 1 1"
EX_MATRIX = "\n".join([
    "1 1 1 1 1 1 1 1", "1 1 1 1 1 1 1 0", "1 1 1 1 1 0 0 0", "1 1 1 1 0 0 0 0",
    "1 1 1 0 0 0 0 0", "1 1 0 0 0 0 0 0", "1 1 0 0 0 0 0 0", "1 0 0 0 0 0 0 0",
    "0 1 0 0 0 0 0 0", "0 0 1 0 0 0 0 0", "0 0 0 1 0 0 0 0", "0 0 0 0 0 0 0 1",
])


def render_rarefaction():
    st.title("🌿 Rarefaction & Extrapolation")
    st.caption(
        "After Colwell, Mao & Chang (2004) and Colwell et al. (2012) - the "
        "mathematics behind EstimateS."
    )

    with st.sidebar:
        st.header("Data type")
        mode = st.radio(
            "What kind of data do you have?",
            ["Individual-based (abundance vector)",
             "Sample-based (species x samples matrix)"],
            help="Individual-based rarefies by individuals; sample-based "
                 "rarefies by sampling units (e.g. quadrats).")
        incidence = mode.startswith("Sample")

        st.header("Input method")
        method = st.radio("How to provide data?",
                          ["Paste", "Upload CSV/TSV", "Use example"])

        st.header("Options")
        do_extrap = st.checkbox("Extrapolate beyond the reference sample", True)
        extrap_factor = st.slider("Extrapolate to ... x reference size",
                                  1.0, 3.0, 2.0, 0.5, disabled=not do_extrap,
                                  help="EstimateS advises not going past 2-3x.")
        ci_level = st.select_slider("Confidence level", [0.90, 0.95, 0.99], 0.95)
        n_knots = st.slider("Curve resolution (points)", 20, 200, 80, 10)

    counts, err = None, None
    col_in, col_out = st.columns([1, 1.6])

    with col_in:
        st.subheader("Input")
        try:
            if method == "Use example":
                sample = EX_MATRIX if incidence else EX_ABUND
                st.code(sample, language=None)
                counts = (R.counts_from_incidence(parse_matrix(sample))
                          if incidence else
                          R.counts_from_abundances(parse_vector(sample)))
            elif method == "Paste":
                if incidence:
                    txt = st.text_area("Paste species x samples matrix "
                                       "(rows = species, columns = units)",
                                       EX_MATRIX, height=240)
                    if txt.strip():
                        counts = R.counts_from_incidence(parse_matrix(txt))
                else:
                    txt = st.text_area("Paste abundance counts (one per species)",
                                       EX_ABUND, height=160)
                    if txt.strip():
                        counts = R.counts_from_abundances(parse_vector(txt))
            else:
                up = st.file_uploader("Upload CSV or TSV", ["csv", "tsv", "txt"])
                has_header = st.checkbox("First row is a header", False)
                index_col = st.checkbox("First column is a label (species names)",
                                        False)
                if up is not None:
                    df = read_uploaded_df(up, has_header)
                    if index_col:
                        df = df.iloc[:, 1:]
                    mat = df.apply(pd.to_numeric, errors="coerce").fillna(0)\
                            .to_numpy(dtype=float)
                    counts = (R.counts_from_incidence(mat) if incidence else
                              R.counts_from_abundances(
                                  mat[:, 0] if mat.ndim == 2 else mat.ravel()))
        except Exception as e:  # noqa: BLE001
            err = str(e)

        if err:
            st.error(f"Could not parse the data: {err}")
        if counts is not None:
            unit = "sampling units" if incidence else "individuals"
            est_name = "Chao2" if incidence else "Chao1"
            est_val = R.chao2(counts) if incidence else R.chao1(counts)
            st.markdown("**Reference sample**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Observed species", counts.S_obs)
            m2.metric(f"Total {unit}", counts.N)
            m3.metric(est_name, f"{est_val:.1f}")
            rare = "Uniques (Q1)" if incidence else "Singletons (f1)"
            dup = "Duplicates (Q2)" if incidence else "Doubletons (f2)"
            st.caption(f"{rare}: {counts.f1}  |  {dup}: {counts.f2}")

    with col_out:
        st.subheader("Rarefaction curve")
        if counts is None:
            st.info("Enter data on the left to compute a curve.")
        else:
            extrap_to = int(round(counts.N * extrap_factor)) if do_extrap else None
            curve = R.compute_curve(counts, incidence=incidence,
                                    extrapolate_to=extrap_to,
                                    n_knots=n_knots, ci=ci_level)
            xlab = "Number of sampling units" if incidence else "Number of individuals"
            rare_mask = ~curve.is_extrap
            ext_mask = curve.is_extrap

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=np.concatenate([curve.x, curve.x[::-1]]),
                y=np.concatenate([curve.hi, curve.lo[::-1]]),
                fill="toself", fillcolor="rgba(46,139,87,0.15)",
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                name=f"{int(ci_level*100)}% CI"))
            fig.add_trace(go.Scatter(
                x=curve.x[rare_mask], y=curve.y[rare_mask], mode="lines",
                line=dict(color=GREEN, width=3), name="Rarefaction (interpolated)"))
            if ext_mask.any():
                join = np.where(curve.x <= counts.N)[0][-1]
                xe = np.concatenate([[curve.x[join]], curve.x[ext_mask]])
                ye = np.concatenate([[curve.y[join]], curve.y[ext_mask]])
                fig.add_trace(go.Scatter(
                    x=xe, y=ye, mode="lines",
                    line=dict(color=GREEN, width=3, dash="dash"),
                    name="Extrapolation"))
            fig.add_trace(go.Scatter(
                x=[counts.N], y=[counts.S_obs], mode="markers",
                marker=dict(color="darkgreen", size=11), name="Reference sample"))
            est_label = "Chao2" if incidence else "Chao1"
            fig.add_hline(y=curve.estimator,
                          line=dict(color=RED, dash="dash", width=2),
                          annotation_text=f"{est_label} = {curve.estimator:.2f}",
                          annotation_position="top left",
                          annotation_font=dict(color=RED))
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=RED, dash="dash", width=2),
                name=f"{est_label} (asymptote)"))
            fig.update_layout(
                xaxis_title=xlab, yaxis_title="Species richness (S est)",
                template="simple_white", height=520,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")

            df = pd.DataFrame({
                xlab: curve.x.astype(int),
                "S_est": np.round(curve.y, 4),
                "SD": np.round(curve.sd, 4),
                f"CI_low_{int(ci_level*100)}": np.round(curve.lo, 4),
                f"CI_high_{int(ci_level*100)}": np.round(curve.hi, 4),
                "type": np.where(curve.is_extrap, "extrapolated", "rarefaction"),
            })
            with st.expander("Results table"):
                st.dataframe(df, width="stretch", hide_index=True)
            st.download_button("Download results (CSV)",
                               df.to_csv(index=False).encode(),
                               "rarefaction_results.csv", "text/csv")

    with st.expander("About the methods & how to cite"):
        st.markdown(
            "- Individual-based rarefaction - Hurlbert (1971); Colwell et al. "
            "2012, Eq. 4.\n"
            "- Sample-based rarefaction - Mao Tau; Colwell, Mao & Chang 2004; "
            "Colwell et al. 2012, Eq. 17.\n"
            "- Extrapolation - Chao1 (individuals, Eq. 9) or Chao2 (samples, "
            "Eq. 18) as the asymptotic target.\n"
            "- CIs - unconditional variance, so the interval stays open at the "
            "reference sample.\n\n"
            "Please cite: Colwell, R. K. 2013. EstimateS, Version 9. "
            "http://purl.oclc.org/estimates"
        )


# --------------------------------------------------------------------------- #
#  DIVERSITY
# --------------------------------------------------------------------------- #
EX_DIV = "\n".join([
    "Grass A,40,10,55",
    "Sedge B,20,15,10",
    "Herb C,15,25,5",
    "Forb D,10,20,0",
    "Shrub E,8,18,2",
    "Fern F,5,12,0",
    "Moss G,2,0,28",
])


def _fmt_div_rows(rows, count_label):
    data = []
    for name, d in rows:
        data.append({
            "Sample": name,
            "Richness (S)": d.S,
            count_label: round(d.total, 2),
            "Shannon H'": round(d.shannon_H, 4),
            "exp(H')": round(d.shannon_exp, 3),
            "Simpson D": round(d.simpson_D, 4),
            "1 - D": round(d.gini_simpson, 4),
            "1 / D": round(d.inv_simpson, 3),
            "Evenness J'": (round(d.pielou_J, 4)
                            if d.pielou_J == d.pielou_J else float("nan")),
        })
    return pd.DataFrame(data)


def render_diversity():
    st.title("📊 Diversity Calculator")
    st.caption("Shannon (H') and Simpson diversity for counts or percent-cover data.")

    with st.sidebar:
        st.header("Data type")
        dtype = st.radio("Values are...",
                         ["Individual counts", "Percent cover (or any cover value)"])
        count_label = "Total individuals" if dtype.startswith("Individual") \
            else "Total cover"
        st.header("Input method")
        method = st.radio("How to provide data?",
                          ["Paste", "Upload CSV/TSV", "Use example"])
        st.caption("Rows = species. Columns = one or more samples/sites.")

    st.subheader("Input")
    st.markdown(
        "Enter a species x samples table: one row per species, one column per "
        "sample/site. A first column of species names and a header row of "
        "sample names are optional but recommended."
    )

    mat, sample_names, err = None, None, None
    try:
        if method == "Use example":
            st.code("species, Site 1, Site 2, Site 3\n" + EX_DIV, language=None)
            df = pd.read_csv(io.StringIO("sp,Site 1,Site 2,Site 3\n" + EX_DIV),
                             header=0)
            sample_names = list(df.columns[1:])
            mat = df.iloc[:, 1:].to_numpy(dtype=float)
        elif method == "Paste":
            txt = st.text_area(
                "Paste data (comma/space/tab separated). Include a species "
                "column and header row if you like.",
                "species, Site 1, Site 2, Site 3\n" + EX_DIV, height=220)
            has_header = st.checkbox("First row is a header (sample names)", True)
            has_labels = st.checkbox("First column is species names", True)
            if txt.strip():
                lines = [",".join(l.replace("\t", ",").replace(";", ",").split(","))
                         for l in txt.strip().splitlines()]
                df = pd.read_csv(io.StringIO("\n".join(lines)),
                                 header=0 if has_header else None)
                if has_labels:
                    sample_names = list(map(str, df.columns[1:])) if has_header \
                        else None
                    df = df.iloc[:, 1:]
                elif has_header:
                    sample_names = list(map(str, df.columns))
                mat = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        else:
            up = st.file_uploader("Upload CSV or TSV", ["csv", "tsv", "txt"])
            has_header = st.checkbox("First row is a header (sample names)", True)
            has_labels = st.checkbox("First column is species names", True)
            if up is not None:
                df = read_uploaded_df(up, has_header)
                if has_labels:
                    sample_names = list(map(str, df.columns[1:])) if has_header \
                        else None
                    df = df.iloc[:, 1:]
                elif has_header:
                    sample_names = list(map(str, df.columns))
                mat = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    except Exception as e:  # noqa: BLE001
        err = str(e)

    if err:
        st.error(f"Could not parse the data: {err}")
    if mat is None:
        st.info("Enter data above to compute diversity indices.")
        return

    mat = np.nan_to_num(mat, nan=0.0)
    rows = DIV.diversity_table(mat, sample_names=sample_names,
                               include_pooled=mat.shape[1] > 1)
    table = _fmt_div_rows(rows, count_label)

    st.subheader("Results")
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button("Download indices (CSV)",
                       table.to_csv(index=False).encode(),
                       "diversity_indices.csv", "text/csv")

    plot_rows = [r for r in rows if not r[0].startswith("Pooled")]
    if len(plot_rows) > 1:
        names = [r[0] for r in plot_rows]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Shannon H'", x=names,
                             y=[r[1].shannon_H for r in plot_rows],
                             marker_color=GREEN))
        fig.add_trace(go.Bar(name="Simpson 1 - D", x=names,
                             y=[r[1].gini_simpson for r in plot_rows],
                             marker_color=BLUE))
        fig.update_layout(barmode="group", template="simple_white", height=420,
                          yaxis_title="Diversity", margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, width="stretch")

    with st.expander("How to read these indices"):
        st.markdown(
            "- Richness (S) - number of species present.\n"
            "- Shannon H' - accounts for richness and evenness; higher = more "
            "diverse. exp(H') expresses it as an effective number of equally-"
            "common species.\n"
            "- Simpson D - dominance (probability two individuals are the same "
            "species). 1 - D is diversity (higher = more diverse); 1 / D is the "
            "effective number of common species.\n"
            "- Evenness J' - how equal the abundances are (0-1; 1 = perfectly "
            "even).\n\n"
            "Counts and percent cover use the same formulas - both are converted "
            "to proportions first. References: Magurran (2004); Jost (2006)."
        )


# --------------------------------------------------------------------------- #
#  ONE-WAY ANOVA
# --------------------------------------------------------------------------- #
EX_ANOVA = "\n".join([
    "Control, Low, High",
    "64, 78, 75",
    "72, 91, 93",
    "68, 97, 78",
    "77, 82, 71",
    "56, 85, 63",
    "95, 77, 76",
])


def _groups_from_wide(df, has_header):
    if has_header:
        labels = [str(c) for c in df.columns]
    else:
        labels = [f"Group {i+1}" for i in range(df.shape[1])]
    groups = []
    for i in range(df.shape[1]):
        col = pd.to_numeric(df.iloc[:, i], errors="coerce").to_numpy(dtype=float)
        col = col[np.isfinite(col)]
        groups.append(col)
    return labels, groups


def render_anova():
    st.title("🧪 One-way ANOVA")
    st.caption("Compare a numeric measurement across two or more groups.")

    with st.sidebar:
        st.header("Data layout")
        layout = st.radio("How is your data arranged?",
                          ["Wide - one column per group",
                           "Long - a value column + a group column"])
        st.header("Input method")
        method = st.radio("How to provide data?",
                          ["Paste", "Upload CSV/TSV", "Use example"])
        alpha = st.select_slider("Significance level (alpha)", [0.10, 0.05, 0.01], 0.05)

    labels, groups, err = None, None, None
    st.subheader("Input")

    try:
        if method == "Use example":
            st.code(EX_ANOVA, language=None)
            df = pd.read_csv(io.StringIO(EX_ANOVA.replace(", ", ",")), header=0)
            labels, groups = _groups_from_wide(df, True)
        elif method == "Paste":
            if layout.startswith("Wide"):
                txt = st.text_area("Paste groups as columns (header row = group "
                                   "names).", EX_ANOVA, height=200)
                has_header = st.checkbox("First row is a header (group names)", True)
                if txt.strip():
                    lines = [",".join(l.replace("\t", ",").replace(";", ",").split(","))
                             for l in txt.strip().splitlines()]
                    df = pd.read_csv(io.StringIO("\n".join(lines)),
                                     header=0 if has_header else None)
                    labels, groups = _groups_from_wide(df, has_header)
            else:
                txt = st.text_area("Paste two columns: value, group",
                                   "value,group\n64,Control\n72,Control\n"
                                   "78,Low\n91,Low\n75,High\n93,High", height=200)
                if txt.strip():
                    df = pd.read_csv(io.StringIO(txt), header=0)
                    vcol, gcol = df.columns[0], df.columns[1]
                    labels = list(dict.fromkeys(df[gcol].astype(str)))
                    groups = [pd.to_numeric(df[df[gcol].astype(str) == lab][vcol],
                                            errors="coerce").dropna().to_numpy()
                              for lab in labels]
        else:
            up = st.file_uploader("Upload CSV or TSV", ["csv", "tsv", "txt"])
            has_header = st.checkbox("First row is a header", True)
            if up is not None:
                df = read_uploaded_df(up, has_header)
                if layout.startswith("Wide"):
                    labels, groups = _groups_from_wide(df, has_header)
                else:
                    cols = list(df.columns)
                    c1, c2 = st.columns(2)
                    vcol = c1.selectbox("Value column", cols, 0)
                    gcol = c2.selectbox("Group column", cols,
                                        1 if len(cols) > 1 else 0)
                    labels = list(dict.fromkeys(df[gcol].astype(str)))
                    groups = [pd.to_numeric(df[df[gcol].astype(str) == lab][vcol],
                                            errors="coerce").dropna().to_numpy()
                              for lab in labels]
    except Exception as e:  # noqa: BLE001
        err = str(e)

    if err:
        st.error(f"Could not parse the data: {err}")
        return
    if not groups:
        st.info("Enter data above to run the ANOVA.")
        return

    keep = [(lab, g) for lab, g in zip(labels, groups) if g.size >= 1]
    labels = [lab for lab, _ in keep]
    groups = [g for _, g in keep]
    if len(groups) < 2:
        st.error("Need at least two groups with data.")
        return
    if any(g.size < 2 for g in groups):
        st.warning("Some groups have fewer than 2 observations; results may be "
                   "unreliable and assumption tests may not run.")

    st.subheader("Group distributions")
    fig = go.Figure()
    for lab, g in zip(labels, groups):
        fig.add_trace(go.Box(y=g, name=str(lab), boxmean=True,
                             marker_color=GREEN, boxpoints="all",
                             jitter=0.4, pointpos=0))
    fig.update_layout(template="simple_white", height=420, showlegend=False,
                      yaxis_title="Value", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")

    res = AN.one_way_anova(groups, labels)
    st.subheader("One-way ANOVA")
    m1, m2, m3 = st.columns(3)
    m1.metric("F", f"{res.F:.3f}")
    m2.metric("p-value", f"{res.p:.4g}")
    m3.metric("eta squared", f"{res.eta_sq:.3f}")

    verdict = ("**significant** - at least one group mean differs"
               if res.p < alpha else
               "**not significant** - no detected difference between group means")
    st.markdown(f"At alpha = {alpha}, the result is {verdict} (F({res.df_between}, "
                f"{res.df_within}) = {res.F:.3f}, p = {res.p:.4g}).")

    anova_tbl = pd.DataFrame([
        {"Source": "Between groups", "SS": res.ss_between, "df": res.df_between,
         "MS": res.ms_between, "F": res.F, "p": res.p},
        {"Source": "Within groups", "SS": res.ss_within, "df": res.df_within,
         "MS": res.ms_within, "F": np.nan, "p": np.nan},
        {"Source": "Total", "SS": res.ss_total, "df": res.df_between + res.df_within,
         "MS": np.nan, "F": np.nan, "p": np.nan},
    ])
    st.dataframe(anova_tbl.round(4), width="stretch", hide_index=True)

    means_tbl = pd.DataFrame([
        {"Group": lab, "n": res.group_ns[lab], "mean": round(res.group_means[lab], 4)}
        for lab in labels])
    with st.expander("Group means"):
        st.dataframe(means_tbl, width="stretch", hide_index=True)

    st.subheader("Assumption checks")
    lev = AN.levene_test(groups)
    sha = AN.shapiro_test(groups)
    ca, cb = st.columns(2)
    with ca:
        ok = lev["equal_variances"]
        st.markdown(f"**Equal variances (Levene)** - p = {lev['p']:.4g}  \n"
                    f"{'assumption met' if ok else 'WARNING: variances differ'}")
    with cb:
        if sha.get("normal") is None:
            st.markdown("**Normality (Shapiro-Wilk)** - not enough data to test")
        else:
            okn = sha["normal"]
            st.markdown(f"**Normal residuals (Shapiro-Wilk)** - p = {sha['p']:.4g}  \n"
                        f"{'assumption met' if okn else 'WARNING: non-normal residuals'}")
    assumptions_ok = lev["equal_variances"] and (sha.get("normal") in (True, None))
    if not assumptions_ok:
        st.warning("One or more ANOVA assumptions look violated. Consider the "
                   "non-parametric Kruskal-Wallis test below.")

    st.subheader("Post-hoc: Tukey HSD (parametric)")
    st.caption("Which specific groups differ, controlling for multiple comparisons.")
    try:
        tuk = pd.DataFrame(AN.tukey_hsd(groups, labels))
        tuk = tuk.rename(columns={"meandiff": "mean diff", "p_adj": "p (adj)",
                                  "ci_low": "CI low", "ci_high": "CI high"})
        st.dataframe(tuk.round(4), width="stretch", hide_index=True)
    except Exception as e:  # noqa: BLE001
        st.info(f"Tukey HSD unavailable for this data ({e}).")

    with st.expander("Non-parametric alternative (Kruskal-Wallis + Dunn)",
                     expanded=not assumptions_ok):
        kw = AN.kruskal_wallis(groups, labels)
        st.markdown(f"**Kruskal-Wallis**: H({kw['df']}) = {kw['H']:.3f}, "
                    f"p = {kw['p']:.4g}, epsilon squared = {kw['epsilon_sq']:.3f} - "
                    f"{'**significant**' if kw['p'] < alpha else 'not significant'} "
                    f"at alpha = {alpha}.")
        st.caption("Ranks-based; makes no normality/equal-variance assumption.")
        dunn = pd.DataFrame(AN.dunn_test(groups, labels))
        dunn = dunn.rename(columns={"p_adj": "p (adj, Bonferroni)"})
        st.markdown("**Dunn's post-hoc:**")
        st.dataframe(dunn.round(4), width="stretch", hide_index=True)

    with st.expander("About the tests"):
        st.markdown(
            "- One-way ANOVA tests whether group means differ, assuming roughly "
            "normal residuals and similar variances.\n"
            "- eta squared is the proportion of variance explained by the "
            "grouping.\n"
            "- Tukey HSD gives adjusted p-values for every pair of groups.\n"
            "- Levene checks equal variances; Shapiro-Wilk checks normality of "
            "residuals.\n"
            "- If assumptions fail, Kruskal-Wallis (with Dunn's post-hoc) compares "
            "groups using ranks instead of raw values."
        )
