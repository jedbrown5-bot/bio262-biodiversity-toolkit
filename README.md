# 🌿 BIO262 Biodiversity Toolkit

A small Streamlit teaching hub with three tools, reached from a welcome screen
and a sidebar:

1. **Rarefaction** — individual- and sample-based rarefaction & extrapolation
   with confidence intervals, reproducing the core methods of **EstimateS**
   (Colwell, Mao & Chang 2004; Colwell et al. 2012).
2. **Diversity** — **Shannon (H′)** and **Simpson** diversity indices from
   either **individual counts** or **percent-cover** data, for one sample or
   many sites side-by-side.
3. **One-way ANOVA** — F-test with a full SS/df/MS/F/p table, **Tukey HSD**
   post-hoc, **Levene** + **Shapiro–Wilk** assumption checks, a boxplot, and a
   non-parametric **Kruskal–Wallis + Dunn's** fallback.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501). If
`streamlit` isn't on your PATH, use `python -m streamlit run app.py`.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Entry point — welcome screen + navigation |
| `toolkit.py` | The three page implementations (UI) |
| `rarefaction.py` | Rarefaction / extrapolation / Chao maths |
| `diversity.py` | Shannon & Simpson indices |
| `anova.py` | ANOVA, Tukey, Levene, Shapiro, Kruskal–Wallis, Dunn |
| `verify.py` | Validation of rarefaction vs Monte-Carlo |
| `requirements.txt` | Python dependencies |

## Input formats

**Rarefaction (individual-based)** — a vector of species abundances:
```
40 20 15 12 10 8 6 5 4 3 3 2 2 2 1 1 1 1 1 1
```

**Rarefaction (sample-based) & Diversity** — a species × samples table (one row
per species, one column per sample/site); values may be counts, cover, or 0/1.

**ANOVA** — either *wide* (one column per group, header = group names) or *long*
(a value column and a group column). Paste or upload CSV/TSV.

## Methods & citation

Clean-room implementations of published estimators; not affiliated with the
EstimateS source code. Please cite:

> Colwell, R. K. 2013. *EstimateS*, Version 9. http://purl.oclc.org/estimates

> Colwell, R. K., et al. 2012. Models and estimators linking individual-based
> and sample-based rarefaction, extrapolation and comparison of assemblages.
> *Journal of Plant Ecology* 5:3–21.

> Magurran, A. E. 2004. *Measuring Biological Diversity.* Blackwell.  ·
> Jost, L. 2006. Entropy and diversity. *Oikos* 113:363–375.
