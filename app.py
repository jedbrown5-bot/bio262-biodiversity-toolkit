"""
BIO262 Biodiversity Toolkit
===========================
Entry point. Run with:  streamlit run app.py

Three tools (see toolkit.py for the page implementations):
  🌿 Rarefaction   -- rarefaction & extrapolation (EstimateS-style)
  📊 Diversity     -- Shannon & Simpson diversity (counts or % cover)
  🧪 One-way ANOVA -- ANOVA + Tukey HSD, assumption checks, non-parametric fallback
"""

from __future__ import annotations

import streamlit as st

import toolkit

st.set_page_config(page_title="BIO262 Biodiversity Toolkit", page_icon="🌿",
                   layout="wide")

rare_pg = st.Page(toolkit.render_rarefaction, title="Rarefaction", icon="🌿",
                  url_path="rarefaction")
div_pg = st.Page(toolkit.render_diversity, title="Diversity", icon="📊",
                 url_path="diversity")
anova_pg = st.Page(toolkit.render_anova, title="One-way ANOVA", icon="🧪",
                   url_path="anova")
home_pg = st.Page(lambda: toolkit.render_home(rare_pg, div_pg, anova_pg),
                  title="Home", icon="🏠", default=True)

pg = st.navigation({"BIO262 Toolkit": [home_pg],
                    "Tools": [rare_pg, div_pg, anova_pg]})
pg.run()
