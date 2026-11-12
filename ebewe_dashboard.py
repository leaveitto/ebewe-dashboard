"""
EBEWE Program — Building Compliance Analytics
City of Los Angeles, Department of Building and Safety

Three phases over one dataset. Preliminary describes what the file contains and
what is wrong with it; Midterm diagnoses what the compliance label actually
measures; Final fits a model that predicts non-compliance and ranks filings for
review. The cleaning pipeline mirrors EBEWE_Analysis.ipynb (Sections 3-5
cleaning, Section 6 descriptive statistics, Section 7 figures, Section 9 model).

Every statistic, figure and model on every page is computed from the loaded CSV
at run time. Nothing is precomputed or hardcoded, so a refreshed file flows
through the whole application — including the model — without edits here.

Run locally:   streamlit run ebewe_dashboard.py
"""

import io
import glob
import os
from contextlib import contextmanager

import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="EBEWE Descriptive Dashboard",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Palette — one meaning per hue
# ---------------------------------------------------------------------------
# An earlier version used RED for eight unrelated things: NOT COMPLIED, high
# missingness, the incompleteness series, the "Mean" bar, flagged outliers,
# below-threshold property types, the reference line itself, and below-baseline
# agents. A reader cannot learn what a colour means if it means eight things.
#
# Each token below has exactly one job. CIVIC and OCHRE are the encoding pair and
# are distinguishable under deuteranopia and protanopia, which red/green is not —
# and red/green would also import a moral reading ("green = good") into a chart
# where the honest finding is that compliance measures filing completeness rather
# than performance.
# The tokens come in two sets. An earlier version hardcoded the light one, which
# forced a pale background and dark text no matter what theme was selected — so
# choosing Dark produced Streamlit's dark chrome with this stylesheet's light
# ground underneath it, and the sidebar went unreadable. The theme is read from
# Streamlit and the matching set is used.
#
# Only the neutrals flip. CIVIC and OCHRE keep their meaning in both themes and are
# lightened on dark so they hold contrast against a dark ground rather than sinking
# into it. Both remain distinguishable under deuteranopia and protanopia, which
# red/green is not — and red/green would import a moral reading ("green = good")
# into charts whose honest finding is that compliance measures filing completeness
# rather than performance.
try:
    _THEME = st.context.theme.type or "light"
except Exception:          # older Streamlit, or no theme information available
    _THEME = "light"
IS_DARK = _THEME == "dark"

# One hue family, not two. The neutrals were cool blue-greys (hue 216-220 degrees)
# while the accent is warm gold (42 degrees), so any chart putting a neutral series
# beside a gold one showed two unrelated palettes. The neutrals below are the same
# lightnesses pulled into the gold family at very low saturation (32-43 degrees), so
# grey-for-context and gold-for-focus read as one system.
if IS_DARK:
    INK = "#F0EDE8"        # text, axes, reference lines
    SLATE = "#A49B91"      # secondary text
    CONTEXT = "#7E766D"    # any non-focal series; the measured quantity, COMPLIED
    OCHRE = "#E8B74F"      # the focal state: NOT COMPLIED, incomplete, below threshold
    YELLOW = "#FFC93C"     # interface state only — selected, buttons, pills. Never a mark.
    # Validated dark: monotone lightness, adjacent dL >= 0.06, light end 2.07:1, 8 deg spread.
    TIER_SEQUENCE = ["#F0DCA8", "#D0A338", "#B4871F", "#8C6610", "#64490B"]
else:
    INK = "#23201C"
    SLATE = "#7A736A"
    CONTEXT = "#474038"
    OCHRE = "#B07D0A"
    # Two yellows, because one cannot do both jobs. #FFC93C is what makes the interface
    # feel like itself, and it works on buttons and selected rows because they are large
    # filled shapes. As a chart line on white it measures 1.54:1 and disappears. OCHRE is
    # the same hue darkened to 3.63:1, so a line or a small marker still reads.
    YELLOW = "#FFC93C"
    # Ordered steps for charts needing more than two categories (the per-tier series).
    # The previous ramp ended at #E9CE8C, which measures 1.50:1 against the page and was
    # effectively invisible. This one is validated: monotone lightness, adjacent dL
    # >= 0.06, light end 2.28:1, hue spread 4 degrees.
    TIER_SEQUENCE = ["#3A2A06", "#64490B", "#8C6610", "#B4871F", "#D0A338"]

# Chart marks are one hue, separated by lightness only. Encoding two series as a
# neutral against a gold made the neutral look like a second palette; here the darker
# step simply reads as the one being pointed at. Each triple is validated: CVD
# separation and normal-vision dE above the floor, and every step at or above 3:1
# against its own surface, so a mark still reads without relying on its value label.
if IS_DARK:
    MARK_DEEP = "#F2E0B0"       # the series carrying the finding
    MARK = "#D8AC45"            # a single series with nothing to contrast against
    MARK_SOFT = "#A87C1A"       # context: the series the reader is not asked to look at
else:
    # #2E2206 cleared 3:1 at every step but sat at 12% lightness, which reads as
    # charcoal rather than as a dark gold. These three are unmistakably one hue.
    # The trade: the light step is 2.75:1, under the 3:1 floor, which the validator
    # permits only with relief -- a visible value label or legend entry. Every chart
    # using MARK_SOFT has one, so the condition is met rather than waived.
    MARK_DEEP = "#564005"
    MARK = "#8A6610"
    MARK_SOFT = "#C0931F"

# A correlation runs -1 to +1, so it needs a diverging scale rather than the ordered
# ramp. Built from the same gold as every other mark, through a near-white midpoint,
# so a named Plotly scale cannot reintroduce a second hue family.
CORR_SCALE = [
    [0.00, MARK_DEEP], [0.25, MARK], [0.45, "#E8DCC4"],
    [0.50, "#F6F5F3"],
    [0.55, "#E8DCC4"], [0.75, MARK], [1.00, MARK_DEEP],
]

# INK and SLATE are text, axes and reference lines only. They are never a mark; that
# is what MARK_* is for, and mixing the two is how the neutral/gold clash started.
CIVIC = CONTEXT
BLUE = MARK
RED = MARK_DEEP
DARK_BLUE = MARK_DEEP

# ----------------------------------------------------------------------------
# Typography and page styling
# ----------------------------------------------------------------------------
# Source Serif for headings puts this in a document register rather than a
# product one — it is a written analysis with figures, not an operations console.
# IBM Plex Sans carries the data; its tabular figures keep decimal points aligned
# in the metric rows, which Streamlit's default face does not.
#
# Mono is kept for dataset field names only. That is a semantic distinction the
# analysis depends on: `propertyType` is a column in the file, "property type" is
# the concept. Losing it would flatten a difference the prose relies on.
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
    <style>
      /* NOTE: no blank lines in this stylesheet. Streamlit renders this through a
         markdown parser first, and a blank line followed by indented text is markdown's
         code-block syntax — it closes the <style> element and prints the remaining CSS
         onto the page as body text. Keep every rule on a contiguous run of lines. */
      /* This sheet sets no background and no text colour. That is deliberate. An earlier
         version pinned both to a light palette, so choosing Dark gave Streamlit's dark
         chrome over a forced light ground and an unreadable sidebar. Branching in Python
         on st.context.theme does not fix it either: Streamlit documents that value as
         possibly incorrect at the moment the user changes theme, and a Python branch only
         takes effect on the next rerun.
         Inheriting instead means the browser repaints the moment the toggle is used, with
         no rerun and nothing to keep in sync. Every surface below is derived from the
         inherited text colour with color-mix, so it tracks whichever theme is active. */
      html, body, .stApp, .stMarkdown, .stCaption, .stDataFrame,
      [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
      .stTabs [data-testid="stTab"], [data-testid="stWidgetLabel"] {
        font-family: "IBM Plex Sans", system-ui, sans-serif;
      }
      /* Icons keep their own font whatever else is set. Restored explicitly so a future
         change to the rule above cannot silently break them again. */
      [data-testid="stIconMaterial"], .material-icons, .material-icons-outlined,
      [class*="material-symbols"], [class*="material-icons"], span[data-testid*="Icon"] {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined",
                     "Material Icons" !important;
      }
      h1, h2, h3, h4 {
        font-family: "Source Serif 4", Georgia, serif;
        font-weight: 600; letter-spacing: -0.01em;
      }
      h1 { font-size: 2.1rem; line-height: 1.15; }
      h2 { font-size: 1.45rem; margin-top: 0.4rem; }
      h3 { font-size: 1.15rem; }
      /* Measure by role, not one rule for everything. Narrative prose keeps a reading
         measure; captions sit under the thing they describe and take its width; callouts
         run wider than prose but not edge to edge. */
      .stMarkdown p { max-width: none; line-height: 1.55; }
      .stAlert p { max-width: none; line-height: 1.5; }
      [data-testid="stCaptionContainer"] p,
      [data-testid="stCaptionContainer"] { max-width: none; line-height: 1.5; }
      [data-testid="stCaptionContainer"] { opacity: 0.72; font-size: 0.86rem; }
      code, .stMarkdown code {
        font-family: "IBM Plex Mono", monospace; font-size: 0.86em;
        background: color-mix(in srgb, currentColor 8%, transparent);
        padding: 0.08em 0.32em; border-radius: 2px;
      }
      [data-testid="stMetricValue"] { font-weight: 600; font-variant-numeric: tabular-nums; }
      [data-testid="stMetricLabel"] { opacity: 0.72; font-size: 0.82rem; }
      /* Callouts read as margin notes rather than product notifications: a rule on the
         leading edge, no fill, no shadow. The rule is mixed from the inherited text
         colour so it stays visible on either ground. */
      .stAlert {
        background: transparent !important; border: 0 !important;
        border-left: 2px solid color-mix(in srgb, currentColor 22%, transparent) !important;
        border-radius: 0 !important; padding-left: 0.9rem !important;
        box-shadow: none !important;
      }
      hr { border-color: color-mix(in srgb, currentColor 15%, transparent); }
      [data-testid="stSidebar"] {
        border-right: 1px solid color-mix(in srgb, currentColor 15%, transparent);
      }
      .stTabs [data-testid="stTab"] { font-size: 0.94rem; }
      /* Print. These pages get exported to PDF as the submitted artefact, so print is a
         real output rather than an afterthought. Side-by-side columns kept their screen
         widths on paper, which cut the Overview donut off at the page edge; stacking them
         lets every section print the way the single-column tabs already did. Blocks were
         also free to split across a page boundary, which dropped the "Values that cannot
         be company names" table on top of the heading beneath it. */
      @media print {
        [data-testid="stHorizontalBlock"] { display: block !important; }
        [data-testid="stColumn"] {
          width: 100% !important; flex: none !important;
          min-width: 100% !important; margin-bottom: 1rem;
        }
        [data-testid="stPlotlyChart"], .stPlotlyChart, .js-plotly-plot,
        [data-testid="stDataFrame"], .stAlert, [data-testid="stMetric"] {
          break-inside: avoid; page-break-inside: avoid;
        }
        h1, h2, h3, h4 { break-after: avoid; page-break-after: avoid; }
        .stMarkdown p, .stAlert p { max-width: none; }
        [data-testid="stSidebar"] { break-before: page; }
        /* Plotly renders a fixed-width SVG sized to the browser viewport. On paper that
           width is not reduced, so wide charts and tables ran off the right edge — whole
           bars lost, not just their labels. Constraining both to the page is what actually
           fixes the exported PDFs; the axis headroom on individual figures never was the
           cause. */
        .js-plotly-plot, .js-plotly-plot .plotly, .js-plotly-plot .svg-container,
        .js-plotly-plot svg, [data-testid="stPlotlyChart"] {
          max-width: 100% !important; width: 100% !important;
        }
        [data-testid="stDataFrame"], [data-testid="stDataFrame"] > div {
          max-width: 100% !important; width: 100% !important; overflow: visible !important;
        }
      }
      </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Ground and interface — light mode only
# ----------------------------------------------------------------------------
# A grey page with white cards floating on it. The separation is what lets a card be a
# unit: the border does almost nothing and the ground does the work. Light mode only —
# the dark tokens above handle the other case, and pinning colours unconditionally is
# what broke dark mode in an earlier version.
#
# Same no-blank-lines rule as the sheet above: a blank line followed by indented text is
# markdown's code-block syntax and would close the <style> element early.
if not IS_DARK:
    st.markdown(
        """
        <style>
          .stApp { background-color: #F6F5F3; }
          /* The rail is a stack of cards sitting on the page colour, as in the
             structure mockup, rather than one flat white column. The groups are
             found by what they contain, so no extra wrappers are needed. */
          section[data-testid="stSidebar"] { background-color: #F6F5F3;
            border-right: 1px solid #EBE8E5; }
          section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            padding: 1rem 0.85rem 2rem; display: flex; flex-direction: column; gap: 0.7rem; }
          /* Streamlit puts an unnamed wrapper between stSidebarUserContent and the
             block, so the rail's groups are reached through it, not as its children.
             Verified against the live DOM rather than assumed. */
          section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]
            [data-testid="stVerticalBlock"] { gap: 0.7rem; }
          /* Groups are marked in Python with railmark, the same trick card() uses in
             the main area: a blanket rule on every element container turned eleven
             separate widgets into eleven cards. */
          section[data-testid="stSidebar"]
            [data-testid="stElementContainer"]:has(.railmark) { display: none; }
          section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(
            > [data-testid="stElementContainer"] .railmark) {
            background: #FFFFFF; border: 1px solid #EBE8E5; border-radius: 14px;
            padding: 0.85rem 0.9rem; gap: 0.5rem; }
          /* the phase nav supplies its own card shapes, so it keeps the page colour */
          section[data-testid="stSidebar"] .st-key-phase { background: transparent;
            border: 0 !important; padding: 0 !important; }
          section[data-testid="stSidebar"] hr { display: none; }
          [data-testid="stElementContainer"]:has(> div > div > .cardmark) { display: none; }
          [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .cardmark) {
            background: #FFFFFF; border: 1px solid #EBE8E5 !important; border-radius: 16px;
            padding: 1.35rem 1.5rem 1.45rem; box-shadow: 0 1px 2px rgba(31,35,41,.05);
            display: flex; flex-direction: column; }
          /* Panels size to their content. Stretching every card in a row to the
             tallest one turns the difference into a band of empty card, which is
             the clutter: space spent on nothing. Tops align, bottoms are ragged,
             and every pixel inside a panel is content. */
          [data-testid="stHorizontalBlock"] { align-items: flex-start; }
          [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .cardmark)
            > [data-testid="stElementContainer"]:has([data-testid="stButton"]) {
            margin-top: auto; padding-top: 0.6rem; }
          [data-testid="stButton"] button { width: 100%; }
          [data-testid="stMain"] [data-testid="stBaseButton-secondary"] {
            background: #FFC93C; border: none; color: #23201C; font-weight: 600;
            border-radius: 10px; }
          [data-testid="stMain"] [data-testid="stBaseButton-secondary"]:hover {
            background: #F0B82C; color: #23201C; }
          [data-testid="stMetric"] { background: #F6F5F3; border-radius: 12px;
            padding: 0.7rem 0.9rem 0.8rem; }
          [data-testid="stDataFrame"] { border-radius: 12px; border: 1px solid #EBE8E5; }
          hr { border-color: #EBE8E5 !important; margin: 0.9rem 0 !important; }
          [data-testid="stTabs"] { background: #FFFFFF; border: 1px solid #EBE8E5;
            border-radius: 16px; padding: 1.3rem 1.5rem 1.6rem;
            box-shadow: 0 1px 2px rgba(31,35,41,.05); }
          [data-testid="stTabs"] [data-testid="stMetric"] { background: #F6F5F3; }
          /* Pills, as in the structure mockup. An underline strip reads as browser
             chrome; these read as a control belonging to the page.
             This Streamlit build emits data-testid="stTab" with role="tab" and a
             .react-aria-SelectionIndicator child -- there is no data-baseweb="tab"
             here at all, so selectors written against that name match nothing.
             Checked against the live DOM. */
          /* Folder tabs. The strip carries one continuous edge the full width of the
             panel; each tab is rounded on its top corners only and sits on that edge,
             and the selected one overlaps it by a pixel so the line breaks where the
             open folder is. The loose rule that used to float under the pills is that
             edge now, so it reads as structure rather than as a stray border. */
          .stTabs [role="tablist"] { gap: 0.25rem !important; flex-wrap: wrap;
                   width: 100%; padding: 0 !important; align-items: flex-end;
                   border-bottom: 1px solid #E2DDD5; }
          .stTabs .react-aria-SelectionIndicator { display: none !important; }
          /* A transparent bottom border still has the tab's own background painted
             behind it, so the strip's edge was being erased under every closed tab
             and only reappeared past the last one. Closed tabs carry the edge colour
             themselves; only the open one breaks it. */
          .stTabs [data-testid="stTab"] { padding: 0.42rem 1rem 0.46rem !important;
                   background: #F1EEE9; border: 1px solid #E2DDD5;
                   border-bottom: 1px solid #E2DDD5;
                   border-radius: 10px 10px 0 0; margin-bottom: -1px;
                   height: auto !important; transition: background .12s ease; }
          .stTabs [data-testid="stTab"] p { font-size: 0.8rem !important;
                   font-weight: 600; color: #7A736A; }
          .stTabs [data-testid="stTab"]:hover { background: #E8E2D8; }
          .stTabs [data-testid="stTab"]:hover p { color: #23201C; }
          /* the open folder: yellow, one step taller, and it cuts the edge */
          .stTabs [data-testid="stTab"][aria-selected="true"] { background: #FFC93C;
                   border-color: #F0B82C; border-bottom-color: #FFC93C;
                   padding-top: 0.56rem !important; }
          .stTabs [data-testid="stTab"][aria-selected="true"] p { color: #23201C; }
          .st-key-phase, .st-key-phase [data-testid="stRadio"] { width: 100%; }
          .st-key-phase [data-testid="stRadioGroup"] { gap: 0.4rem; width: 100%;
            align-items: stretch; }
          .st-key-phase [data-testid="stRadioGroup"] > * { background: #F6F5F3;
            border-radius: 12px; padding: 0.55rem 0.75rem 0.6rem; margin: 0; width: 100%;
            box-sizing: border-box; transition: background 120ms ease; }
          .st-key-phase [data-testid="stRadioGroup"] > *:hover { background: #F1EFEC; }
          .st-key-phase [data-testid="stRadioGroup"] > *:has(input:checked) {
            background: #FFC93C; }
          .st-key-phase [data-testid="stRadioGroup"] > *:active { transform: scale(0.985); }
          .st-key-phase [data-testid="stRadioOption"] { margin: 0; cursor: pointer; }
          .st-key-phase [data-testid="stRadioOption"] > div > div:not([data-testid]) {
            display: none; }
          .st-key-phase [data-testid="stRadioOption"] > div { gap: 0; }
          .st-key-phase [data-testid="stRadioCaption"] { font-size: 0.68rem; opacity: 0.72; }
          /* Streamlit's default side padding is 80px each way. On a 1200px main area
             that is 160px -- 13% of the width -- spent before anything renders, and the
             cards inside add their own. 32px is enough to keep content off the chrome. */
          .block-container { padding: 2rem 2rem 3rem !important; max-width: 1500px; }
          @media (max-width: 900px) {
            .block-container { padding: 1.4rem 1rem 2rem !important; }
          }
          [data-testid="stVerticalBlock"] { gap: 0.75rem; }
          [data-testid="stTabs"] [data-testid="stVerticalBlock"] { gap: 1rem; }
          h1 { font-size: 1.6rem !important; }
          h2 { font-size: 1.22rem !important; margin: 0.2rem 0 0.1rem !important; }
          h3 { font-size: 1rem !important; margin: 0.6rem 0 0.1rem !important; }
          /* same trap as the rule below: (0,1,1) beats every single-class size in the
             scale, so it must exclude the classed paragraphs too. */
          .stMarkdown p:not([class]) { font-size: 0.93rem; line-height: 1.5;
                   max-width: none; }
          .stMarkdown p[class] { line-height: 1.45; max-width: none; }
          [data-testid="stMetricValue"] { font-size: 1.55rem; }
          [data-testid="stMetricLabel"] { font-size: 0.72rem !important; opacity: 0.68;
            text-transform: uppercase; letter-spacing: 0.055em; }
          [data-testid="stAlertContainer"] { border-radius: 12px; border: 1px solid #EBE8E5;
            border-left: 4px solid #B07D0A; background: #FFFFFF; }
          [data-testid="stAlertContainer"] p { font-size: 0.9rem; }
          .statband { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
            background: #EBE8E5; border: 1px solid #EBE8E5; border-radius: 14px;
            overflow: hidden; }
          .statband .bcell { background: #FFFFFF; padding: 1rem 1.15rem 1.05rem; }
          .statband .bcell.hi { background: #FDF4E0; }
          .stMain .statband .bcell span { display: block; font-size: 0.66rem; letter-spacing: 0.06em;
            text-transform: uppercase; color: #7A736A; margin-bottom: 0.15rem; }
          .stMain .statband .bcell b { font-size: 1.22rem; font-weight: 700;
                   letter-spacing: -0.025em; }
          .pullq { background: #FDF4E0; border-left: 4px solid #B07D0A;
            border-radius: 0 14px 14px 0; padding: 1.05rem 1.35rem;
            margin: 0.35rem 0 0.2rem; }
          .srule { display: flex; gap: 0.85rem; align-items: baseline;
                   border-top: 2px solid #23201C; padding-top: 0.7rem;
                   margin: 2.9rem 0 1.5rem; }
          /* the first rule in a tab should not sit hard against the strip */
          [data-testid="stTabPanel"] [data-testid="stElementContainer"]:first-child
            .srule { margin-top: 1.3rem; }
          .stMain .srule .snum { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em;
                   color: #B07D0A; font-variant-numeric: tabular-nums; }
          .stMain .srule h3 { margin: 0 !important; font-size: 1.08rem !important;
                   font-weight: 700; letter-spacing: -0.015em; padding: 0 !important; }
          .stMain .srule p { margin: 0.2rem 0 0; font-size: 0.84rem; color: #7A736A; }
          /* One type scale, applied everywhere. Before this, st.metric, st.caption,
             st.info and the custom helpers each set their own size, so nine text
             treatments sat within a few tenths of a rem of each other and nothing
             read as more important than anything else. */
          [data-testid="stMetricValue"] { font-size: 1.22rem !important;
                   letter-spacing: -0.025em; font-weight: 700 !important; }
          [data-testid="stMetricLabel"] p { font-size: 0.66rem !important;
                   letter-spacing: 0.06em; text-transform: uppercase; font-weight: 600; }
          [data-testid="stMetricDelta"] { font-size: 0.72rem !important; }
          .stAlert p, .stAlert li { font-size: 0.84rem !important; line-height: 1.5; }
          [data-testid="stCaptionContainer"] p { font-size: 0.78rem !important; }
          /* :not([class]) matters. This rule is (0,2,1); .claim, .sublead, .pullq p,
             .srule p and p.lead are all (0,1,0) or (0,1,1), so without it this one
             silently won and every custom size in the scale was being ignored. Plain
             markdown paragraphs carry no class; the custom ones all do. */
          .stMain .stMarkdown p:not([class]), .stMain .stMarkdown li { font-size: 0.88rem; }
          .stMain [data-testid="stExpander"] summary p { font-size: 0.75rem !important;
                   letter-spacing: 0.05em; text-transform: uppercase; font-weight: 600;
                   color: #7A736A; }
          .stMain [data-testid="stExpander"] details { border: 0; border-radius: 0;
                   background: transparent; }
          .stMain [data-testid="stExpander"] summary { border-top: 1px dashed #DFDCD9;
                   padding-top: 0.4rem; }
          /* a caption immediately followed by a disclosure reads as one unit */
          .stMain [data-testid="stElementContainer"]:has(> [data-testid="stCaptionContainer"])
            + [data-testid="stElementContainer"]:has([data-testid="stExpander"]) {
            margin-top: -0.35rem; }
          .stMain [data-testid="stExpander"] summary { padding-left: 0; }
          .stMain [data-testid="stExpander"] summary svg { width: 0.8rem; height: 0.8rem; }
          /* kv rows carry an inline bar: label, track, value. The track column is
             what makes three numbers legible as a comparison without a chart. */
          .kv { display: flex; flex-direction: column; gap: 0.56rem;
                   margin: 0.62rem 0 0.55rem; }
          .stMain .kv .r { display: grid; grid-template-columns: 8.6em 1fr 3.9em; gap: 0.5rem;
                   align-items: center; font-size: 0.72rem; color: #7A736A; }
          .kv .r > span:first-child { overflow: hidden; text-overflow: ellipsis;
                   white-space: nowrap; }
          .kv .r.norail { grid-template-columns: 1fr auto; }
          .kv .t { display: block; background: #EBE8E5; border-radius: 4px; height: 9px;
                   overflow: hidden; }
          .kv .f { display: block; height: 100%; border-radius: 4px; background: #8A6610; }
          .kv .f.mute { background: #D7C9AE; }
          .stMain .kv .v { text-align: right; font-variant-numeric: tabular-nums; color: #23201C;
                   font-size: 0.78rem; font-weight: 700; letter-spacing: -0.015em;
                   white-space: nowrap; }
          table.mini { width: 100%; border-collapse: collapse; font-size: 0.76rem;
                   margin: 0.7rem 0 0.4rem; }
          .miniwrap { width: 100%; overflow-x: auto; }
          /* A panel header: the section rule's information, inside the card. */
          .stMain dl.defs { display: grid; grid-template-columns: auto 1fr;
                   gap: 0.28rem 0.9rem; font-size: 0.76rem; margin: 0.4rem 0 0.2rem; }
          .stMain dl.defs dt { color: #7A736A; }
          .stMain dl.defs dd { margin: 0; overflow-wrap: anywhere; min-width: 0; }
          .stMain dl.defs dd code { white-space: normal; overflow-wrap: anywhere;
                   display: inline-block; max-width: 100%; }
          .stMain .ph { display: flex; align-items: baseline; gap: 0.5rem;
                   margin: 0 0 0.1rem; }
          .stMain .ph em { font-style: normal; font-size: 0.62rem; font-weight: 700;
                   letter-spacing: 0.08em; color: #8A6610; font-variant-numeric: tabular-nums; }
          .stMain .ph h4 { margin: 0 !important; font-size: 0.9rem !important;
                   font-weight: 700; letter-spacing: -0.012em; padding: 0 !important; }
          .stMain .ph .pnote { margin-left: auto; font-size: 0.66rem; color: #7A736A;
                   white-space: nowrap; }
          /* A long table scrolls in place rather than lengthening the page. */
          .stMain .scrollbox { width: 100%; overflow: auto; border: 1px solid #EBE8E5;
                   border-radius: 8px; margin: 0.55rem 0 0.2rem;
                   max-height: var(--sb, 220px); }

          .stMain .scrollbox table.mini { margin: 0; }
          .stMain table.mini.sticky thead th { position: sticky; top: 0; z-index: 1;
                   background: #FFFFFF; box-shadow: inset 0 -1px 0 #EBE8E5; }
          /* In a scrolling table the text columns wrap so the numeric ones are not
             pushed out of the panel; nowrap stays on the numbers. */
          .stMain table.mini.sticky td:not(.n) { white-space: normal; }
          .stMain table.mini.sticky td:first-child { white-space: nowrap; }
          .stMain table.mini th, table.mini td { padding: 0.44rem 0.62rem; text-align: left;
                   border-bottom: 1px solid #EBE8E5; white-space: nowrap; }
          .stMain table.mini th { color: #7A736A; font-weight: 500; font-size: 0.66rem;
                   text-transform: uppercase; letter-spacing: 0.05em; }
          table.mini td.n, table.mini th.n { text-align: right;
                   font-variant-numeric: tabular-nums; }
          table.mini tr.focus td { background: #FDF4E0; font-weight: 600; color: #23201C; }
          .stMain table.mini code { font-size: 0.72rem; }
          .dbar { display: inline-block; width: 46px; height: 8px; border-radius: 4px;
                   background: #EBE8E5; overflow: hidden; vertical-align: middle;
                   margin-right: 0.4rem; }
          .dbar > span { display: block; height: 100%; background: #8A6610;
                   border-radius: 4px; }
          .dnum { font-variant-numeric: tabular-nums; }
          table.mini tbody tr:last-child td { border-bottom: none; }
          .kv-legacy { margin-bottom: 0.55rem; }
          .stMain .phasecard h4 { margin: 0 !important; font-size: 1.02rem !important;
                   font-weight: 700; letter-spacing: -0.015em; padding: 0 !important; }
          .stMain .phasecard p { margin: 0.25rem 0 0; font-size: 0.82rem; color: #7A736A; }
          .stMain .pullq > b { display: block; font-size: 0.95rem; margin-bottom: 0.15rem; }
          .stMain .pullq p b { font-weight: 700; color: #23201C; }
          .stMain .pullq p { margin: 0; font-size: 0.88rem; color: #61584D; }
          .stMain .claim { font-size: 1.85rem; font-weight: 700; letter-spacing: -0.03em;
            line-height: 1.2; margin: 0.35rem 0 0.5rem; }
          .stMain .claim em { font-style: normal; color: #B07D0A; }
          .stMain .eyebrow { font-size: 0.66rem; letter-spacing: 0.11em; text-transform: uppercase;
            color: #7A736A; font-weight: 600; }
          .stMain .sublead { font-size: 0.88rem; color: #61584D; margin: 0; }
          .pagehead h2 { margin: 0.1rem 0 0.15rem !important; font-size: 1.7rem !important;
            letter-spacing: -0.025em; }
          /* the lead sits with the title rather than floating at the top of its half */
          .stMain [data-testid="stHorizontalBlock"]:has(.pagehead) { align-items: center; }
          .pagehead .standing { font-size: 0.8rem; opacity: 0.6; margin: 0 0 0.5rem; }
          .stMain p.lead { font-size: 0.95rem; line-height: 1.55; max-width: none; margin: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Chart defaults, so every figure inherits the same face and axis weight rather
# than each one re-specifying it.
PLOT_FONT = {"family": "IBM Plex Sans, system-ui, sans-serif", "size": 12, "color": INK}
# Gridlines and axis rules are semi-transparent mid-greys rather than fixed hexes, so
# they read against either ground without depending on the theme value being current.
# Plotly cannot inherit CSS, so figures are the one place the theme has to be resolved
# in Python — and st.context.theme is documented as possibly stale at the moment of a
# toggle. Keeping the structural furniture theme-agnostic means a stale value can at
# worst tint the text, never leave a chart with invisible axes.
# One step off the surface, hairline, solid. Dashed grid reads as a threshold when
# it is only a grid, and a heavy axis rule competes with the marks.
GRID_RGBA = "rgba(128,120,110,0.16)"
AXIS_RGBA = "rgba(128,120,110,0.28)"
SURFACE = "#FFFFFF"      # the card the figures sit on; the colour of the mark gaps

PLOT_LAYOUT = {
    "font": PLOT_FONT,
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    # No axis rules. The gridlines already establish the plane, and a line on top of
    # them is a second statement of the same thing.
    "xaxis": {"gridcolor": GRID_RGBA, "zerolinecolor": AXIS_RGBA, "zerolinewidth": 1,
              "linecolor": "rgba(0,0,0,0)", "ticks": "",
              "tickfont": {"size": 11, "color": SLATE},
              "title": {"font": {"size": 11, "color": SLATE}, "standoff": 12}},
    "yaxis": {"gridcolor": GRID_RGBA, "zerolinecolor": AXIS_RGBA, "zerolinewidth": 1,
              "linecolor": "rgba(0,0,0,0)", "ticks": "",
              "tickfont": {"size": 11, "color": SLATE},
              "title": {"font": {"size": 11, "color": SLATE}, "standoff": 12}},
    "title_font": {"family": "Source Serif 4, Georgia, serif", "size": 15, "color": INK},
    # A legend below the plot, laid out horizontally, with the swatch carrying
    # identity and the text in an ink token rather than the series colour.
    "legend": {"font": {"size": 11, "color": SLATE}, "orientation": "h",
               "yanchor": "top", "y": -0.16, "x": 0, "xanchor": "left",
               "bgcolor": "rgba(0,0,0,0)", "itemsizing": "constant"},
    # Bars never fill their slot: the leftover band is the air between them.
    "bargap": 0.38,
    "bargroupgap": 0.12,
    "hoverlabel": {"bgcolor": "#FFFFFF", "bordercolor": "rgba(128,120,110,0.28)",
                   "font": {"family": "IBM Plex Sans, system-ui, sans-serif",
                            "size": 12, "color": INK}},
    # No margin here. A fixed top margin clips any figure that has a title, and a
    # fixed bottom one clips a legend placed below the plot. chart() sizes both
    # from what the figure actually carries.
}

def chart(fig, **kwargs):
    """Render a figure with the shared theme applied.

    Every figure goes through here so the type face, grid weight and axis colour
    are set in one place. update_layout merges rather than replaces, so a
    figure's own range, tickangle or reversed axis survives this call.

    The title font is applied only when the figure actually has a title. Setting
    it unconditionally materialises a title object with no text, which Plotly
    renders as the literal word "undefined" above every untitled chart — the
    frequency chart, the missingness bars, Figures 3 to 5 and several others.
    An earlier fix moved from a title dict to title_font and did not help,
    because the cause is the empty title object rather than how it is spelled.
    """
    layout = dict(PLOT_LAYOUT)
    has_title = bool(getattr(fig.layout.title, "text", None))
    title_font = layout.pop("title_font", None)
    if title_font and has_title:
        layout["title_font"] = title_font
    fig.update_layout(**layout)

    # Margins are measured from what the figure carries, not fixed. A title needs
    # room above the plot and a legend below it needs room under; one set of
    # numbers for every figure clips whichever it did not allow for.
    _legend_below = (fig.layout.showlegend is not False
                     and sum(1 for t in fig.data if t.showlegend is not False) > 1)
    # layout.margin is a plotly Margin object, not a mapping: read its sides.
    _m = fig.layout.margin
    fig.update_layout(margin={
        "l": _m.l if _m.l is not None else 8,
        "r": _m.r if _m.r is not None else 16,
        "t": _m.t if _m.t is not None else (46 if has_title else 10),
        "b": _m.b if _m.b is not None else (64 if _legend_below else 12),
    })
    if has_title:
        # Left-aligned to the plot area, sitting clear of it.
        fig.update_layout(title={"x": 0, "xref": "paper", "xanchor": "left",
                                 "y": 1, "yref": "container", "yanchor": "top",
                                 "pad": {"t": 6, "b": 14}})

    # Mark specs, applied centrally so no figure can miss them.
    for tr in fig.data:
        t = tr.type
        if t == "bar":
            # 4px rounded data-end, capped thickness, and a 2px surface gap between
            # touching bars -- the gap separates them, not a stroke around each.
            tr.update(marker_cornerradius=4,
                      marker_line={"width": 2, "color": SURFACE})
            if getattr(tr, "textposition", None) in (None, "auto"):
                tr.textposition = "outside"
            tr.textfont = {"size": 11, "color": INK}
            tr.cliponaxis = False
        elif t in ("scatter", "scattergl"):
            mode = tr.mode or ""
            if "lines" in mode:
                tr.line.width = tr.line.width or 2
                tr.line.shape = tr.line.shape or "linear"
                tr.update(line_simplify=False)
            if "markers" in mode:
                # >=8px with a 2px surface ring so points stay legible where
                # two series cross.
                tr.marker.size = max(tr.marker.size or 0, 8)
                tr.marker.line = {"width": 2, "color": SURFACE}
        elif t == "histogram":
            tr.update(marker_line={"width": 1, "color": SURFACE})

    # A crosshair on anything time-like, per-mark hover elsewhere.
    if fig.layout.hovermode is None:
        has_line = any(getattr(tr, "mode", "") and "lines" in tr.mode for tr in fig.data)
        fig.update_layout(hovermode="x unified" if has_line else "closest")
        if has_line:
            fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1,
                             spikedash="solid", spikecolor=AXIS_RGBA)

    kwargs.setdefault("width", "stretch")
    # The modebar floats over the top-right of the plot area on hover and prints
    # into screenshots. Nothing here needs pan, lasso or autoscale.
    kwargs.setdefault("config", {"displayModeBar": False, "staticPlot": False})
    return st.plotly_chart(fig, **kwargs)


# ----------------------------------------------------------------------------
# Pipeline constants — mirrored exactly from the notebook
# ----------------------------------------------------------------------------
RENAME_MAP = {
    "BUILDING ADDRESS": "buildingAddress",
    "BUILDING ID": "buildingId",
    "CARBON DIOXIDE EMISSIONS (Metric Ton CO2e)": "co2Emissions",
    "COMPLIANCE STATUS": "complianceStatus",
    "% DIFFERENCE FROM NATIONAL MEDIAN SOURCE EUI": "pctDiffNationalSourceEui",
    "% DIFFERENCE FROM NATIONAL MEDIAN SITE EUI": "pctDiffNationalSiteEui",
    "ENERGY STAR SCORE": "energyStarScore",
    "ENERGY STAR CERTIFICATION - ELIGIBILITY": "energyStarCertEligibility",
    "ENERGY STAR CERTIFICATION - LAST APPROVAL DATE": "energyStarCertLastApproval",
    "ENERGY STAR CERTIFICATION - YEAR(S) CERTIFIED": "energyStarCertYears",
    "ENTITY RESPONSIBLE FOR BENCHMARK": "entityResponsible",
    "GROSS BUILDING FLOOR AREA (ft²)": "grossFloorArea",
    "INDOOR WATER USE (kgal)": "indoorWaterUse",
    "INDOOR WATER USE INTENSITY (gal/ft²)": "indoorWaterUseIntensity",
    "NUMBER OF BUILDINGS": "numberOfBuildings",
    "OCCUPANCY": "occupancy",
    "OUTDOOR WATER USE (kgal)": "outdoorWaterUse",
    "POSTAL CODE": "postalCode",
    "PROGRAM YEAR": "programYear",
    "PROPERTY TYPE": "propertyType",
    "SITE ENERGY USE INTENSITY (EUI) (kBtu/ft²)": "siteEui",
    "Source EUI (kBtu/ft²)": "sourceEui",
    "TOTAL WATER USE (kgal)": "totalWaterUse",
    "WEATHER NORMALIZED SITE ENERGY USE INTENSITY (EUI) (kBtu/ft²)": "weatherNormSiteEui",
    "WEATHER NORMALIZED SOURCE ENERGY USE INTENSITY (EUI) (kBtu/ft²)": "weatherNormSourceEui",
    "YEAR BUILT": "yearBuilt",
    "AIN": "ain",
    "LADBS Building Category": "ladbsBuildingCategory",
}

NUMERIC_COLS = [
    "co2Emissions", "pctDiffNationalSourceEui", "pctDiffNationalSiteEui",
    "energyStarScore", "grossFloorArea", "indoorWaterUse", "indoorWaterUseIntensity",
    "numberOfBuildings", "occupancy", "outdoorWaterUse", "siteEui", "sourceEui",
    "totalWaterUse", "weatherNormSiteEui", "weatherNormSourceEui", "yearBuilt",
]

STRUCTURAL_COLS = ["propertyType", "yearBuilt", "grossFloorArea", "occupancy",
                   "entityResponsible", "numberOfBuildings"]

# French-locale propertyType labels. Section 4.5 already maps French Non/Oui in
# energyStarCertEligibility; the same submissions carry French property types, which
# split a category across two labels in every groupby. The source encodes the same
# label two ways — a proper "a-grave" and a U+FFFD replacement character — so the map
# is keyed on a form that strips non-alphanumerics rather than on literal strings.
# Explicit map, not a fuzzy rule: only these three labels are touched. Mirrors §4.6.
PROPERTY_TYPE_FR = {
    "IMMEUBLE LOGEMENTS MULTIPLES": "MULTIFAMILY HOUSING",
    "BUREAU": "OFFICE",
    "STATIONNEMENT": "PARKING",
}

PLAUSIBILITY_BOUNDS = {
    "siteEui": {"floor": 0, "ceiling": 2000},
    "sourceEui": {"floor": 0, "ceiling": 2000},
    "co2Emissions": {"floor": 0, "ceiling": 50000},
    "grossFloorArea": {"floor": 100, "ceiling": 5_000_000},
}

OUTLIER_COLS = ["co2Emissions", "siteEui", "sourceEui", "grossFloorArea"]

CORR_COLS = ["siteEui", "sourceEui", "co2Emissions", "ghgIntensityPer1kSqft",
             "energyStarScore", "grossFloorArea", "buildingAge"]

AGE_ORDER = ["1930 or Earlier", "1931-1950", "1951-1970", "1971-1990", "1991-Present"]

MIN_PLAUSIBLE_AREA = 1000
# A property-type rate computed on a handful of filings is noise, not a segment
# finding. Types below this are charted but excluded from spread and flagging.
PHASE_HOME = "Home"
PHASE_PRELIM = "Preliminary"
PHASE_MID = "Midterm"
PHASE_FINAL = "Final"
PHASES = [PHASE_HOME, PHASE_PRELIM, PHASE_MID, PHASE_FINAL]
# The stage rides as a caption rather than inside the label, which gives the nav its
# two-line card without a CSS hack and keeps the label short enough to read as a title.
PHASE_CAPTIONS = ["Overview", "Descriptive", "Diagnostic", "Predictive"]
# Material icons rather than emoji: they inherit the text colour, so they go dark on the
# selected yellow card instead of staying a fixed hue.
PHASE_ICONS = {
    PHASE_HOME: ":material/dashboard:",
    PHASE_PRELIM: ":material/table_chart:",
    PHASE_MID: ":material/troubleshoot:",
    PHASE_FINAL: ":material/insights:",
}

MIN_TYPE_FILINGS = 100
# An operator counts as a driver of a category's deficit only if it is large enough to
# move the category AND materially below baseline. "Materially" is a declared judgement,
# expressed as a share of the category's own deficit so it scales. Mirrors §6.7.3.
MIN_OPERATOR_FILINGS = 100
DEFICIT_SHARE = 0.25
STRUCTURAL_PAIRS = {
    frozenset(["siteEui", "ghgIntensityPer1kSqft"]),
    frozenset(["sourceEui", "ghgIntensityPer1kSqft"]),
}


# ----------------------------------------------------------------------------
# Cleaning pipeline (notebook Sections 3-5)
# ----------------------------------------------------------------------------
def _pad_axis(fig, values, axis="x", pad=0.18):
    """Outside bar labels get clipped at the plot edge unless the axis is padded.
    Called on every chart using textposition='outside'."""
    finite = [v for v in values if pd.notna(v)]
    if not finite:
        return fig
    hi, lo = max(finite), min(finite)
    upper = hi + abs(hi) * pad if hi else 1
    lower = min(0, lo)
    fig.update_layout(**{f"{axis}axis": {"range": [lower, upper]}})
    return fig


def _age_bucket(age):
    if pd.isnull(age):
        return "UNKNOWN"
    elif age <= 25:
        return "1991-Present"
    elif age <= 45:
        return "1971-1990"
    elif age <= 65:
        return "1951-1970"
    elif age <= 85:
        return "1931-1950"
    return "1930 or Earlier"


def _flag_outliers_iqr(df, columns):
    """Flag, never modify. Lower fence clamped at 0 — these are physical quantities."""
    report = []
    for col in columns:
        valid = df[col].notna()
        q1 = df.loc[valid, col].quantile(0.25)
        q3 = df.loc[valid, col].quantile(0.75)
        iqr = q3 - q1
        lower_fence = max(0, q1 - 1.5 * iqr)
        upper_fence = q3 + 1.5 * iqr
        is_outlier = ((df[col] > upper_fence) | (df[col] < lower_fence)) & valid
        df[col + "IsOutlier"] = is_outlier
        report.append({
            "Column": col,
            "Lower Fence": round(lower_fence, 2),
            "Upper Fence": round(upper_fence, 2),
            "Flagged (not modified)": int(is_outlier.sum()),
            "Non-null N": int(valid.sum()),
            "% Flagged": round(is_outlier.sum() / valid.sum() * 100, 1) if valid.sum() else np.nan,
        })
    return df, pd.DataFrame(report)


@st.cache_data(show_spinner="Running the notebook's cleaning pipeline…")
def load_and_clean(raw_bytes: bytes):
    """Sections 3-5, in order. Returns the cleaned frame plus a diagnostics dict."""
    diag = {}

    # pandas cannot infer compression from an in-memory buffer, so sniff the magic bytes.
    # This is what lets a gzipped CSV be committed to the repo instead of a 24 MB plain one.
    compression = "gzip" if raw_bytes[:2] == b"\x1f\x8b" else None
    df = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False, compression=compression)
    diag["raw_shape"] = df.shape

    # 3.1 rename
    df.rename(columns=RENAME_MAP, inplace=True)
    missing_expected = [c for c in RENAME_MAP.values() if c not in df.columns]
    diag["unmapped_columns"] = missing_expected

    # 4.1 duplicates
    n_dupes = int(df.duplicated().sum())
    diag["duplicates"] = n_dupes
    if n_dupes:
        df = df.drop_duplicates()

    # 4.2 "Not Available" -> real nulls, then the missingness report
    df.replace(["Not Available", "not available", "NOT AVAILABLE", "N/A", "n/a", ""],
               np.nan, inplace=True)
    missing_counts = df.isnull().sum()
    diag["missing_report"] = (
        pd.DataFrame({
            "Missing Count": missing_counts,
            "Missing %": (missing_counts / len(df) * 100).round(2),
        })
        .query("`Missing Count` > 0")
        .sort_values("Missing %", ascending=False)
    )

    # 4.3 text -> float, NaN preserved (no imputation in the Preliminary phase)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(r"[\$,%]", "", regex=True)
                .str.strip()
                .replace(["nan", "NaN", ""], np.nan)
                .astype(float)
            )

    # 4.4 the incomplete-filing subpopulation
    df["isIncompleteFiling"] = df[STRUCTURAL_COLS].isnull().all(axis=1)
    diag["n_incomplete"] = int(df["isIncompleteFiling"].sum())
    # Which fields actually share the flag's null mask, verified rather than asserted.
    # An earlier version of this app stated "five" while its own missingness chart
    # showed six bars at the same rate; numberOfBuildings has the identical mask.
    diag["comissing_fields"] = sorted(
        c for c in df.columns
        if c != "isIncompleteFiling"
        and df[c].isnull().equals(df["isIncompleteFiling"])
    )
    diag["incomplete_crosstab"] = (
        df.groupby("isIncompleteFiling")["complianceStatus"]
        .value_counts(normalize=True).mul(100).round(1)
        .rename("% of group").reset_index()
    )
    diag["incomplete_compliant_rate"] = float(
        df.loc[df["isIncompleteFiling"], "complianceStatus"].eq("COMPLIED").mean()
    )
    diag["n_incomplete_complied"] = int(
        df.loc[df["isIncompleteFiling"], "complianceStatus"].eq("COMPLIED").sum()
    )
    # Wide form so each cell can be surfaced as its own metric rather than a
    # dataframe whose "% of group" column gets truncated at narrow widths.
    # Not rounded here. Rounding at construction fixes the precision for every consumer:
    # the COMPLIED cell became exactly 0.04, so the Overview metric read 0.04% while the
    # headline sentence above it — computed from the raw counts — said 0.035%. Same 11
    # filings, two figures on one screen. Each display site now chooses its own precision
    # from the full value.
    diag["incomplete_crosstab_wide"] = (
        pd.crosstab(df["isIncompleteFiling"], df["complianceStatus"], normalize="index")
        .mul(100)
        .reindex(columns=["COMPLIED", "NOT COMPLIED"], fill_value=0.0)
        .reindex(index=[False, True], fill_value=0.0)
    )

    # 4.5 target integrity, then locale fix, then the generic categorical fill
    diag["missing_target"] = int(df["complianceStatus"].isnull().sum())
    locale_fix = {"Non": "No", "Oui": "Yes", "non": "No", "oui": "Yes"}
    diag["n_locale_fixed"] = int(df["energyStarCertEligibility"].isin(locale_fix).sum())
    df["energyStarCertEligibility"] = df["energyStarCertEligibility"].replace(locale_fix)
    for col in ["propertyType", "entityResponsible", "energyStarCertEligibility",
                "ladbsBuildingCategory"]:
        df[col] = df[col].fillna("UNKNOWN")

    # 4.6 standardize text
    for col in ["propertyType", "complianceStatus", "entityResponsible", "ladbsBuildingCategory"]:
        df[col] = df[col].astype(str).str.strip().str.upper()

    # 4.6b French-locale propertyType labels, remapped before the field is used as a
    # grouping key. Reported so the repair is visible rather than silent.
    _pt_before = df["propertyType"].copy()
    _pt_keys = df["propertyType"].map(
        lambda v: re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]", " ", str(v))).strip())
    df["propertyType"] = np.where(_pt_keys.isin(PROPERTY_TYPE_FR),
                                  _pt_keys.map(PROPERTY_TYPE_FR).fillna(df["propertyType"]),
                                  df["propertyType"])
    _pt_changed = _pt_before != df["propertyType"]
    diag["locale_property_types"] = (
        pd.DataFrame({"From": _pt_before[_pt_changed], "To": df.loc[_pt_changed, "propertyType"]})
        .value_counts().rename("Filings").reset_index() if _pt_changed.any()
        else pd.DataFrame(columns=["From", "To", "Filings"]))
    diag["n_locale_property_types"] = int(_pt_changed.sum())
    diag["property_types_before"] = int(_pt_before.nunique())
    diag["property_types_after"] = int(df["propertyType"].nunique())

    # 4.6b two targeted repairs to entityResponsible, applied before the field is used
    # as a grouping key anywhere. .str.strip() removes whitespace but not other
    # characters, so a value such as "_LANDMARK PROPERTY SERVICES" survives it.
    _before = df["entityResponsible"].copy()
    df["entityResponsible"] = df["entityResponsible"].str.replace(
        r"^[^A-Z0-9]+|[^A-Z0-9]+$", "", regex=True)
    diag["n_punct_stripped"] = int((_before != df["entityResponsible"]).sum())

    # Explicit alias map for pairs the mechanical rule in 4.6c cannot reach — labels
    # differing by a real word, where only a human can say whether two firms are one.
    # Empty by design: the one entry it used to hold (BAY EFFICIENCY, LLC) differs only
    # by a legal form, so 4.6c merges it without a judgement. Kept as the place a hand
    # merge belongs if one of the 28 remaining groups is ever verified.
    ENTITY_ALIASES = {}
    _present = {k: v for k, v in ENTITY_ALIASES.items()
                if k in set(df["entityResponsible"].unique())}
    diag["aliases_applied"] = [
        (k, v, int((df["entityResponsible"] == k).sum()), int((df["entityResponsible"] == v).sum()))
        for k, v in _present.items()]
    if _present:
        df["entityResponsible"] = df["entityResponsible"].replace(_present)

    # 4.6c legal-form merge. A group whose labels differ ONLY by punctuation and legal
    # form is one organisation spelled two ways. A group where a real word differs
    # (HARPER MANAGEMENT vs HARPER ENTERPRISES, LP) may be two filers, and 4.8 reports
    # those instead. The test is mechanical — no group is adjudicated by hand — so the
    # merge is auditable and reversible by editing _LEGAL alone.
    _LEGAL = {"LLC", "L", "C", "INC", "INCORPORATED", "CORP", "CORPORATION", "CO",
              "COMPANY", "LP", "LLP", "LTD", "LIMITED", "THE", "TRUST", "REIT"}
    _lc = df.loc[df["entityResponsible"] != "UNKNOWN", "entityResponsible"].value_counts()
    _lf = pd.DataFrame({"raw": _lc.index, "filings": _lc.values})
    _lf["toks"] = _lf["raw"].astype(str).str.upper().str.replace(
        r"[^A-Z0-9 ]", " ", regex=True).str.split()
    _lf["key"] = _lf["toks"].apply(lambda t: " ".join(w for w in t if w not in _LEGAL))
    _lf = _lf[_lf["key"] != ""]

    _legal_map = {}
    for _k, _g in _lf.groupby("key"):
        if len(_g) < 2:
            continue
        _sets = [set(t) for t in _g["toks"]]
        if set().union(*_sets) - set.intersection(*_sets) <= _LEGAL:
            _canon = _g.sort_values("filings", ascending=False)["raw"].iloc[0]
            for _r in _g["raw"]:
                if _r != _canon:
                    _legal_map[_r] = _canon

    diag["legal_merge_labels"] = len(_legal_map)
    diag["legal_merge_groups"] = len(set(_legal_map.values()))
    if _legal_map:
        df["entityResponsible"] = df["entityResponsible"].replace(_legal_map)
    
    diag["n_entities"] = int(df.loc[df["entityResponsible"] != "UNKNOWN",
                                    "entityResponsible"].nunique())

    # 5.1 sanity-bound yearBuilt, derive buildingAge
    current_year = df["programYear"].max()
    invalid_year = (df["yearBuilt"] < 1800) | (df["yearBuilt"] > current_year)
    diag["n_invalid_yearbuilt"] = int(invalid_year.sum())
    df.loc[invalid_year, "yearBuilt"] = np.nan
    df["buildingAge"] = df["programYear"] - df["yearBuilt"]
    df.loc[df["buildingAge"] < 0, "buildingAge"] = np.nan

    # 5.2 plausibility bounds — data-entry errors, not statistical outliers
    plaus_report = []
    for col, b in PLAUSIBILITY_BOUNDS.items():
        invalid = (df[col] <= b["floor"]) | (df[col] > b["ceiling"])
        n_invalid = int(invalid.sum())
        n_valid_before = int(df[col].notna().sum())
        df.loc[invalid, col] = np.nan
        plaus_report.append({
            "Column": col,
            "Floor (excl.)": b["floor"],
            "Ceiling": b["ceiling"],
            "Nulled as data-entry error": n_invalid,
            "% of non-null values": round(n_invalid / n_valid_before * 100, 2) if n_valid_before else np.nan,
        })
    diag["plaus_report"] = pd.DataFrame(plaus_report)

    # 5.3 IQR flagging, computed on plausibility-bounded data
    df, outlier_report = _flag_outliers_iqr(df, OUTLIER_COLS)

    # 5.4 feature engineering
    valid_area = df["grossFloorArea"] >= MIN_PLAUSIBLE_AREA
    df["ghgIntensityPer1kSqft"] = np.where(
        valid_area, (df["co2Emissions"] / df["grossFloorArea"]) * 1000, np.nan
    )
    df, ghg_report = _flag_outliers_iqr(df, ["ghgIntensityPer1kSqft"])
    diag["outlier_report"] = pd.concat([outlier_report, ghg_report], ignore_index=True)

    df["ageBucket"] = df["buildingAge"].apply(_age_bucket)
    df["isCompliant"] = (df["complianceStatus"] == "COMPLIED").astype(int)
    df["postalCode"] = df["postalCode"].astype(str).str.split(".").str[0]

    # 4.7 values in entityResponsible that cannot be company names. Reported, never
    # repaired: the true entity is unrecoverable, and assigning them to UNKNOWN would
    # merge them with the incomplete filings, which are almost never compliant.
    _ent = df.loc[df["entityResponsible"] != "UNKNOWN", "entityResponsible"]
    _counts = _ent.value_counts()
    _strong = {
        "Numeric only": lambda x: x.isdigit(),
        "No letters": lambda x: not any(c.isalpha() for c in x),
        "Underscore": lambda x: "_" in x,
    }
    _flag = {}
    for lbl, test in _strong.items():
        for v in _counts.index:
            if test(str(v)):
                _flag.setdefault(v, []).append(lbl)
    diag["nonname"] = (pd.DataFrame(
        [{"Value": v, "Flagged as": ", ".join(k), "Filings": int(_counts[v])}
         for v, k in _flag.items()]).sort_values("Filings", ascending=False)
        if _flag else pd.DataFrame(columns=["Value", "Flagged as", "Filings"]))
    diag["nonname_share"] = (diag["nonname"]["Filings"].sum() / _counts.sum() * 100
                             if len(diag["nonname"]) else 0.0)

    # 4.8 labels that may refer to one organisation. Detected, not merged.
    _SUFFIX = (r"\b(LLC|L\.L\.C\.|INC|INC\.|CORP|CORPORATION|CO|COMPANY|LP|LLP|LTD|GROUP|"
               r"HOLDINGS|PARTNERS|ENTERPRISES|PROPERTIES|MANAGEMENT|SERVICES|ASSOCIATES|"
               r"TRUST|REIT)\b")
    _norm = pd.DataFrame({"raw": _counts.index, "filings": _counts.values})
    _norm["key"] = (_norm["raw"].astype(str).str.upper()
                    .str.replace(r"[^A-Z0-9 ]", " ", regex=True)
                    .str.replace(_SUFFIX, " ", regex=True)
                    .str.replace(r"\s+", " ", regex=True).str.strip())
    _norm = _norm[_norm["key"] != ""]
    _grp = _norm.groupby("key").filter(lambda g: len(g) > 1)
    diag["dupe_groups"] = int(_grp["key"].nunique()) if len(_grp) else 0
    diag["dupe_labels"] = int(len(_grp))
    diag["dupe_share"] = (_grp["filings"].sum() / _counts.sum() * 100) if len(_grp) else 0.0
    diag["dupe_detail"] = (_grp.sort_values(["key", "filings"], ascending=[True, False])
                           if len(_grp) else pd.DataFrame(columns=["raw", "filings", "key"]))

    # 6.8 availability of the fields not described elsewhere, plus the duplicate check
    _UND = ["occupancy", "numberOfBuildings", "totalWaterUse", "weatherNormSiteEui",
            "weatherNormSourceEui", "pctDiffNationalSourceEui", "pctDiffNationalSiteEui",
            "indoorWaterUse", "indoorWaterUseIntensity", "outdoorWaterUse"]
    _av = []
    for c in _UND:
        if c in df.columns:
            n = int(df[c].notna().sum())
            med = df[c].dropna().median() if n else np.nan
            mx = df[c].max() if n else np.nan
            _av.append({"Field": c, "Present": n, "Present %": round(n / len(df) * 100, 1),
                        "Median": round(med, 2) if pd.notna(med) else np.nan,
                        "Max": mx,
                        "Max / median": (abs(mx / med) if med else np.nan)})
    diag["availability"] = pd.DataFrame(_av).sort_values("Present %", ascending=False)

    a, b = "pctDiffNationalSiteEui", "pctDiffNationalSourceEui"
    if a in df.columns and b in df.columns:
        _both = df[[a, b]].dropna()
        diag["pctdiff_n"] = int(len(_both))
        diag["pctdiff_same"] = int((_both[a] == _both[b]).sum()) if len(_both) else 0

    # 6.9 structural features from the coverage tier — populated on every filing,
    # unlike propertyType, so usable across the whole dataset.
    _cat = df["ladbsBuildingCategory"]
    df["isCityOwned"] = _cat.str.contains("CITY OWNED", na=False)
    df["sizeBand"] = (_cat.str.replace(r"\s*\(CITY OWNED BUILDINGS?\)", "", regex=True)
                          .str.strip())

    return df, diag


def find_bundled_csv():
    """Look for a CSV shipped alongside the app, so a deployed copy needs no upload."""
    for pattern in ("data/*.csv.gz", "data/*.csv", "*.csv.gz"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None


# ----------------------------------------------------------------------------
# Sidebar — data source and filters
# ----------------------------------------------------------------------------
def rail():
    """A bordered group in the sidebar, matching the main area's card()."""
    c = st.sidebar.container()
    c.markdown("<span class='railmark'></span>", unsafe_allow_html=True)
    return c


# Two groups, as in the structure mockup: identity and source, then the phase nav.
_rail_top = rail()
_rail_top.title("EBEWE Dashboard")
_rail_top.caption("City of Los Angeles · LADBS building benchmarking filings")

bundled = find_bundled_csv()
uploaded = _rail_top.file_uploader(
    "Upload the EBEWE CSV",
    type=["csv", "gz"],
    help="The same file used in Colab. Optional if a copy is bundled with the app.",
)

raw_bytes = None
source_label = None
if uploaded is not None:
    raw_bytes = uploaded.getvalue()
    source_label = uploaded.name
elif bundled:
    with open(bundled, "rb") as fh:
        raw_bytes = fh.read()
    source_label = os.path.basename(bundled)

if raw_bytes is None:
    st.title("EBEWE Program — Building Compliance Analytics")
    st.info(
        "Upload the EBEWE CSV in the sidebar to begin. The app runs the full cleaning "
        "pipeline from the Preliminary notebook (Sections 3–5) on upload — the file does "
        "not need to be pre-cleaned."
    )
    st.stop()

df, diag = load_and_clean(raw_bytes)

# The key lets the Home page's buttons move the selection: a callback writes
# st.session_state["phase"] before Streamlit re-instantiates this widget.
phase = st.sidebar.radio(
    "Phase",
    PHASES,
    captions=PHASE_CAPTIONS,
    format_func=lambda p: f"{PHASE_ICONS[p]} {p}",
    key="phase",
    help="Home summarises all three phases. Preliminary describes the data, Midterm "
         "diagnoses what the compliance label measures, Final fits the model.",
)
_rail_filters = rail()
years = sorted(df["programYear"].dropna().unique().astype(int))
year_range = _rail_filters.select_slider(
    "Program year range",
    options=years,
    value=(years[0], years[-1]),
)

all_types = sorted(t for t in df["propertyType"].unique() if t != "UNKNOWN")
top_12 = list(df[~df["isIncompleteFiling"]]["propertyType"].value_counts().head(12).index)
type_mode = _rail_filters.radio(
    "Property types (complete filings)",
    ["Top 12 by filing count", "All types", "Choose specific types"],
    help="Segment-level views only. Twelve chips pinned open crowds the sidebar, "
         "so the common cases are presets.",
)
if type_mode == "Top 12 by filing count":
    chosen_types = top_12
elif type_mode == "All types":
    chosen_types = []
else:
    chosen_types = _rail_filters.multiselect("Select types", options=all_types, default=[])
    if not chosen_types:
        _rail_filters.caption("No types selected — showing all.")

_rail_src = rail()
_rail_src.caption(f"Source file: `{source_label}`")
_rail_src.caption(
    f"{diag['raw_shape'][0]:,} rows × {diag['raw_shape'][1]} columns as loaded · "
    f"{diag['duplicates']:,} duplicate rows"
)
_rail_src.download_button(
    "Download cleaned dataset (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="ebewe_cleaned.csv",
    mime="text/csv",
)

# Apply filters
mask = df["programYear"].between(year_range[0], year_range[1])
dff = df[mask].copy()
dff_complete = dff[~dff["isIncompleteFiling"]].copy()
if chosen_types:
    dff_seg = dff_complete[dff_complete["propertyType"].isin(chosen_types)].copy()
else:
    dff_seg = dff_complete.copy()

if len(dff) == 0:
    st.error("No filings in the selected year range.")
    st.stop()

filtered = (year_range[0], year_range[1]) != (years[0], years[-1])

# ----------------------------------------------------------------------------
# Forward-looking phases
# ----------------------------------------------------------------------------
def band(cells, highlight=None):
    """A hairline-separated row of figures, read as one strip rather than four widgets."""
    html = "<div class='statband'>"
    for i, (label, value) in enumerate(cells):
        hi = " hi" if i == highlight else ""
        html += f"<div class='bcell{hi}'><span>{label}</span><b>{value}</b></div>"
    st.markdown(html + "</div>", unsafe_allow_html=True)


def _inline_md(text):
    """Convert the markdown spans that do not survive inside a raw HTML block.

    st.markdown with unsafe_allow_html=True does not parse markdown inside a block
    element, so `**bold**` and `code` reach the browser as literal characters. Every
    helper that emits its own HTML runs its text through this first.
    """
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", text)


@contextmanager
def panel(num="", title="", note=""):
    """A bordered panel carrying its own header.

    A section rule divides a column; a panel is a box that can sit beside another
    box. That is the whole difference between a page that only grows downwards and
    a grid, so the header moves inside the card rather than ruling across the page.
    """
    c = card()
    with c:
        bits = ""
        if num:
            bits += f"<em>{num}</em>"
        if title:
            bits += f"<h4>{title}</h4>"
        if note:
            bits += f"<span class='pnote'>{note}</span>"
        if bits:
            st.markdown(f"<div class='ph'>{bits}</div>", unsafe_allow_html=True)
        yield c


def scrollbox(headers, rows, height=200, focus=None, numeric_from=1):
    """A table that scrolls inside its panel instead of lengthening the page.

    Ten or more rows of a supporting table push everything after them off screen
    for a reader who only wanted the chart above. Fixed height, sticky header.
    """
    h = "".join(f"<th class=\"{'n' if i >= numeric_from else ''}\">{c}</th>"
                for i, c in enumerate(headers))
    body = ""
    for i, r in enumerate(rows):
        cls = " class='focus'" if focus is not None and i == focus else ""
        tds = "".join(f"<td class=\"{'n' if j >= numeric_from else ''}\">{c}</td>"
                      for j, c in enumerate(r))
        body += f"<tr{cls}>{tds}</tr>"
    # height is a floor, not a cap. Cards in a row stretch to the tallest, and a
    # fixed max-height turned that extra height into padding; as a flex floor the
    # table grows into it and the space becomes rows instead of nothing.
    st.markdown(
        f"<div class='scrollbox' style='--sb:{height}px'>"
        f"<table class='mini sticky'><thead><tr>{h}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>", unsafe_allow_html=True)


def section(num, title, kicker=None):
    """A numbered rule between blocks.

    Tabs are a flat device: every tab opens the same way, so a long tab reads as one
    undifferentiated column. The number gives the eye somewhere to land and tells the
    reader how far through the argument they are.
    """
    k = f"<p>{_inline_md(kicker)}</p>" if kicker else ""
    st.markdown(
        f"<div class='srule'><span class='snum'>{num}</span>"
        f"<div><h3>{title}</h3>{k}</div></div>",
        unsafe_allow_html=True)


def evidence(ratio=1.7, note_first=False):
    """Return (evidence_column, note_column) for a figure and its reading.

    A caption under a chart is read after the reader has already formed an impression
    of it. Beside the chart, the sentence and the shape it describes are in view at
    once. note_first flips the order so consecutive blocks do not read as a column
    with a permanent sidebar.
    """
    if note_first:
        a, b = st.columns([1, ratio], gap="large")
        return b, a
    a, b = st.columns([ratio, 1], gap="large")
    return a, b


def kv(rows, scale=None):
    """Label, bar, value rows inside a card.

    A three- or four-row comparison does not need a plotting library: the bar is the
    comparison, and drawn inline it costs no axis, no legend and no vertical space a
    chart would claim. rows are (label, value_text, fraction_0_to_1, muted) with the
    last two optional; pass scale to compute the fraction from a raw number instead.
    """
    html = "<div class='kv'>"
    for row in rows:
        label, value = row[0], row[1]
        frac = row[2] if len(row) > 2 else None
        muted = row[3] if len(row) > 3 else False
        if frac is None and scale:
            frac = None
        if frac is None:
            html += (f"<div class='r norail'><span>{label}</span>"
                     f"<b class='v'>{value}</b></div>")
        else:
            w = max(0.0, min(1.0, float(frac))) * 100
            html += (f"<div class='r'><span>{label}</span>"
                     f"<span class='t'><span class='f{' mute' if muted else ''}' "
                     f"style='width:{w:.1f}%'></span></span>"
                     f"<b class='v'>{value}</b></div>")
    st.markdown(html + "</div>", unsafe_allow_html=True)


def minitable(headers, rows, focus=None, numeric_from=1):
    """A compact table for three to six rows.

    st.dataframe brings a scroll container, a toolbar and its own type scale for data
    that is really a short list. This renders the same rows at the page's own scale,
    with the row that carries the point highlighted rather than left for the reader
    to find.
    """
    h = "".join(f"<th class=\"{'n' if i >= numeric_from else ''}\">{c}</th>"
                for i, c in enumerate(headers))
    body = ""
    for i, r in enumerate(rows):
        cls = " class='focus'" if focus is not None and i == focus else ""
        tds = "".join(f"<td class=\"{'n' if j >= numeric_from else ''}\">{c}</td>"
                      for j, c in enumerate(r))
        body += f"<tr{cls}>{tds}</tr>"
    # nowrap keeps numbers on one line, which means a wide table overflows its
    # column and the last column is simply cut off. The wrapper scrolls instead.
    st.markdown(f"<div class='miniwrap'><table class='mini'><thead><tr>{h}</tr></thead>"
                f"<tbody>{body}</tbody></table></div>", unsafe_allow_html=True)


def aside(caption_text, body, label="How this is computed"):
    """A short caption with its disclosure directly beneath it.

    This used to split into two columns to save a row. Six of its seven call sites
    are already inside a column or a card, so the split nested a column inside a
    column and the disclosure rendered about sixty pixels wide, with its label
    running one letter per line. A saved row is not worth that; the two are kept
    tight vertically instead, which reads as one unit without the risk.
    """
    st.caption(caption_text)
    method(body, label)

def method(body, label="How this is computed"):
    """Put the methodological passage behind a disclosure.

    A dashboard has one focal path per section: the figure, then one sentence saying
    what it shows. The reasoning behind a choice matters for the write-up and for a
    reader who challenges a number, but set at the same weight as the finding it
    competes with it, and the page reads as undifferentiated prose.
    """
    with st.expander(label):
        st.markdown(body)


def pull(headline, body):
    """The sentence that changes how the numbers above it should be read."""
    st.markdown(
        f"<div class='pullq'><b>{_inline_md(headline)}</b>"
        f"<p>{_inline_md(body)}</p></div>", unsafe_allow_html=True)


def card():
    """A bordered container that CSS can actually find.

    st.container(border=True) renders as a plain stVerticalBlock with a border; there is
    no test id distinguishing it from every other block, and the emotion class that does
    is regenerated between Streamlit versions. Dropping an empty marker inside and
    selecting the block that :has() it is version-independent.
    """
    c = st.container(border=True)
    c.markdown("<span class='cardmark'></span>", unsafe_allow_html=True)
    return c


def _phase_header(title, standing, blurb):
    """Eyebrow, title, standing, lead — across the card's width rather than down it."""
    stage, _, rest = title.partition(" — ")
    with card():
        left, right = st.columns([1, 2.1], gap="large")
        with left:
            st.markdown(
                f"<div class='pagehead'><span class='eyebrow'>{rest.upper() or 'OVERVIEW'}"
                f"</span><h2>{stage}</h2><p class='standing'>{standing}</p></div>",
                unsafe_allow_html=True,
            )
        with right:
            # Raw HTML bypasses Streamlit's markdown pass, so inline code spans would
            # arrive as literal backticks.
            lead = _inline_md(blurb)
            st.markdown(f"<p class='lead'>{lead}</p>", unsafe_allow_html=True)



# ----------------------------------------------------------------------------
# Final-phase model — fitted live, cached
# ----------------------------------------------------------------------------
# Fitted on load like every other figure here, rather than shipped as saved
# predictions. A refresh of the published file therefore flows through the model
# as well as the descriptive tabs, which is the claim this app makes everywhere
# else and would be false on this page alone if the numbers were frozen.
MODEL_CONT = ["grossFloorArea", "buildingAge", "siteEui", "energyStarScore",
              "numberOfBuildings", "occupancy"]
MODEL_CAT = ["ladbsBuildingCategory", "sizeBand", "propertyType"]
TE_SMOOTH = 20
ENTITY_COLS = ["entityTE", "entityFreq"]
GB_PARAMS = dict(max_iter=300, learning_rate=0.05, min_samples_leaf=100,
                 max_leaf_nodes=15, l2_regularization=5.0, early_stopping=True,
                 validation_fraction=0.15, random_state=42)


@st.cache_resource(show_spinner="Fitting the model — once per file, about 12 seconds…")
def fit_model(_df, cache_key):
    """Fit, evaluate and ablate the compliance classifier.

    `_df` is unhashed (the leading underscore tells Streamlit to skip it);
    `cache_key` carries the identity of the loaded file instead.
    """
    from sklearn.model_selection import GroupShuffleSplit, GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, average_precision_score,
                                 confusion_matrix)

    # Complete filings only. Across all filings the model recovers
    # isIncompleteFiling rather than anything about the building (Section 4.4).
    m = _df[~_df["isIncompleteFiling"]].copy()
    m["y"] = (m["complianceStatus"] == "NOT COMPLIED").astype(int)

    # Grouped on buildingId. With several filings per building a random split puts
    # the same building on both sides and inflates every metric reported below.
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
                  .split(m, m["y"], groups=m["buildingId"]))
    train, test = m.iloc[tr].copy(), m.iloc[te].copy()

    def smoothed(frame, col, base):
        g = frame.groupby(col)["y"].agg(["sum", "count"])
        return (g["sum"] + TE_SMOOTH * base) / (g["count"] + TE_SMOOTH)

    # entityResponsible carries the widest compliance spread in the data and far too
    # many levels to one-hot. Encoded out of fold and grouped by building, so no row
    # informs its own encoding — the one place leakage would otherwise re-enter.
    prior = train["y"].mean()
    train["entityTE"] = np.nan
    for fit_i, enc_i in GroupKFold(n_splits=5).split(train, train["y"],
                                                     groups=train["buildingId"]):
        f = train.iloc[fit_i]
        mp = smoothed(f, "entityResponsible", f["y"].mean())
        train.iloc[enc_i, train.columns.get_loc("entityTE")] = (
            train.iloc[enc_i]["entityResponsible"].map(mp).fillna(f["y"].mean()).values)
    full_map = smoothed(train, "entityResponsible", prior)
    test["entityTE"] = test["entityResponsible"].map(full_map).fillna(prior)
    freq = train["entityResponsible"].value_counts()
    train["entityFreq"] = train["entityResponsible"].map(freq).fillna(0)
    test["entityFreq"] = test["entityResponsible"].map(freq).fillna(0)
    unseen = float((~test["entityResponsible"].isin(full_map.index)).mean())

    def matrix(frame):
        X = frame[MODEL_CONT + ENTITY_COLS + ["programYear"]].copy()
        X["isCityOwned"] = frame["isCityOwned"].astype(int)
        for c in MODEL_CONT:
            X[c + "_isna"] = frame[c].isna().astype(int)
        return pd.concat(
            [X, pd.get_dummies(frame[MODEL_CAT].astype(str), columns=MODEL_CAT)], axis=1)

    X_tr, X_te = matrix(train), matrix(test)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
    X_tr, X_te = X_tr.astype(float), X_te.astype(float)
    y_tr, y_te = train["y"].values, test["y"].values
    base_rate = float(y_te.mean())

    def score(name, proba, thr=0.5):
        pred = (proba >= thr).astype(int)
        _, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
        return {"Model": name, "Accuracy": accuracy_score(y_te, pred),
                "Precision": precision_score(y_te, pred, zero_division=0),
                "Recall": recall_score(y_te, pred, zero_division=0),
                "F1": f1_score(y_te, pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_te, proba),
                "PR-AUC": average_precision_score(y_te, proba),
                "Caught": int(tp), "False alarms": int(fp), "Missed": int(fn)}

    m1 = Pipeline([("impute", SimpleImputer(strategy="median")),
                   ("scale", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=2000,
                                              class_weight="balanced"))]).fit(X_tr, y_tr)
    # Regularisation chosen on the train-test AUC gap rather than the test score:
    # an unregularised fit reached the same test AUC with twice the gap.
    m2 = HistGradientBoostingClassifier(**GB_PARAMS).fit(X_tr, y_tr)
    # The ablation withholds the two entity columns and changes nothing else, so the
    # difference measures what knowing the filer is worth rather than implying it.
    m3 = HistGradientBoostingClassifier(**GB_PARAMS).fit(
        X_tr.drop(columns=ENTITY_COLS), y_tr)
    p1 = m1.predict_proba(X_te)[:, 1]
    p2 = m2.predict_proba(X_te)[:, 1]
    p3 = m3.predict_proba(X_te.drop(columns=ENTITY_COLS))[:, 1]

    comparison = pd.DataFrame([
        {"Model": "Baseline — predict all COMPLIED",
         "Accuracy": accuracy_score(y_te, np.zeros_like(y_te)), "Precision": 0.0,
         "Recall": 0.0, "F1": 0.0, "ROC-AUC": 0.5, "PR-AUC": base_rate,
         "Caught": 0, "False alarms": 0, "Missed": int(y_te.sum())},
        score("Logistic regression", p1),
        score("Gradient boosting", p2),
        score("Gradient boosting, no entity features", p3)])

    # Importance for the columns this page names, with each indicator family permuted
    # as a block. Permuting all columns five times each costs about fifty seconds,
    # nearly all of it on indicators no reader looks at.
    rng = np.random.default_rng(42)
    base_auc = roc_auc_score(y_te, p2)
    NAMED = ["entityTE", "programYear", "entityFreq", "buildingAge", "grossFloorArea",
             "isCityOwned", "occupancy", "siteEui", "energyStarScore"]
    PRETTY = {"entityTE": "Responsible entity (encoded)", "programYear": "Program year",
              "entityFreq": "Entity filing volume", "buildingAge": "Building age",
              "grossFloorArea": "Gross floor area", "isCityOwned": "City-owned",
              "occupancy": "Occupancy", "siteEui": "Site EUI",
              "energyStarScore": "ENERGY STAR score"}
    GROUPS = {"All property-type columns": "propertyType",
              "All coverage-tier columns": "ladbsBuildingCategory",
              "All size-band columns": "sizeBand"}

    def drop_for(cols, repeats=3):
        losses = []
        for _ in range(repeats):
            Xp = X_te.copy()
            idx = rng.permutation(len(Xp))
            for c in cols:
                Xp[c] = Xp[c].values[idx]
            losses.append(base_auc - roc_auc_score(y_te, m2.predict_proba(Xp)[:, 1]))
        return float(np.mean(losses)), float(np.std(losses))

    rows = []
    for c in NAMED:
        mean, sd = drop_for([c])
        rows.append({"Feature": PRETTY[c], "AUC drop": mean, "SD": sd})
    for label, prefix in GROUPS.items():
        cols = [c for c in X_te.columns if c.startswith(prefix)]
        if cols:
            mean, sd = drop_for(cols)
            rows.append({"Feature": label, "AUC drop": mean, "SD": sd})
    importance = (pd.DataFrame(rows).sort_values("AUC drop", ascending=False)
                  .reset_index(drop=True))

    ranked = (test.assign(risk=p2)
              .sort_values("risk", ascending=False)
              .loc[:, ["programYear", "buildingAddress", "propertyType",
                       "entityResponsible", "ladbsBuildingCategory", "risk",
                       "complianceStatus"]])
    order = np.argsort(-p2)
    curve = {n: int(y_te[order[:n]].sum()) for n in (100, 250, 500, 1000, 1500, 2000)}

    return {
        "comparison": comparison, "importance": importance, "ranked": ranked,
        "curve": curve, "proba": p2, "y_test": y_te, "base_rate": base_rate,
        "n_train": len(train), "n_test": len(test), "n_pos": int(y_te.sum()),
        "train_buildings": int(train["buildingId"].nunique()),
        "test_buildings": int(test["buildingId"].nunique()),
        "unseen_entity_share": unseen,
        "auc_full": base_auc, "auc_noentity": roc_auc_score(y_te, p3),
        "pr_full": average_precision_score(y_te, p2),
        "pr_noentity": average_precision_score(y_te, p3),
        "gap_m1": roc_auc_score(y_tr, m1.predict_proba(X_tr)[:, 1]) - roc_auc_score(y_te, p1),
        "gap_m2": roc_auc_score(y_tr, m2.predict_proba(X_tr)[:, 1]) - base_auc,
    }


# ----------------------------------------------------------------------------
# Dataset description — portal metadata plus what is computed from the file
# ----------------------------------------------------------------------------
# The descriptions come from the LADBS open-data portal and cannot be derived from
# the data, so they are stored. Everything beside them — type, fill rate, range —
# is computed from whatever file is loaded, so an uploaded refresh describes itself.
COLUMN_NOTES = {
    "buildingAddress": "Address of building.",
    "buildingId": "Identifier for the building.",
    "co2Emissions": "Total GHG emissions (metric tons CO2e).",
    "complianceStatus": "The status of the record in regard to the EBEWE Ordinance.",
    "pctDiffNationalSourceEui": "Percent difference from national median source EUI.",
    "pctDiffNationalSiteEui": "Percent difference from national median site EUI.",
    "energyStarScore": "ENERGY STAR score.",
    "energyStarCertEligibility": "ENERGY STAR certification — eligibility.",
    "energyStarCertLastApproval": "ENERGY STAR certification — last approval date.",
    "energyStarCertYears": "ENERGY STAR certification — year(s) certified.",
    "entityResponsible": "Organization responsible for the benchmark filing.",
    "grossFloorArea": "Gross building floor area, self-reported (ft²).",
    "indoorWaterUse": "Indoor water use, all sources (kgal).",
    "indoorWaterUseIntensity": "Indoor water use intensity, all sources (gal/ft²).",
    "numberOfBuildings": "Number of buildings covered by the filing.",
    "ladbsBuildingCategory": "Ordinance coverage tier: size band and ownership.",
    "programYear": "Compliance year the filing reports for.",
    "propertyType": "Primary property function.",
    "siteEui": "Site energy use intensity (kBtu/ft²).",
    "sourceEui": "Source energy use intensity (kBtu/ft²).",
    "occupancy": "Reported occupancy (%).",
    "yearBuilt": "Year the building was built.",
    "postalCode": "Postal code.",
    "ain": "Last three digits of the Assessor Identification Number.",
}
ENGINEERED_NOTES = {
    "isIncompleteFiling": "Engineered (4.4). True when all six structural fields are "
                          "absent on the same row — an abandoned submission.",
    "isCompliant": "Engineered (5.4). complianceStatus read as a boolean.",
    "buildingAge": "Engineered (5.1). programYear minus yearBuilt, bounded.",
    "ghgIntensityPer1kSqft": "Engineered (5.4). Emissions normalised by floor area.",
    "ageBucket": "Engineered (5.4). Building age in bands.",
    "isCityOwned": "Engineered. Derived from the coverage tier.",
    "sizeBand": "Engineered. Size portion of the coverage tier.",
}


@st.cache_data(show_spinner=False)
def _headline_facts(_df, cache_key):
    """Figures the Home page quotes. Computed once, read by the cards below.

    Every number on Home comes from here rather than being typed into prose, so a
    refreshed file moves the cards and the phases together instead of leaving the
    summary asserting last month's figures.
    """
    out = {}
    out["filings"] = len(_df)
    out["buildings"] = int(_df["buildingId"].nunique())
    out["incomplete_share"] = float(_df["isIncompleteFiling"].mean() * 100)
    out["n_incomplete"] = int(_df["isIncompleteFiling"].sum())
    complete = _df[~_df["isIncompleteFiling"]]
    out["complete_rate"] = float(complete["isCompliant"].mean() * 100)
    out["n_complete"] = len(complete)
    out["n_noncompliant_complete"] = int((complete["isCompliant"] == 0).sum())
    inc = _df[_df["isIncompleteFiling"]]
    out["incomplete_notcomplied"] = float((inc["isCompliant"] == 0).mean() * 100)

    by_year_all = _df.groupby("programYear")["isCompliant"].mean() * 100
    by_year_complete = complete.groupby("programYear")["isCompliant"].mean() * 100
    out["first_year"], out["last_year"] = int(by_year_all.index[0]), int(by_year_all.index[-1])
    out["fall_all"] = float(by_year_all.iloc[0] - by_year_all.iloc[-1])
    out["fall_complete"] = float(by_year_complete.iloc[0] - by_year_complete.iloc[-1])
    out["attributable"] = (100 * (1 - out["fall_complete"] / out["fall_all"])
                           if out["fall_all"] else float("nan"))

    # sizeBand, not ladbsBuildingCategory: the Coverage Tier tab reports the spread
    # across size bands, and the Home card must not contradict the tab it links to.
    tier_all = _df.groupby("sizeBand")["isCompliant"].mean() * 100
    tier_complete = complete.groupby("sizeBand")["isCompliant"].mean() * 100
    out["tier_spread_all"] = float(tier_all.max() - tier_all.min())
    out["tier_spread_complete"] = float(tier_complete.max() - tier_complete.min())

    ent = complete[complete["entityResponsible"] != "UNKNOWN"]
    counts = ent.groupby("entityResponsible")["isCompliant"].agg(["mean", "size"])
    big = counts[counts["size"] >= MIN_TYPE_FILINGS]
    out["n_big_entities"] = len(big)
    out["entity_spread"] = float((big["mean"].max() - big["mean"].min()) * 100) if len(big) else float("nan")
    types = complete.groupby("propertyType")["isCompliant"].agg(["mean", "size"])
    types = types[types["size"] >= MIN_TYPE_FILINGS]
    out["type_spread"] = float((types["mean"].max() - types["mean"].min()) * 100) if len(types) else float("nan")

    out["majority_complete"] = max(out["complete_rate"], 100 - out["complete_rate"])
    rule = (~_df["isIncompleteFiling"])
    out["one_rule_acc"] = float((rule == _df["isCompliant"]).mean() * 100)
    return out


def _goto(target):
    """Switch phase from a button. Set through a callback so the widget key can be
    written before Streamlit re-instantiates the radio on the next run."""
    st.session_state["phase"] = target


def render_home(source_label):
    facts = _headline_facts(df, (source_label, len(df), int(df["programYear"].max())))

    # Hero: the claim stated as a proportion before it is stated as a number. The filled
    # share of the bar is the finding; the figure beside it only confirms it.
    with card():
        hl, hr = st.columns([1.15, 1], gap="large")
        with hl:
            st.markdown(
                "<span class='eyebrow'>The finding this dashboard exists for</span>"
                f"<p class='claim'>{facts['incomplete_share']:.0f} per cent of filings are "
                "marked NOT COMPLIED because the report was <em>never finished</em> — not "
                "because the building failed.</p>"
                "<p class='sublead'>Compliance and paperwork share one label. Each "
                "phase separates a different part of that mixture.</p>",
                unsafe_allow_html=True,
            )
        with hr:
            _inc = facts["incomplete_share"]
            _w1 = 330 * _inc / 100
            _w2 = 330 * facts["complete_rate"] / 100
            st.markdown(
                f"""<svg viewBox="0 0 330 150" width="100%" height="150" role="img"
 aria-label="Of all filings, {_inc:.1f} per cent are structurally incomplete; among
 complete filings {facts['complete_rate']:.2f} per cent are compliant.">
<text x="2" y="14" font-size="10" fill="{SLATE}">ALL FILINGS · {facts['filings']:,}</text>
<rect x="0" y="22" width="330" height="34" rx="6" fill="#EBE8E5"/>
<rect x="0" y="22" width="{_w1:.0f}" height="34" rx="6" fill="{MARK}"/>
<text x="9" y="44" font-size="12" fill="#FFFFFF" font-weight="600">{_inc:.1f}%</text>
<text x="{_w1 + 12:.0f}" y="44" font-size="11" fill="{INK}">incomplete · {facts['n_incomplete']:,}</text>
<text x="2" y="84" font-size="10" fill="{SLATE}">COMPLETE FILINGS · {facts['n_complete']:,}</text>
<rect x="0" y="92" width="330" height="34" rx="6" fill="#EBE8E5"/>
<rect x="0" y="92" width="{_w2:.0f}" height="34" rx="6" fill="{MARK_DEEP}"/>
<text x="9" y="114" font-size="12" fill="#FFFFFF" font-weight="600">{facts['complete_rate']:.2f}% compliant</text>
</svg>""",
                unsafe_allow_html=True,
            )

    band([
        ("Filings", f"{facts['filings']:,}"),
        ("Buildings", f"{facts['buildings']:,}"),
        ("Program years", f"{facts['first_year']}\u2013{facts['last_year']}"),
        ("Structurally incomplete", f"{facts['incomplete_share']:.1f}%"),
    ], highlight=3)

    pull("Who this is for",
         "The LADBS program analyst who plans enforcement each program year, and the "
         "manager who reports compliance performance to the City.")

    st.divider()
    section("\u2192", "The three phases",
            "Each answers a question the next one depends on. Every figure is computed "
            "from the loaded file.")

    c1, c2, c3 = st.columns(3, gap="large")
    _cards = [
        (c1, "01", "Preliminary", "descriptive",
         "What does the dataset contain, and what is wrong with it?",
         # Each pair is drawn against the same 0-100 scale, so the bars are
         # comparable down the card and across the three cards.
         [("Incomplete", f"{facts['incomplete_share']:.1f}%",
           facts['incomplete_share'] / 100, True),
          ("Compliant", f"{facts['complete_rate']:.1f}%",
           facts['complete_rate'] / 100, False),
          ("Incompl. failed", f"{facts['incomplete_notcomplied']:.1f}%",
           facts['incomplete_notcomplied'] / 100, True)],
         "Data Quality · Descriptive Stats · Key Figures",
         "Open Preliminary", "go_prelim", PHASE_PRELIM),
        (c2, "02", "Midterm", "diagnostic",
         "Is compliance falling, and what actually separates it?",
         # Reported against surviving, on one scale, so the shrinkage is the picture.
         [("Headline fall", f"{facts['fall_all']:.1f}", facts['fall_all'] / 100, False),
          ("Complete only", f"{facts['fall_complete']:.1f}",
           facts['fall_complete'] / 100, True),
          ("Tier spread", f"{facts['tier_spread_all']:.1f}",
           facts['tier_spread_all'] / 100, False),
          ("Tier, complete", f"{facts['tier_spread_complete']:.1f}",
           facts['tier_spread_complete'] / 100, True)],
         "Compliance Trend · Coverage Tier · Responsible Entity · Correlations",
         "Open Midterm", "go_mid", PHASE_MID),
        (c3, "03", "Final", "predictive",
         "Can non-compliance be predicted well enough to target audits?",
         [("Bar to beat", f"{facts['majority_complete']:.1f}%",
           facts['majority_complete'] / 100, True),
          ("One-column rule", f"{facts['one_rule_acc']:.1f}%",
           facts['one_rule_acc'] / 100, True),
          ("Modelled on", f"{facts['n_complete']:,}"),
          ("Target group", f"{facts['n_noncompliant_complete']:,}")],
         "Model comparison · feature importance · ranked review list",
         "Open Final", "go_final", PHASE_FINAL),
    ]
    for _col, _n, _name, _kind, _q, _rows, _tabs, _label, _key, _phase in _cards:
        with _col:
            with card():
                st.markdown(
                    f"<div class='phasecard'><span class='eyebrow'>{_n} \u00b7 "
                    f"{_kind.upper()}</span><h4>{_name}</h4><p>{_q}</p></div>",
                    unsafe_allow_html=True)
                kv(_rows)
                st.caption(_tabs)
                st.button(_label, key=_key, width="stretch",
                          on_click=_goto, args=(_phase,))

    st.divider()
    with st.expander("About this dataset", expanded=False):
        raw_rows, raw_cols = diag.get("raw_shape", (len(df), len(df.columns)))
        band([
            ("Rows, as published", f"{raw_rows:,}"),
            ("Columns, as published", f"{raw_cols}"),
            ("Duplicate rows", f"{diag.get('duplicates', 0):,}"),
            ("Filings per building",
             f"{raw_rows / max(1, df['buildingId'].nunique()):.2f}"),
        ], highlight=2)

        # Two panels rather than band, prose, table stacked. The provenance is four
        # facts, which is a list and not a paragraph, and the dictionary scrolls in
        # place so reading it does not move everything below it.
        _pa, _pb = st.columns([1, 2], gap="medium")
        with _pa:
            with panel(title="Provenance"):
                st.markdown(
                    "<dl class='defs'>"
                    "<dt>Programme</dt><dd>EBEWE, City of Los Angeles</dd>"
                    "<dt>Publisher</dt><dd>LADBS open data portal</dd>"
                    "<dt>Licence</dt><dd>Public Domain</dd>"
                    f"<dt>Loaded</dt><dd><code>{source_label}</code></dd>"
                    "</dl>", unsafe_allow_html=True)
                pull("A row is one filing",
                     f"Not one building. {facts['filings']:,} filings cover "
                     f"{facts['buildings']:,} buildings across program years "
                     f"{facts['first_year']} to {facts['last_year']}, so a building "
                     f"appears once per year it reports.")
        with _pb:
            src_cols = [c for c in df.columns if c in COLUMN_NOTES]
            eng_cols = [c for c in df.columns if c in ENGINEERED_NOTES]
            table = pd.DataFrame({
                "Column": src_cols + eng_cols,
                "Description": [COLUMN_NOTES.get(c, "") for c in src_cols]
                               + [ENGINEERED_NOTES[c] for c in eng_cols],
                "Type": [str(df[c].dtype) for c in src_cols + eng_cols],
                "Filled %": [round(df[c].notna().mean() * 100, 1)
                             for c in src_cols + eng_cols],
            })
            _bar = lambda p: (
                f"<span class='dbar'><span style='width:{p:.0f}%'></span></span>"
                f"<span class='dnum'>{p:.0f}</span>")
            with panel(title="Data dictionary",
                       note=f"{len(table)} columns \u00b7 scrolls in place"):
                scrollbox(["Column", "Description", "Type", "Filled %"],
                          [(r["Column"], r["Description"], f"<code>{r['Type']}</code>",
                            _bar(r["Filled %"])) for _, r in table.iterrows()],
                          height=300, numeric_from=3)
                st.caption(
                    "Descriptions are the portal's. Type and fill rate are measured "
                    "on the loaded file after cleaning."
                )


def render_final(source_label):
    res = fit_model(df, (source_label, len(df), int(df["programYear"].max())))
    comp = res["comparison"]
    gb = comp[comp["Model"] == "Gradient boosting"].iloc[0]
    nb = comp[comp["Model"] == "Gradient boosting, no entity features"].iloc[0]
    base = comp.iloc[0]

    _phase_header(
        "Final — predictive modelling",
        "Fitted on load. Every figure on this page is produced by re-running the model "
        "on the file currently loaded.",
        "The Midterm showed the organisation filing a report separates compliance far "
        "more sharply than anything about the building. Is that enough to direct a "
        "limited number of audits?",
    )

    # The setup is five numbers, not a paragraph. The band states them once; the
    # caption below carries only the part that is a design decision, not a figure.
    band([
        ("Modelled on", f"{res['n_train'] + res['n_test']:,} complete filings"),
        ("Train", f"{res['n_train']:,} \u00b7 {res['train_buildings']:,} bldgs"),
        ("Test", f"{res['n_test']:,} \u00b7 {res['test_buildings']:,} bldgs"),
        ("Test base rate", f"{res['base_rate'] * 100:.2f}%"),
    ], highlight=3)
    aside(
        f"Target: **NOT COMPLIED**. Split **by building**, not by row.",
        f"A random row split would leak the same building's history into the test fold. "
        f"The test fold holds {res['n_pos']:,} non-compliant filings.",
        "Why the split is by building")

    section("01", "Three models against the baseline")
    # Four rows. The chosen model is highlighted rather than left to be found by
    # reading down the PR-AUC column.
    _best = int(comp["PR-AUC"].astype(float).idxmax())
    minitable(
        ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC",
         "Caught", "False alarms", "Missed"],
        [(r["Model"], f"{r['Accuracy']:.4f}", f"{r['Precision']:.4f}",
          f"{r['Recall']:.4f}", f"{r['F1']:.4f}", f"{r['ROC-AUC']:.4f}",
          f"{r['PR-AUC']:.4f}", f"{int(r['Caught']):,}",
          f"{int(r['False alarms']):,}", f"{int(r['Missed']):,}")
         for _, r in comp.iterrows()],
        focus=_best)
    pull(
        "The baseline is accurate and useless",
        f"Predicting COMPLIED for every filing scores **{base['Accuracy'] * 100:.2f}%** "
        f"accuracy and catches **none** of the {int(base['Missed']):,} non-compliant "
        f"filings.")
    # Two disclosures stacked took two full-width rows to show two labels.
    _m1, _m2 = st.columns(2, gap="medium")
    with _m1:
        method(
            f"Gradient boosting reaches **{gb['Accuracy'] * 100:.2f}%** accuracy, barely "
            f"higher, while catching **{int(gb['Caught']):,}** of them at "
            f"**{gb['Precision'] * 100:.1f}%** precision. Accuracy cannot tell those two "
            f"apart, which is why PR-AUC \u2014 **{base['PR-AUC']:.4f}** against "
            f"**{gb['PR-AUC']:.4f}** \u2014 carries the comparison instead.",
            "Why PR-AUC and not accuracy")
    with _m2:
        method(
            f"ROC-AUC on train against test: logistic regression {res['gap_m1']:+.4f}, "
            f"gradient boosting {res['gap_m2']:+.4f}. The boosting hyperparameters were "
            f"chosen on this gap rather than on the test score, since choosing on the "
            f"test score is itself a form of leakage.",
            "Overfitting check")

    section("02", "What the model uses")
    # Chart and reading sit side by side: the claim is about the chart, and a reader
    # who has to scroll past the bars to reach it loses the comparison.
    gl, gr = st.columns([1.35, 1], gap="large")
    imp = res["importance"].sort_values("AUC drop")
    fig = go.Figure(go.Bar(
        x=imp["AUC drop"], y=imp["Feature"], orientation="h",
        error_x=dict(type="data", array=imp["SD"], color=SLATE, thickness=1, width=3),
        marker_color=MARK,
        text=[("<0.001" if v < 0.001 else f"{v:.3f}") for v in imp["AUC drop"]],
        textposition="outside", cliponaxis=False))
    fig.update_layout(height=430, xaxis_title="ROC-AUC lost when the column is shuffled",
                      yaxis_title="", margin={"l": 10, "r": 60, "t": 10, "b": 40})
    with gl:
        chart(fig)
    ent_auc = (1 - (nb["ROC-AUC"] - 0.5) / (gb["ROC-AUC"] - 0.5)) * 100
    ent_pr = (1 - (nb["PR-AUC"] - res["base_rate"]) / (gb["PR-AUC"] - res["base_rate"])) * 100
    with gr:
        st.info(
            f"**Who files, not what is filed.** Measured above chance, the responsible "
            f"entity accounts for **{ent_auc:.1f}%** of ROC-AUC and **{ent_pr:.1f}%** of "
            f"PR-AUC."
        )
        method(
            f"Shuffling the responsible entity costs more ROC-AUC than every other column "
            f"combined, and all property-type indicators together cost almost nothing. "
            f"Refitting with the entity features withheld \u2014 the last row of the table "
            f"above \u2014 drops ROC-AUC from **{gb['ROC-AUC']:.4f}** to "
            f"**{nb['ROC-AUC']:.4f}** and PR-AUC from **{gb['PR-AUC']:.4f}** to "
            f"**{nb['PR-AUC']:.4f}**. That model still beats random selection, so the "
            f"building characteristics are not worthless \u2014 but it misses most of what "
            f"the full model catches.", "How the share is measured")

    section("03", "The review list",
            "A ranking, not a verdict \u2014 what a review list of a given size would "
            "contain.")
    # Control and consequence on the left, the list itself on the right: the reader
    # moves the slider and watches three numbers, not a table reflowing under them.
    cl, cr = st.columns([1, 2.15], gap="large")
    with cl:
        with card():
            cap = st.slider("Filings the analyst can review", min_value=50,
                            max_value=min(2000, res["n_test"]), value=250, step=50)
            order = np.argsort(-res["proba"])
            hits = int(res["y_test"][order[:cap]].sum())
            kv([
                ("Found", f"{hits:,}", hits / max(1, cap), False),
                ("Hit rate", f"{hits / cap * 100:.1f}%", hits / max(1, cap), False),
                ("Random would be", f"{res['base_rate'] * 100:.1f}%",
                 res['base_rate'], True),
                ("Of all non-compliant", f"{hits / res['n_pos'] * 100:.1f}%",
                 hits / max(1, res['n_pos']), True),
            ])
            st.caption(
                f"Reviewing {cap:,} filings at random would be expected to find about "
                f"{cap * res['base_rate']:.0f}."
            )
    with cr:
        st.dataframe(res["ranked"].head(cap).style.format({"risk": "{:.3f}"}),
                     width="stretch", hide_index=True, height=430)
        st.caption("Ranked by predicted risk on the held-out fold.")


# ----------------------------------------------------------------------------
# Phase routing
# ----------------------------------------------------------------------------
# Home and Final write their own page and stop. Preliminary and Midterm share the
# header and tab machinery below, because both are built from the same filtered
# frames and differ only in which tabs they open.
if phase == PHASE_HOME:
    render_home(source_label)
    st.stop()

if phase == PHASE_FINAL:
    render_final(source_label)
    st.stop()

# ----------------------------------------------------------------------------
# Shared header — Preliminary and Midterm
# ----------------------------------------------------------------------------
# No KPI row here. Home states the dataset-level figures, and repeating the same four
# tiles on every phase answers no question the phase is asking. The filter notice below
# covers the one case where the active selection changes what the page shows.

if filtered:
    st.warning(
        f"Filtered to program years {year_range[0]}–{year_range[1]}. Figures reflect the "
        "filtered subset, not the full dataset.",
    )

# The compliance series is computed once here and read by both the Midterm summary
# and the Compliance Trend tab. Computing it twice is how a summary drifts from the
# page it summarises.
yearly_overall = dff.groupby("programYear")["isCompliant"].mean() * 100
yearly_complete = dff_complete.groupby("programYear")["isCompliant"].mean() * 100
yearly_incomplete = dff.groupby("programYear")["isIncompleteFiling"].mean() * 100

if len(yearly_overall) >= 2:
    _ov_first, _ov_last = yearly_overall.index[0], yearly_overall.index[-1]
    _ov_drop = yearly_overall.iloc[0] - yearly_overall.iloc[-1]
    _cm_drop = (yearly_complete.iloc[0] - yearly_complete.iloc[-1]
                if len(yearly_complete) >= 2 else float("nan"))
else:
    _ov_first = _ov_last = yearly_overall.index[0] if len(yearly_overall) else "n/a"
    _ov_drop = _cm_drop = float("nan")


if phase == PHASE_PRELIM:
    _phase_header(
        "Preliminary — descriptive",
        "What the file contains, and what is wrong with it.",
        "Nothing here explains compliance \u2014 that is the Midterm. But nothing in the "
        "Midterm is trustworthy without it.",
    )
    # The band restates nothing from Home: these are properties of the file in view
    # (the year filter applies), which is what the three tabs below are about.
    _n_null_cols = int((diag["missing_report"]["Missing %"] > 0).sum())
    band([
        ("Rows in view", f"{len(dff):,}"),
        # The sidebar reports the as-loaded width; this is the width after the
        # engineered columns are added, so the label has to say which one it is.
        ("Columns after engineering", f"{dff.shape[1]}"),
        ("Columns with nulls", f"{_n_null_cols} of {dff.shape[1]}"),
        ("Structurally incomplete", f"{dff['isIncompleteFiling'].mean() * 100:.1f}%"),
    ], highlight=2)
    tabs = st.tabs([
        "Data Quality",
        "Descriptive Stats",
        "Key Figures",
    ])

    with tabs[0]:
        # A grid, not a stack. The evidence that carries the argument holds the left
        # column across two panels' worth of height; its reading and the second
        # figure sit beside it, so the whole data-quality case is one screen.
        _L, _R = st.columns([1.5, 1], gap="medium")

        miss = diag["missing_report"].reset_index(names="Column")
        _com = diag.get("comissing_fields", [])
        _m = miss.sort_values("Missing %")
        fig = px.bar(
            _m,
            x="Missing %", y="Column", orientation="h",
            color=_m["Column"].isin(_com),
            # Colour marks membership in the co-missing set, not a missingness threshold.
            # The six identical masks are the finding; every other column is context.
            color_discrete_map={True: MARK_DEEP, False: MARK_SOFT},
            # A number on all 21 bars is chaos and goes unread. The six that share
            # the mask are the finding, so only those are labelled; the axis carries
            # the rest and the tooltip has every value.
            text=[f"{v:.1f}%" if c in _com else "" for v, c in
                  zip(_m["Missing %"], _m["Column"])],
        )
        fig.update_layout(height=560, showlegend=False, xaxis_title="Missing (%)",
                          xaxis_ticksuffix="%")
        fig.update_traces(textposition="outside", cliponaxis=False)
        _pad_axis(fig, miss["Missing %"])
        with _L:
            with panel("01", "Missingness by column", f"{len(miss)} columns"):
                chart(fig)
                method(
                    f"The fields are {', '.join('`%s`' % c for c in _com)}. An "
                    f"identical mask, not merely a similar rate, is what justifies "
                    f"treating these rows as one event rather than six independent "
                    f"gaps; the flag built from it is `isIncompleteFiling`. "
                    f"Missingness is computed after the `Not Available` placeholder "
                    f"is standardized to a true null (Section 4.2) \u2014 left as-is, "
                    f"pandas treats that string as a valid category."
                )

        inc_by_year = df.groupby("programYear")["isIncompleteFiling"].mean().mul(100).round(1)
        fig2 = px.line(
            x=inc_by_year.index, y=inc_by_year.values, markers=True,
            labels={"x": "Program year", "y": "% incomplete"},
        )
        fig2.update_traces(line_color=MARK)
        fig2.update_layout(height=250, yaxis_ticksuffix="%")
        with _R:
            with panel():
                pull(f"{len(_com)} fields, one mask",
                     f"Null on exactly the same {diag['n_incomplete']:,} records "
                     f"\u2014 no exceptions either way. That pattern is "
                     f"`isIncompleteFiling`.")
            with panel("02", "Incomplete rate over time"):
                chart(fig2)
                st.caption(
                    "A rising rate makes a naive compliance trend look like falling "
                    "performance."
                )

        # Three supporting reports, one row of thirds. They answer the same question
        # from different angles, so they belong beside each other.
        _pb, _ob, _fr = st.columns(3, gap="medium")
        with _pb:
          with panel("03", "Nulled", "5.2 bounds"):
            st.caption("Data-entry errors at both ends.")
            _pr = diag["plaus_report"]
            scrollbox(
                ["Column", "Ceiling", "Nulled"],
                [(r["Column"], f"{r['Ceiling']:,.0f}",
                  f"{r['Nulled as data-entry error']:,}")
                 for _, r in _pr.iterrows()], height=160,
                focus=int(_pr["Nulled as data-entry error"].astype(float).idxmax())
                if len(_pr) else None)
        with _ob:
          with panel("04", "Flagged only", "IQR fences"):
            st.caption("Unusual but real buildings. Never removed.")
            _orp = diag["outlier_report"]
            _fcol = next((c for c in _orp.columns if "lag" in c), _orp.columns[-1])
            _ucol = next((c for c in _orp.columns if "Upper" in c), None)
            scrollbox(
                ["Column", "Fence", "Flagged"],
                [(r["Column"],
                  f"{r[_ucol]:,.2f}" if _ucol is not None else "\u2014",
                  f"{int(r[_fcol]):,}")
                 for _, r in _orp.iterrows()], height=160,
                focus=int(_orp[_fcol].astype(float).idxmax()) if len(_orp) else None)
        with _fr:
          with panel("05", "Other repairs"):
            if diag.get("n_locale_property_types"):
                st.caption(
                    f"{diag['n_locale_property_types']} filings split one category "
                    f"across two labels. Distinct types: "
                    f"{diag['property_types_before']} before, "
                    f"{diag['property_types_after']} after."
                )
                _lp = diag["locale_property_types"]
                scrollbox(["From", "To", "n"],
                          [(r["From"], r["To"], f"{int(r['Filings']):,}")
                           for _, r in _lp.iterrows()], height=130, numeric_from=2)
            st.caption(
                f"{diag['n_invalid_yearbuilt']} implausible `yearBuilt` values nulled "
                f"before `buildingAge` was derived; {diag['n_locale_fixed']} "
                f"`Non`/`Oui` entries mapped to English before any generic fill."
            )
            if diag["missing_target"] == 0:
                st.success(f"Target integrity: 0 missing `complianceStatus`.")
            else:
                st.error(f"{diag['missing_target']} missing complianceStatus values.")
            method(
                "The published file encodes the same French label two ways \u2014 once "
                "with a proper accent and once with a replacement character \u2014 so "
                "the map is keyed on a form that ignores non-alphanumerics rather "
                "than on literal strings."
            )

        section("06", "Entity field repairs and audits",
                "The field that carries the most signal, so the one worth auditing.")

        band([
            ("Distinct entities", f"{diag.get('n_entities', 0):,}"),
            ("Punctuation stripped", f"{diag.get('n_punct_stripped', 0):,}"),
            ("Legal-form merges", f"{diag.get('legal_merge_labels', 0):,}"),
            ("Merged into", f"{diag.get('legal_merge_groups', 0)} names"),
        ], highlight=2)
        method(
            f"{diag.get('legal_merge_labels', 0)} labels differing only in punctuation and "
            f"legal form were merged into {diag.get('legal_merge_groups', 0)} names \u2014 "
            "3D INVESTMENTS, LLC into 3D INVESTMENTS, and so on. The canonical name is "
            "whichever spelling carries the most filings. The rule is mechanical: a group "
            "merges only when every differing token is a legal form, so no group is "
            "adjudicated by hand.", "How the merge rule works")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Values that cannot be company names**")
            if len(diag.get("nonname", [])):
                _nn = diag["nonname"].head(8)
                minitable(["Value", "Flagged as", "Filings"],
                          [(r["Value"], r["Flagged as"], f"{int(r['Filings']):,}")
                           for _, r in _nn.iterrows()], numeric_from=2)
                aside(
                    f"{diag['nonname_share']:.2f}% of named-entity filings. Not repaired.",
                    "The true entity is unrecoverable, and assigning these to UNKNOWN "
                    "would merge them with the incomplete filings, which are almost never "
                    "compliant. A code groups as reliably as a name; only readability is "
                    "lost.", "Why these are left as they are")
            else:
                st.caption("None detected in this refresh.")
        with c2:
            st.markdown("**Labels that may be one organisation**")
            kv([("Candidate groups", f"{diag.get('dupe_groups', 0):,}"),
                ("Labels involved", f"{diag.get('dupe_labels', 0):,}"),
                ("Share of filings", f"{diag.get('dupe_share', 0):.1f}%")])
            aside(
                "What remains after the legal-form merge. Reported, not merged.",
                "These groups share a normalised key but differ by a real word \u2014 "
                "ANCHOR PACIFICA COMPANY against ANCHOR PACIFICA MANAGEMENT COMPANY, or "
                "2014 E 15TH ST, LLC against 2014 E 15TH ST ASSOCIATES, LLC. A shared key "
                "shows two names are similar; a company and its management arm are often "
                "separate filers.", "Why these are not merged")
            if len(diag.get("dupe_detail", [])):
                with st.expander("Show candidate groups"):
                    st.dataframe(diag["dupe_detail"], width="stretch", hide_index=True)

        section("05", "Availability of fields not described elsewhere",
                "Ten columns pass through unused. Recorded so an exclusion is not an "
                "oversight.")
        av = diag.get("availability", pd.DataFrame())
        if len(av):
            _ev, _nt = evidence(ratio=1.55)
            fig = px.bar(av.sort_values("Present %"), x="Present %", y="Field", orientation="h",
                         text="Present %")
            fig.update_traces(marker_color=MARK, texttemplate="%{text:.1f}%",
                              textposition="outside", cliponaxis=False)
            fig.update_layout(height=420, xaxis_range=[0, 100], yaxis_title="")
            with _ev:
                chart(fig)

            # Fields never plausibility-bounded still carry the errors 5.2 exists to remove.
            unb = av[~av["Field"].isin(PLAUSIBILITY_BOUNDS.keys())].copy()
            bad = unb[unb["Max / median"] > 1000]
            water = av[av["Field"].str.contains("ater")]
            with _nt:
                if len(bad):
                    st.warning(
                        f"**{len(bad)} field(s) never passed through the Section 5.2 "
                        f"plausibility bounds.** Read their medians and quartiles, not "
                        f"their means."
                    )
                if len(water) and water["Present %"].max() > 25 > water["Present %"].min():
                    st.info(
                        f"**The water fields split.** `totalWaterUse` reaches "
                        f"{water['Present %'].max():.1f}%, its components only "
                        f"{water[water['Present %'] < 25]['Present %'].max():.1f}% \u2014 "
                        f"aggregate analysis is supported, the indoor/outdoor breakdown is "
                        f"not."
                    )

        if diag.get("pctdiff_n"):

            # Four numeric columns do not fit a one-third column: the last one was
            # simply clipped, and a scroll wrapper cannot help because the column
            # clips first. It belongs at full width.
            if len(bad):
                with st.expander("Which fields, and by how much"):
                    minitable(
                        ["Field", "Max", "Median", "Max / median"],
                        [(f"<code>{r.Field}</code>", f"{r.Max:,.0f}",
                          f"{r.Median:,.1f}",
                          f"{r.Max / r.Median:,.0f}\u00d7" if r.Median else "\u2014")
                         for r in bad.itertuples()])
                    st.caption("Any modelling use must extend the bounding first.")
            same, n = diag["pctdiff_same"], diag["pctdiff_n"]
            pct = same / n * 100
            if pct > 99:
                st.error(
                    f"**`pctDiffNationalSiteEui` and `pctDiffNationalSourceEui` are duplicates.** "
                    f"Identical on {same:,} of {n:,} rows ({pct:.2f}%); the {n - same} exceptions "
                    "differ by 0.10, which is rounding at the source. Site and source energy differ "
                    "by definition, so two identical columns indicate a publishing artefact. Using "
                    "both in a model double-counts one measurement.",
                )

    # ----------------------------------------------------------------------------
    # Tab 3 — Descriptive Stats
    # ----------------------------------------------------------------------------
    with tabs[1]:
        section("01", "Frequency \u2014 filing volume by property type (6.1)")
        _ev, _nt = evidence(ratio=1.75)
        # A silent .head(10) dropped two categories while the caption below used the full
        # selection as its denominator. Cap at 12 to match the Top-12 preset and say so when
        # the selection is wider, rather than truncating without a note.
        _vol_all = dff_seg["propertyType"].value_counts()
        vol = _vol_all.head(12)
        if len(vol):
            fig = px.bar(x=vol.values, y=vol.index, orientation="h",
                         labels={"x": "Number of Filings", "y": ""}, text=vol.values)
            fig.update_traces(marker_color=MARK, textposition="outside",
                              texttemplate="%{text:,}", cliponaxis=False)
            fig.update_layout(height=max(380, 32 * len(vol)),
                              yaxis={"autorange": "reversed"})
            _pad_axis(fig, vol.values)
            with _ev:
                chart(fig)
            with _nt:
              if len(_vol_all) > len(vol):
                st.caption(
                    f"Showing the {len(vol)} largest of {len(_vol_all)} types in the current "
                    f"selection. The percentages below are computed on all "
                    f"{len(dff_seg):,} filings in the selection, not on the plotted subset."
                )
              if len(vol) >= 2:
                # Share is computed against every complete filing in the selection, not just the
                # plotted bars, so the sentence cannot overstate dominance when the chart is capped.
                share = vol.iloc[0] / len(dff_seg) * 100 if len(dff_seg) else 0
                ratio = vol.iloc[0] / vol.iloc[1]
                if ratio >= 1.5:
                    lead = (f"**{vol.index[0].title()} dominates by volume** \u2014 "
                            f"{ratio:.1f}\u00d7 the next largest category "
                            f"({vol.index[1].title()}), and {share:.0f}% of all "
                            f"{len(dff_seg):,} complete filings in the current selection.")
                else:
                    lead = (f"**{vol.index[0].title()} leads** at {share:.0f}% of all "
                            f"{len(dff_seg):,} complete filings in the current selection, "
                            f"but only {ratio:.2f}\u00d7 {vol.index[1].title()} \u2014 no "
                            f"single type dominates this selection.")
                pull("What the volumes mean for the Midterm", lead)
                method(
                    "Where one property type is far more common, overall accuracy can "
                    "look strong while performance on the minority types stays poor. The "
                    "share is computed against every complete filing in the selection, "
                    "not just the plotted bars, so the sentence cannot overstate dominance "
                    "when the chart is capped at 12.", "Why this matters downstream")

        # Table and chart are two views of one comparison; stacking them makes the
        # reader hold four numbers in memory to see the gap the chart already draws.
        section("02", "Central tendency \u2014 mean vs. median (6.2)",
                "The gap between them is the skew that survived cleaning.")
        rows = []
        for col in ["siteEui", "sourceEui", "co2Emissions", "energyStarScore"]:
            rows.append({
                "Metric": col,
                "Mean": round(dff[col].mean(), 2),
                "Median": round(dff[col].median(), 2),
                "n (non-null)": int(dff[col].notna().sum()),
            })
        central = pd.DataFrame(rows)
        # vertical_alignment="top" so the table header and the plot frame start on
        # the same line; without it the shorter column centres against the taller.
        _ct, _cc = st.columns([1, 1.25], gap="large", vertical_alignment="top")
        with _ct:
            minitable(["Metric", "Mean", "Median", "n"],
                      [(r["Metric"], f"{r['Mean']:,.2f}", f"{r['Median']:,.2f}",
                        f"{int(r['n (non-null)']):,}") for _, r in central.iterrows()])
        c2 = _cc
        fig = go.Figure()
        # Mean and median are two estimators, not a good one and a bad one.
        fig.add_bar(name="Mean", x=central["Metric"], y=central["Mean"], marker_color=MARK_SOFT)
        fig.add_bar(name="Median", x=central["Metric"], y=central["Median"], marker_color=MARK_DEEP)
        fig.update_layout(barmode="group", height=300,
                          margin={"l": 8, "r": 8, "t": 8, "b": 40},
                          legend={"orientation": "h", "y": -0.22})
        c2.plotly_chart(fig, width="stretch")
        method(
            "The first three run the same direction \u2014 mean above median, the "
            "signature of a right-skewed distribution pulled up by a small number of very "
            "energy-intensive buildings. `energyStarScore` runs the other way, mean "
            "*below* median, because it is a bounded 1\u2013100 percentile score rather "
            "than an unbounded physical quantity. Reporting only the mean would "
            "misrepresent both cases, in opposite directions.", "Reading the two columns")

        section("03", "Dispersion \u2014 range and standard deviation (6.3)",
                "The flag rate sits beside the standard deviation so a modeller can see "
                "how much of the tail IQR capping would touch.")
        rows = []
        for col in ["siteEui", "sourceEui", "co2Emissions"]:
            flag_col = col + "IsOutlier"
            n_valid = int(dff[col].notna().sum())
            rows.append({
                "Metric": col,
                "Min": round(dff[col].min(), 2),
                "Max": round(dff[col].max(), 2),
                "Range": round(dff[col].max() - dff[col].min(), 2),
                "Std Dev": round(dff[col].std(), 2),
                "IQR-flagged": int(dff[flag_col].sum()),
                "% flagged": round(dff[flag_col].sum() / n_valid * 100, 1) if n_valid else np.nan,
            })
        minitable(["Metric", "Max", "Std dev", "IQR-flagged", "% flagged"],
                  [(r["Metric"], f"{r['Max']:,.0f}", f"{r['Std Dev']:,.1f}",
                    f"{int(r['IQR-flagged']):,}", f"{r['% flagged']:.1f}%")
                   for r in rows])

        section("04", "Median Site EUI by property type")
        _ev, _nt = evidence(ratio=1.75)
        med = (dff_seg.groupby("propertyType")["siteEui"].median()
               .dropna().sort_values(ascending=False))
        if len(med):
            fig = px.bar(x=med.values, y=med.index, orientation="h",
                         labels={"x": "Median Site EUI (kBtu/ft²)", "y": ""},
                         text=med.round(1))
            # One series, so the hue encodes nothing — the bar length already carries the
            # value. GREEN previously coloured this chart, which gave that token a second
            # meaning on top of the one it was declared for.
            fig.update_traces(marker_color=MARK, textposition="outside", cliponaxis=False)
            fig.update_layout(height=max(380, 30 * len(med)), yaxis={"autorange": "reversed"})
            _pad_axis(fig, med.values)
            with _ev:
                chart(fig)
        if len(med) >= 2:
            hi_name, hi_val = med.index[0].title(), med.iloc[0]
            lo_name, lo_val = med.index[-1].title(), med.iloc[-1]
            ratio = hi_val / lo_val if lo_val else float("nan")
            with _nt:
                st.info(
                    f"**Use, not efficiency.** {hi_name} reaches {hi_val:,.1f} "
                    f"kBtu/ft\u00b2, {ratio:.0f}\u00d7 {lo_name} at {lo_val:,.1f}. "
                    f"Percentiles below are therefore computed within type."
                )

        section("05", "Position \u2014 percentile within property type (6.4)",
                "One citywide ranking would penalise an efficient warehouse for being a "
                "warehouse.")
        dff_seg["siteEuiPercentileInType"] = (
            dff_seg.groupby("propertyType")["siteEui"].rank(pct=True).round(3)
        )
        c1, c2 = st.columns([1, 1.9], gap="large")
        with c1:
            type_options = sorted(dff_seg["propertyType"].dropna().unique())
            if type_options:
                picked = st.selectbox("Property type", type_options)
                sub = dff_seg[dff_seg["propertyType"] == picked]["siteEui"].dropna()
                if len(sub):
                    st.metric("Buildings in type", f"{len(sub):,}")
                    q = sub.quantile([0.25, 0.5, 0.75])
                    st.write("**Quartile boundaries (Site EUI)**")
                    st.write(pd.DataFrame({"Percentile": ["25th", "50th", "75th"],
                                           "Site EUI": q.round(2).values}))
                    lookup = st.number_input(
                        "Look up a Site EUI value", min_value=0.0,
                        value=float(round(sub.median(), 1)), step=1.0,
                    )
                    pct = (sub < lookup).mean() * 100
                    st.success(f"{lookup:,.1f} kBtu/ft² sits at the **{pct:.0f}th percentile** "
                               f"within {picked}.")
        with c2:
            sample = (dff_seg[["propertyType", "siteEui", "siteEuiPercentileInType"]]
                      .dropna())
            if len(sample):
                st.write("**Sample of buildings with their within-type percentile**")
                st.dataframe(sample.sample(min(12, len(sample)), random_state=42),
                             width="stretch")

    # ----------------------------------------------------------------------------
    # Tab 4 — Figures 1-4
    # ----------------------------------------------------------------------------
    with tabs[2]:
        section("Fig. 1", "Site EUI distribution")
        eui = dff[["siteEui", "siteEuiIsOutlier"]].dropna(subset=["siteEui"])
        c1, c2 = st.columns(2, gap="large")
        with c1:
            fig = go.Figure()
            fig.add_histogram(x=eui.loc[~eui["siteEuiIsOutlier"], "siteEui"], nbinsx=60,
                              name="Within IQR fences", marker_color=MARK_SOFT)
            fig.add_histogram(x=eui.loc[eui["siteEuiIsOutlier"], "siteEui"], nbinsx=30,
                              name="Flagged outlier (not removed)", marker_color=MARK_DEEP)
            # Median and mean sit ~11 units apart on a 2,000-unit axis, so their labels
            # overlap into illegible text if both are placed at the default position.
            fig.add_vline(x=eui["siteEui"].median(), line_dash="dash", line_color=INK)
            fig.add_vline(x=eui["siteEui"].mean(), line_dash="dot", line_color=INK)
            fig.add_annotation(x=0.98, y=0.98, xref="paper", yref="paper",
                               xanchor="right", showarrow=False, align="right",
                               text=(f"<b>Median</b> {eui['siteEui'].median():,.1f}"
                                     f" &nbsp;<span style='color:{INK}'>— —</span><br>"
                                     f"<b>Mean</b> {eui['siteEui'].mean():,.1f}"
                                     f" &nbsp;<span style='color:black'>· · ·</span>"),
                               bgcolor="rgba(255,255,255,0.85)", bordercolor="#CCCCCC",
                               borderwidth=1, borderpad=6)
            fig.update_layout(barmode="overlay", height=420,
                              title=f"Site EUI (n={len(eui):,} reported values)",
                              xaxis_title="Site EUI (kBtu/ft²)", yaxis_title="Buildings",
                              legend={"orientation": "h", "y": -0.25})
            chart(fig)
        with c2:
            fig = px.histogram(x=np.log1p(eui["siteEui"].clip(lower=0)), nbins=60)
            fig.update_traces(marker_color=MARK)
            fig.update_layout(height=420, title="Log-transformed (viewing aid only)",
                              xaxis_title="log(1 + Site EUI)", yaxis_title="Buildings")
            chart(fig)
        st.caption("A viewing aid only \u2014 no `siteEui` value is modified by this panel.")

        section("Fig. 2", "Compliance rate by property type")
        _ev, _nt = evidence(ratio=1.9, note_first=True)
        _type_stats = dff_seg.groupby("propertyType")["isCompliant"].agg(["mean", "count"])
        _type_stats["mean"] *= 100
        _small = _type_stats[_type_stats["count"] < MIN_TYPE_FILINGS]
        comp_rate = (_type_stats[_type_stats["count"] >= MIN_TYPE_FILINGS]["mean"]
                     .sort_values(ascending=False))
        if len(comp_rate):
            colors = [MARK_DEEP if v < 80 else MARK_SOFT for v in comp_rate.values]
            fig = go.Figure(go.Bar(x=comp_rate.values, y=comp_rate.index, orientation="h",
                                   marker_color=colors, text=comp_rate.round(1),
                                   texttemplate="%{text}%", textposition="outside",
                                   cliponaxis=False))
            # Both labels used to sit inside the plot, landing on the bars and on
            # each other. Above the plot area they annotate the lines without
            # competing with the data.
            fig.add_vline(x=comp_rate.mean(), line_dash="dash", line_width=1,
                          line_color=AXIS_RGBA,
                          annotation_text=f"Avg {comp_rate.mean():.1f}%",
                          annotation_position="top", annotation_yshift=10,
                          annotation_font={"size": 10, "color": SLATE})
            fig.add_vline(x=80, line_dash="dot", line_width=1, line_color=AXIS_RGBA,
                          annotation_text="80% reference",
                          annotation_position="top", annotation_yshift=26,
                          annotation_font={"size": 10, "color": SLATE})
            # Explicit top margin: the two reference-line labels sit above the plot,
            # and without a title chart() would leave only 10px and clip them.
            fig.update_layout(height=max(380, 30 * len(comp_rate)),
                              margin={"t": 46},
                              xaxis_title="% Compliant", yaxis={"autorange": "reversed"},
                              xaxis_range=[0, 118],   # 18% headroom, matching _pad_axis
                              )  # no title: the note beside the chart says this,
            # and a title here would sit under the two reference-line labels
            with _ev:
                chart(fig)
        with _nt:
            if True:
                st.markdown(
                    "**The 80% line is a reading aid, not a target.** The ordinance sets "
                    "deadlines and fees, not a compliance-rate goal."
                )
                if len(_small):
                    aside(
                        f"{len(_small)} type(s) under {MIN_TYPE_FILINGS} filings excluded.",
                        "A fixed line is used rather than the citywide average so a bar's "
                    "colour changes only when that property type's own rate changes. "
                    f"Types with fewer than {MIN_TYPE_FILINGS} complete filings are "
                    "excluded because a rate computed on a handful of filings swings "
                    "across the line on one building.", "Why a fixed line")

        c1, c2 = st.container(), st.container()
        with c1:
            section("Fig. 3", "Median Site EUI by building age")
            aged = dff[dff["ageBucket"] != "UNKNOWN"]
            age_eui = aged.groupby("ageBucket")["siteEui"].median().reindex(AGE_ORDER)
            age_n = aged["ageBucket"].value_counts().reindex(AGE_ORDER)
            fig = px.bar(x=AGE_ORDER, y=age_eui.values,
                         labels={"x": "Building Age Bucket", "y": "Median Site EUI (kBtu/ft²)"},
                         text=age_eui.round(1))
            fig.update_traces(marker_color=MARK, textposition="outside", cliponaxis=False)
            fig.update_layout(height=420)
            _pad_axis(fig, age_eui.values, axis="y", pad=0.12)
            _ev, _nt = evidence(ratio=2.3)
            with _ev:
                chart(fig)
            with _nt:
                st.caption("Median, not mean \u2014 a handful of real but extreme buildings "
                           "would misrepresent the typical building of an era.")
            st.write(age_n.rename("n per bucket").to_frame().T)
        with c2:
            section("Fig. 4", "Top 15 postal codes by total emissions")
            # Summed across every program year in the current range, so at more than one
            # filing per building this stacks repeated annual readings rather than measuring
            # a single year. The axis is labelled with the range it covers, and the ratio is
            # computed below rather than quoted — an earlier version hardcoded "~8.4", the
            # same untraced figure §6.5.1 used to carry.
            zips = (dff.groupby("postalCode")["co2Emissions"].sum()
                    .sort_values(ascending=False).head(15))
            _yr_lo, _yr_hi = int(dff["programYear"].min()), int(dff["programYear"].max())
            _n_bldg = dff["buildingId"].nunique()
            _fpb = len(dff) / _n_bldg if _n_bldg else float("nan")
            fig = px.bar(x=zips.values, y=zips.index, orientation="h",
                         labels={"x": f"Total CO2e, {_yr_lo}\u2013{_yr_hi} (Metric Tons)",
                                 "y": "Postal Code"})
            # Single series again; no encoding, so no second hue.
            fig.update_traces(marker_color=MARK)
            fig.update_layout(height=560, yaxis={"autorange": "reversed", "type": "category"})
            _ev, _nt = evidence(ratio=1.8)
            with _ev:
                chart(fig)
            with _nt:
                aside(
                    f"Total, not average \u2014 retrofit targeting follows total carbon.",
                    f"The bars sum every filing from {_yr_lo} to {_yr_hi} \u2014 "
                    f"{_n_bldg:,} buildings at {_fpb:.2f} filings each \u2014 so a postal "
                    f"code with more years of coverage accumulates more. The top ranks are "
                    f"stable; the tail shifts if a single year is used instead.",
                    "What the sum covers")

    # ----------------------------------------------------------------------------
    # Tab 5 — Compliance Trend (Figure 5 + Section 6.5)
    # ----------------------------------------------------------------------------

else:
    _phase_header(
        "Midterm — diagnostic",
        "What the compliance label actually measures.",
        f"`complianceStatus` records two unrelated events under one label, and no field "
        f"separates them. The reported fall of {_ov_drop:.1f} points becomes "
        f"{_cm_drop:.1f} once filing completeness is held constant.",
    )
    # The band is the diagnostic claim in four numbers: the reported fall, the fall
    # that survives the control, the share that was filing behaviour, and the rival
    # explanation that does survive it.
    _attrib = (100 * (1 - _cm_drop / _ov_drop)) if _ov_drop else float("nan")
    _ent = dff_complete[dff_complete["entityResponsible"] != "UNKNOWN"]
    _ec = _ent.groupby("entityResponsible")["isCompliant"].agg(["mean", "size"])
    _ec = _ec[_ec["size"] >= MIN_TYPE_FILINGS]
    band([
        ("Reported fall", f"{_ov_drop:.1f} pts"),
        ("Fall among complete filings", f"{_cm_drop:.1f} pts"),
        ("Attributable to filing", f"{_attrib:.0f}%"),
        ("Spread across organisations",
         f"{(_ec['mean'].max() - _ec['mean'].min()) * 100:.1f} pts" if len(_ec) else "n/a"),
    ], highlight=3)
    tabs = st.tabs([
        "Compliance Trend",
        "Coverage Tier",
        "Responsible Entity",
        "Correlations",
    ])

    with tabs[0]:
        # The trend chart is the argument, so it holds the wide column; the caveat
        # sits beside it rather than under it.
        _ev, _nt = st.columns([2.0, 1], gap="medium")
        # series computed once above the tabs; see the note there

        # One axis. All three series are percentages of filings, so the second
        # y-scale this used to carry was never measuring anything the first could
        # not -- and a dual axis lets any two lines be made to cross wherever the
        # ranges are set, which is exactly the comparison this figure is about.
        fig = go.Figure()
        _series = [
            (yearly_overall, "Overall", MARK_SOFT, "solid"),
            (yearly_complete, "Complete filings only", MARK_DEEP, "solid"),
            (yearly_incomplete, "Incomplete filing rate", MARK, "dot"),
        ]
        for _ser, _name, _col, _dash in _series:
            fig.add_trace(go.Scatter(
                x=_ser.index, y=_ser.values, mode="lines+markers", name=_name,
                line={"color": _col, "width": 2, "dash": _dash},
                hovertemplate="%{y:.1f}%<extra>" + _name + "</extra>"))
        # Endpoint labels on the two series the figure is about. Labelling all three
        # collides: the overall rate and the incomplete rate converge at the right
        # edge, and nudging a label off its own line reads as noise. The third stays
        # in the legend and the tooltip.
        for _ser, _name, _col, _ in _series[:2]:
            fig.add_annotation(x=_ser.index[-1], y=_ser.values[-1],
                               text=f"{_ser.values[-1]:.0f}%", showarrow=False,
                               xanchor="left", xshift=10,
                               font={"size": 11, "color": INK})
        fig.update_layout(height=430, yaxis_title="% of filings",
                          yaxis_ticksuffix="%", xaxis_title="Program year",
                          margin={"l": 8, "r": 56, "t": 8, "b": 8})
        with _ev:
            with panel("Fig. 5", "Is compliance declining, or is filing completeness?"):
                chart(fig)
                st.caption("The finding is the gap, and how it widens.")
        with _nt:
          with panel(title="Read with care"):
            # The three figures this tab would otherwise tile here are already in the
            # band above the tabs, which stays on screen. Repeating them is the
            # redundancy, not the restatement of the caveat below.
            if len(yearly_overall) >= 2:
                st.warning(
                    "**Endpoint figures overstate their own precision.** On a fixed set "
                    "of buildings the same share ranges from roughly 40% to 84%."
                )
                method(
                    "The building roster is not constant \u2014 the ordinance phased in "
                    "by size, so early years cover a fraction of the buildings later "
                    "years do \u2014 and the final program year may still be partially "
                    "processed. The range depends on the cohort and end year chosen.",
                    "Why the range is that wide")


        # Balanced panel, anchored the same way as notebook Section 6.5.1: start at the
        # first year the roster reaches 95% of its peak, rather than at the earliest year
        # selected. Early program years cover a fraction of the buildings later ones do, so
        # requiring presence in those years shrinks the panel to the earliest cohort only
        # and yields different figures from the notebook for the same finding.
        _roster = dff.groupby("programYear")["buildingId"].nunique()
        _start = _roster[_roster >= 0.95 * _roster.max()].index[0]
        _win = dff[dff["programYear"] >= _start]
        _yrs = sorted(_win["programYear"].unique())
        _per = _win.groupby("buildingId")["programYear"].nunique()
        _panel = _per[_per == len(_yrs)].index

        if len(_panel) < 50 or len(_yrs) < 4:
            st.info(
                f"Only {len(_panel):,} buildings appear in all {len(_yrs)} years from {_start} — "
                "too few to compare. Widen the program-year range to see this view.",
            )
        else:
            _g = _win[_win["buildingId"].isin(_panel)]
            _gc = _g[~_g["isIncompleteFiling"]]
            _bal = pd.DataFrame({
                "Overall %": (_g.groupby("programYear")["isCompliant"].mean() * 100).round(1),
                "Complete %": (_gc.groupby("programYear")["isCompliant"].mean() * 100).round(1),
                "Incomplete %": (_g.groupby("programYear")["isIncompleteFiling"].mean() * 100).round(1),
            })
            _c = _bal["Complete %"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=_bal.index, y=_bal["Complete %"], mode="lines+markers",
                                     name="Complete filings only",
                                     line={"color": MARK_DEEP, "width": 3}))
            fig.add_trace(go.Scatter(x=_bal.index, y=_bal["Overall %"], mode="lines+markers",
                                     name="Overall", line={"color": MARK_SOFT, "width": 2, "dash": "dot"}))
            fig.update_layout(height=420, xaxis_title="Program Year", yaxis_title="% Compliant",
                              title=f"{len(_panel):,} buildings held constant, {_yrs[0]}–{_yrs[-1]}",
                              legend={"orientation": "h", "y": -0.2})
            _ev, _nt = evidence(ratio=1.75)
            with _ev:
                with panel("06", "The same trend on a fixed set of buildings",
                           f"{len(_panel):,} buildings"):
                    chart(fig)

            # The shape, stated from the data. The final-year fall is excluded when locating
            # the break, since it is usually the steepest drop and is reported separately.
            _d = np.diff(_c.values)
            if len(_d) >= 3:
                _bi = int(np.argmin(_d[:-1])) + 1
                _mid = _c.iloc[_bi:-1]
                with _nt:
                    # Three readings of one line: stacked beside the chart they annotate
                    # the shape; tiled under it they read as a scorecard of their own.
                    _mx = max(abs(_d[_bi-1]), abs(_mid.iloc[-1] - _mid.iloc[0]),
                              abs(_d[-1]), 1e-9)
                    kv([
                        (f"Break at {_c.index[_bi]}", f"{_d[_bi-1]:+.1f}",
                         abs(_d[_bi-1]) / _mx, False),
                        (f"{_mid.index[0]}\u2013{_mid.index[-1]}",
                         f"{_mid.iloc[-1] - _mid.iloc[0]:+.1f}",
                         abs(_mid.iloc[-1] - _mid.iloc[0]) / _mx, True),
                        (f"Final yr {_c.index[-1]}", f"{_d[-1]:+.1f}",
                         abs(_d[-1]) / _mx, False),
                    ])
            if len(_d) >= 3 and _mid.iloc[-1] > _mid.iloc[0]:
                pull(
                    "Held constant, compliance improved",
                    f"Compliance among complete filings rose "
                    f"**{_mid.iloc[-1] - _mid.iloc[0]:.1f} points** between "
                    f"{_mid.index[0]} and {_mid.index[-1]}. A break, a recovery, then a "
                    f"final-year drop \u2014 not a steady decline.")
                method(
                    "An endpoint comparison conceals this because a line drawn between "
                    "the first and last points passes straight over the recovery. The "
                    "window starts where roster coverage matures, matching Section 6.5.1 "
                    "of the notebook, rather than at the earliest year selected.",
                    "Why the endpoint figure disagrees")

        # The two year-by-year tables answer the same question on two rosters, so
        # they sit side by side and scroll inside their panels. Stacked and
        # full-height they were twenty rows that pushed the page down.
        trend_table = pd.DataFrame({
            "Overall Compliance %": yearly_overall.round(1),
            "Compliance % (complete filings only)": yearly_complete.round(1),
            "Incomplete Filing %": yearly_incomplete.round(1),
            "Buildings in roster": dff.groupby("programYear")["buildingId"].nunique(),
        })
        _t1, _t2 = st.columns(2, gap="medium")
        with _t1:
            with panel("07", "All filings", "unbalanced roster"):
                st.caption(
                    "Where the roster grows, a rate change partly reflects a change "
                    "in which buildings are measured."
                )
                scrollbox(["Year", "Overall %", "Complete %", "Buildings"],
                          [(str(y), f"{r['Overall Compliance %']:.1f}",
                            f"{r['Compliance % (complete filings only)']:.1f}",
                            f"{int(r['Buildings in roster']):,}")
                           for y, r in trend_table.iterrows()], height=220)
        with _t2:
            with panel("08", "Balanced panel", "same buildings every year"):
                st.caption("Roster held constant, so the movement is behaviour.")
                if "_bal" in dir():
                    scrollbox(["Year", "Overall %", "Complete %", "Incomplete %"],
                              [(str(y), f"{r['Overall %']:.1f}",
                                f"{r['Complete %']:.1f}", f"{r['Incomplete %']:.1f}")
                               for y, r in _bal.iterrows()], height=220)
                else:
                    st.caption("Not available for the current year range.")

    # ----------------------------------------------------------------------------
    # Tab 6 — Coverage Tier (Section 6.9)
    # ----------------------------------------------------------------------------
    with tabs[1]:
        section("01", "Coverage tier predicts filing, not compliance",
                "The only structural field populated on every record, incomplete filings "
                "included.")
        # Both disclosures for this section share one row rather than taking two.
        _d1, _d2 = st.columns(2, gap="medium")
        with _d1:
            method(
                "`ladbsBuildingCategory` is the ordinance's own size and ownership "
                "segmentation. Unlike "
                f"{', '.join(diag.get('comissing_fields', STRUCTURAL_COLS))} \u2014 all "
                "missing on the structurally incomplete filings \u2014 it is present on "
                "every row, which is why it can answer which buildings file incompletely "
                "at all.",
                "Why this field and not another")
        # The ordinance floor is 20,000 sq ft for private buildings and 7,500 for city-owned
        # ones, so the tiers below 20,000 exist only because city-owned buildings are covered
        # lower. Checked against the data rather than asserted from the ordinance text.
        _own = pd.crosstab(dff["sizeBand"], dff["isCityOwned"])
        for _c in (False, True):
            if _c not in _own.columns:
                _own[_c] = 0
        _small = [b for b in _own.index if str(b).startswith(("7,500", "15,000"))]
        if _small:
            _priv = int(_own.loc[_small, False].sum())
            _city = int(_own.loc[_small, True].sum())
            with _d2:
                if _priv == 0:
                    method(
                        f"The ordinance covers privately owned buildings at 20,000 sq ft "
                        f"and city-owned buildings at 7,500, so the tiers below 20,000 sq "
                        f"ft exist only because city ownership lowers the floor. That "
                        f"holds here: {', '.join(_small)} together hold {_city:,} filings "
                        f"and not one is privately owned. A compliance gap at the small "
                        f"end of the ranking below is therefore an ownership difference "
                        f"as well as a size one.",
                        "Why the small tiers are city-owned")
                else:
                    st.caption(
                        f"{_priv:,} privately owned filing(s) sit below the 20,000 sq ft "
                        f"floor, alongside {_city:,} city-owned \u2014 size and ownership "
                        f"are not fully entangled at the small end here."
                    )

        _bands = (dff.groupby("sizeBand")["isIncompleteFiling"].size()
                  .sort_values(ascending=False).index.tolist())
        if len(_bands) < 2:
            st.info(
                f"Only one coverage tier in the current selection ({_bands[0] if _bands else 'none'}), "
                "so there is no cross-tier comparison to make. Widen the filters.",
            )
        else:
            _all = (dff.groupby("sizeBand")["isCompliant"].mean() * 100).reindex(_bands)
            _cmp = (dff_complete.groupby("sizeBand")["isCompliant"].mean() * 100).reindex(_bands)
            _inc = (dff.groupby("sizeBand")["isIncompleteFiling"].mean() * 100).reindex(_bands)
            _sa, _sc = _all.max() - _all.min(), _cmp.max() - _cmp.min()

            band([
                ("Spread, all filings", f"{_sa:.1f} pts"),
                ("Spread, complete only", f"{_sc:.1f} pts"),
                ("Spread, incompleteness", f"{_inc.max() - _inc.min():.1f} pts"),
                ("Tiers compared", f"{len(_bands)}"),
            ], highlight=1)

            c1, c2 = st.columns(2, gap="large")
            with c1:
                fig = go.Figure()
                fig.add_bar(y=_bands, x=_all.values, orientation="h", name="All filings",
                            marker_color=MARK_SOFT, text=_all.round(1), textposition="outside",
                            cliponaxis=False)
                fig.add_bar(y=_bands, x=_cmp.values, orientation="h",
                            name="Complete filings only", marker_color=MARK_DEEP,
                            text=_cmp.round(1), textposition="outside", cliponaxis=False)
                fig.update_layout(height=430, barmode="group", xaxis_range=[0, 128],
                                  xaxis_title="% Compliant", yaxis={"autorange": "reversed"},
                                  title="A. Compliance by coverage tier",
                                  legend={"orientation": "h", "y": -0.2})
                chart(fig)
            with c2:
                fig = go.Figure(go.Bar(y=_bands, x=_inc.values, orientation="h",
                                       marker_color=MARK, text=_inc.round(1),
                                       textposition="outside", cliponaxis=False))
                # Default annotation position lands on the top bar's value label.
                fig.add_vline(x=dff["isIncompleteFiling"].mean() * 100, line_dash="dash",
                              line_width=1, line_color=AXIS_RGBA,
                              annotation_position="top", annotation_yshift=10,
                              annotation_font={"size": 10, "color": SLATE},
                              annotation_text=f"overall {dff['isIncompleteFiling'].mean()*100:.1f}%")
                fig.update_layout(height=430, xaxis_range=[0, _inc.max() * 1.3],
                                  margin={"t": 62},
                                  xaxis_title="% of filings structurally incomplete",
                                  yaxis={"autorange": "reversed"},
                                  title="B. Filing incompleteness by coverage tier")
                chart(fig)

            # The conclusion holds only if the spread genuinely collapses. State it either way.
            if _sa > 0 and _sc < _sa / 2:
                pull(
                    "Coverage tier predicts filing, not compliance",
                    f"Compliance varies **{_sa:.1f} points** across coverage tiers, but only "
                    f"**{_sc:.1f} points** among complete filings. Panel B is the mechanism: "
                    f"incompleteness runs from {_inc.min():.1f}% for {_inc.idxmin()} to "
                    f"{_inc.max():.1f}% for {_inc.idxmax()}. Because incomplete filings are "
                    f"almost never compliant, that pattern surfaces in the headline rate as "
                    f"though it were a difference in performance.")
            else:
                pull(
                    "The spread does not collapse",
                    f"Excluding incomplete filings moves the compliance spread from "
                    f"**{_sa:.1f}** to **{_sc:.1f} points**. Coverage tier is associated "
                    f"with compliance beyond its association with filing completeness, so "
                    f"the two panels are separate findings rather than one mechanism.")

            section("02", "Incompleteness by tier and program year",
                    "Tiers moving together means a programme change; one tier alone means "
                    "a phase-in date.")
            _ev, _nt = evidence(ratio=1.9)
            _piv = (dff.pivot_table(index="programYear", columns="sizeBand",
                                    values="isIncompleteFiling", aggfunc="mean") * 100).round(1)
            if len(_piv) >= 2 and _piv.shape[1] >= 2:
                fig = go.Figure()
                for _i, _band in enumerate(_piv.columns):
                    # Coverage tier is ordered by building size, so the series take an
                    # ordered ramp rather than Plotly's categorical cycle — which put an
                    # arbitrary red beside an arbitrary green on a size variable and
                    # invited a good/bad reading of what is just "bigger" and "smaller".
                    fig.add_trace(go.Scatter(
                        x=_piv.index, y=_piv[_band], mode="lines+markers", name=_band,
                        line={"color": TIER_SEQUENCE[_i % len(TIER_SEQUENCE)], "width": 2},
                        marker={"size": 6}))
                fig.update_layout(height=420, xaxis_title="Program Year",
                                  yaxis_title="% structurally incomplete",
                                  legend={"orientation": "h", "y": -0.25})
                with _ev:
                    chart(fig)

                _yoy = _piv.diff()
                _rose = (_yoy > 0).sum(axis=1)
                _broad = _rose[_rose == _rose.max()]
                with _nt:
                    if _rose.max() == _piv.shape[1]:
                        st.caption(
                            f"All {_piv.shape[1]} tiers rose together in "
                            f"{', '.join(str(y) for y in _broad.index)}. Breadth alone does "
                            "not identify one year \u2014 compare magnitude and evenness "
                            "before reading any single year as a programme-wide event."
                        )
                    st.info(
                        "`isCityOwned` and `sizeBand` derive from this field, so they are "
                        "the only structural features usable across all filings."
                    )
                    method(
                        "The 2019 break has one documented cause \u2014 deadlines for "
                        "Program Years 2019\u20132021 were tolled. The size-tier phase-in "
                        "is not a second one: the 20,000\u201349,999 sq ft band steps at "
                        "2018, visible in the chart.", "On the 2019 break")
            with st.expander("Incompleteness by tier and year, as a table"):
                minitable(["Year"] + [str(c) for c in _piv.columns],
                          [tuple([str(y)] + [f"{v:.1f}" for v in r])
                           for y, r in _piv.iterrows()])

    # ----------------------------------------------------------------------------
    # Tab 7 — Responsible Entity (Section 6.7)
    # ----------------------------------------------------------------------------
    with tabs[2]:
        section("01", "Who files matters more than what is filed",
                "Complete filings only \u2014 incomplete ones carry UNKNOWN.")

        MIN_FILINGS = 100
        _ag = (dff_complete[dff_complete["entityResponsible"] != "UNKNOWN"]
               .groupby("entityResponsible")["isCompliant"]
               .agg(["mean", "count"]).rename(columns={"mean": "compliance", "count": "filings"}))
        _ag["compliance"] = (_ag["compliance"] * 100).round(1)
        _big = _ag[_ag["filings"] >= MIN_FILINGS].sort_values("compliance")

        if len(_big) < 3:
            st.info(
                f"Only {len(_big)} agents have {MIN_FILINGS}+ complete filings in the current "
                "selection. Widen the filters to compare agents.",
            )
        else:
            _base = dff_complete["isCompliant"].mean() * 100
            # The two spreads do not share a denominator: agents are ranked across every
            # complete filing, property types across the current sidebar selection. An earlier
            # version described them as "the same basis", which was not true. Both bases are
            # now stated, and the all-types figure is computed alongside so the reader can see
            # how much the selection is doing.
            _tstat = dff_seg.groupby("propertyType")["isCompliant"].agg(["mean", "count"])
            _tstat["mean"] *= 100
            _types = _tstat[_tstat["count"] >= MIN_TYPE_FILINGS]["mean"]
            _tall = dff_complete.groupby("propertyType")["isCompliant"].agg(["mean", "count"])
            _tall["mean"] *= 100
            _tall = _tall[_tall["count"] >= MIN_TYPE_FILINGS]["mean"]

            # One decimal, not zero. The per-agent rates are rounded to 1dp above, so this
            # spread is exactly 98.5 on the current data — and Python's round-half-to-even
            # rendered that as "98" while the table below said 98.5. Two figures for one
            # quantity, differing only by how they were displayed.
            band([
                (f"Agents with {MIN_FILINGS}+ filings", f"{len(_big):,}"),
                ("Spread across agents",
                 f"{_big['compliance'].max() - _big['compliance'].min():.1f} pts"),
                ("Spread across property types",
                 f"{_types.max() - _types.min():.1f} pts" if len(_types) > 1 else "n/a"),
                ("Baseline, complete filings", f"{_base:.1f}%"),
            ], highlight=1)
            # The property-type spread is dominated by whichever single category sits below
            # the benchmark; report it with and without, since the decomposition below shows
            # that outlier is itself an operator effect.
            if len(_types) > 2:
                _wo = _types.drop(_types.idxmin())
                # The bases were previously described in a 106-word paragraph. They are two
                # measurements with different denominators, which is a comparison — so it is
                # shown as one, with the warning about comparability kept as the lead.
                aside(
                        "Different denominators \u2014 not comparable as ratios. Both "
                        "bases are stated.",
                        "Agents are ranked across every complete filing; property types "
                        "across the current sidebar selection. Incomplete filings are "
                        "excluded throughout, since they carry UNKNOWN and are almost "
                        "never compliant, so including them would manufacture the "
                        "variation this view measures.", "What each row is computed over")
                minitable(
                    ["Spread", "Range", "Basis", "Population"],
                    [("Across responsible agents",
                      f"{_big['compliance'].max() - _big['compliance'].min():.1f} pts",
                      f"agents with {MIN_FILINGS}+ complete filings",
                      f"all {len(dff_complete):,} complete filings"),
                     ("Across property types (selection)",
                      f"{_types.max() - _types.min():.1f} pts",
                      f"types with {MIN_TYPE_FILINGS}+ filings",
                      f"{len(_types)} types in the selection"),
                     ("Across property types (all qualifying)",
                      f"{_tall.max() - _tall.min():.1f} pts",
                      f"types with {MIN_TYPE_FILINGS}+ filings",
                      f"{len(_tall)} types")],
                    numeric_from=1)
                pull(
                    "The entity separates compliance more sharply than building function",
                    f"Drop the lowest type ({_types.idxmin().title()}, "
                    f"{_types.min():.1f}%) and the remaining {len(_wo)} span only "
                    f"**{_wo.max() - _wo.min():.1f} points**.")
            else:
                st.caption(
                    f"Too few property types to report a spread. The agent spread covers "
                    f"all {len(dff_complete):,} complete filings.")

            _show = pd.concat([_big.head(10), _big.tail(10)]).drop_duplicates()
            fig = go.Figure(go.Bar(
                x=_show["compliance"], y=_show.index, orientation="h",
                marker_color=[MARK_DEEP if v < _base else MARK_SOFT for v in _show["compliance"]],
                text=_show["compliance"], texttemplate="%{text}%",
                textposition="outside", cliponaxis=False,
                customdata=_show["filings"],
                hovertemplate="%{y}<br>%{x}% compliant<br>%{customdata:,} filings<extra></extra>"))
            fig.add_vline(x=_base, line_dash="dash", line_width=1, line_color=AXIS_RGBA,
                          annotation_text=f"Baseline {_base:.1f}%",
                          annotation_position="top", annotation_yshift=10,
                          annotation_font={"size": 10, "color": SLATE})
            fig.update_layout(height=max(420, 26 * len(_show)), xaxis_range=[0, 118],
                              margin={"t": 56},
                              xaxis_title="% Compliant", yaxis={"autorange": "reversed"},
                              title="Lowest and highest compliance among high-volume agents")
            section("02", "The agents at each end")
            chart(fig)

            section("03", "Does a pooled agent rate describe any actual year?")
            _yrs = sorted(dff_complete["programYear"].unique())
            if len(_yrs) >= 4:
                _mid = _yrs[len(_yrs) // 2]
                _rows = []
                for a in _big.index:
                    g = dff_complete[dff_complete["entityResponsible"] == a]
                    e = g[g["programYear"] < _mid]["isCompliant"]
                    l = g[g["programYear"] >= _mid]["isCompliant"]
                    if len(e) >= 20 and len(l) >= 20:
                        _rows.append({"Agent": a,
                                      f"Before {_mid} %": round(e.mean() * 100, 1),
                                      f"Before {_mid} n": len(e),
                                      f"From {_mid} %": round(l.mean() * 100, 1),
                                      f"From {_mid} n": len(l),
                                      "Change": round((l.mean() - e.mean()) * 100, 1),
                                      "Pooled %": round(g["isCompliant"].mean() * 100, 1)})
                if _rows:
                    _reg = pd.DataFrame(_rows).sort_values("Change")
                    _broke = _reg[_reg["Change"].abs() >= 30]
                    minitable(["Agent", "Before 2021 %", "n", "From 2021 %", "n",
                               "Change", "Pooled %"],
                              [(r["Agent"], f"{r['Before 2021 %']:.1f}",
                                f"{int(r['Before 2021 n']):,}",
                                f"{r['From 2021 %']:.1f}", f"{int(r['From 2021 n']):,}",
                                f"{r['Change']:+.1f}", f"{r['Pooled %']:.1f}")
                               for _, r in _reg.head(10).iterrows()],
                              focus=0)
                    # One sentence repeated per agent said the same thing three times.
                    # The claim is made once; the agents it applies to are the table.
                    if len(_broke):
                        pull(
                            "A pooled rate describes neither period",
                            f"**{len(_broke)} agent(s)** move by 30 points or more across "
                            f"{_mid}. A static per-agent encoding would be wrong on both "
                            f"sides of the break; a lagged feature \u2014 the same "
                            f"building's compliance last year \u2014 would not.")
                        minitable(
                            ["Agent", f"Before {_mid}", f"From {_mid}", "Pooled"],
                            [(r["Agent"], f"{r[f'Before {_mid} %']:.1f}%",
                              f"{r[f'From {_mid} %']:.1f}%", f"{r['Pooled %']:.1f}%")
                             for _, r in _broke.iterrows()])
                    if _broke.empty:
                        st.caption("No agent changes by 30+ points across this window; the pooled "
                                   "rates can be read at face value.")
            else:
                st.caption("Widen the program-year range to compare agents across periods.")

            st.divider()
            section("04", "Is the lowest property type a category or its operators?")
            _flagged = _types[_types < 80].sort_values()
            if _flagged.empty:
                st.info("No property type falls below the 80% reference line in the current "
                        "selection.")
            else:
                _t = _flagged.index[0]
                _cat = dff_complete[dff_complete["propertyType"] == _t]
                _ops = (_cat.groupby("entityResponsible")["isCompliant"]
                        .agg(["mean", "count"]).sort_values("count", ascending=False))
                _ops["mean"] = (_ops["mean"] * 100).round(1)
                # Match notebook §6.7.3. A bare `mean < _base` treats an operator 0.2 points
                # below baseline as equivalent to one 41 points below, which swept VERT ENERGY
                # GROUP, INC (92.1% against a 92.2% baseline) in as a self-storage "driver".
                # The margin is a declared share of the category's own deficit, so it scales
                # with the category and the baseline instead of being a bare number of points.
                _cat_rate = _cat["isCompliant"].mean() * 100
                _deficit = _base - _cat_rate
                _margin = DEFICIT_SHARE * _deficit
                _large = _ops[_ops["count"] >= MIN_OPERATOR_FILINGS]
                _drv = _large[_large["mean"] < _base - _margin].index.tolist()
                _sens = {sh: _large[_large["mean"] < _base - sh * _deficit].index.tolist()
                         for sh in (0.10, 0.25, 0.50, 0.75)}
                _unguarded = _large[_large["mean"] < _base].index.tolist()

                if not _drv:
                    st.caption(f"No single large operator accounts for the {_t} deficit — it looks "
                               "like a genuine property-type effect.")
                else:
                    _isd = _cat["entityResponsible"].isin(_drv)
                    _sp = _cat.groupby(_isd)["isCompliant"].agg(["mean", "count"])
                    _sp["mean"] = (_sp["mean"] * 100).round(1)
                    _rest_r, _rest_n = _sp.loc[False, "mean"], _sp.loc[False, "count"]
                    _drv_r, _drv_n = _sp.loc[True, "mean"], _sp.loc[True, "count"]
                    _tot = _rest_n + _drv_n
                    _cf = (_rest_n * _rest_r + _drv_n * _base) / _tot

                    k1, k2, k3 = st.columns(3)
                    k1.metric(f"{_t} overall", f"{_cat['isCompliant'].mean()*100:.1f}%")
                    k2.metric(f"{len(_drv)} large operator(s)",
                              f"{_drv_r:.1f}%", f"{_drv_n:,} filings", delta_color="off")
                    k3.metric("Everyone else in the category",
                              f"{_rest_r:.1f}%", f"{_rest_n:,} filings", delta_color="off")
                    pull(
                        "An operator effect presenting as a building-function effect",
                        f"Those operators hold {_drv_n/_tot*100:.0f}% of the category. Had "
                        f"they filed at baseline, {_t} would sit at **{_cf:.1f}%** and "
                        f"would not fall below the 80% line in Figure 2 at all.")
                    method(
                        f"The remaining filings sit {abs(_base-_rest_r):.1f} points from "
                        f"the {_base:.1f}% baseline, against a "
                        f"{abs(_rest_r-_drv_r):.1f}-point gap between the two groups.",
                        "The two groups side by side")
                    # This was a paragraph. It is tabular data — four thresholds, the rate
                    # each implies, and the set each selects — and serialising it into prose
                    # buried the only thing that matters: the set does not change. As a table
                    # the constancy is visible in one pass down the last column.
                    aside(
                        f"Driver rule: {MIN_OPERATOR_FILINGS}+ filings and at least "
                        f"{_margin:.1f} points below the {_base:.1f}% baseline.",
                        f"The margin is {DEFICIT_SHARE:.0%} of the category's own "
                        f"{_deficit:.1f}-point deficit \u2014 a judgement, so its effect is "
                        f"shown in full below. If the selected set moved across this range "
                        f"the decomposition would be fragile and should not be read as a "
                        f"finding.", "How the margin was chosen")
                    # Column order matters here. "Selected" carries the whole finding — the set
                    # is the same at every margin and gains a third operator only at 0% — and it
                    # was last, so it was the column that truncated. Every row then displayed the
                    # same two names and the one row that differs looked identical to the four
                    # that do not, which is the opposite of what the table is for. It now sits
                    # second. The "Operators" count is dropped as redundant: the names are there
                    # to be counted, and it was occupying width the names needed.
                    _sens_rows = [
                        {"Margin": f"{sh:.0%} of deficit",
                         "Selected": ", ".join(a) if a else "none",
                         "Threshold": f"{_base - sh*_deficit:.1f}%"}
                        for sh, a in _sens.items()
                    ]
                    _sens_rows.append({
                        "Margin": "0% (unguarded)",
                        "Selected": ", ".join(_unguarded) if _unguarded else "none",
                        "Threshold": f"{_base:.1f}%"})
                    st.dataframe(pd.DataFrame(_sens_rows), width="stretch", hide_index=True)
                    st.dataframe(_ops.head(8).rename(
                        columns={"mean": "compliance %", "count": "filings"}), width="stretch")

            section("05", "Is the agent effect an outsourcing effect?",
                    "Vendors, managers, owners \u2014 three levels instead of thousands, if "
                    "compliance tracked the split.")
            VENDOR_KEYS = ["ENERGY", "EFFICIENCY", "CONSERVICE", "REALPAGE", "YARDI", "WEGOWISE",
                           "CODEGREEN", "CARLETON", "UTILITY MANAGEMENT", "CONSULTING",
                           "BENCHMARK", "VERT", "SERVIDYNE", "UL VERIFICATION"]
            MANAGER_KEYS = ["MANAGEMENT", "RESIDENTIAL SERVICES", "COMMUNITIES", "PROPERTIES",
                            "REAL ESTATE", "APARTMENTS", "CBRE", "MOSS &", "BERGLAS"]
            OWNER_SET = {"PUBLIC STORAGE", "EXTRA SPACE STORAGE", "STORAGE ETC", "PRICE SELF STORAGE",
                         "EZ STORAGE RELATED COMPANIES", "PROLOGIS", "REXFORD INDUSTRIAL",
                         "EQUITY RESIDENTIAL", "KAISER FOUNDATION HOSPITALS",
                         "LOS ANGELES WORLD AIRPORTS", "LOS ANGELES DEPARTMENT OF WATER AND POWER",
                         "CITY OF LOS ANGELES, RECREATION AND PARKS", "LOS ANGELES PUBLIC LIBRARY",
                         "UNIVERSITY OF SOUTHERN CALIFORNIA", "PARAMOUNT STUDIOS", "KROGER",
                         "FRED LEEDS PROPERTIES, INC", "ECE_COLA_GSD"}

            def _kind(name):
                u = str(name).upper()
                if name in OWNER_SET:
                    return "OWNER"
                if any(k in u for k in VENDOR_KEYS):
                    return "VENDOR"
                if any(k in u for k in MANAGER_KEYS):
                    return "MANAGER"
                return "UNCLASSIFIED"

            _typed = _big.copy()
            _typed["type"] = [_kind(a) for a in _typed.index]
            _known = _typed[_typed["type"] != "UNCLASSIFIED"]
            _movers = list(_broke["Agent"]) if "_broke" in dir() and len(_broke) else []

            if _known["type"].nunique() < 2:
                st.info("Too few classified agents in the current selection to compare filer types.")
            else:
                def _by_type(exclude=()):
                    rows = []
                    for t, grp in _known.groupby("type"):
                        ags = [a for a in grp.index if a not in exclude]
                        if not ags:
                            continue
                        f = dff_complete[dff_complete["entityResponsible"].isin(ags)]
                        rows.append({"Filer type": t, "Agents": len(ags), "Filings": len(f),
                                     "Compliance %": round(f["isCompliant"].mean() * 100, 1)})
                    return pd.DataFrame(rows).sort_values("Compliance %")

                _t1, _t2 = _by_type(), _by_type(exclude=_movers)
                c1, c2 = st.columns(2)
                c1.markdown("**All classified agents**")
                c1.dataframe(_t1, width="stretch", hide_index=True)
                c2.markdown(f"**Excluding the {len(_movers)} regime-changing agent(s)**"
                            if _movers else "**No regime-changing agents to exclude**")
                c2.dataframe(_t2, width="stretch", hide_index=True)

                if len(_t2) >= 2:
                    _sp1 = _t1["Compliance %"].max() - _t1["Compliance %"].min()
                    _sp2 = _t2["Compliance %"].max() - _t2["Compliance %"].min()
                    # The test is self-filed against outsourced, not the max-minus-min spread.
                    # The spread lands near 5 points and moves either side of it depending on how
                    # a handful of borderline agents are named, so it is reported but not used as
                    # a decision rule. An earlier version keyed a verdict off a 5-point cutoff and
                    # returned the opposite conclusion to the notebook on a 0.2-point difference.
                    #
                    # Outsourced pools VENDOR and MANAGER: a property manager filing on an
                    # owner's behalf is outsourcing as much as a compliance vendor is. An earlier
                    # version tested OWNER against VENDOR alone, which dropped managers from the
                    # outsourced side of a hypothesis about outsourcing. Mirrors §6.7.4.
                    _d = _t2.set_index("Filer type")["Compliance %"].to_dict()
                    _n = _t2.set_index("Filer type")["Filings"].to_dict()
                    if "OWNER" in _d:
                        _out_types = [t for t in ("VENDOR", "MANAGER") if t in _d]
                        _out_ags = [a for a, r in _known.iterrows()
                                    if r["type"] in _out_types and a not in _movers]
                        _out_f = dff_complete[dff_complete["entityResponsible"].isin(_out_ags)]
                        _out_r = _out_f["isCompliant"].mean() * 100 if len(_out_f) else float("nan")
                        _ov = abs(_d["OWNER"] - _out_r)
                        _mgr = _d.get("MANAGER")
                        _mgr_n = int(_n.get("MANAGER", 0))
                        # One paragraph previously carried the finding, the numbers behind it,
                        # a robustness check and the Midterm consequence. They are four
                        # different kinds of statement and a reader has to separate them
                        # anyway; doing it here means the finding is legible at a glance and
                        # the support is available rather than in the way.
                        st.warning(
                            f"**Outsourcing does not explain the agent effect.** Self-filed "
                            f"{_d['OWNER']:.1f}% against outsourced {_out_r:.1f}% — a gap of "
                            f"{_ov:.1f} points"
                            + (f", both within a point of the {_base:.1f}% baseline."
                               if max(abs(_d["OWNER"] - _base), abs(_out_r - _base)) < 1
                               else f", against a {_base:.1f}% baseline."),
                        )
                        _fc1, _fc2 = st.columns(2)
                        _fc1.markdown(
                            f"Self-filed — OWNER  \n**{int(_n['OWNER']):,}** filings")
                        _fc2.markdown(
                            f"Outsourced — {' + '.join(_out_types)}  \n"
                            f"**{len(_out_f):,}** filings")
                        _notes = []
                        if _out_r <= _d["OWNER"] + 1:
                            _notes.append("Paying someone else to file is not associated with a "
                                          "better outcome here.")
                        else:
                            _notes.append("The two sit close enough that filer type does not "
                                          "separate them usefully.")
                        if "VENDOR" in _d:
                            _notes.append(
                                f"Vendors alone sit at {_d['VENDOR']:.1f}%, "
                                + ("at or below baseline" if _d["VENDOR"] <= _base else "above baseline")
                                + ", so this is not an artefact of pooling them with managers.")
                        if _mgr is not None:
                            _notes.append(
                                f"Manager-filed buildings do sit higher ({_mgr:.1f}% on "
                                f"{_mgr_n:,} filings), but on a small base with most of that group "
                                f"near 100% — suggestive, not a finding.")
                        _notes.append("**For the Midterm:** filer type is not a viable "
                                      "low-cardinality substitute. A model must carry agent "
                                      "identity, with high-volume agents retained and the rest "
                                      "bucketed.")
                        # Six stacked paragraphs. The one that changes what a modeller
                        # does stays visible; the supporting observations go behind a
                        # disclosure, where they can be read in full if challenged.
                        st.caption(_notes[-1])
                        with st.expander("The supporting observations"):
                            st.markdown("\n".join(f"- {n}" for n in _notes[:-1]))
                            st.caption(
                                f"Widest gap between any two filer types: {_sp1:.1f} points "
                                f"across all classified agents, {_sp2:.1f} with the "
                                f"regime-changers removed. That depends on which side a few "
                                f"ambiguously named firms are placed on, so it is not used "
                                f"to decide the question."
                            )
                _unc = _typed[_typed["type"] == "UNCLASSIFIED"]
                if len(_unc):
                    _uf = dff_complete[dff_complete["entityResponsible"].isin(_unc.index)]
                    _ur = _uf["isCompliant"].mean() * 100 if len(_uf) else float("nan")
                    aside(
                        f"{len(_unc)} agent(s) unclassified, covering {len(_uf):,} filings "
                        f"at {_ur:.1f}% compliance.",
                        "They sit high, but the effect of classifying them depends on "
                        "which group they join \u2014 the highest would widen the gap, the "
                        "lowest would narrow it. This residual is a limitation of the "
                        "classification, not something the result is demonstrably robust "
                        "to.", "What the unclassified agents do to the result")

    # ----------------------------------------------------------------------------
    # Tab 8 — Correlations (Figure 6 + Section 6.6)
    # ----------------------------------------------------------------------------
    with tabs[3]:
        section("Fig. 6", "Correlation heatmap")
        _ev, _nt = evidence(ratio=1.9)
        corr_cols = CORR_COLS + ["isCompliant"]
        corr = dff[corr_cols].corr().round(2)
        masked = corr.mask(np.triu(np.ones(corr.shape), k=1).astype(bool))
        fig = px.imshow(masked, text_auto=True, zmin=-1, zmax=1,
                        color_continuous_scale=CORR_SCALE, aspect="auto")
        fig.update_layout(
            height=640,
            title="Lower triangle only — the upper half is redundant",
            margin={"l": 140, "r": 40, "b": 160, "t": 60},
            xaxis={"tickangle": -45, "side": "bottom", "automargin": True},
            yaxis={"automargin": True},
            coloraxis_colorbar={"thickness": 14},
        )
        with _ev:
            chart(fig)

        with _nt:
            st.warning(
                "**Not all strong correlations are the same kind of finding.** `siteEui` ↔ `sourceEui` "
                "(r ≈ 0.94) is *empirical* — two distinct measurements that move together because "
                "efficient buildings tend to be efficient on both. `siteEui` ↔ `ghgIntensityPer1kSqft` "
                "(r ≈ 0.98) is *structural*: GHG intensity is calculated from the same energy data Site "
                "EUI already represents. No amount of cleaning will reduce the second one, and treating "
                "it as a data-quality problem would be a mistake.",
            )

        section("02", "Flagged pairs (|r| > 0.80)")
        flagged = []
        for i in range(len(CORR_COLS)):
            for j in range(i + 1, len(CORR_COLS)):
                a, b = CORR_COLS[i], CORR_COLS[j]
                r = corr.loc[a, b]
                if abs(r) > 0.80:
                    flagged.append({
                        "Pair": f"{a} ↔ {b}",
                        "r": r,
                        "Kind": "STRUCTURAL (definitional)" if frozenset([a, b]) in STRUCTURAL_PAIRS
                                else "EMPIRICAL",
                    })
        st.dataframe(pd.DataFrame(flagged) if flagged else pd.DataFrame({"Pair": ["None"]}),
                     width="stretch", hide_index=True)
        st.markdown(
            "**Modeling recommendation for Midterm:** what `siteEui` and `ghgIntensityPer1kSqft` "
            "share is normalisation by floor area, not a common input — `co2Emissions` itself "
            "correlates with `siteEui` at only r ≈ 0.42, and with `grossFloorArea` at r ≈ 0.44, "
            "because it is a building-size quantity while the two intensities are size-neutral. "
            "Use *either* the EUI fields *or* `ghgIntensityPer1kSqft` in a given model — including "
            "both adds no information, it just inflates the standard errors on both coefficients."
        )

        section("03", "Correlation with the classification target")
        target_corr = (dff[corr_cols].corr()["isCompliant"].drop("isCompliant")
                       .sort_values(ascending=False).round(3))
        fig = px.bar(x=target_corr.values, y=target_corr.index, orientation="h",
                     labels={"x": "r with isCompliant", "y": ""}, text=target_corr.values)
        fig.update_traces(marker_color=MARK, textposition="outside", cliponaxis=False)
        fig.update_layout(height=380, xaxis_range=[-0.1, 0.1],
                          yaxis={"autorange": "reversed"})
        chart(fig)
        st.error(
            "Every continuous feature correlates with `isCompliant` at |r| < 0.05. Combined with "
            "the finding that `isIncompleteFiling` almost perfectly separates the two classes, "
            "compliance looks driven by categorical and structural factors — property type, "
            "responsible entity, filing completeness — rather than by energy performance itself. "
            "A Midterm model built only on continuous energy metrics is unlikely to perform well.",
        )
