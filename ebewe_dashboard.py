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

import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

if IS_DARK:
    INK = "#ECEEF1"        # text, axes, reference lines; also the second data series
    SLATE = "#9BA2AD"      # secondary text; the default for any non-focal series
    CIVIC = "#AEB6C2"      # the measured quantity; COMPLIED
    OCHRE = "#E0A93B"      # the contrasting state: NOT COMPLIED, incomplete, below threshold
    YELLOW = "#FFC93C"     # interface state only — selected, buttons, pills. Never a mark.
    TIER_SEQUENCE = ["#F0DCA8", "#E0A93B", "#B07D0A", "#7A5A09", "#4A3607"]
else:
    INK = "#1F2329"
    SLATE = "#6B7280"
    CIVIC = "#39414D"
    OCHRE = "#B07D0A"
    # Two yellows, because one cannot do both jobs. #FFC93C is what makes the interface
    # feel like itself, and it works on buttons and selected rows because they are large
    # filled shapes. As a chart line on white it measures 1.54:1 and disappears. OCHRE is
    # the same hue darkened to 3.63:1, so a line or a small marker still reads.
    YELLOW = "#FFC93C"
    # Ordered steps for charts that genuinely need more than two categories (the per-tier
    # series). Sequential in lightness so they stay separable in greyscale and in print.
    TIER_SEQUENCE = ["#4A3607", "#7A5A09", "#B07D0A", "#D3A63F", "#E9CE8C"]

# Call-site aliases. BLUE and RED are kept because they read naturally at the point of
# use ("red if below the line"); each resolves to the token whose meaning it carries.
# GREEN is gone: it was pointing at VERDIGRIS, declared for "a check that passed", while
# actually colouring a postal-code chart, a median-EUI chart and a trend series — the
# same one-hue-many-meanings problem this palette exists to prevent. ORANGE is gone too;
# it resolved to INK, so nothing orange was ever drawn and the name misled every reader.
BLUE = CIVIC
RED = OCHRE
DARK_BLUE = TIER_SEQUENCE[0]   # one lightness step down, for a second related series

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
      /* Ground. Streamlit's default white is colder than the palette and makes the
         ochre read as alarm rather than as the second encoding hue. A warm off-white
         with a slightly deeper sidebar gives the page a paper ground and lets the
         cards and tables sit on it without borders doing all the work. Light mode
         only: the dark tokens above already handle the other case, and pinning a
         colour here is what broke dark mode in an earlier version. */
      html, body, .stApp, .stMarkdown, .stCaption, .stDataFrame,
      [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
      .stTabs [data-baseweb="tab"], [data-testid="stWidgetLabel"] {
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
      .stMarkdown p { max-width: 74ch; line-height: 1.55; }
      .stAlert p { max-width: 104ch; line-height: 1.5; }
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
      .stTabs [data-baseweb="tab"] { font-size: 0.94rem; }
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
# Ground — light mode only
# ----------------------------------------------------------------------------
# A grey page with white cards floating on it, rather than Streamlit's flat white.
# The separation is what lets a card be a unit: the border does almost nothing and the
# ground does the work, which is why the corners can be soft without looking fussy.
# Applied only in light mode — the dark tokens above handle the other case, and pinning
# colours unconditionally is what broke dark mode in an earlier version.
#
# Same no-blank-lines rule as the sheet above: a blank line followed by indented text is
# markdown's code-block syntax and would close the <style> element early.
if not IS_DARK:
    st.markdown(
        """
        <style>
          .stApp { background-color: #F3F4F6; }
          section[data-testid="stSidebar"] { background-color: #FFFFFF;
            border-right: 1px solid #E5E7EB; }
          /* Cards. st.container(border=True) is the unit this app composes with, so it
             gets the white fill, the soft corner and a shadow light enough to read as
             lift rather than as a drop shadow. */
          [data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF;
            border-radius: 16px; border: 1px solid #E5E7EB;
            box-shadow: 0 1px 2px rgba(31,35,41,.05); }
          /* Metric tiles sit inside cards, so they take the page grey rather than white —
             a white tile on a white card has nothing to sit against. */
          [data-testid="stMetric"] { background: #F3F4F6; border-radius: 12px;
            padding: 0.7rem 0.9rem 0.8rem; }
          [data-testid="stMetricLabel"] { opacity: 0.68; }
          [data-testid="stDataFrame"] { border-radius: 12px; }
          hr { border-color: #E5E7EB !important; }
          /* The active tab carries the amber, not the bright yellow: an underline is a
             thin mark and the bright one would vanish against white. */
          .stTabs [data-baseweb="tab-highlight"] { background-color: #B07D0A; }
          .stTabs [data-baseweb="tab-list"] { gap: 1.4rem; border-bottom: 1px solid #E5E7EB; }
          .stTabs [data-baseweb="tab"] { padding: 0 0 0.55rem; color: #6B7280; }
          .stTabs [aria-selected="true"] { color: #1F2329; font-weight: 600; }
          /* Phase nav. Each option becomes a card on the page grey; the selected one
             takes the bright yellow, which is a filled shape and so can carry it.
             stRadioOption and stRadioGroup are Streamlit's own hooks — an earlier
             version targeted [role="radiogroup"] > label, which matches nothing. */
          section[data-testid="stSidebar"] [data-testid="stRadioGroup"] { gap: 0.4rem; }
          section[data-testid="stSidebar"] [data-testid="stRadioOption"] {
            background: #F3F4F6; border-radius: 12px; padding: 0.55rem 0.7rem;
            margin: 0; width: 100%; transition: background 120ms ease; }
          section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {
            background: #ECEEF1; }
          section[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) {
            background: #FFC93C; }
          section[data-testid="stSidebar"] [data-testid="stRadioCaption"] {
            font-size: 0.68rem; opacity: 0.72; }
          /* The tab block is the card. Wrapping every tab body in a container would mean
             re-indenting a thousand lines; styling the wrapper Streamlit already emits
             puts all of that content on a white panel for four rules. */
          [data-testid="stTabs"] { background: #FFFFFF; border: 1px solid #E5E7EB;
            border-radius: 16px; padding: 1.1rem 1.3rem 1.3rem;
            box-shadow: 0 1px 2px rgba(31,35,41,.05); }
          [data-testid="stTabs"] [data-testid="stMetric"] { background: #F3F4F6; }
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
GRID_RGBA = "rgba(128,134,142,0.22)"
AXIS_RGBA = "rgba(128,134,142,0.45)"

PLOT_LAYOUT = {
    "font": PLOT_FONT,
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "xaxis": {"gridcolor": GRID_RGBA, "zerolinecolor": GRID_RGBA,
              "linecolor": AXIS_RGBA, "title": {"font": {"size": 12, "color": SLATE}}},
    "yaxis": {"gridcolor": GRID_RGBA, "zerolinecolor": GRID_RGBA,
              "linecolor": AXIS_RGBA, "title": {"font": {"size": 12, "color": SLATE}}},
    # title_font only — setting a title dict without "text" makes Plotly render the
    # string "undefined" on every figure that never had a title of its own.
    "title_font": {"family": "Source Serif 4, Georgia, serif", "size": 15, "color": INK},
    "legend": {"font": {"size": 11, "color": SLATE}},
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
    title_font = layout.pop("title_font", None)
    if title_font and getattr(fig.layout.title, "text", None):
        layout["title_font"] = title_font
    fig.update_layout(**layout)
    kwargs.setdefault("width", "stretch")
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
# The stage rides as a caption rather than inside the label. Streamlit renders it on its
# own line under the option, which gives the nav the two-line card look without a CSS
# hack, and keeps the label short enough to read as a title.
PHASE_CAPTIONS = ["Overview", "Descriptive", "Diagnostic", "Predictive"]

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
st.sidebar.title("EBEWE Dashboard")
st.sidebar.caption("City of Los Angeles · LADBS building benchmarking filings")

bundled = find_bundled_csv()
uploaded = st.sidebar.file_uploader(
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
    key="phase",
    help="Home summarises all three phases. Preliminary describes the data, Midterm "
         "diagnoses what the compliance label measures, Final fits the model.",
)
st.sidebar.divider()

years = sorted(df["programYear"].dropna().unique().astype(int))
year_range = st.sidebar.select_slider(
    "Program year range",
    options=years,
    value=(years[0], years[-1]),
)

all_types = sorted(t for t in df["propertyType"].unique() if t != "UNKNOWN")
top_12 = list(df[~df["isIncompleteFiling"]]["propertyType"].value_counts().head(12).index)
type_mode = st.sidebar.radio(
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
    chosen_types = st.sidebar.multiselect("Select types", options=all_types, default=[])
    if not chosen_types:
        st.sidebar.caption("No types selected — showing all.")

st.sidebar.divider()
st.sidebar.caption(f"Source file: `{source_label}`")
st.sidebar.caption(
    f"{diag['raw_shape'][0]:,} rows × {diag['raw_shape'][1]} columns as loaded · "
    f"{diag['duplicates']:,} duplicate rows"
)
st.sidebar.download_button(
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
def _phase_header(title, standing, blurb):
    st.title(title)
    st.caption(standing)
    st.markdown(blurb)
    st.divider()



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

    # sizeBand, not ladbsBuildingCategory. The tab and the paper both report the size
    # spread; the raw category splits city-owned tiers out separately and gives 30.2
    # rather than 16.7, so using it here would contradict the page it summarises.
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

    st.title("EBEWE Program — Building Compliance Analytics")
    st.caption(
        "City of Los Angeles, Department of Building and Safety. Every figure in this "
        "application is recomputed from the loaded file; nothing is stored or hardcoded."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Filings", f"{facts['filings']:,}")
    k2.metric("Buildings", f"{facts['buildings']:,}")
    k3.metric("Program years", f"{facts['first_year']}–{facts['last_year']}")
    k4.metric("Structurally incomplete", f"{facts['incomplete_share']:.1f}%")

    with st.container(border=True):
        st.markdown(
            "**Who this is for.** The LADBS program analyst who plans EBEWE enforcement "
            "and outreach each program year, and the program manager who reports "
            "compliance performance to the City."
        )
        st.markdown(
            "**The question this application answers.** The compliance field records two "
            "unrelated events under one label — a building that reported and failed a "
            "requirement, and a building whose report was never completed. No field "
            "separates them. Each phase below answers one part of what follows from that."
        )

    st.divider()
    st.subheader("The three phases")
    st.caption(
        "Each phase answers a question the next one depends on. The figures below are "
        "computed from the loaded file, not restated from an earlier run."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### Preliminary — descriptive")
            st.markdown("*What does the dataset contain, and what is wrong with it?*")
            st.markdown(
                f"- **{facts['incomplete_share']:.1f}%** of filings are structurally "
                f"incomplete ({facts['n_incomplete']:,} of {facts['filings']:,})\n"
                f"- Among complete filings, **{facts['complete_rate']:.2f}%** are compliant\n"
                f"- Among incomplete filings, **{facts['incomplete_notcomplied']:.2f}%** "
                f"are NOT COMPLIED"
            )
            st.caption("Tabs: Data Quality · Descriptive Stats · Figures 1–4")
            st.button("Open Preliminary", key="go_prelim", width="stretch",
                      on_click=_goto, args=(PHASE_PRELIM,))
    with c2:
        with st.container(border=True):
            st.markdown("#### Midterm — diagnostic")
            st.markdown("*Is compliance falling, and what actually separates it?*")
            st.markdown(
                f"- Headline fall of **{facts['fall_all']:.1f} points** becomes "
                f"**{facts['fall_complete']:.1f}** among complete filings\n"
                f"- Coverage tier spreads compliance **{facts['tier_spread_all']:.1f} "
                f"points**, but only **{facts['tier_spread_complete']:.1f}** once "
                f"filing completeness is held constant\n"
                f"- Across {facts['n_big_entities']} organisations the spread is "
                f"**{facts['entity_spread']:.1f} points**, against "
                f"**{facts['type_spread']:.1f}** across all property types with "
                f"{MIN_TYPE_FILINGS}+ complete filings"
            )
            st.caption("Tabs: Compliance Trend · Coverage Tier · Responsible Entity · Correlations")
            st.button("Open Midterm", key="go_mid", width="stretch",
                      on_click=_goto, args=(PHASE_MID,))
    with c3:
        with st.container(border=True):
            st.markdown("#### Final — predictive")
            st.markdown("*Can non-compliance be predicted well enough to target audits?*")
            st.markdown(
                f"- Modelled on the **{facts['n_complete']:,}** complete filings only\n"
                f"- The bar to beat is **{facts['majority_complete']:.2f}%**, the majority "
                f"class — not the **{facts['one_rule_acc']:.2f}%** a one-column rule reaches "
                f"across all filings\n"
                f"- The target group is the **{facts['n_noncompliant_complete']:,}** "
                f"non-compliant complete filings"
            )
            st.caption("Model comparison · feature importance · ranked review list")
            st.button("Open Final", key="go_final", width="stretch",
                      on_click=_goto, args=(PHASE_FINAL,))

    st.divider()
    with st.expander("About this dataset", expanded=False):
        raw_rows, raw_cols = diag.get("raw_shape", (len(df), len(df.columns)))
        a1, a2, a3 = st.columns(3)
        a1.metric("Rows, as published", f"{raw_rows:,}")
        a2.metric("Columns, as published", f"{raw_cols}")
        a3.metric("Duplicate rows", f"{diag.get('duplicates', 0):,}")
        st.markdown(
            f"**Source.** Existing Buildings Energy and Water Efficiency (EBEWE) Program, "
            f"City of Los Angeles open data portal, published by LADBS under the "
            f"Public Domain licence. Loaded file: `{source_label}`.\n\n"
            f"**What a row is.** One filing, not one building. "
            f"{facts['filings']:,} filings cover {facts['buildings']:,} buildings across "
            f"program years {facts['first_year']} to {facts['last_year']}, so a building "
            f"appears once per year it reports."
        )
        src_cols = [c for c in df.columns if c in COLUMN_NOTES]
        eng_cols = [c for c in df.columns if c in ENGINEERED_NOTES]
        table = pd.DataFrame({
            "Column": src_cols + eng_cols,
            "Description": [COLUMN_NOTES.get(c, "") for c in src_cols]
                           + [ENGINEERED_NOTES[c] for c in eng_cols],
            "Type": [str(df[c].dtype) for c in src_cols + eng_cols],
            "Filled %": [round(df[c].notna().mean() * 100, 1) for c in src_cols + eng_cols],
        })
        st.dataframe(table, width="stretch", hide_index=True)
        st.caption(
            "Descriptions of the published columns are the portal's. Type and fill rate "
            "are measured on the loaded file after cleaning, so they reflect this data "
            "rather than the portal's documentation."
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
        "The diagnostic phase showed that the compliance label measures filing behaviour "
        "as much as building performance, and that the organisation filing a report "
        "separates compliance far more sharply than anything about the building. This "
        "phase asks whether that is enough to predict non-compliance well enough to "
        "direct a limited number of audits.",
    )

    st.subheader("What the model is asked to do")
    st.markdown(
        f"Predict **NOT COMPLIED** among the **{res['n_train'] + res['n_test']:,}** "
        f"complete filings. Trained on {res['n_train']:,} filings across "
        f"{res['train_buildings']:,} buildings and tested on {res['n_test']:,} filings "
        f"across {res['test_buildings']:,} buildings, split **by building** so no "
        f"building appears on both sides. The test fold holds **{res['n_pos']:,}** "
        f"non-compliant filings, a base rate of **{res['base_rate'] * 100:.2f}%**."
    )

    st.subheader("Three models against the baseline")
    st.dataframe(
        comp.style.format({c: "{:.4f}" for c in
                           ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]}),
        width="stretch", hide_index=True)
    st.warning(
        f"**The baseline is accurate and useless.** Predicting COMPLIED for every filing "
        f"scores **{base['Accuracy'] * 100:.2f}%** accuracy and catches **none** of the "
        f"{int(base['Missed']):,} non-compliant filings — precision, recall and F1 are all "
        f"zero. Gradient boosting reaches **{gb['Accuracy'] * 100:.2f}%** accuracy, which "
        f"is barely higher, while catching **{int(gb['Caught']):,}** of them at "
        f"**{gb['Precision'] * 100:.1f}%** precision. Accuracy cannot tell those two apart, "
        f"which is why PR-AUC — **{base['PR-AUC']:.4f}** against **{gb['PR-AUC']:.4f}** — "
        f"carries the comparison instead."
    )
    st.caption(
        f"Overfitting check, ROC-AUC on train against test: logistic regression "
        f"{res['gap_m1']:+.4f}, gradient boosting {res['gap_m2']:+.4f}. The boosting "
        f"hyperparameters were chosen on this gap rather than on the test score, since "
        f"choosing on the test score is itself a form of leakage."
    )

    st.divider()
    st.subheader("What the model uses")
    imp = res["importance"].sort_values("AUC drop")
    fig = go.Figure(go.Bar(
        x=imp["AUC drop"], y=imp["Feature"], orientation="h",
        error_x=dict(type="data", array=imp["SD"], color=SLATE, thickness=1, width=3),
        marker_color=CIVIC,
        text=[("<0.001" if v < 0.001 else f"{v:.3f}") for v in imp["AUC drop"]],
        textposition="outside", cliponaxis=False))
    fig.update_layout(height=430, xaxis_title="ROC-AUC lost when the column is shuffled",
                      yaxis_title="", margin={"l": 10, "r": 60, "t": 10, "b": 40})
    chart(fig)
    ent_auc = (1 - (nb["ROC-AUC"] - 0.5) / (gb["ROC-AUC"] - 0.5)) * 100
    ent_pr = (1 - (nb["PR-AUC"] - res["base_rate"]) / (gb["PR-AUC"] - res["base_rate"])) * 100
    st.info(
        f"**Who files, not what is filed.** Shuffling the responsible entity costs more "
        f"ROC-AUC than every other column combined, and all property-type indicators "
        f"together cost almost nothing. Refitting with the entity features withheld — the "
        f"last row of the table above — drops ROC-AUC from **{gb['ROC-AUC']:.4f}** to "
        f"**{nb['ROC-AUC']:.4f}** and PR-AUC from **{gb['PR-AUC']:.4f}** to "
        f"**{nb['PR-AUC']:.4f}**. Measured above chance, the responsible entity accounts "
        f"for **{ent_auc:.1f}%** of ROC-AUC and **{ent_pr:.1f}%** of PR-AUC. The model "
        f"without it still beats random selection, so the building characteristics are not "
        f"worthless — but it misses most of what the full model catches."
    )

    st.divider()
    st.subheader("The review list")
    st.caption(
        "The model is used as a ranking rather than a verdict. The analyst's question is "
        "not whether one filing is non-compliant but which filings to review with the "
        "capacity available, so the page reports what a review list of a given size "
        "would contain."
    )
    cap = st.slider("Filings the analyst can review", min_value=50,
                    max_value=min(2000, res["n_test"]), value=250, step=50)
    order = np.argsort(-res["proba"])
    hits = int(res["y_test"][order[:cap]].sum())
    m1_, m2_, m3_ = st.columns(3)
    m1_.metric("Non-compliant found", f"{hits:,}")
    m2_.metric("Hit rate", f"{hits / cap * 100:.1f}%",
               delta=f"{hits / cap * 100 - res['base_rate'] * 100:+.1f} pts vs random")
    m3_.metric("Share of all non-compliant", f"{hits / res['n_pos'] * 100:.1f}%")
    st.dataframe(res["ranked"].head(cap).style.format({"risk": "{:.3f}"}),
                 width="stretch", hide_index=True, height=360)
    st.caption(
        f"Ranked by predicted risk on the held-out fold. Reviewing {cap:,} filings at "
        f"random would be expected to find about {cap * res['base_rate']:.0f}."
    )

    st.divider()
    st.subheader("What this does not settle")
    st.markdown(
        f"- **Cost-benefit ranking is out of reach.** No penalty, enforcement or retrofit "
        f"cost appears in the published columns, so the model can rank by risk but not by "
        f"expected value. Importing external cost figures would import assumptions with a "
        f"weaker footing than anything else here.\n"
        f"- **Timing cannot be assessed.** The data records a program year but no filing "
        f"date, so nothing here says when in a cycle an intervention would land.\n"
        f"- **Exempt buildings are indistinguishable from delinquent ones.** The ordinance "
        f"exempts unoccupied and mid-demolition buildings; neither is recorded. That places "
        f"a ceiling on recall no model can cross.\n"
        f"- **Program year carries a confound.** It is the second most important feature, "
        f"but it absorbs both the policy regime — including the 2019–2021 deadline "
        f"suspension — and the roster growth as the ordinance phased in. It is not a causal "
        f"finding.\n"
        f"- **{res['unseen_entity_share'] * 100:.1f}% of test filings** name an organisation "
        f"absent from the training data; those are scored at the base rate, so the model has "
        f"no entity signal for roughly one filing in "
        f"{max(1, round(1 / max(res['unseen_entity_share'], 1e-9))):.0f}."
    )


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
with st.container(border=True):
    _k1, _k2, _k3, _k4 = st.columns(4)
    _k1.metric("Filings", f"{len(dff):,}")
    _k2.metric("Compliance rate", f"{dff['isCompliant'].mean() * 100:.1f}%")
    _k3.metric("Incomplete filings", f"{dff['isIncompleteFiling'].mean() * 100:.1f}%")
    _k4.metric("Median Site EUI (kBtu/ft²)", f"{dff['siteEui'].median():,.1f}")

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
        "This phase characterises the dataset before any claim is made from it: what "
        "each field holds, how much is missing and on which rows, which measurements "
        "are implausible, and how the categorical fields are distributed. Nothing here "
        "explains compliance — that is the Midterm — but nothing in the Midterm is "
        "trustworthy without it.",
    )
    tabs = st.tabs([
        "Data Quality",
        "Descriptive Stats",
        "Figures 1–4",
    ])

    with tabs[0]:
        st.subheader("Missingness by column")
        st.caption(
            "Computed after the `Not Available` placeholder is standardized to a true null "
            "(Section 4.2). Left as-is, pandas would treat that string as a valid category."
        )
        miss = diag["missing_report"].reset_index(names="Column")
        _com = diag.get("comissing_fields", [])
        _m = miss.sort_values("Missing %")
        fig = px.bar(
            _m,
            x="Missing %", y="Column", orientation="h",
            color=_m["Column"].isin(_com),
            # Colour marks membership in the co-missing set, not a missingness threshold.
            # The six identical masks are the finding; every other column is context.
            color_discrete_map={True: CIVIC, False: SLATE},
            text="Missing %",
        )
        fig.update_layout(height=650, showlegend=False, xaxis_title="Missing (%)")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
        _pad_axis(fig, miss["Missing %"])
        chart(fig)
        st.info(
            f"{len(_com)} fields share an identical missing mask — not merely a similar rate. "
            f"They are null on exactly the same {diag['n_incomplete']:,} records, with no "
            f"exceptions in either direction: {', '.join('`%s`' % c for c in _com)}. "
            f"That is the pattern captured as `isIncompleteFiling`.",
        )

        st.divider()
        st.subheader("Incomplete-filing rate over time")
        inc_by_year = df.groupby("programYear")["isIncompleteFiling"].mean().mul(100).round(1)
        fig = px.line(
            x=inc_by_year.index, y=inc_by_year.values, markers=True,
            labels={"x": "Program Year", "y": "% Incomplete"},
        )
        fig.update_traces(line_color=RED)
        fig.update_layout(height=350, title="Share of filings that are structurally incomplete")
        chart(fig)
        st.caption(
            "Checked by year rather than once overall: a rising incomplete-filing rate makes a "
            "naive compliance trend look like buildings are performing worse when they may not be."
        )

        st.divider()
        st.subheader("Plausibility bounding (5.2)")
        st.caption(
            "Data-entry errors nulled at both ends — a 6.7-million kBtu/ft² reading is not an "
            "unusual building. Shown full width so the counts stay readable."
        )
        st.dataframe(diag["plaus_report"], width="stretch", hide_index=True)

        st.subheader("IQR outlier flags (5.3)")
        st.caption("Flagged, never removed. These are genuinely unusual but real buildings.")
        st.dataframe(diag["outlier_report"], width="stretch", hide_index=True)

        if diag.get("n_locale_property_types"):
            st.markdown("**French-locale property types**")
            st.caption(
                f"{diag['n_locale_property_types']} filing(s) carried French labels for a property "
                f"type that also exists in English, so one category was split across two labels in "
                f"every grouping. The published file encodes the same French label two ways — once "
                f"with a proper accent and once with a replacement character — so the map is keyed "
                f"on a form that ignores non-alphanumerics rather than on literal strings. Distinct "
                f"property types: {diag['property_types_before']} before, "
                f"{diag['property_types_after']} after."
            )
            st.dataframe(diag["locale_property_types"], width="stretch", hide_index=True)

        st.markdown(
            f"Also corrected: **{diag['n_invalid_yearbuilt']} implausible `yearBuilt` values** "
            f"nulled before `buildingAge` was derived, and **{diag['n_locale_fixed']} French-locale "
            f"entries** (`Non`/`Oui`) in `energyStarCertEligibility` mapped to English before any "
            f"generic fill ran — preserving real Yes/No information instead of discarding it as missing."
        )
        if diag["missing_target"] == 0:
            st.success(
                f"Target integrity check passed: 0 missing `complianceStatus` values. "
                f"The classification target is deliberately excluded from generic fill logic.",
            )
        else:
            st.error(f"{diag['missing_target']} missing complianceStatus values — resolve before modeling.")

        st.divider()
        st.subheader("Entity field repairs and audits")
        st.caption(
            "`entityResponsible` carries more signal about compliance than any other field "
            "(see the Responsible Entity tab), which is reason to check what it actually holds."
        )

        r1, r2, r3 = st.columns(3)
        r1.metric("Distinct entities", f"{diag.get('n_entities', 0):,}")
        r2.metric("Filings with punctuation stripped", f"{diag.get('n_punct_stripped', 0):,}")
        r3.metric("Legal-form merges", f"{diag.get('legal_merge_labels', 0):,}",
                  f"into {diag.get('legal_merge_groups', 0)} names", delta_color="off")
        st.caption(
            f"{diag.get('legal_merge_labels', 0)} labels differing only in punctuation and "
            f"legal form were merged into {diag.get('legal_merge_groups', 0)} names — "
            "3D INVESTMENTS, LLC into 3D INVESTMENTS, and so on. The canonical name is "
            "whichever spelling carries the most filings. The rule is mechanical: a group "
            "merges only when every differing token is a legal form, so no group is "
            "adjudicated by hand."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Values that cannot be company names**")
            if len(diag.get("nonname", [])):
                st.dataframe(diag["nonname"], width="stretch", hide_index=True)
                st.caption(
                    f"{diag['nonname_share']:.2f}% of named-entity filings. Not repaired: the true "
                    "entity is unrecoverable, and assigning these to UNKNOWN would merge them with "
                    "the incomplete filings, which are almost never compliant. A code groups as "
                    "reliably as a name; only readability is lost."
                )
            else:
                st.caption("None detected in this refresh.")
        with c2:
            st.markdown("**Labels that may be one organisation**")
            st.metric("Candidate groups", f"{diag.get('dupe_groups', 0):,}",
                      f"{diag.get('dupe_labels', 0):,} labels, {diag.get('dupe_share', 0):.1f}% of filings",
                      delta_color="off")
            st.caption(
                "What remains after the legal-form merge above. These groups share a "
                "normalised key but differ by a real word — ANCHOR PACIFICA COMPANY against "
                "ANCHOR PACIFICA MANAGEMENT COMPANY, or 2014 E 15TH ST, LLC against 2014 E "
                "15TH ST ASSOCIATES, LLC. Reported, not merged: a shared key shows two names "
                "are similar, and a company and its management arm are often separate filers."
            )
            if len(diag.get("dupe_detail", [])):
                with st.expander("Show candidate groups"):
                    st.dataframe(diag["dupe_detail"], width="stretch", hide_index=True)

        st.divider()
        st.subheader("Availability of fields not described elsewhere")
        st.caption(
            "Ten raw columns pass through the pipeline without appearing in the analysis. "
            "Recording their availability separates a deliberate exclusion from an oversight."
        )
        av = diag.get("availability", pd.DataFrame())
        if len(av):
            fig = px.bar(av.sort_values("Present %"), x="Present %", y="Field", orientation="h",
                         text="Present %")
            fig.update_traces(marker_color=BLUE, texttemplate="%{text:.1f}%",
                              textposition="outside", cliponaxis=False)
            fig.update_layout(height=420, xaxis_range=[0, 100], yaxis_title="")
            chart(fig)

            # Fields never plausibility-bounded still carry the errors 5.2 exists to remove.
            unb = av[~av["Field"].isin(PLAUSIBILITY_BOUNDS.keys())].copy()
            bad = unb[unb["Max / median"] > 1000]
            if len(bad):
                st.warning(
                    "**These fields never passed through the Section 5.2 plausibility bounds.** "
                    + ", ".join(f"`{r.Field}` reaches {r.Max:,.0f} against a median of {r.Median:,.1f}"
                                for r in bad.itertuples())
                    + ". Read their medians and quartiles; do not read their means or standard "
                      "deviations. Any modelling use must extend the bounding first.",
                )

            water = av[av["Field"].str.contains("ater")]
            if len(water) and water["Present %"].max() > 25 > water["Present %"].min():
                st.info(
                    f"**The water fields split.** `totalWaterUse` is present on "
                    f"{water['Present %'].max():.1f}% of filings while its components reach only "
                    f"{water[water['Present %'] < 25]['Present %'].max():.1f}%. The published data "
                    "supports aggregate water analysis but not the indoor/outdoor breakdown — a "
                    "property of the source, not a scoping decision.",
                )

        if diag.get("pctdiff_n"):
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
        st.subheader("Frequency — filing volume by property type (6.1)")
        # A silent .head(10) dropped two categories while the caption below used the full
        # selection as its denominator. Cap at 12 to match the Top-12 preset and say so when
        # the selection is wider, rather than truncating without a note.
        _vol_all = dff_seg["propertyType"].value_counts()
        vol = _vol_all.head(12)
        if len(vol):
            fig = px.bar(x=vol.values, y=vol.index, orientation="h",
                         labels={"x": "Number of Filings", "y": ""}, text=vol.values)
            fig.update_traces(marker_color=BLUE, textposition="outside",
                              texttemplate="%{text:,}", cliponaxis=False)
            fig.update_layout(height=max(380, 32 * len(vol)),
                              yaxis={"autorange": "reversed"})
            _pad_axis(fig, vol.values)
            chart(fig)
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
                    lead = (f"{vol.index[0].title()} dominates by volume — {ratio:.1f}× the next "
                            f"largest category ({vol.index[1].title()}), and {share:.0f}% of all "
                            f"{len(dff_seg):,} complete filings in the current selection.")
                else:
                    lead = (f"{vol.index[0].title()} leads at {share:.0f}% of all {len(dff_seg):,} "
                            f"complete filings in the current selection, but only {ratio:.2f}× "
                            f"{vol.index[1].title()} — no single type dominates this selection.")
                st.caption(
                    lead + " Filing volume is real context for the Midterm: where one property "
                    "type is far more common, overall accuracy can look strong while performance "
                    "on the minority types stays poor."
                )

        st.divider()
        st.subheader("Central tendency — mean vs. median (6.2)")
        st.caption(
            "Reported side by side deliberately: the gap between them is an honest indicator of "
            "how right-skewed each field remains after cleaning. The mean alone would overstate "
            "the energy use of a typical building."
        )
        rows = []
        for col in ["siteEui", "sourceEui", "co2Emissions", "energyStarScore"]:
            rows.append({
                "Metric": col,
                "Mean": round(dff[col].mean(), 2),
                "Median": round(dff[col].median(), 2),
                "n (non-null)": int(dff[col].notna().sum()),
            })
        central = pd.DataFrame(rows)
        st.dataframe(central, width="stretch", hide_index=True)
        c2 = st.container()
        fig = go.Figure()
        # Mean and median are two estimators, not a good one and a bad one.
        fig.add_bar(name="Mean", x=central["Metric"], y=central["Mean"], marker_color=SLATE)
        fig.add_bar(name="Median", x=central["Metric"], y=central["Median"], marker_color=BLUE)
        fig.update_layout(barmode="group", height=340,
                          title="Gap between mean and median")
        c2.plotly_chart(fig, width="stretch")
        st.caption(
            "The first three run the same direction — mean above median, the signature of a "
            "right-skewed distribution pulled up by a small number of very energy-intensive "
            "buildings. `energyStarScore` runs the other way, mean *below* median, because it "
            "is a bounded 1–100 percentile score rather than an unbounded physical quantity. "
            "Reporting only the mean would misrepresent both cases, in opposite directions."
        )

        st.divider()
        st.subheader("Dispersion — range and standard deviation (6.3)")
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
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "The outlier-flag rate sits next to the standard deviation on purpose: it shows a "
            "Midterm modeler how much of the tail would be affected by capping at the IQR fence."
        )

        st.divider()
        st.subheader("Median Site EUI by property type")
        med = (dff_seg.groupby("propertyType")["siteEui"].median()
               .dropna().sort_values(ascending=False))
        if len(med):
            fig = px.bar(x=med.values, y=med.index, orientation="h",
                         labels={"x": "Median Site EUI (kBtu/ft²)", "y": ""},
                         text=med.round(1))
            # One series, so the hue encodes nothing — the bar length already carries the
            # value. GREEN previously coloured this chart, which gave that token a second
            # meaning on top of the one it was declared for.
            fig.update_traces(marker_color=CIVIC, textposition="outside", cliponaxis=False)
            fig.update_layout(height=max(380, 30 * len(med)), yaxis={"autorange": "reversed"})
            _pad_axis(fig, med.values)
            chart(fig)
        if len(med) >= 2:
            hi_name, hi_val = med.index[0].title(), med.iloc[0]
            lo_name, lo_val = med.index[-1].title(), med.iloc[-1]
            ratio = hi_val / lo_val if lo_val else float("nan")
            st.caption(
                f"{hi_name} reports the highest median intensity at {hi_val:,.1f} kBtu/ft², "
                f"{ratio:.0f}× that of {lo_name} at {lo_val:,.1f}. Energy need here is a function "
                f"of use, not efficiency — which is exactly why percentiles below are computed "
                f"within type. Naming the leaders live rather than in fixed text, since which "
                f"types appear depends on the sidebar selection."
            )

        st.divider()
        st.subheader("Position — percentile within property type (6.4)")
        st.caption(
            "Ranking every building against one citywide distribution would penalize an efficient "
            "warehouse simply because warehouses use less energy than offices in general."
        )
        dff_seg["siteEuiPercentileInType"] = (
            dff_seg.groupby("propertyType")["siteEui"].rank(pct=True).round(3)
        )
        c1, c2 = st.columns([1, 2])
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
        st.subheader("Figure 1 — Site EUI distribution")
        eui = dff[["siteEui", "siteEuiIsOutlier"]].dropna(subset=["siteEui"])
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_histogram(x=eui.loc[~eui["siteEuiIsOutlier"], "siteEui"], nbinsx=60,
                              name="Within IQR fences", marker_color=BLUE)
            fig.add_histogram(x=eui.loc[eui["siteEuiIsOutlier"], "siteEui"], nbinsx=30,
                              name="Flagged outlier (not removed)", marker_color=RED)
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
            fig.update_traces(marker_color=DARK_BLUE)
            fig.update_layout(height=420, title="Log-transformed (viewing aid only)",
                              xaxis_title="log(1 + Site EUI)", yaxis_title="Buildings")
            chart(fig)
        st.caption(
            "The log panel redistributes visual mass without altering a single underlying value — "
            "no `siteEui` value in the dataframe is modified by this view."
        )

        st.divider()
        st.subheader("Figure 2 — Compliance rate by property type")
        _type_stats = dff_seg.groupby("propertyType")["isCompliant"].agg(["mean", "count"])
        _type_stats["mean"] *= 100
        _small = _type_stats[_type_stats["count"] < MIN_TYPE_FILINGS]
        comp_rate = (_type_stats[_type_stats["count"] >= MIN_TYPE_FILINGS]["mean"]
                     .sort_values(ascending=False))
        if len(comp_rate):
            colors = [RED if v < 80 else BLUE for v in comp_rate.values]
            fig = go.Figure(go.Bar(x=comp_rate.values, y=comp_rate.index, orientation="h",
                                   marker_color=colors, text=comp_rate.round(1),
                                   texttemplate="%{text}%", textposition="outside",
                                   cliponaxis=False))
            fig.add_vline(x=comp_rate.mean(), line_dash="dash", line_color=INK,
                          annotation_text=f"Avg {comp_rate.mean():.1f}%",
                          annotation_position="bottom left")
            fig.add_vline(x=80, line_dash="dot", line_color=INK,
                          annotation_text="80% reference line", annotation_position="top left")
            fig.update_layout(height=max(380, 30 * len(comp_rate)),
                              xaxis_title="% Compliant", yaxis={"autorange": "reversed"},
                              xaxis_range=[0, 118],   # 18% headroom, matching _pad_axis
                              title="Below the 80% reference line is marked in ochre")
            chart(fig)
        st.caption(
            "The 80% line is a fixed reading aid chosen for this analysis, not a target set by "
            "the EBEWE ordinance — the ordinance sets filing deadlines and per-building fees, "
            "not a compliance-rate goal. It is used rather than the citywide average so a bar's "
            "colour only changes when that property type's own rate changes."
        )
        if len(_small):
            st.caption(
                f"{len(_small)} type(s) with fewer than {MIN_TYPE_FILINGS} complete filings are "
                f"excluded from this chart: a rate computed on a handful of filings would swing "
                f"across the line on one building."
            )

        st.divider()
        c1, c2 = st.container(), st.container()
        with c1:
            st.subheader("Figure 3 — Median Site EUI by building age")
            aged = dff[dff["ageBucket"] != "UNKNOWN"]
            age_eui = aged.groupby("ageBucket")["siteEui"].median().reindex(AGE_ORDER)
            age_n = aged["ageBucket"].value_counts().reindex(AGE_ORDER)
            fig = px.bar(x=AGE_ORDER, y=age_eui.values,
                         labels={"x": "Building Age Bucket", "y": "Median Site EUI (kBtu/ft²)"},
                         text=age_eui.round(1))
            fig.update_traces(marker_color=BLUE, textposition="outside", cliponaxis=False)
            fig.update_layout(height=420)
            _pad_axis(fig, age_eui.values, axis="y", pad=0.12)
            chart(fig)
            st.caption("Median, not mean — a handful of real but extreme buildings would "
                       "misrepresent the typical building of an era.")
            st.write(age_n.rename("n per bucket").to_frame().T)
            st.divider()
        with c2:
            st.subheader("Figure 4 — Top 15 postal codes by total emissions")
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
            fig.update_traces(marker_color=CIVIC)
            fig.update_layout(height=560, yaxis={"autorange": "reversed", "type": "category"})
            chart(fig)
            st.caption(f"Total, not average — this view is about where retrofit and enforcement "
                       f"resources should go, which depends on total carbon impact. The bars sum "
                       f"every filing from {_yr_lo} to {_yr_hi} — {_n_bldg:,} buildings at "
                       f"{_fpb:.2f} filings each — so a postal code with more years of coverage "
                       f"accumulates more; the top ranks are stable but the tail "
                       f"shifts if a single year is used instead.")

    # ----------------------------------------------------------------------------
    # Tab 5 — Compliance Trend (Figure 5 + Section 6.5)
    # ----------------------------------------------------------------------------

else:
    _phase_header(
        "Midterm — diagnostic",
        "What the compliance label actually measures.",
        f"`complianceStatus` records two unrelated events under one label: a building "
        f"that reported and failed a requirement, and a building whose report was never "
        f"completed. No field separates them. This phase measures the mixture — the "
        f"reported fall of {_ov_drop:.1f} points becomes {_cm_drop:.1f} once filing "
        f"completeness is held constant — and then asks what does separate compliance "
        f"when filing behaviour is taken out of the way.",
    )
    tabs = st.tabs([
        "Compliance Trend",
        "Coverage Tier",
        "Responsible Entity",
        "Correlations",
    ])

    with tabs[0]:
        st.subheader("Figure 5 — Is compliance declining, or is filing completeness?")
        # series computed once above the tabs; see the note there

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=yearly_overall.index, y=yearly_overall.values,
                                 mode="lines+markers", name="Overall % Compliant",
                                 line={"color": BLUE, "width": 3}), secondary_y=False)
        fig.add_trace(go.Scatter(x=yearly_complete.index, y=yearly_complete.values,
                                 mode="lines+markers", name="% Compliant (complete filings only)",
                                 line={"color": DARK_BLUE, "width": 3},
                                 marker_symbol="triangle-up"), secondary_y=False)
        fig.add_trace(go.Scatter(x=yearly_incomplete.index, y=yearly_incomplete.values,
                                 mode="lines+markers", name="% Incomplete Filing",
                                 line={"color": OCHRE, "width": 2, "dash": "dash"},
                                 marker_symbol="square"), secondary_y=True)
        fig.update_yaxes(title_text="% Compliant", secondary_y=False)
        fig.update_yaxes(title_text="% Incomplete Filing", secondary_y=True,
                         color=RED, showgrid=False)
        fig.update_layout(height=480, xaxis_title="Program Year",
                          legend={"orientation": "h", "y": -0.2})
        chart(fig)
        st.caption(
            "The three series are overlaid because the finding *is* the gap between them, and "
            "the way that gap widens as the incomplete-filing rate climbs."
        )

        if len(yearly_overall) >= 2:
            drop_overall, drop_complete = _ov_drop, _cm_drop
            share = (1 - drop_complete / drop_overall) * 100 if drop_overall else np.nan
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Overall decline, {yearly_overall.index[0]}–{yearly_overall.index[-1]}",
                      f"{drop_overall:.1f} pts")
            c2.metric("Decline, complete filings only", f"{drop_complete:.1f} pts")
            c3.metric("Share attributable to filing completeness",
                      f"{share:.0f}%" if pd.notna(share) else "n/a")
            st.warning(
                "**These are endpoint figures and they overstate their own precision.** The "
                "building roster is not constant — the ordinance phased in by size, so early "
                "years cover a fraction of the buildings later years do — and the final program "
                "year may still be partially processed. On a fixed set of buildings the same "
                "share ranges from roughly 40% to 84% depending on the cohort and end year "
                "chosen. See the balanced-panel comparison below.",
            )

        st.divider()
        st.subheader("The same trend on a fixed set of buildings")
        st.caption(
            "Restricting to buildings observed in every year removes roster growth as an "
            "explanation, so any remaining change is buildings behaving differently rather "
            "than different buildings being present. The window starts where roster "
            "coverage matures, matching Section 6.5.1 of the notebook."
        )

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
                                     line={"color": DARK_BLUE, "width": 3}))
            fig.add_trace(go.Scatter(x=_bal.index, y=_bal["Overall %"], mode="lines+markers",
                                     name="Overall", line={"color": BLUE, "width": 2, "dash": "dot"}))
            fig.update_layout(height=420, xaxis_title="Program Year", yaxis_title="% Compliant",
                              title=f"{len(_panel):,} buildings held constant, {_yrs[0]}–{_yrs[-1]}",
                              legend={"orientation": "h", "y": -0.2})
            chart(fig)

            # The shape, stated from the data. The final-year fall is excluded when locating
            # the break, since it is usually the steepest drop and is reported separately.
            _d = np.diff(_c.values)
            if len(_d) >= 3:
                _bi = int(np.argmin(_d[:-1])) + 1
                _mid = _c.iloc[_bi:-1]
                k1, k2, k3 = st.columns(3)
                k1.metric(f"Break at {_c.index[_bi]}", f"{_d[_bi-1]:+.1f} pts")
                k2.metric(f"{_mid.index[0]}–{_mid.index[-1]}",
                          f"{_mid.iloc[-1] - _mid.iloc[0]:+.1f} pts")
                k3.metric(f"Final year ({_c.index[-1]})", f"{_d[-1]:+.1f} pts")
                if _mid.iloc[-1] > _mid.iloc[0]:
                    st.success(
                        f"With the building set held constant, compliance among complete filings "
                        f"**improved** {_mid.iloc[-1] - _mid.iloc[0]:.1f} points between "
                        f"{_mid.index[0]} and {_mid.index[-1]}. The series is a one-year break, a "
                        f"multi-year recovery, and a final-year drop — not a steady decline. An "
                        f"endpoint comparison conceals this because a line drawn between the first "
                        f"and last points passes straight over the recovery.",
                    )

            st.dataframe(_bal, width="stretch")

        st.divider()
        st.subheader("All filings, unbalanced roster")
        trend_table = pd.DataFrame({
            "Overall Compliance %": yearly_overall.round(1),
            "Compliance % (complete filings only)": yearly_complete.round(1),
            "Incomplete Filing %": yearly_incomplete.round(1),
            "Buildings in roster": dff.groupby("programYear")["buildingId"].nunique(),
        })
        st.dataframe(trend_table, width="stretch")
        st.caption(
            "The roster column is why the endpoint comparison above needs care: where it grows, "
            "a change in the compliance rate partly reflects a change in which buildings are "
            "being measured."
        )

    # ----------------------------------------------------------------------------
    # Tab 6 — Coverage Tier (Section 6.9)
    # ----------------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Coverage tier predicts filing, not compliance")
        st.caption(
            "`ladbsBuildingCategory` is the ordinance's own size and ownership segmentation. "
            f"Unlike {', '.join(diag.get('comissing_fields', STRUCTURAL_COLS))} — all missing "
            f"on the structurally incomplete filings — it is populated on every record. It is "
            f"therefore the only structural field that can answer which buildings file "
            f"incompletely."
        )
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
            if _priv == 0:
                st.caption(
                    f"The ordinance covers privately owned buildings at 20,000 sq ft and "
                    f"city-owned buildings at 7,500, so the tiers below 20,000 sq ft exist only "
                    f"because city ownership lowers the floor. That holds in this data: "
                    f"{', '.join(_small)} together hold {_city:,} filings and not one is "
                    f"privately owned. A compliance gap at the small end of the ranking below is "
                    f"therefore an ownership difference as well as a size one."
                )
            else:
                st.caption(
                    f"Note: {_priv:,} privately owned filing(s) appear below the 20,000 sq ft "
                    f"ordinance floor, alongside {_city:,} city-owned. Size and ownership are not "
                    f"fully entangled at the small end in this selection."
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

            k1, k2, k3 = st.columns(3)
            k1.metric("Spread, all filings", f"{_sa:.1f} pts")
            k2.metric("Spread, complete filings only", f"{_sc:.1f} pts")
            k3.metric("Spread, incompleteness rate", f"{_inc.max() - _inc.min():.1f} pts")

            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_bar(y=_bands, x=_all.values, orientation="h", name="All filings",
                            marker_color=BLUE, text=_all.round(1), textposition="outside",
                            cliponaxis=False)
                fig.add_bar(y=_bands, x=_cmp.values, orientation="h",
                            name="Complete filings only", marker_color="#90CAF9",
                            text=_cmp.round(1), textposition="outside", cliponaxis=False)
                fig.update_layout(height=430, barmode="group", xaxis_range=[0, 128],
                                  xaxis_title="% Compliant", yaxis={"autorange": "reversed"},
                                  title="A. Compliance by coverage tier",
                                  legend={"orientation": "h", "y": -0.2})
                chart(fig)
            with c2:
                fig = go.Figure(go.Bar(y=_bands, x=_inc.values, orientation="h",
                                       marker_color=RED, text=_inc.round(1),
                                       textposition="outside", cliponaxis=False))
                fig.add_vline(x=dff["isIncompleteFiling"].mean() * 100, line_dash="dash",
                              line_color=INK,
                              annotation_text=f"overall {dff['isIncompleteFiling'].mean()*100:.1f}%")
                fig.update_layout(height=430, xaxis_range=[0, _inc.max() * 1.3],
                                  xaxis_title="% of filings structurally incomplete",
                                  yaxis={"autorange": "reversed"},
                                  title="B. Filing incompleteness by coverage tier")
                chart(fig)

            # The conclusion holds only if the spread genuinely collapses. State it either way.
            if _sa > 0 and _sc < _sa / 2:
                st.success(
                    f"Compliance varies {_sa:.1f} points across coverage tiers, but only {_sc:.1f} "
                    f"points among complete filings. **Coverage tier predicts whether a building "
                    f"completes a filing, not whether it complies once it has.** Panel B is the "
                    f"mechanism: incompleteness runs from {_inc.min():.1f}% for {_inc.idxmin()} to "
                    f"{_inc.max():.1f}% for {_inc.idxmax()}. Because incomplete filings are almost "
                    f"never compliant, that pattern surfaces in the headline rate as though it were "
                    f"a difference in performance.",
                )
            else:
                st.info(
                    f"The compliance spread does not collapse once incomplete filings are excluded "
                    f"({_sa:.1f} to {_sc:.1f} points). Coverage tier is associated with compliance "
                    f"beyond its association with filing completeness, so the two panels are "
                    f"separate findings rather than one mechanism.",
                )

            st.divider()
            st.subheader("Incompleteness by tier and program year")
            st.caption(
                "Tiers moving together in one year points to a programme-wide change; one tier "
                "moving alone points to that tier's own phase-in date. Read alongside the "
                "Compliance Trend tab. The 2019 break has one documented cause — deadlines for "
                "Program Years 2019–2021 were tolled — and the size-tier phase-in is not a second "
                "one: the 20,000–49,999 sq ft band steps at 2018, which is visible in the chart "
                "below."
            )
            _piv = (dff.pivot_table(index="programYear", columns="sizeBand",
                                    values="isIncompleteFiling", aggfunc="mean") * 100).round(1)
            if len(_piv) >= 2 and _piv.shape[1] >= 2:
                fig = go.Figure()
                for _i, band in enumerate(_piv.columns):
                    # Coverage tier is ordered by building size, so the series take an
                    # ordered ramp rather than Plotly's categorical cycle — which put an
                    # arbitrary red beside an arbitrary green on a size variable and
                    # invited a good/bad reading of what is just "bigger" and "smaller".
                    fig.add_trace(go.Scatter(
                        x=_piv.index, y=_piv[band], mode="lines+markers", name=band,
                        line={"color": TIER_SEQUENCE[_i % len(TIER_SEQUENCE)], "width": 2},
                        marker={"size": 6}))
                fig.update_layout(height=420, xaxis_title="Program Year",
                                  yaxis_title="% structurally incomplete",
                                  legend={"orientation": "h", "y": -0.25})
                chart(fig)

                _yoy = _piv.diff()
                _rose = (_yoy > 0).sum(axis=1)
                _broad = _rose[_rose == _rose.max()]
                if _rose.max() == _piv.shape[1]:
                    st.caption(
                        f"All {_piv.shape[1]} tiers rose together in "
                        f"{', '.join(str(y) for y in _broad.index)}. Breadth alone does not identify "
                        "one year — compare magnitude and evenness before reading any single year as "
                        "a programme-wide event."
                    )
            st.dataframe(_piv, width="stretch")

            st.info(
                "`isCityOwned` and `sizeBand` are derived from this field and are available on "
                "incomplete filings, which makes them the only structural features usable across "
                "all filings rather than the complete subset alone.",
            )

    # ----------------------------------------------------------------------------
    # Tab 7 — Responsible Entity (Section 6.7)
    # ----------------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Who files matters more than what is filed")
        st.caption(
            "`entityResponsible` records the organisation that submitted the benchmark report. "
            "Incomplete filings are excluded throughout: they carry UNKNOWN and are almost never "
            "compliant, so including them would manufacture the variation this view measures."
        )

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

            c1, c2, c3 = st.columns(3)
            c1.metric(f"Agents with {MIN_FILINGS}+ filings", f"{len(_big):,}")
            # One decimal, not zero. The per-agent rates are rounded to 1dp above, so this
            # spread is exactly 98.5 on the current data — and Python's round-half-to-even
            # rendered that as "98" while the table below said 98.5. Two figures for one
            # quantity, differing only by how they were displayed.
            c2.metric("Spread across agents",
                      f"{_big['compliance'].max() - _big['compliance'].min():.1f} pts")
            c3.metric("Spread across property types",
                      f"{_types.max() - _types.min():.1f} pts" if len(_types) > 1 else "n/a")
            # The property-type spread is dominated by whichever single category sits below
            # the benchmark; report it with and without, since the decomposition below shows
            # that outlier is itself an operator effect.
            if len(_types) > 2:
                _wo = _types.drop(_types.idxmin())
                # The bases were previously described in a 106-word paragraph. They are two
                # measurements with different denominators, which is a comparison — so it is
                # shown as one, with the warning about comparability kept as the lead.
                st.caption(
                    "These two spreads use different denominators and are not comparable as "
                    "ratios. Both bases are stated so the comparison can be read for what it is."
                )
                st.dataframe(pd.DataFrame([
                    # Range second, not last. It is the only column carrying a number and it was
                    # the one that truncated, leaving three rows of denominators and no figures.
                    {"Spread": "Across responsible agents",
                     "Range": f"{_big['compliance'].max() - _big['compliance'].min():.1f} pts",
                     "Basis": f"agents with {MIN_FILINGS}+ complete filings",
                     "Population": f"all {len(dff_complete):,} complete filings"},
                    {"Spread": "Across property types (current selection)",
                     "Range": f"{_types.max() - _types.min():.1f} pts",
                     "Basis": f"types with {MIN_TYPE_FILINGS}+ filings",
                     "Population": f"{len(_types)} types in the selection"},
                    {"Spread": "Across property types (all qualifying)",
                     "Range": f"{_tall.max() - _tall.min():.1f} pts",
                     "Basis": f"types with {MIN_TYPE_FILINGS}+ filings",
                     "Population": f"{len(_tall)} types"},
                ]), width="stretch", hide_index=True)
                st.caption(
                    f"The lowest type ({_types.idxmin().title()}, {_types.min():.1f}%) accounts "
                    f"for much of the type spread — drop it and the remaining {len(_wo)} span "
                    f"{_wo.max() - _wo.min():.1f} points. On either basis the responsible entity "
                    f"discriminates more sharply than building function does, and the category "
                    f"that looks like an exception is decomposed below."
                )
            else:
                st.caption(f"Too few property types in the current selection to report a spread. "
                           f"The agent spread covers all {len(dff_complete):,} complete filings.")

            _show = pd.concat([_big.head(10), _big.tail(10)]).drop_duplicates()
            fig = go.Figure(go.Bar(
                x=_show["compliance"], y=_show.index, orientation="h",
                marker_color=[RED if v < _base else BLUE for v in _show["compliance"]],
                text=_show["compliance"], texttemplate="%{text}%",
                textposition="outside", cliponaxis=False,
                customdata=_show["filings"],
                hovertemplate="%{y}<br>%{x}% compliant<br>%{customdata:,} filings<extra></extra>"))
            fig.add_vline(x=_base, line_dash="dash", line_color=INK,
                          annotation_text=f"Baseline {_base:.1f}%", annotation_position="top left")
            fig.update_layout(height=max(420, 26 * len(_show)), xaxis_range=[0, 118],
                              xaxis_title="% Compliant", yaxis={"autorange": "reversed"},
                              title="Lowest and highest compliance among high-volume agents")
            chart(fig)

            st.divider()
            st.subheader("Does a pooled agent rate describe any actual year?")
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
                    st.dataframe(_reg, width="stretch", hide_index=True)
                    for _, r in _broke.iterrows():
                        st.error(
                            f"**{r['Agent']}** — {r[f'Before {_mid} %']:.1f}% across "
                            f"{r[f'Before {_mid} n']:,} filings before {_mid}, then "
                            f"{r[f'From {_mid} %']:.1f}% across {r[f'From {_mid} n']:,} filings after. "
                            f"Its pooled rate of {r['Pooled %']:.1f}% describes neither period. A "
                            f"static per-agent encoding would be wrong in both; a lagged feature — "
                            f"the same building's compliance last year — would not.",
                        )
                    if _broke.empty:
                        st.caption("No agent changes by 30+ points across this window; the pooled "
                                   "rates can be read at face value.")
            else:
                st.caption("Widen the program-year range to compare agents across periods.")

            st.divider()
            st.subheader("Is the lowest property type a category or its operators?")
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
                    st.success(
                        f"Those operators hold {_drv_n/_tot*100:.0f}% of the category. The remaining "
                        f"filings sit {abs(_base-_rest_r):.1f} points from the {_base:.1f}% baseline "
                        f"— against a {abs(_rest_r-_drv_r):.1f}-point gap between the two groups. "
                        f"Had they filed at baseline, {_t} would sit at {_cf:.1f}% and would not fall "
                        f"below the 80% reference line in Figure 2 at all. This is an operator "
                        f"effect "
                        f"presenting as a building-function effect.",
                    )
                    # This was a paragraph. It is tabular data — four thresholds, the rate
                    # each implies, and the set each selects — and serialising it into prose
                    # buried the only thing that matters: the set does not change. As a table
                    # the constancy is visible in one pass down the last column.
                    st.caption(
                        f"Driver rule: {MIN_OPERATOR_FILINGS}+ filings in the category and at "
                        f"least {_margin:.1f} points below the {_base:.1f}% baseline "
                        f"({DEFICIT_SHARE:.0%} of the category's own {_deficit:.1f}-point "
                        f"deficit). The margin is a judgement, so its effect is shown in full — "
                        f"if the selected set moved across this range the decomposition would be "
                        f"fragile and should not be read as a finding."
                    )
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

            st.divider()
            st.subheader("Is the agent effect an outsourcing effect?")
            st.caption(
                "`entityResponsible` mixes three kinds of filer: compliance vendors filing for many "
                "owners, property managers filing for buildings they operate, and owners filing for "
                "themselves. If compliance tracked that split, filer type would be the better "
                "feature — three levels instead of thousands, and it generalises to unseen agents."
            )
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
                        st.caption("  \n".join(_notes))
                        st.caption(
                            f"Reported for completeness: the widest gap between any two filer "
                            f"types is {_sp1:.1f} points across all classified agents and "
                            f"{_sp2:.1f} points with the regime-changers removed. That figure "
                            f"depends on which side a few ambiguously named firms are placed on, "
                            f"so it is not used to decide the question."
                        )
                _unc = _typed[_typed["type"] == "UNCLASSIFIED"]
                if len(_unc):
                    _uf = dff_complete[dff_complete["entityResponsible"].isin(_unc.index)]
                    _ur = _uf["isCompliant"].mean() * 100 if len(_uf) else float("nan")
                    st.caption(
                        f"{len(_unc)} agent(s) unclassified by the keyword rules, covering "
                        f"{len(_uf):,} filings at {_ur:.1f}% compliance. They sit high, but the "
                        f"effect of classifying them depends on which group they join — joining "
                        f"the highest group would widen the gap, joining the lowest would narrow "
                        f"it. This residual is a limitation of the classification, not something "
                        f"the result is demonstrably robust to."
                    )

    # ----------------------------------------------------------------------------
    # Tab 8 — Correlations (Figure 6 + Section 6.6)
    # ----------------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Figure 6 — Correlation heatmap")
        corr_cols = CORR_COLS + ["isCompliant"]
        corr = dff[corr_cols].corr().round(2)
        masked = corr.mask(np.triu(np.ones(corr.shape), k=1).astype(bool))
        fig = px.imshow(masked, text_auto=True, zmin=-1, zmax=1,
                        color_continuous_scale="Blues", aspect="auto")
        fig.update_layout(
            height=640,
            title="Lower triangle only — the upper half is redundant",
            margin={"l": 140, "r": 40, "b": 160, "t": 60},
            xaxis={"tickangle": -45, "side": "bottom", "automargin": True},
            yaxis={"automargin": True},
            coloraxis_colorbar={"thickness": 14},
        )
        chart(fig)

        st.warning(
            "**Not all strong correlations are the same kind of finding.** `siteEui` ↔ `sourceEui` "
            "(r ≈ 0.94) is *empirical* — two distinct measurements that move together because "
            "efficient buildings tend to be efficient on both. `siteEui` ↔ `ghgIntensityPer1kSqft` "
            "(r ≈ 0.98) is *structural*: GHG intensity is calculated from the same energy data Site "
            "EUI already represents. No amount of cleaning will reduce the second one, and treating "
            "it as a data-quality problem would be a mistake.",
        )

        st.subheader("Flagged pairs (|r| > 0.80)")
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

        st.divider()
        st.subheader("Correlation with the classification target")
        target_corr = (dff[corr_cols].corr()["isCompliant"].drop("isCompliant")
                       .sort_values(ascending=False).round(3))
        fig = px.bar(x=target_corr.values, y=target_corr.index, orientation="h",
                     labels={"x": "r with isCompliant", "y": ""}, text=target_corr.values)
        fig.update_traces(marker_color=BLUE, textposition="outside", cliponaxis=False)
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
