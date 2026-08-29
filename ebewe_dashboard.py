"""
EBEWE Program — Descriptive & Diagnostic Dashboard
City of Los Angeles, Department of Building and Safety

Preliminary phase deliverable. Every statistic and figure here is computed live
from the uploaded CSV using the same pipeline as EBEWE_Prelim_Analysis_v5.ipynb
(Sections 3-5 cleaning, Section 6 descriptive stats, Section 7 figures).
Nothing is hardcoded, so a future data refresh flows straight through.

Run locally:   streamlit run ebewe_dashboard.py
"""

import io
import glob
import os

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
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

BLUE = "#1565C0"
DARK_BLUE = "#0D47A1"
RED = "#D32F2F"
GREEN = "#2E7D32"
ORANGE = "#F57C00"

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

STRUCTURAL_COLS = ["propertyType", "yearBuilt", "grossFloorArea", "occupancy", "entityResponsible"]

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
    diag["incomplete_crosstab_wide"] = (
        pd.crosstab(df["isIncompleteFiling"], df["complianceStatus"], normalize="index")
        .mul(100).round(1)
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
    st.title("EBEWE Program — Descriptive & Diagnostic Dashboard")
    st.info(
        "Upload the EBEWE CSV in the sidebar to begin. The app runs the full cleaning "
        "pipeline from the Preliminary notebook (Sections 3–5) on upload — the file does "
        "not need to be pre-cleaned."
    )
    st.stop()

df, diag = load_and_clean(raw_bytes)

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
# Header
# ----------------------------------------------------------------------------
st.title("EBEWE Program — Descriptive & Diagnostic Dashboard")
st.caption(
    "Every number below is recomputed live from the uploaded file. "
    "Nothing is hardcoded, so a data refresh flows straight through."
)
if filtered:
    st.warning(
        f"Filtered to program years {year_range[0]}–{year_range[1]}. Figures reflect the "
        "filtered subset, not the full dataset.",
        icon="⚠️",
    )

tabs = st.tabs([
    "Overview",
    "Data Quality",
    "Descriptive Stats",
    "Figures 1–4",
    "Compliance Trend",
    "Correlations",
])

# ----------------------------------------------------------------------------
# Tab 1 — Overview
# ----------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total filings", f"{len(dff):,}")
    c2.metric("Compliance rate", f"{dff['isCompliant'].mean() * 100:.1f}%")
    c3.metric("Incomplete filings", f"{dff['isIncompleteFiling'].mean() * 100:.1f}%")
    c4.metric("Median Site EUI (kBtu/ft²)", f"{dff['siteEui'].median():,.1f}")

    left, right = st.columns([1, 1])

    with left:
        counts = dff["complianceStatus"].value_counts()
        fig = px.pie(
            values=counts.values,
            names=counts.index,
            hole=0.45,
            color=counts.index,
            color_discrete_map={"COMPLIED": BLUE, "NOT COMPLIED": RED},
        )
        fig.update_layout(title="Compliance status, all filings", height=380)
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("The headline finding")
        n_inc = diag["n_incomplete"]
        n_inc_complied = diag["n_incomplete_complied"]
        # Phrased conditionally: the zero is a live finding, not a fixed claim. If a future
        # refresh introduces a compliant incomplete filing, this sentence must not lie.
        if n_inc_complied == 0:
            headline_line = (
                f"None of the {n_inc:,} incomplete filings is recorded as COMPLIED."
            )
        else:
            headline_line = (
                f"{n_inc_complied:,} of {n_inc:,} incomplete filings are recorded as "
                f"COMPLIED ({n_inc_complied / n_inc * 100:.1f}%) — this was 0% at the "
                f"time of the Preliminary analysis, so the finding has changed."
            )
        st.markdown(
            f"""
`isIncompleteFiling` flags records where **propertyType, yearBuilt, grossFloorArea,
occupancy, and entityResponsible are all missing together** — not independently. That
joint pattern points to a submission that was never completed, rather than five
unrelated data gaps.

**{headline_line}**

So `complianceStatus` is partly a filing-completeness flag, not a pure measure of
energy performance. That single fact reshapes how the compliance trend on the next
tabs should be read — and what a Midterm classifier can honestly be built on.
            """
        )

    st.divider()
    st.subheader("Compliance rate — incomplete vs. complete filings")
    ct = diag["incomplete_crosstab_wide"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Complete filings — COMPLIED", f"{ct.loc[False, 'COMPLIED']:.1f}%")
    k2.metric("Complete filings — NOT COMPLIED", f"{ct.loc[False, 'NOT COMPLIED']:.1f}%")
    k3.metric("Incomplete filings — COMPLIED", f"{ct.loc[True, 'COMPLIED']:.1f}%")
    k4.metric("Incomplete filings — NOT COMPLIED", f"{ct.loc[True, 'NOT COMPLIED']:.1f}%")
    st.caption(
        "Computed on the full dataset (Section 4.4), not the filtered subset, since it "
        "characterizes the data as a whole."
    )

# ----------------------------------------------------------------------------
# Tab 2 — Data Quality
# ----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Missingness by column")
    st.caption(
        "Computed after the `Not Available` placeholder is standardized to a true null "
        "(Section 4.2). Left as-is, pandas would treat that string as a valid category."
    )
    miss = diag["missing_report"].reset_index(names="Column")
    fig = px.bar(
        miss.sort_values("Missing %"),
        x="Missing %", y="Column", orientation="h",
        color=miss.sort_values("Missing %")["Missing %"] > 50,
        color_discrete_map={True: RED, False: BLUE},
        text="Missing %",
    )
    fig.update_layout(height=650, showlegend=False, xaxis_title="Missing (%)")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    _pad_axis(fig, miss["Missing %"])
    st.plotly_chart(fig, width="stretch")
    st.info(
        f"Five structural fields share an identical missing rate — they are missing on the "
        f"same {diag['n_incomplete']:,} records. That is the pattern captured as "
        f"`isIncompleteFiling`.",
        icon="🔍",
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
    st.plotly_chart(fig, width="stretch")
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
            icon="✅",
        )
    else:
        st.error(f"{diag['missing_target']} missing complianceStatus values — resolve before modeling.")

# ----------------------------------------------------------------------------
# Tab 3 — Descriptive Stats
# ----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Frequency — filing volume by property type (6.1)")
    vol = dff_seg["propertyType"].value_counts().head(10)
    if len(vol):
        fig = px.bar(x=vol.values, y=vol.index, orientation="h",
                     labels={"x": "Number of Filings", "y": ""}, text=vol.values)
        fig.update_traces(marker_color=BLUE, textposition="outside",
                          texttemplate="%{text:,}", cliponaxis=False)
        fig.update_layout(height=max(380, 32 * len(vol)),
                          yaxis={"autorange": "reversed"})
        _pad_axis(fig, vol.values)
        st.plotly_chart(fig, width="stretch")
        if len(vol) >= 2:
            st.caption(
                f"{vol.index[0].title()} dominates by volume — {vol.iloc[0] / vol.iloc[1]:.1f}× "
                f"the next largest category ({vol.index[1].title()}). That imbalance is real "
                f"context for the Midterm: a classifier trained on this data sees far more "
                f"examples of one property type than of all the others combined."
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
    fig.add_bar(name="Mean", x=central["Metric"], y=central["Mean"], marker_color=RED)
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
        fig.update_traces(marker_color=GREEN, textposition="outside", cliponaxis=False)
        fig.update_layout(height=max(380, 30 * len(med)), yaxis={"autorange": "reversed"})
        _pad_axis(fig, med.values)
        st.plotly_chart(fig, width="stretch")
    st.caption(
        "Offices and retail stores report roughly ten times the intensity per square foot of "
        "self-storage or parking. Energy need here is a function of use, not efficiency — "
        "which is exactly why percentiles below are computed within type."
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
with tabs[3]:
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
        fig.add_vline(x=eui["siteEui"].median(), line_dash="dash", line_color=ORANGE)
        fig.add_vline(x=eui["siteEui"].mean(), line_dash="dot", line_color="black")
        fig.add_annotation(x=0.98, y=0.98, xref="paper", yref="paper",
                           xanchor="right", showarrow=False, align="right",
                           text=(f"<b>Median</b> {eui['siteEui'].median():,.1f}"
                                 f" &nbsp;<span style='color:{ORANGE}'>— —</span><br>"
                                 f"<b>Mean</b> {eui['siteEui'].mean():,.1f}"
                                 f" &nbsp;<span style='color:black'>· · ·</span>"),
                           bgcolor="rgba(255,255,255,0.85)", bordercolor="#CCCCCC",
                           borderwidth=1, borderpad=6)
        fig.update_layout(barmode="overlay", height=420,
                          title=f"Site EUI (n={len(eui):,} reported values)",
                          xaxis_title="Site EUI (kBtu/ft²)", yaxis_title="Buildings",
                          legend={"orientation": "h", "y": -0.25})
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.histogram(x=np.log1p(eui["siteEui"].clip(lower=0)), nbins=60)
        fig.update_traces(marker_color=DARK_BLUE)
        fig.update_layout(height=420, title="Log-transformed (viewing aid only)",
                          xaxis_title="log(1 + Site EUI)", yaxis_title="Buildings")
        st.plotly_chart(fig, width="stretch")
    st.caption(
        "The log panel redistributes visual mass without altering a single underlying value — "
        "no `siteEui` value in the dataframe is modified by this view."
    )

    st.divider()
    st.subheader("Figure 2 — Compliance rate by property type")
    comp_rate = (dff_seg.groupby("propertyType")["isCompliant"].mean()
                 .sort_values(ascending=False) * 100)
    if len(comp_rate):
        colors = [RED if v < 80 else BLUE for v in comp_rate.values]
        fig = go.Figure(go.Bar(x=comp_rate.values, y=comp_rate.index, orientation="h",
                               marker_color=colors, text=comp_rate.round(1),
                               texttemplate="%{text}%", textposition="outside",
                               cliponaxis=False))
        fig.add_vline(x=comp_rate.mean(), line_dash="dash", line_color=ORANGE,
                      annotation_text=f"Avg {comp_rate.mean():.1f}%",
                      annotation_position="bottom left")
        fig.add_vline(x=80, line_dash="dot", line_color=RED,
                      annotation_text="80% benchmark", annotation_position="top left")
        fig.update_layout(height=max(380, 30 * len(comp_rate)),
                          xaxis_title="% Compliant", yaxis={"autorange": "reversed"},
                          xaxis_range=[0, 108],
                          title="Red = below the 80% policy benchmark")
        st.plotly_chart(fig, width="stretch")
    st.caption(
        "An 80% threshold is used rather than the citywide average, so a bar's color only "
        "changes when that property type's own rate changes."
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
        st.plotly_chart(fig, width="stretch")
        st.caption("Median, not mean — a handful of real but extreme buildings would "
                   "misrepresent the typical building of an era.")
        st.write(age_n.rename("n per bucket").to_frame().T)
        st.divider()
    with c2:
        st.subheader("Figure 4 — Top 15 postal codes by total emissions")
        zips = (dff.groupby("postalCode")["co2Emissions"].sum()
                .sort_values(ascending=False).head(15))
        fig = px.bar(x=zips.values, y=zips.index, orientation="h",
                     labels={"x": "Total CO2e (Metric Tons)", "y": "Postal Code"})
        fig.update_traces(marker_color=GREEN)
        fig.update_layout(height=560, yaxis={"autorange": "reversed", "type": "category"})
        st.plotly_chart(fig, width="stretch")
        st.caption("Total, not average — this view is about where retrofit and enforcement "
                   "resources should go, which depends on total carbon impact.")

# ----------------------------------------------------------------------------
# Tab 5 — Compliance Trend (Figure 5 + Section 6.5)
# ----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Figure 5 — Is compliance declining, or is filing completeness?")
    yearly_overall = dff.groupby("programYear")["isCompliant"].mean() * 100
    yearly_complete = dff_complete.groupby("programYear")["isCompliant"].mean() * 100
    yearly_incomplete = dff.groupby("programYear")["isIncompleteFiling"].mean() * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=yearly_overall.index, y=yearly_overall.values,
                             mode="lines+markers", name="Overall % Compliant",
                             line={"color": BLUE, "width": 3}), secondary_y=False)
    fig.add_trace(go.Scatter(x=yearly_complete.index, y=yearly_complete.values,
                             mode="lines+markers", name="% Compliant (complete filings only)",
                             line={"color": GREEN, "width": 3},
                             marker_symbol="triangle-up"), secondary_y=False)
    fig.add_trace(go.Scatter(x=yearly_incomplete.index, y=yearly_incomplete.values,
                             mode="lines+markers", name="% Incomplete Filing",
                             line={"color": RED, "width": 2, "dash": "dash"},
                             marker_symbol="square"), secondary_y=True)
    fig.update_yaxes(title_text="% Compliant", secondary_y=False)
    fig.update_yaxes(title_text="% Incomplete Filing", secondary_y=True,
                     color=RED, showgrid=False)
    fig.update_layout(height=480, xaxis_title="Program Year",
                      legend={"orientation": "h", "y": -0.2})
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "The three series are overlaid because the finding *is* the gap between them, and "
        "the way that gap widens as the incomplete-filing rate climbs."
    )

    if len(yearly_overall) >= 2:
        drop_overall = yearly_overall.iloc[0] - yearly_overall.iloc[-1]
        drop_complete = yearly_complete.iloc[0] - yearly_complete.iloc[-1]
        share = (1 - drop_complete / drop_overall) * 100 if drop_overall else np.nan
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Overall decline, {yearly_overall.index[0]}–{yearly_overall.index[-1]}",
                  f"{drop_overall:.1f} pts")
        c2.metric("Decline, complete filings only", f"{drop_complete:.1f} pts")
        c3.metric("Share attributable to filing completeness",
                  f"{share:.0f}%" if pd.notna(share) else "n/a")
        st.info(
            "Read the blue line alone and the story is that buildings are performing worse. "
            "The green line says most of that gap is an artifact of how many filings were "
            "never completed. Anyone reading the raw compliance trend as a performance "
            "story is reading it wrong.",
            icon="📉",
        )

    st.divider()
    trend_table = pd.DataFrame({
        "Overall Compliance %": yearly_overall.round(1),
        "Compliance % (complete filings only)": yearly_complete.round(1),
        "Incomplete Filing %": yearly_incomplete.round(1),
    })
    st.dataframe(trend_table, width="stretch")

# ----------------------------------------------------------------------------
# Tab 6 — Correlations (Figure 6 + Section 6.6)
# ----------------------------------------------------------------------------
with tabs[5]:
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
    st.plotly_chart(fig, width="stretch")

    st.warning(
        "**Not all strong correlations are the same kind of finding.** `siteEui` ↔ `sourceEui` "
        "(r ≈ 0.94) is *empirical* — two distinct measurements that move together because "
        "efficient buildings tend to be efficient on both. `siteEui` ↔ `ghgIntensityPer1kSqft` "
        "(r ≈ 0.98) is *structural*: GHG intensity is calculated from the same energy data Site "
        "EUI already represents. No amount of cleaning will reduce the second one, and treating "
        "it as a data-quality problem would be a mistake.",
        icon="⚠️",
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
        "**Modeling recommendation for Midterm:** `co2Emissions` is the shared input behind "
        "both the EUI fields and the GHG-intensity field. Use *either* the EUI fields *or* "
        "`ghgIntensityPer1kSqft` in a given model — including both adds no information, it "
        "just inflates the standard errors on both coefficients."
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
    st.plotly_chart(fig, width="stretch")
    st.error(
        "Every continuous feature correlates with `isCompliant` at |r| < 0.05. Combined with "
        "the finding that `isIncompleteFiling` almost perfectly separates the two classes, "
        "compliance looks driven by categorical and structural factors — property type, "
        "responsible entity, filing completeness — rather than by energy performance itself. "
        "A Midterm model built only on continuous energy metrics is unlikely to perform well.",
        icon="🎯",
    )

st.divider()
st.caption(
    "EBEWE Preliminary Dashboard · pipeline mirrors EBEWE_Prelim_Analysis_v5.ipynb "
    "(Sections 3–7) · data: Los Angeles Open Data Portal, LADBS (public domain)"
)
