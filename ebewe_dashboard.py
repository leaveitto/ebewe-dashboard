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
        .mul(100).round(2)
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

    # 4.6b two targeted repairs to entityResponsible, applied before the field is used
    # as a grouping key anywhere. .str.strip() removes whitespace but not other
    # characters, so a value such as "_LANDMARK PROPERTY SERVICES" survives it.
    _before = df["entityResponsible"].copy()
    df["entityResponsible"] = df["entityResponsible"].str.replace(
        r"^[^A-Z0-9]+|[^A-Z0-9]+$", "", regex=True)
    diag["n_punct_stripped"] = int((_before != df["entityResponsible"]).sum())

    # Explicit alias map rather than a normalisation rule: a rule that merged on legal
    # form alone would also merge distinct firms sharing a stem. Each entry is a
    # judgement that can be read and reversed.
    ENTITY_ALIASES = {"BAY EFFICIENCY, LLC": "BAY EFFICIENCY"}
    _present = {k: v for k, v in ENTITY_ALIASES.items()
                if k in set(df["entityResponsible"].unique())}
    diag["aliases_applied"] = [
        (k, v, int((df["entityResponsible"] == k).sum()), int((df["entityResponsible"] == v).sum()))
        for k, v in _present.items()]
    if _present:
        df["entityResponsible"] = df["entityResponsible"].replace(_present)
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
    "Coverage Tier",
    "Responsible Entity",
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
        pct_complied = n_inc_complied / n_inc * 100 if n_inc else 0.0
        sep_pct = 100 - pct_complied
        if n_inc_complied == 0:
            headline_line = (
                f"None of the {n_inc:,} incomplete filings is recorded as COMPLIED."
            )
        else:
            # Deliberately NOT rounded to one decimal. At this magnitude a single
            # decimal renders 0.04% as "0.0%", which is what makes a near-total
            # separation look like a literal one.
            headline_line = (
                f"Only {n_inc_complied:,} of {n_inc:,} incomplete filings — "
                f"{pct_complied:.3f}% — are recorded as COMPLIED."
            )
        st.markdown(
            f"""
`isIncompleteFiling` flags records where **propertyType, yearBuilt, grossFloorArea,
occupancy, and entityResponsible are all missing together** — not independently. That
joint pattern points to a submission that was never completed, rather than five
unrelated data gaps.

**{headline_line}**

That is near-total separation, not literal separation — the distinction matters,
because "never" is a claim a single counterexample refutes while "{sep_pct:.2f}% of the
time" survives a data refresh.

So `complianceStatus` is partly a filing-completeness flag, not a pure measure of
energy performance. That single fact reshapes how the compliance trend on the next
tabs should be read — and what a Midterm classifier can honestly be built on.
            """
        )

    st.divider()
    st.subheader("Compliance rate — incomplete vs. complete filings")
    ct = diag["incomplete_crosstab_wide"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Complete filings — COMPLIED", f"{ct.loc[False, 'COMPLIED']:.2f}%")
    k2.metric("Complete filings — NOT COMPLIED", f"{ct.loc[False, 'NOT COMPLIED']:.2f}%")
    k3.metric("Incomplete filings — COMPLIED", f"{ct.loc[True, 'COMPLIED']:.2f}%",
              help="Two decimals on purpose — at one decimal this rounds to 0.0% and "
                   "reads as a literal zero when it is not one.")
    k4.metric("Incomplete filings — NOT COMPLIED", f"{ct.loc[True, 'NOT COMPLIED']:.2f}%")
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

    st.divider()
    st.subheader("Entity field repairs and audits")
    st.caption(
        "`entityResponsible` carries more signal about compliance than any other field "
        "(see the Responsible Entity tab), which is reason to check what it actually holds."
    )

    r1, r2, r3 = st.columns(3)
    r1.metric("Distinct entities", f"{diag.get('n_entities', 0):,}")
    r2.metric("Filings with punctuation stripped", f"{diag.get('n_punct_stripped', 0):,}")
    r3.metric("Aliases merged", f"{len(diag.get('aliases_applied', []))}")
    for src_name, dst_name, n_src, n_dst in diag.get("aliases_applied", []):
        st.caption(f"Merged **{src_name}** ({n_src:,} filings) into **{dst_name}** "
                   f"({n_dst:,}) — one firm previously counted as two agents.")

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
            "Detected by normalising away punctuation and legal form. Reported, not merged — "
            "a shared key shows two names are similar, not that two organisations are the same."
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
        st.plotly_chart(fig, width="stretch")

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
                icon="⚠️",
            )

        water = av[av["Field"].str.contains("ater")]
        if len(water) and water["Present %"].max() > 25 > water["Present %"].min():
            st.info(
                f"**The water fields split.** `totalWaterUse` is present on "
                f"{water['Present %'].max():.1f}% of filings while its components reach only "
                f"{water[water['Present %'] < 25]['Present %'].max():.1f}%. The published data "
                "supports aggregate water analysis but not the indoor/outdoor breakdown — a "
                "property of the source, not a scoping decision.",
                icon="💧",
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
                icon="🔁",
            )

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
            # Share is computed against every complete filing, not just the plotted top 10,
            # so the sentence cannot overstate dominance when the chart is filtered.
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
        st.warning(
            "**These are endpoint figures and they overstate their own precision.** The "
            "building roster is not constant — the ordinance phased in by size, so early "
            "years cover a fraction of the buildings later years do — and the final program "
            "year may still be partially processed. On a fixed set of buildings the same "
            "share ranges from roughly 40% to 84% depending on the cohort and end year "
            "chosen. See the balanced-panel comparison below.",
            icon="⚠️",
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
            icon="ℹ️",
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
                                 name="Complete filings only", line={"color": GREEN, "width": 3}))
        fig.add_trace(go.Scatter(x=_bal.index, y=_bal["Overall %"], mode="lines+markers",
                                 name="Overall", line={"color": BLUE, "width": 2, "dash": "dot"}))
        fig.update_layout(height=420, xaxis_title="Program Year", yaxis_title="% Compliant",
                          title=f"{len(_panel):,} buildings held constant, {_yrs[0]}–{_yrs[-1]}",
                          legend={"orientation": "h", "y": -0.2})
        st.plotly_chart(fig, width="stretch")

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
                    icon="📈",
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
with tabs[5]:
    st.subheader("Coverage tier predicts filing, not compliance")
    st.caption(
        "`ladbsBuildingCategory` is the ordinance's own size and ownership segmentation. "
        "Unlike propertyType, yearBuilt, grossFloorArea, occupancy and entityResponsible — "
        "all missing on the structurally incomplete filings — it is populated on every "
        "record. It is therefore the only structural field that can answer which buildings "
        "file incompletely."
    )

    _bands = (dff.groupby("sizeBand")["isIncompleteFiling"].size()
              .sort_values(ascending=False).index.tolist())
    if len(_bands) < 2:
        st.info(
            f"Only one coverage tier in the current selection ({_bands[0] if _bands else 'none'}), "
            "so there is no cross-tier comparison to make. Widen the filters.",
            icon="ℹ️",
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
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = go.Figure(go.Bar(y=_bands, x=_inc.values, orientation="h",
                                   marker_color=RED, text=_inc.round(1),
                                   textposition="outside", cliponaxis=False))
            fig.add_vline(x=dff["isIncompleteFiling"].mean() * 100, line_dash="dash",
                          line_color=ORANGE,
                          annotation_text=f"overall {dff['isIncompleteFiling'].mean()*100:.1f}%")
            fig.update_layout(height=430, xaxis_range=[0, _inc.max() * 1.3],
                              xaxis_title="% of filings structurally incomplete",
                              yaxis={"autorange": "reversed"},
                              title="B. Filing incompleteness by coverage tier")
            st.plotly_chart(fig, width="stretch")

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
                icon="🔍",
            )
        else:
            st.info(
                f"The compliance spread does not collapse once incomplete filings are excluded "
                f"({_sa:.1f} to {_sc:.1f} points). Coverage tier is associated with compliance "
                f"beyond its association with filing completeness, so the two panels are "
                f"separate findings rather than one mechanism.",
                icon="ℹ️",
            )

        st.divider()
        st.subheader("Incompleteness by tier and program year")
        st.caption(
            "Tiers moving together in one year points to a programme-wide change; one tier "
            "moving alone points to that tier's own phase-in date. Read alongside the "
            "Compliance Trend tab, where the 2019 break is documented as a deadline suspension."
        )
        _piv = (dff.pivot_table(index="programYear", columns="sizeBand",
                                values="isIncompleteFiling", aggfunc="mean") * 100).round(1)
        if len(_piv) >= 2 and _piv.shape[1] >= 2:
            fig = go.Figure()
            for band in _piv.columns:
                fig.add_trace(go.Scatter(x=_piv.index, y=_piv[band], mode="lines+markers",
                                         name=band))
            fig.update_layout(height=420, xaxis_title="Program Year",
                              yaxis_title="% structurally incomplete",
                              legend={"orientation": "h", "y": -0.25})
            st.plotly_chart(fig, width="stretch")

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
            icon="🧩",
        )

# ----------------------------------------------------------------------------
# Tab 7 — Responsible Entity (Section 6.7)
# ----------------------------------------------------------------------------
with tabs[6]:
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
            icon="ℹ️",
        )
    else:
        _base = dff_complete["isCompliant"].mean() * 100
        _types = dff_seg.groupby("propertyType")["isCompliant"].mean() * 100

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Agents with {MIN_FILINGS}+ filings", f"{len(_big):,}")
        c2.metric("Spread across agents",
                  f"{_big['compliance'].max() - _big['compliance'].min():.0f} pts")
        c3.metric("Spread across property types",
                  f"{_types.max() - _types.min():.0f} pts" if len(_types) > 1 else "n/a")
        # The property-type spread is dominated by whichever single category sits below
        # the benchmark; report it with and without, since the decomposition below shows
        # that outlier is itself an operator effect.
        if len(_types) > 2:
            _wo = _types.drop(_types.idxmin())
            st.caption(
                f"Same comparison, same basis. The property-type spread is "
                f"{_types.max() - _types.min():.0f} points, but almost all of it is one "
                f"category ({_types.idxmin().title()}, {_types.min():.1f}%) — drop it and the "
                f"remaining eleven span {_wo.max() - _wo.min():.0f} points. The responsible "
                f"entity discriminates far more sharply than building function does, and the "
                f"one category that looks like an exception is decomposed below."
            )
        else:
            st.caption("Same comparison, same basis. The responsible entity discriminates "
                       "far more sharply than building function does.")

        _show = pd.concat([_big.head(10), _big.tail(10)]).drop_duplicates()
        fig = go.Figure(go.Bar(
            x=_show["compliance"], y=_show.index, orientation="h",
            marker_color=[RED if v < _base else BLUE for v in _show["compliance"]],
            text=_show["compliance"], texttemplate="%{text}%",
            textposition="outside", cliponaxis=False,
            customdata=_show["filings"],
            hovertemplate="%{y}<br>%{x}% compliant<br>%{customdata:,} filings<extra></extra>"))
        fig.add_vline(x=_base, line_dash="dash", line_color=ORANGE,
                      annotation_text=f"Baseline {_base:.1f}%", annotation_position="top left")
        fig.update_layout(height=max(420, 26 * len(_show)), xaxis_range=[0, 118],
                          xaxis_title="% Compliant", yaxis={"autorange": "reversed"},
                          title="Lowest and highest compliance among high-volume agents")
        st.plotly_chart(fig, width="stretch")

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
                        icon="⚠️",
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
            st.info("No property type falls below the 80% benchmark in the current selection.",
                    icon="ℹ️")
        else:
            _t = _flagged.index[0]
            _cat = dff_complete[dff_complete["propertyType"] == _t]
            _ops = (_cat.groupby("entityResponsible")["isCompliant"]
                    .agg(["mean", "count"]).sort_values("count", ascending=False))
            _ops["mean"] = (_ops["mean"] * 100).round(1)
            _drv = _ops[(_ops["count"] >= 100) & (_ops["mean"] < _base)].index.tolist()

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
                    f"below the 80% line in Figure 2 at all. This is an operator effect "
                    f"presenting as a building-function effect.",
                    icon="🔍",
                )
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
            st.info("Too few classified agents in the current selection to compare filer types.",
                    icon="ℹ️")
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
                if _sp2 < 5:
                    st.warning(
                        f"**The outsourcing hypothesis does not hold.** The gap across filer "
                        f"types falls from {_sp1:.1f} to {_sp2:.1f} points once the "
                        f"regime-changing agents are removed, so the apparent effect is those "
                        f"specific firms rather than a general property of who files. Filer type "
                        f"is not a viable low-cardinality substitute: a model must carry agent "
                        f"identity itself, with high-volume agents retained and the rest bucketed.",
                        icon="⚠️",
                    )
                else:
                    st.success(
                        f"The filer-type gap survives removal of the regime-changing agents "
                        f"({_sp1:.1f} to {_sp2:.1f} points), so it is not driven by those firms "
                        f"alone. Filer type is worth testing as a low-cardinality feature "
                        f"alongside agent identity.",
                        icon="✅",
                    )
            _unc = _typed[_typed["type"] == "UNCLASSIFIED"]
            if len(_unc):
                st.caption(f"{len(_unc)} agent(s) unclassified by the keyword rules. Because they "
                           "sit at high compliance, classifying them would narrow the gap further "
                           "rather than widen it.")

# ----------------------------------------------------------------------------
# Tab 8 — Correlations (Figure 6 + Section 6.6)
# ----------------------------------------------------------------------------
with tabs[7]:
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
    "EBEWE Preliminary Dashboard · pipeline mirrors EBEWE_Prelim_Analysis_v21.ipynb "
    "(Sections 3–9) · data: Los Angeles Open Data Portal, LADBS (public domain)"
)
