"""
Analisis exhaustivo de mitigacion de sesgo en GENERO y PROFESION.

Genera todos los artefactos del capitulo `analisis.tex`:
- CSVs extendidos (metricas por config, subgrupos BOLD, pruebas estadisticas,
  bootstrap, ANOVA, correlaciones entre benchmarks).
- Tablas LaTeX (tab_ex_*).
- Figuras (ex_*) en 300 DPI.

Uso:
    python resultados/genero_profesion/analisis/analizar_exhaustivo_gp.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    HAS_SM = True
except Exception:
    HAS_SM = False

warnings.filterwarnings("ignore")
np.random.seed(20260527)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = BASE_DIR / "resultados" / "dataset_mitigacion_tesis.csv"
FIG_DIR = BASE_DIR / "resultados" / "genero_profesion" / "figuras"
DAT_DIR = BASE_DIR / "resultados" / "genero_profesion" / "analisis" / "datos"
TAB_DIR = BASE_DIR / "resultados" / "genero_profesion" / "analisis" / "tablas"
for d in (FIG_DIR, DAT_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Tipografia de figuras (pt). Parte del tribunal imprime la memoria en A4; ejes,
# leyendas y valores sobre barras deben leerse sin ampliar. Subir FS_* aqui y
# regenerar los PNG; no compensar solo reduciendo width en \includegraphics.
FS_VALUE = 12
FS_VALUE_DENSE = 10
FS_TICK = 12
FS_TITLE = 14
FS_SUPTITLE = 15
FS_LEGEND = 13
FS_AXIS = 13

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "cm",
    "font.size": FS_AXIS,
    "axes.titlesize": FS_TITLE,
    "axes.labelsize": FS_AXIS,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "legend.fontsize": FS_LEGEND,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

VALUE_BBOX = dict(
    boxstyle="round,pad=0.25", facecolor="white", alpha=0.95,
    edgecolor="#CBD5E0", linewidth=0.5,
)

C_TEXT = "#2D3436"
C_SPINE = "#4A5568"
C_GRID = "#E2E8F0"
C_EDGE = "#FFFFFF"
C_GENDER = "#2C5282"
C_PROFESSION = "#A0826D"
C_SOCIO = "#6B5B4F"
C_TECH_CDA = "#B8A090"
C_TECH_DEBIAS = "#2C5282"
C_FAM_GPT2 = "#2C5282"
C_FAM_LLAMA = "#744C4E"
C_POS = "#2F855A"
C_NEG = "#9B2C2C"

MODEL_SHORT = {
    "gpt2": "GPT-2 Base",
    "gpt2-medium": "GPT-2 Medium",
    "gpt2-xl": "GPT-2 XL",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama 3.2-1B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama 3.2-3B",
}
MODELS = list(MODEL_SHORT.values())
BIAS_TYPES = ["gender", "profession"]
BIAS_LABEL = {"gender": "Género", "profession": "Profesión"}
TECH_LABEL = {"cda": "Dataset", "debias": "Dataset + prompt"}
TECH_SHORT = {"cda": "D", "debias": "D+p"}
FAM_LABEL = {"gpt2": "GPT-2", "llama": "Llama"}

PROF_SUBGROUPS = [
    "artistic_occupations", "computer_occupations", "corporate_titles",
    "dance_occupations", "engineering_branches", "entertainer_occupations",
    "film_and_television_occupations", "healthcare_occupations",
    "industrial_occupations", "mental_health_occupations",
    "metalworking_occupations", "nursing_specialties",
    "professional_driver_types", "railway_industry_occupations",
    "scientific_occupations", "sewing_occupations",
    "theatre_personnel", "writing_occupations",
]
GENDER_SUBGROUPS = ["American_actors", "American_actresses"]


def save_fig(fig, name):
    fig.savefig(
        FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight",
        pad_inches=0.18, facecolor="white", edgecolor="none",
    )
    plt.close(fig)


def style_axis(ax, ylabel="", xlabel="", title=""):
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, pad=10, loc="left")
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.7, color=C_GRID)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_SPINE)


def add_bar_labels(ax, bars, fmt="{:.1f}", fontsize=None, offset=None,
                   fontweight="bold"):
    if fontsize is None:
        fontsize = FS_VALUE
    heights = [b.get_height() for b in bars]
    if not heights:
        return
    span = max(heights) - min(heights) if heights else 1.0
    abs_max = max(abs(h) for h in heights) if heights else 1.0
    if offset is not None:
        pad = offset
    else:
        pad = max(abs_max * 0.06, span * 0.08, abs_max * 0.025)
    for bar in bars:
        h = bar.get_height()
        if h is None or (isinstance(h, float) and np.isnan(h)):
            continue
        va = "bottom" if h >= 0 else "top"
        y = h + pad if h >= 0 else h - pad
        ax.annotate(
            fmt.format(h), (bar.get_x() + bar.get_width() / 2, y),
            ha="center", va=va, fontsize=fontsize, fontweight=fontweight,
            color=C_TEXT, bbox=VALUE_BBOX, clip_on=False, zorder=5,
        )


def fix_ylim(ax, values, label_pad=1.0):
    flat = [v for v in values if v is not None and not (
        isinstance(v, float) and np.isnan(v))]
    if not flat:
        return
    vmin, vmax = float(min(flat)), float(max(flat))
    extra = max(label_pad, abs(vmax) * 0.25, abs(vmin) * 0.25)
    if vmin < -0.05:
        ax.set_ylim(vmin - max(0.5, abs(vmin) * 0.14), vmax + extra)
        ax.axhline(0, color=C_SPINE, linestyle="--", linewidth=0.5, zorder=1)
    else:
        ax.set_ylim(0, vmax + extra)


def tech_legend_handles():
    return [
        Patch(facecolor=C_TECH_CDA, edgecolor=C_EDGE, label="Dataset"),
        Patch(facecolor=C_TECH_DEBIAS, edgecolor=C_EDGE, label="Dataset + prompt"),
    ]


def category_legend_handles():
    return [
        Patch(facecolor=C_GENDER, edgecolor=C_EDGE, label="Género"),
        Patch(facecolor=C_PROFESSION, edgecolor=C_EDGE, label="Profesión"),
        Patch(facecolor=C_SOCIO, edgecolor=C_EDGE, label="Socio (repr. prof.)"),
    ]


def familia_line_legend_handles():
    return [
        Line2D([0], [0], color=C_TEXT, ls="-", marker="o", markersize=8,
               markeredgecolor=C_SPINE, label="GPT-2"),
        Line2D([0], [0], color=C_TEXT, ls="--", marker="s", markersize=8,
               markeredgecolor=C_SPINE, label="Llama"),
    ]


def baseline_lora_legend_handles():
    return [
        Patch(facecolor=C_TECH_CDA, alpha=0.55, edgecolor=C_SPINE, label="Baseline"),
        Patch(facecolor=C_TECH_DEBIAS, alpha=0.55, edgecolor=C_SPINE, label="LoRA"),
    ]


def legend_below_fig(fig, handles, ncol=2, y=0.01, fontsize=None):
    """Leyenda bajo la figura (nunca loc='best' dentro del eje: tapa barras/valores)."""
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
        ncol=ncol, frameon=True, fontsize=fontsize or FS_LEGEND, columnspacing=1.4,
    )


def finish_fig(fig, bottom=0.22, top=0.92, left=0.07, right=0.98,
               hspace=None, wspace=None):
    kwargs = dict(bottom=bottom, top=top, left=left, right=right)
    if hspace is not None:
        kwargs["hspace"] = hspace
    if wspace is not None:
        kwargs["wspace"] = wspace
    fig.subplots_adjust(**kwargs)


def fmt_config_tex(row):
    tech = TECH_LABEL[row["technique"]]
    return f"{row['model_short']}, {tech}, {int(row['lora_trainable_pct'])}\\% LoRA"


def fmt_config_short(row):
    return (f"{row['model_short']} | {TECH_SHORT[row['technique']]} | "
            f"{int(row['lora_trainable_pct'])}%")

# =============================================================================
# Carga y computo de metricas extendidas
# =============================================================================

def load_and_compute():
    df = pd.read_csv(DATA_PATH)
    df["model_short"] = df["model_base"].map(MODEL_SHORT)
    df["family_label"] = df["family"].map(FAM_LABEL)
    sset = df[df["suite_kind"] == "crowspairs_stereoset"].copy()
    bold = df[df["suite_kind"] == "bold"].copy()

    # ----- StereoSet: SS, ICAT, LMS, distancia al ideal por categoria/tarea
    for bt in BIAS_TYPES:
        for task in ("inter", "intra"):
            ss_b = f"sset_{task}_blm_en_bbt_{bt}_ss"
            ss_l = f"sset_{task}_lrm_en_bbt_{bt}_ss"
            ic_b = f"sset_{task}_blm_en_bbt_{bt}_icat"
            ic_l = f"sset_{task}_lrm_en_bbt_{bt}_icat"
            lm_b = f"sset_{task}_blm_en_bbt_{bt}_lms"
            lm_l = f"sset_{task}_lrm_en_bbt_{bt}_lms"
            sset[f"delta_ss_{task}_{bt}"] = sset[ss_b] - sset[ss_l]
            sset[f"delta_icat_{task}_{bt}"] = sset[ic_l] - sset[ic_b]
            sset[f"delta_lms_{task}_{bt}"] = sset[lm_b] - sset[lm_l]
            sset[f"dist_ss_{task}_{bt}_blm"] = (sset[ss_b] - 50.0).abs()
            sset[f"dist_ss_{task}_{bt}_lrm"] = (sset[ss_l] - 50.0).abs()
            sset[f"delta_dist_ss_{task}_{bt}"] = (
                sset[f"dist_ss_{task}_{bt}_blm"] - sset[f"dist_ss_{task}_{bt}_lrm"]
            )

    # ----- CrowS-Pairs: STS y distancia al ideal
    for cat in ("gender", "socioeconomic"):
        sset[f"delta_csp_{cat}"] = (
            sset[f"csp_blm_en_bbt_{cat}_sts"] - sset[f"csp_lrm_en_bbt_{cat}_sts"]
        )
        sset[f"dist_csp_{cat}_blm"] = (sset[f"csp_blm_en_bbt_{cat}_sts"] - 50.0).abs()
        sset[f"dist_csp_{cat}_lrm"] = (sset[f"csp_lrm_en_bbt_{cat}_sts"] - 50.0).abs()
        sset[f"delta_dist_csp_{cat}"] = (
            sset[f"dist_csp_{cat}_blm"] - sset[f"dist_csp_{cat}_lrm"]
        )

    # ----- Metricas compuestas GP
    sset["M_stereo_gp"] = (
        sset["delta_ss_inter_gender"] + sset["delta_ss_intra_gender"]
        + sset["delta_ss_inter_profession"] + sset["delta_ss_intra_profession"]
    ) / 4.0
    sset["M_gp"] = (
        sset["delta_ss_inter_gender"] + sset["delta_ss_intra_gender"]
        + sset["delta_ss_inter_profession"] + sset["delta_ss_intra_profession"]
        + sset["delta_csp_gender"] + sset["delta_csp_socioeconomic"]
    ) / 6.0

    # ----- Coste linguistico
    sset["delta_lms_inter_global"] = (
        sset["sset_inter_blm_en_lms"] - sset["sset_inter_lrm_en_lms"]
    )
    sset["delta_lms_intra_global"] = (
        sset["sset_intra_blm_en_lms"] - sset["sset_intra_lrm_en_lms"]
    )
    sset["delta_lms_avg"] = (
        sset["delta_lms_inter_global"] + sset["delta_lms_intra_global"]
    ) / 2.0

    # ----- MEI: clasico, estricto y normalizado
    sset["MEI_gp"] = sset["M_gp"] / sset["delta_lms_avg"].abs().clip(lower=0.05)
    sset["MEI_gp_strict"] = sset["M_gp"] / sset["delta_lms_avg"].abs().clip(lower=0.5)
    mei_min = sset["MEI_gp"].min()
    mei_max = sset["MEI_gp"].max()
    rng = max(mei_max - mei_min, 1e-6)
    sset["MEI_gp_norm"] = (sset["MEI_gp"] - mei_min) / rng

    # ----- BOLD: pareadas por dominio
    for dom in BIAS_TYPES:
        bold[f"d_rggap_{dom}"] = bold[f"bold_blm_{dom}_rg_gap"] - bold[f"bold_lrm_{dom}_rg_gap"]
        bold[f"d_regard_{dom}"] = bold[f"bold_lrm_{dom}_avg_regard"] - bold[f"bold_blm_{dom}_avg_regard"]
        bold[f"d_neg_{dom}"] = bold[f"bold_lrm_{dom}_negative_pct"] - bold[f"bold_blm_{dom}_negative_pct"]
        bold[f"d_pos_{dom}"] = bold[f"bold_lrm_{dom}_positive_pct"] - bold[f"bold_blm_{dom}_positive_pct"]
        bold[f"d_neutral_{dom}"] = bold[f"bold_lrm_{dom}_neutral_pct"] - bold[f"bold_blm_{dom}_neutral_pct"]
        bold[f"d_length_{dom}"] = bold[f"bold_lrm_{dom}_avg_length"] - bold[f"bold_blm_{dom}_avg_length"]
        bold[f"d_empty_{dom}"] = bold[f"bold_lrm_{dom}_empty_completions"] - bold[f"bold_blm_{dom}_empty_completions"]

    return sset, bold


# =============================================================================
# Subgrupos BOLD: profesion (18) y genero (2)
# =============================================================================

def build_bold_subgroups(bold):
    rows = []
    meta_cols = ["model_short", "family", "family_label",
                 "technique", "lora_trainable_pct"]
    for _, r in bold.iterrows():
        for sg in PROF_SUBGROUPS:
            base = r.get(f"bold_blm_profession_rg_gm_{sg}", np.nan)
            lora = r.get(f"bold_lrm_profession_rg_gm_{sg}", np.nan)
            rows.append({
                **{c: r[c] for c in meta_cols},
                "domain": "profession",
                "subgroup": sg,
                "regard_blm": base,
                "regard_lrm": lora,
                "delta_regard": (lora - base) if pd.notna(base) and pd.notna(lora) else np.nan,
            })
        for sg in GENDER_SUBGROUPS:
            base = r.get(f"bold_blm_gender_rg_gm_{sg}", np.nan)
            lora = r.get(f"bold_lrm_gender_rg_gm_{sg}", np.nan)
            rows.append({
                **{c: r[c] for c in meta_cols},
                "domain": "gender",
                "subgroup": sg,
                "regard_blm": base,
                "regard_lrm": lora,
                "delta_regard": (lora - base) if pd.notna(base) and pd.notna(lora) else np.nan,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Estadistica inferencial
# =============================================================================

def cohens_d_paired(x_pre, x_post):
    diff = np.asarray(x_pre, dtype=float) - np.asarray(x_post, dtype=float)
    diff = diff[~np.isnan(diff)]
    if diff.size < 2:
        return float("nan")
    s = diff.std(ddof=1)
    if s == 0:
        return float("nan")
    return float(diff.mean() / s)


def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    n_gt = sum((xi > y).sum() for xi in x)
    n_lt = sum((xi < y).sum() for xi in x)
    return float((n_gt - n_lt) / (x.size * y.size))


def wilcoxon_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if x.size < 5:
        return float("nan"), float("nan")
    diff = x - y
    if np.allclose(diff, 0):
        return float("nan"), 1.0
    try:
        w, p = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        return float(w), float(p)
    except Exception:
        return float("nan"), float("nan")


def sign_test_pos(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return 0, 0, float("nan")
    pos = int((v > 0).sum())
    n = int(v.size)
    if n == 0:
        return 0, 0, float("nan")
    p = stats.binomtest(pos, n, 0.5, alternative="greater").pvalue
    return pos, n, float(p)


def bootstrap_ic(values, n_boot=10000, alpha=0.05, agg=np.mean):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(20260527)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    samples = agg(v[idx], axis=1)
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(agg(v)), float(lo), float(hi)


def stat_table_for_pair(label, blm, lrm, group_label="Global"):
    diff = blm - lrm
    w, p = wilcoxon_safe(blm, lrm)
    d = cohens_d_paired(blm, lrm)
    cd = cliffs_delta(blm, lrm)
    pos, n, p_sign = sign_test_pos(diff)
    mean_d, lo, hi = bootstrap_ic(diff)
    return {
        "grupo": group_label,
        "metrica": label,
        "n": int(n),
        "media_delta": mean_d,
        "ic95_lo": lo,
        "ic95_hi": hi,
        "wilcoxon_W": w,
        "wilcoxon_p": p,
        "cohen_d": d,
        "cliff_delta": cd,
        "configs_mejoran": pos,
        "p_signo": p_sign,
    }

# =============================================================================
# ANOVA y correlaciones
# =============================================================================

def anova_decomposition(sset):
    """Descomposicion de varianza M_GP ~ technique * family * lora_pct.

    Devuelve un DataFrame con SS, gl, MS, F, p, eta_p2 por factor.
    Si statsmodels no esta disponible, se calcula una descomposicion manual.
    """
    df = sset[["M_gp", "technique", "family", "lora_trainable_pct"]].copy()
    df["lora"] = df["lora_trainable_pct"].astype(int).astype(str)
    if HAS_SM:
        formula = "M_gp ~ C(technique) * C(family) * C(lora)"
        model = ols(formula, data=df).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        anova = anova.reset_index().rename(columns={
            "index": "factor", "sum_sq": "SS", "df": "gl", "F": "F",
            "PR(>F)": "p"
        })
        ss_total = anova["SS"].sum()
        anova["eta2"] = anova["SS"] / ss_total
        return anova[["factor", "SS", "gl", "F", "p", "eta2"]]
    # Fallback manual: solo efectos principales (ANOVA tipo I)
    rows = []
    grand = df["M_gp"].mean()
    ss_total = ((df["M_gp"] - grand) ** 2).sum()
    for factor in ["technique", "family", "lora"]:
        ss_f = sum(
            len(g) * (g["M_gp"].mean() - grand) ** 2
            for _, g in df.groupby(factor)
        )
        rows.append({
            "factor": factor, "SS": ss_f, "gl": df[factor].nunique() - 1,
            "F": np.nan, "p": np.nan, "eta2": ss_f / ss_total
        })
    rows.append({"factor": "Total", "SS": ss_total, "gl": len(df) - 1,
                 "F": np.nan, "p": np.nan, "eta2": 1.0})
    return pd.DataFrame(rows)


def correlation_benchmarks(sset, bold):
    """Spearman entre metricas clave."""
    metrics = {
        "ΔSS_inter_gen": sset["delta_ss_inter_gender"].values,
        "ΔSS_intra_gen": sset["delta_ss_intra_gender"].values,
        "ΔSS_inter_prof": sset["delta_ss_inter_profession"].values,
        "ΔSS_intra_prof": sset["delta_ss_intra_profession"].values,
        "ΔSTS_gen": sset["delta_csp_gender"].values,
        "ΔSTS_socio": sset["delta_csp_socioeconomic"].values,
    }
    bold_aligned = bold.set_index("exp_group").reindex(sset["exp_group"])
    metrics["Δrg_gap_gen"] = bold_aligned["d_rggap_gender"].values
    metrics["Δrg_gap_prof"] = bold_aligned["d_rggap_profession"].values
    metrics["Δregard_gen"] = bold_aligned["d_regard_gender"].values
    metrics["Δregard_prof"] = bold_aligned["d_regard_profession"].values

    keys = list(metrics.keys())
    n = len(keys)
    rho = np.full((n, n), np.nan)
    pmat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            x, y = metrics[keys[i]], metrics[keys[j]]
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 5:
                continue
            try:
                r, p = stats.spearmanr(x[mask], y[mask])
                rho[i, j] = r
                pmat[i, j] = p
            except Exception:
                pass
    rho_df = pd.DataFrame(rho, index=keys, columns=keys)
    p_df = pd.DataFrame(pmat, index=keys, columns=keys)
    return rho_df, p_df


# =============================================================================
# Tests por familia y por tecnica
# =============================================================================

def stat_table_full(sset, bold):
    """Genera tabla extensa de pruebas pareadas baseline-vs-LoRA."""
    rows = []
    pairs = [
        (r"$\Delta$SS inter género", "sset_inter_blm_en_bbt_gender_ss",
         "sset_inter_lrm_en_bbt_gender_ss", "sset"),
        (r"$\Delta$SS inter profesión", "sset_inter_blm_en_bbt_profession_ss",
         "sset_inter_lrm_en_bbt_profession_ss", "sset"),
        (r"$\Delta$SS intra género", "sset_intra_blm_en_bbt_gender_ss",
         "sset_intra_lrm_en_bbt_gender_ss", "sset"),
        (r"$\Delta$SS intra profesión", "sset_intra_blm_en_bbt_profession_ss",
         "sset_intra_lrm_en_bbt_profession_ss", "sset"),
        (r"$\Delta$STS género", "csp_blm_en_bbt_gender_sts",
         "csp_lrm_en_bbt_gender_sts", "sset"),
        (r"$\Delta$STS socio (repr. prof.)", "csp_blm_en_bbt_socioeconomic_sts",
         "csp_lrm_en_bbt_socioeconomic_sts", "sset"),
        (r"$\Delta$LMS inter (coste)", "sset_inter_blm_en_lms",
         "sset_inter_lrm_en_lms", "sset"),
        (r"$\Delta$LMS intra (coste)", "sset_intra_blm_en_lms",
         "sset_intra_lrm_en_lms", "sset"),
        (r"$\Delta$rg\_gap género (BOLD)", "bold_blm_gender_rg_gap",
         "bold_lrm_gender_rg_gap", "bold"),
        (r"$\Delta$rg\_gap profesión (BOLD)", "bold_blm_profession_rg_gap",
         "bold_lrm_profession_rg_gap", "bold"),
        (r"$\Delta$regard género (BOLD)", "bold_lrm_gender_avg_regard",
         "bold_blm_gender_avg_regard", "bold_inv"),
        (r"$\Delta$regard profesión (BOLD)", "bold_lrm_profession_avg_regard",
         "bold_blm_profession_avg_regard", "bold_inv"),
    ]
    groups = [("Global", lambda d: d)]
    for fam in sorted(sset["family"].unique()):
        groups.append((f"Familia={FAM_LABEL[fam]}", lambda d, f=fam: d[d["family"] == f]))
    for tech in ("cda", "debias"):
        groups.append((f"Técnica={TECH_LABEL[tech]}", lambda d, t=tech: d[d["technique"] == t]))

    for label, c_b, c_l, src in pairs:
        for gname, gfilter in groups:
            base_df = sset if src == "sset" else bold
            sub = gfilter(base_df)
            if src == "bold_inv":
                # c_b=LRM, c_l=BLM: diff = LRM - BLM (= d_regard)
                blm = sub[c_b].values
                lrm = sub[c_l].values
            else:
                blm = sub[c_b].values
                lrm = sub[c_l].values
            row = stat_table_for_pair(label, blm, lrm, gname)
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Bootstrap por celda (familia x tecnica) y por config
# =============================================================================

def bootstrap_by_cell(sset, n_boot=10000):
    rows = []
    for col, mname in [("M_gp", "M_GP"), ("M_stereo_gp", "M_stereo_GP"),
                       ("MEI_gp", "MEI_GP")]:
        for fam in sorted(sset["family"].unique()):
            for tech in ("cda", "debias"):
                sub = sset[(sset["family"] == fam) & (sset["technique"] == tech)]
                if sub.empty:
                    continue
                m, lo, hi = bootstrap_ic(sub[col].values, n_boot=n_boot)
                rows.append({
                    "metrica": mname, "familia": FAM_LABEL[fam],
                    "tecnica": TECH_LABEL[tech], "n": len(sub),
                    "media": m, "ic95_lo": lo, "ic95_hi": hi,
                })
        # globales
        m, lo, hi = bootstrap_ic(sset[col].values, n_boot=n_boot)
        rows.append({
            "metrica": mname, "familia": "Global", "tecnica": "Global",
            "n": len(sset), "media": m, "ic95_lo": lo, "ic95_hi": hi,
        })
    return pd.DataFrame(rows)


def bootstrap_per_config(sset, n_boot=2000):
    """IC bootstrap del M_GP de cada config re-muestreando sus 6 componentes."""
    comps = ["delta_ss_inter_gender", "delta_ss_intra_gender",
             "delta_ss_inter_profession", "delta_ss_intra_profession",
             "delta_csp_gender", "delta_csp_socioeconomic"]
    rng = np.random.default_rng(20260527)
    rows = []
    for _, r in sset.iterrows():
        v = np.array([r[c] for c in comps], dtype=float)
        idx = rng.integers(0, v.size, size=(n_boot, v.size))
        samples = v[idx].mean(axis=1)
        lo, hi = np.quantile(samples, [0.025, 0.975])
        rows.append({
            "model_short": r["model_short"],
            "technique": r["technique"],
            "lora_trainable_pct": r["lora_trainable_pct"],
            "M_gp": r["M_gp"],
            "M_gp_ic_lo": float(lo),
            "M_gp_ic_hi": float(hi),
            "MEI_gp": r["MEI_gp"],
            "delta_lms_avg": r["delta_lms_avg"],
        })
    return pd.DataFrame(rows)

# =============================================================================
# Exportacion de CSVs
# =============================================================================

def export_csvs(sset, bold, subgroups, stat_df, anova_df, rho_df, p_df,
                boot_cells, boot_per_config):
    cols_keep = [
        "model_short", "family", "family_label", "technique",
        "lora_trainable_pct", "exp_group",
        "delta_ss_inter_gender", "delta_ss_inter_profession",
        "delta_ss_intra_gender", "delta_ss_intra_profession",
        "delta_dist_ss_inter_gender", "delta_dist_ss_inter_profession",
        "delta_dist_ss_intra_gender", "delta_dist_ss_intra_profession",
        "delta_icat_inter_gender", "delta_icat_inter_profession",
        "delta_icat_intra_gender", "delta_icat_intra_profession",
        "delta_lms_inter_gender", "delta_lms_inter_profession",
        "delta_lms_intra_gender", "delta_lms_intra_profession",
        "delta_csp_gender", "delta_csp_socioeconomic",
        "delta_dist_csp_gender", "delta_dist_csp_socioeconomic",
        "M_gp", "M_stereo_gp", "delta_lms_avg",
        "delta_lms_inter_global", "delta_lms_intra_global",
        "MEI_gp", "MEI_gp_strict", "MEI_gp_norm",
    ]
    sset[cols_keep].to_csv(DAT_DIR / "metricas_extendidas_gp.csv",
                           index=False, encoding="utf-8")

    bold_cols = ["model_short", "family", "family_label", "technique",
                 "lora_trainable_pct", "exp_group"]
    for dom in BIAS_TYPES:
        for k in ("avg_regard", "rg_gap", "avg_length", "negative_pct",
                  "positive_pct", "neutral_pct", "empty_completions",
                  "total_completions"):
            for p in ("blm", "lrm"):
                col = f"bold_{p}_{dom}_{k}"
                if col in bold.columns:
                    bold_cols.append(col)
        bold_cols.extend([f"d_rggap_{dom}", f"d_regard_{dom}",
                          f"d_neg_{dom}", f"d_pos_{dom}", f"d_neutral_{dom}",
                          f"d_length_{dom}", f"d_empty_{dom}"])
    bold[[c for c in bold_cols if c in bold.columns]].to_csv(
        DAT_DIR / "bold_metricas_gp.csv", index=False, encoding="utf-8"
    )

    subgroups.to_csv(DAT_DIR / "bold_subgrupos_gp.csv",
                     index=False, encoding="utf-8")
    stat_df.to_csv(DAT_DIR / "pruebas_estadisticas.csv",
                   index=False, encoding="utf-8")
    anova_df.to_csv(DAT_DIR / "factores_anova.csv",
                    index=False, encoding="utf-8")
    rho_df.to_csv(DAT_DIR / "correlacion_benchmarks_rho.csv",
                  encoding="utf-8")
    p_df.to_csv(DAT_DIR / "correlacion_benchmarks_p.csv",
                encoding="utf-8")
    boot_cells.to_csv(DAT_DIR / "bootstrap_intervalos.csv",
                      index=False, encoding="utf-8")
    boot_per_config.to_csv(DAT_DIR / "bootstrap_por_config.csv",
                           index=False, encoding="utf-8")


# =============================================================================
# Tablas LaTeX
# =============================================================================

def fmt_p(p):
    if pd.isna(p):
        return "--"
    if p < 0.001:
        return r"\textless 0,001"
    return f"{p:.3f}".replace(".", ",")


def fmt_num(x, decimals=2, plus=False):
    if pd.isna(x):
        return "--"
    s = f"{x:+.{decimals}f}" if plus else f"{x:.{decimals}f}"
    return s.replace(".", ",")


def write_panorama_baseline(sset, bold):
    rows = []
    for m in MODELS:
        sub_s = sset[sset["model_short"] == m]
        sub_b = bold[bold["model_short"] == m]
        if sub_s.empty or sub_b.empty:
            continue
        rows.append({
            "model": m,
            "ss_inter_g": sub_s["sset_inter_blm_en_bbt_gender_ss"].iloc[0],
            "ss_inter_p": sub_s["sset_inter_blm_en_bbt_profession_ss"].iloc[0],
            "ss_intra_g": sub_s["sset_intra_blm_en_bbt_gender_ss"].iloc[0],
            "ss_intra_p": sub_s["sset_intra_blm_en_bbt_profession_ss"].iloc[0],
            "lms_inter": sub_s["sset_inter_blm_en_lms"].iloc[0],
            "lms_intra": sub_s["sset_intra_blm_en_lms"].iloc[0],
            "csp_g": sub_s["csp_blm_en_bbt_gender_sts"].iloc[0],
            "csp_s": sub_s["csp_blm_en_bbt_socioeconomic_sts"].iloc[0],
            "regard_g": sub_b["bold_blm_gender_avg_regard"].iloc[0],
            "regard_p": sub_b["bold_blm_profession_avg_regard"].iloc[0],
            "gap_g": sub_b["bold_blm_gender_rg_gap"].iloc[0],
            "gap_p": sub_b["bold_blm_profession_rg_gap"].iloc[0],
        })
    lines = [
        r"\begin{table}[H]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\caption{Panorama del sesgo \emph{baseline} (sin LoRA) por modelo. StereoSet SS y CrowS-Pairs STS: porcentaje, ideal 50; StereoSet LMS: porcentaje agregado en inglés (ideal 100); BOLD regard: rango [-1,1], ideal +1; BOLD brecha: valor absoluto, ideal 0.}",
        r"\label{tab:ex_panorama_baseline}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}lrrrrrrrrrrrr@{}}",
        r"\toprule",
        r"& \multicolumn{4}{c}{\textbf{StereoSet SS}} & \multicolumn{2}{c}{\textbf{StereoSet LMS}} & \multicolumn{2}{c}{\textbf{CSP STS}} & \multicolumn{2}{c}{\textbf{BOLD regard}} & \multicolumn{2}{c}{\textbf{BOLD brecha}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}",
        r"\textbf{Modelo} & Ig & Ip & Tg & Tp & Li & Ti & gen & soc & gen & prof & gen & prof \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['model']} & {fmt_num(r['ss_inter_g'])} & {fmt_num(r['ss_inter_p'])} & "
            f"{fmt_num(r['ss_intra_g'])} & {fmt_num(r['ss_intra_p'])} & "
            f"{fmt_num(r['lms_inter'])} & {fmt_num(r['lms_intra'])} & "
            f"{fmt_num(r['csp_g'])} & {fmt_num(r['csp_s'])} & "
            f"{fmt_num(r['regard_g'], 3)} & {fmt_num(r['regard_p'], 3)} & "
            f"{fmt_num(r['gap_g'], 3)} & {fmt_num(r['gap_p'], 3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%",
              r"}",
              r"\par\smallskip\footnotesize Ig/Tg = Inter/Intra género; Ip/Tp = Inter/Intra profesión; Li/Ti = LMS \textit{inter}/\textit{intra} agregado (inglés, \textit{baseline} sin LoRA).",
              r"\end{table}"]
    (TAB_DIR / "tab_ex_panorama_baseline.tex").write_text(
        "\n".join(lines), encoding="utf-8")


def write_resumen_categoria(sset, bold):
    n_boot = 10000
    items = []
    cats = [
        (r"StereoSet inter -- género ($\Delta$SS, p.p.)", sset["delta_ss_inter_gender"].values),
        (r"StereoSet inter -- profesión ($\Delta$SS, p.p.)", sset["delta_ss_inter_profession"].values),
        (r"StereoSet intra -- género ($\Delta$SS, p.p.)", sset["delta_ss_intra_gender"].values),
        (r"StereoSet intra -- profesión ($\Delta$SS, p.p.)", sset["delta_ss_intra_profession"].values),
        (r"CrowS-Pairs -- género ($\Delta$STS, p.p.)", sset["delta_csp_gender"].values),
        (r"CrowS-Pairs -- socio/prof. ($\Delta$STS, p.p.)", sset["delta_csp_socioeconomic"].values),
        (r"BOLD -- $\Delta$rg\_gap género", bold["d_rggap_gender"].values),
        (r"BOLD -- $\Delta$rg\_gap profesión", bold["d_rggap_profession"].values),
        (r"BOLD -- $\Delta$regard género", bold["d_regard_gender"].values),
        (r"BOLD -- $\Delta$regard profesión", bold["d_regard_profession"].values),
        (r"$M_{GP}$ (compuesta, p.p.)", sset["M_gp"].values),
        (r"$\mathrm{MEI}_{GP}$", sset["MEI_gp"].values),
    ]
    for label, vals in cats:
        m, lo, hi = bootstrap_ic(vals, n_boot=n_boot)
        med = np.nanmedian(vals)
        vmax = np.nanmax(vals)
        vmin = np.nanmin(vals)
        pos = int((np.asarray(vals) > 0).sum())
        n = int(np.asarray(vals).size)
        items.append((label, m, med, lo, hi, vmax, vmin, pos, n))
    lines = [
        r"\begin{table}[H]",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Resumen por métrica con IC95\% bootstrap (n=30 configs, 10\,000 re-muestras).}",
        r"\label{tab:ex_resumen_categoria}",
        r"\begin{tabularx}{\linewidth}{@{} Y r r r r r r @{}}",
        r"\toprule",
        r"\textbf{Métrica} & \textbf{Media} & \textbf{IC95\%} & \textbf{Med.} & \textbf{Mín.} & \textbf{Máx.} & \textbf{n+/n} \\",
        r"\midrule",
    ]
    for (label, m, med, lo, hi, vmax, vmin, pos, n) in items:
        lines.append(
            f"{label} & {fmt_num(m, 3)} & [{fmt_num(lo, 3)}; {fmt_num(hi, 3)}] & "
            f"{fmt_num(med, 3)} & {fmt_num(vmin, 3)} & {fmt_num(vmax, 3)} & "
            f"{pos}/{n} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}",
              r"\par\smallskip\footnotesize Med. = mediana; n+ = número de configs con $\Delta>0$ (mejora).",
              r"\end{table}"]
    (TAB_DIR / "tab_ex_resumen_categoria.tex").write_text(
        "\n".join(lines), encoding="utf-8")


def write_estadistica_inferencial(stat_df):
    sub = stat_df[stat_df["grupo"] == "Global"]
    lines = [
        r"\begin{table}[H]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Pruebas pareadas baseline vs. LoRA (n=30 configs). Cohen's $d$ pareado, Cliff's $\delta$, Wilcoxon y prueba de signos.}",
        r"\label{tab:ex_estadistica_inferencial}",
        r"\begin{tabularx}{\linewidth}{@{} Y r r r r r r @{}}",
        r"\toprule",
        r"\textbf{Métrica} & \textbf{Media $\Delta$} & \textbf{IC95\%} & \textbf{$d$} & \textbf{$\delta$} & \textbf{$p_W$} & \textbf{n+/30 ($p_s$)} \\",
        r"\midrule",
    ]
    for _, r in sub.iterrows():
        lines.append(
            f"{r['metrica']} & {fmt_num(r['media_delta'], 3)} & "
            f"[{fmt_num(r['ic95_lo'], 3)}; {fmt_num(r['ic95_hi'], 3)}] & "
            f"{fmt_num(r['cohen_d'], 2)} & {fmt_num(r['cliff_delta'], 2)} & "
            f"{fmt_p(r['wilcoxon_p'])} & {int(r['configs_mejoran'])}/{int(r['n'])} ({fmt_p(r['p_signo'])}) \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabularx}",
        r"\par\smallskip\footnotesize $d$ = Cohen; $\delta$ = Cliff; $p_W$ = Wilcoxon; $p_s$ = signos.",
        r"\end{table}",
    ]
    (TAB_DIR / "tab_ex_estadistica_inferencial.tex").write_text(
        "\n".join(lines), encoding="utf-8")


def write_factores_anova(anova_df):
    lines = [
        r"\begin{table}[H]",
        r"\centering\footnotesize",
        r"\caption{Descomposición de varianza de $M_{GP}$ por factor experimental (ANOVA tipo II, modelo $M_{GP}\sim$ técnica $\times$ familia $\times$ \%LoRA).}",
        r"\label{tab:ex_factores_anova}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"\textbf{Factor} & \textbf{SS} & \textbf{gl} & \textbf{F} & \textbf{p} & \textbf{$\eta^2$ (\%)} \\",
        r"\midrule",
    ]
    for _, r in anova_df.iterrows():
        lines.append(
            f"{r['factor']} & {fmt_num(r['SS'], 2)} & {int(r['gl']) if pd.notna(r['gl']) else '--'} & "
            f"{fmt_num(r['F'], 2)} & {fmt_p(r['p'])} & {fmt_num(r['eta2']*100, 1)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "tab_ex_factores_anova.tex").write_text(
        "\n".join(lines), encoding="utf-8")


def write_top10_mei(sset, boot_per_config):
    merged = sset.merge(
        boot_per_config[["model_short", "technique", "lora_trainable_pct",
                         "M_gp_ic_lo", "M_gp_ic_hi"]],
        on=["model_short", "technique", "lora_trainable_pct"], how="left"
    )
    top = merged.sort_values("MEI_gp", ascending=False).head(10)
    lines = [
        r"\begin{table}[H]",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Top-10 configuraciones por MEI$_{GP}$ con IC95\% bootstrap del $M_{GP}$.}",
        r"\label{tab:ex_top10_mei}",
        r"\begin{tabularx}{\linewidth}{@{} c Y r r r r @{}}",
        r"\toprule",
        r"\textbf{\#} & \textbf{Configuración} & \textbf{$M_{GP}$} & \textbf{IC95\%} & \textbf{$\delta_{LMS}$} & \textbf{MEI$_{GP}$} \\",
        r"\midrule",
    ]
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        lines.append(
            f"{i} & {fmt_config_tex(r)} & "
            f"{fmt_num(r['M_gp'], 2)} & "
            f"[{fmt_num(r['M_gp_ic_lo'], 2)}; {fmt_num(r['M_gp_ic_hi'], 2)}] & "
            f"{fmt_num(r['delta_lms_avg'], 2, plus=True)} & "
            f"{fmt_num(r['MEI_gp'], 2)} \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabularx}",
        r"\end{table}",
    ]
    (TAB_DIR / "tab_ex_top10_mei.tex").write_text(
        "\n".join(lines), encoding="utf-8")


def write_familia_x_tecnica(sset, bold, stat_df, boot_cells):
    rows = []
    for fam in sorted(sset["family"].unique()):
        for tech in ("cda", "debias"):
            sub_s = sset[(sset["family"] == fam) & (sset["technique"] == tech)]
            sub_b = bold[(bold["family"] == fam) & (bold["technique"] == tech)]
            rows.append({
                "fam": FAM_LABEL[fam], "tech": TECH_LABEL[tech],
                "n": len(sub_s),
                "M_gp": sub_s["M_gp"].mean(),
                "M_stereo_gp": sub_s["M_stereo_gp"].mean(),
                "MEI_gp": sub_s["MEI_gp"].mean(),
                "delta_lms": sub_s["delta_lms_avg"].mean(),
                "rggap_gen": sub_b["d_rggap_gender"].mean(),
                "rggap_prof": sub_b["d_rggap_profession"].mean(),
            })
    lines = [
        r"\begin{table}[H]",
        r"\centering\footnotesize",
        r"\caption{Promedios por celda familia $\times$ técnica (GPT-2: $n=9$; Llama: $n=6$).}",
        r"\label{tab:ex_familia_x_tecnica}",
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Familia} & \textbf{Técnica} & \textbf{n} & \textbf{$M_{stereo}$} & \textbf{$M_{GP}$} & \textbf{$\delta_{LMS}$} & \textbf{$\Delta$rg gen} & \textbf{$\Delta$rg prof} \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['fam']} & {r['tech']} & {r['n']} & "
            f"{fmt_num(r['M_stereo_gp'], 2)} & {fmt_num(r['M_gp'], 2)} & "
            f"{fmt_num(r['delta_lms'], 2, plus=True)} & "
            f"{fmt_num(r['rggap_gen'], 3, plus=True)} & "
            f"{fmt_num(r['rggap_prof'], 3, plus=True)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "tab_ex_familia_x_tecnica.tex").write_text(
        "\n".join(lines), encoding="utf-8")


def write_subgrupos_profesion_top(subgroups):
    sub = subgroups[subgroups["domain"] == "profession"].copy()
    agg = sub.groupby("subgroup")["delta_regard"].agg(["mean", "std", "count"]).reset_index()
    agg = agg.sort_values("mean", ascending=False)
    top5 = agg.head(5)
    bot5 = agg.tail(5).iloc[::-1]
    lines = [
        r"\begin{table}[H]",
        r"\centering\footnotesize",
        r"\caption{BOLD profesión: mayor y menor $\overline{\Delta\mathrm{regard}}$ por ocupación (30 configs).}",
        r"\label{tab:ex_subgrupos_profesion_top}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"\textbf{Ocupación} & \textbf{$\overline{\Delta\mathrm{regard}}$} & \textbf{DE} & \textbf{n} \\",
        r"\midrule",
        r"\multicolumn{4}{@{}l}{\textit{Top-5 (más mejora regard)}} \\",
        r"\midrule",
    ]
    for _, r in top5.iterrows():
        lines.append(
            f"{r['subgroup'].replace('_', ' ')} & "
            f"{fmt_num(r['mean'], 3, plus=True)} & {fmt_num(r['std'], 3)} & {int(r['count'])} \\\\"
        )
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{@{}l}{\textit{Bottom-5 (más reduce regard)}} \\")
    lines.append(r"\midrule")
    for _, r in bot5.iterrows():
        lines.append(
            f"{r['subgroup'].replace('_', ' ')} & "
            f"{fmt_num(r['mean'], 3, plus=True)} & {fmt_num(r['std'], 3)} & {int(r['count'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "tab_ex_subgrupos_profesion_top.tex").write_text(
        "\n".join(lines), encoding="utf-8")


def write_decision_recomendacion(sset, boot_per_config, stat_df):
    """Recomendacion final: mejor MEI con M_GP significativo y por familia."""
    merged = sset.merge(
        boot_per_config[["model_short", "technique", "lora_trainable_pct",
                         "M_gp_ic_lo", "M_gp_ic_hi"]],
        on=["model_short", "technique", "lora_trainable_pct"], how="left"
    )
    rows = []
    rows.append((r"Mejor $M_{GP}$ global", merged.loc[merged["M_gp"].idxmax()]))
    rows.append((r"Mejor $\mathrm{MEI}_{GP}$ global", merged.loc[merged["MEI_gp"].idxmax()]))
    for fam in sorted(merged["family"].unique()):
        sub = merged[merged["family"] == fam]
        rows.append((rf"Mejor $\mathrm{{MEI}}_{{GP}}$ en {FAM_LABEL[fam]}",
                     sub.loc[sub["MEI_gp"].idxmax()]))
    for tech in ("cda", "debias"):
        sub = merged[merged["technique"] == tech]
        rows.append((rf"Mejor $M_{{GP}}$ con {TECH_LABEL[tech]}",
                     sub.loc[sub["M_gp"].idxmax()]))
    lines = [
        r"\begin{table}[H]",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Configuraciones recomendadas según criterio.}",
        r"\label{tab:ex_decision_recomendacion}",
        r"\begin{tabularx}{\linewidth}{@{} L Y r r r r @{}}",
        r"\toprule",
        r"\textbf{Criterio} & \textbf{Configuración} & \textbf{$M_{GP}$} & \textbf{IC95\%} & \textbf{$\delta_{LMS}$} & \textbf{MEI$_{GP}$} \\",
        r"\midrule",
    ]
    for label, r in rows:
        lines.append(
            f"{label} & {fmt_config_tex(r)} & {fmt_num(r['M_gp'], 2)} & "
            f"[{fmt_num(r['M_gp_ic_lo'], 2)}; {fmt_num(r['M_gp_ic_hi'], 2)}] & "
            f"{fmt_num(r['delta_lms_avg'], 2, plus=True)} & "
            f"{fmt_num(r['MEI_gp'], 2)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    (TAB_DIR / "tab_ex_decision_recomendacion.tex").write_text(
        "\n".join(lines), encoding="utf-8")

# =============================================================================
# Figuras (ex_*)
# =============================================================================

def fig_baseline_panorama(sset, bold):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    items = [
        (axes[0, 0], "StereoSet inter (SS, ideal=50)", [
            ("Género", "sset_inter_blm_en_bbt_gender_ss", C_GENDER),
            ("Profesión", "sset_inter_blm_en_bbt_profession_ss", C_PROFESSION),
        ], sset),
        (axes[0, 1], "StereoSet intra (SS, ideal=50)", [
            ("Género", "sset_intra_blm_en_bbt_gender_ss", C_GENDER),
            ("Profesión", "sset_intra_blm_en_bbt_profession_ss", C_PROFESSION),
        ], sset),
        (axes[1, 0], "CrowS-Pairs (STS, ideal=50)", [
            ("Género", "csp_blm_en_bbt_gender_sts", C_GENDER),
            ("Socio (repr. prof.)", "csp_blm_en_bbt_socioeconomic_sts", C_SOCIO),
        ], sset),
        (axes[1, 1], "BOLD (regard medio, ideal=0)", [
            ("Género", "bold_blm_gender_avg_regard", C_GENDER),
            ("Profesión", "bold_blm_profession_avg_regard", C_PROFESSION),
        ], bold),
    ]
    for ax, title, specs, src in items:
        x = np.arange(len(MODELS))
        w = 0.36
        all_v = []
        bars_all = []
        for i, (label, col, color) in enumerate(specs):
            vals = [src[src["model_short"] == m][col].iloc[0]
                    if not src[src["model_short"] == m].empty else np.nan
                    for m in MODELS]
            offset = (i - 0.5) * w
            b = ax.bar(x + offset, vals, w, color=color, edgecolor=C_EDGE,
                       label=label)
            bars_all.append(b)
            all_v += vals
        is_bold = "ideal=0" in title
        if is_bold:
            flat = [v for v in all_v if v is not None and not (
                isinstance(v, float) and np.isnan(v))]
            vmax = max(flat) if flat else 0.5
            ax.set_ylim(0, vmax + max(0.08, vmax * 0.22))
        else:
            fix_ylim(ax, all_v, label_pad=2.5)
        for b in bars_all:
            add_bar_labels(ax, b, fmt="{:.2f}", fontsize=FS_VALUE_DENSE)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=22, ha="right", fontsize=FS_TICK)
        ax.tick_params(axis="x", labelbottom=True, pad=3)
        if "ideal=50" in title:
            ax.axhline(50, color=C_SPINE, linestyle="--", linewidth=0.7, zorder=2)
        elif is_bold:
            ax.spines["bottom"].set_visible(False)
            ax.axhline(0, color=C_SPINE, linestyle="--", linewidth=0.9, zorder=2)
        style_axis(ax, title=title)
    fig.suptitle("Panorama del sesgo previo a mitigar (modelos baseline)",
                 fontsize=FS_SUPTITLE, fontweight="bold", y=0.98)
    legend_below_fig(fig, category_legend_handles(), ncol=3, y=0.01)
    finish_fig(fig, bottom=0.20, top=0.88, left=0.07, right=0.98, hspace=0.52)
    save_fig(fig, "ex01_baseline_panorama_gp")


def _heatmap(ax, mat, row_labels, col_labels, vmin, vmax, cmap, title,
             fmt="{:+.1f}", sig_mask=None):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=FS_TICK - 1)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=FS_TICK - 1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if pd.isna(v):
                continue
            color = "white" if abs(v) > (vmax - vmin) * 0.55 else C_TEXT
            txt = fmt.format(v)
            if sig_mask is not None and sig_mask[i, j]:
                txt = txt + r"$^{*}$"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=FS_VALUE_DENSE - 1, color=color, fontweight="bold")
    ax.set_title(title, pad=8, loc="left")
    return im


def fig_heatmap_deltaSS(sset, task, fname):
    sub = sset.copy()
    sub["cfg"] = sub.apply(fmt_config_short, axis=1)
    sub = sub.sort_values(["family", "model_short", "lora_trainable_pct", "technique"])
    cols = [f"delta_ss_{task}_gender", f"delta_ss_{task}_profession",
            "delta_csp_gender", "delta_csp_socioeconomic"]
    col_labels = ["ΔSS gen", "ΔSS prof", "ΔSTS gen", "ΔSTS soc/prof"]
    mat = sub[cols].values
    fig, ax = plt.subplots(figsize=(8, 11))
    vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
    _heatmap(ax, mat, sub["cfg"].tolist(), col_labels,
             -vmax, vmax, "RdBu_r",
             f"$\\Delta$ por categoria - StereoSet {task}sentence + CrowS-Pairs",
             fmt="{:+.1f}")
    fig.colorbar(ax.get_images()[0], ax=ax, fraction=0.04, pad=0.02,
                 label="p.p. (positivo = mejora)")
    finish_fig(fig, bottom=0.05, top=0.95, left=0.30, right=0.92)
    save_fig(fig, fname)


def fig_forest_M_GP(boot_per_config):
    df = boot_per_config.copy()
    df["cfg"] = df.apply(fmt_config_short, axis=1)
    df = df.sort_values("M_gp")
    fig, ax = plt.subplots(figsize=(11, 12))
    y = np.arange(len(df))
    colors = [C_TECH_DEBIAS if t == "debias" else C_TECH_CDA for t in df["technique"]]
    ax.errorbar(df["M_gp"], y,
                xerr=[df["M_gp"] - df["M_gp_ic_lo"], df["M_gp_ic_hi"] - df["M_gp"]],
                fmt="none", ecolor=C_SPINE, elinewidth=1.0, capsize=2.5, alpha=0.8)
    ax.scatter(df["M_gp"], y, c=colors, s=70, edgecolors=C_SPINE,
               linewidths=0.6, zorder=3)
    for yi, val in zip(y, df["M_gp"]):
        ax.annotate(f"{val:.2f}", (val, yi), xytext=(8, 0),
                    textcoords="offset points", va="center",
                    fontsize=FS_VALUE_DENSE, fontweight="bold",
                    color=C_TEXT, bbox=VALUE_BBOX)
    ax.axvline(0, color=C_SPINE, linestyle="--", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(df["cfg"], fontsize=FS_TICK - 1)
    style_axis(ax, xlabel="$M_{GP}$ (p.p.) con IC95\\% bootstrap",
               title="Forest plot: mitigacion compuesta por configuracion")
    legend_below_fig(fig, tech_legend_handles(), ncol=2, y=0.01)
    finish_fig(fig, bottom=0.12, top=0.96, left=0.27, right=0.97)
    save_fig(fig, "ex04_forest_M_GP_IC")


def fig_box_familia_tecnica(sset):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    metrics = [
        (axes[0], "Género", ["delta_ss_inter_gender", "delta_ss_intra_gender", "delta_csp_gender"]),
        (axes[1], "Profesión", ["delta_ss_inter_profession", "delta_ss_intra_profession", "delta_csp_socioeconomic"]),
    ]
    pos = 0
    cells = []
    for fam in ("gpt2", "llama"):
        for tech in ("cda", "debias"):
            cells.append((fam, tech, FAM_LABEL[fam], TECH_LABEL[tech]))
    for ax, title, cols in metrics:
        data = []
        labels = []
        positions = []
        cell_colors = []
        for i, (fam, tech, flab, tlab) in enumerate(cells):
            sub = sset[(sset["family"] == fam) & (sset["technique"] == tech)]
            vals = sub[cols].values.flatten()
            data.append(vals[~np.isnan(vals)])
            labels.append(f"{flab}\n{tlab}")
            positions.append(i)
            cell_colors.append(C_TECH_DEBIAS if tech == "debias" else C_TECH_CDA)
        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                        showmeans=True, meanprops=dict(marker="o", markerfacecolor="white",
                                                       markeredgecolor=C_TEXT, markersize=7))
        for patch, c in zip(bp["boxes"], cell_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
            patch.set_edgecolor(C_SPINE)
        for med in bp["medians"]:
            med.set_color(C_TEXT)
            med.set_linewidth(1.2)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=FS_TICK)
        ax.axhline(0, color=C_SPINE, linestyle="--", linewidth=0.7)
        style_axis(ax, ylabel="$\\Delta$ (p.p.)" if ax is axes[0] else "",
                   title=title)
    fig.suptitle("Distribucion de mejoras por familia $\\times$ tecnica (3 metricas SS/STS)",
                 fontsize=FS_SUPTITLE, fontweight="bold", y=1.0)
    finish_fig(fig, bottom=0.18, top=0.90, left=0.07, right=0.98)
    save_fig(fig, "ex05_box_familia_tecnica")


def fig_evolucion_lora(sset):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fams = [("gpt2", "GPT-2"), ("llama", "Llama")]
    cats = [
        ("delta_ss_inter_gender", "ΔSS inter género", C_GENDER),
        ("delta_ss_inter_profession", "ΔSS inter profesión", C_PROFESSION),
        ("delta_ss_intra_gender", "ΔSS intra género", C_GENDER),
        ("delta_ss_intra_profession", "ΔSS intra profesión", C_PROFESSION),
    ]
    for ax, (col, label, color) in zip(axes.flatten(), cats):
        for (fam, flab), ls, marker in zip(fams, ["-", "--"], ["o", "s"]):
            sub = sset[sset["family"] == fam]
            grp = sub.groupby("lora_trainable_pct")[col]
            mean = grp.mean()
            sem = grp.std(ddof=1) / np.sqrt(grp.count())
            ax.plot(mean.index, mean.values, ls, color=color, marker=marker,
                    lw=2.0, markersize=8, label=flab)
            ax.fill_between(mean.index, mean - 1.96 * sem, mean + 1.96 * sem,
                            color=color, alpha=0.12)
            for x_, y_ in zip(mean.index, mean.values):
                ax.annotate(f"{y_:.1f}", (x_, y_), textcoords="offset points",
                            xytext=(0, 12), ha="center",
                            fontsize=FS_VALUE_DENSE, fontweight="bold",
                            color=C_TEXT, bbox=VALUE_BBOX)
        ax.axhline(0, color=C_SPINE, linestyle="--", linewidth=0.6)
        ax.set_xlim(0.7, 3.3)
        ax.set_xticks([1, 2, 3])
        style_axis(ax, ylabel="p.p." if ax in (axes[0, 0], axes[1, 0]) else "",
                   xlabel="% LoRA" if ax in (axes[1, 0], axes[1, 1]) else "",
                   title=label)
    fig.suptitle("Sensibilidad de StereoSet al \\% LoRA por familia (con IC normal aprox.)",
                 fontsize=FS_SUPTITLE, fontweight="bold", y=1.0)
    legend_below_fig(fig, familia_line_legend_handles(), ncol=2, y=0.01)
    finish_fig(fig, bottom=0.14, top=0.88, left=0.07, right=0.98)
    save_fig(fig, "ex06_evolucion_lora")


def fig_pareto(sset):
    fig, ax = plt.subplots(figsize=(11, 7.5))
    for _, row in sset.iterrows():
        color = C_FAM_GPT2 if row["family"] == "gpt2" else C_FAM_LLAMA
        marker = "D" if row["technique"] == "debias" else "o"
        size = 60 + row["lora_trainable_pct"] * 35
        ax.scatter(row["M_gp"], row["delta_lms_avg"], c=color, marker=marker,
                   s=size, alpha=0.78, edgecolors=C_SPINE, linewidth=0.6)
    # Frontera de Pareto: alto M_GP, bajo |delta_lms|
    cand = sset.sort_values("M_gp", ascending=False)
    pareto = []
    best_lms = float("inf")
    for _, r in cand.iterrows():
        v = abs(r["delta_lms_avg"])
        if v < best_lms:
            best_lms = v
            pareto.append(r)
    pareto_df = pd.DataFrame(pareto).sort_values("M_gp")
    ax.plot(pareto_df["M_gp"], pareto_df["delta_lms_avg"],
            "-", color=C_POS, lw=1.5, alpha=0.8, label="Frontera de Pareto")
    for _, r in pareto_df.iterrows():
        ax.annotate(fmt_config_short(r), (r["M_gp"], r["delta_lms_avg"]),
                    xytext=(8, 5), textcoords="offset points",
                    fontsize=FS_VALUE_DENSE, color=C_TEXT,
                    bbox=VALUE_BBOX)
    ax.axhline(0, color=C_GRID, linewidth=0.6)
    ax.axvline(0, color=C_GRID, linewidth=0.6)
    style_axis(ax, ylabel="$\\delta_{LMS}$ (p.p., positivo = perdida)",
               xlabel="$M_{GP}$ (p.p., positivo = mejora)",
               title="Trade-off bias-utility (género+profesión)")
    legend_below_fig(fig, [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FAM_GPT2,
               markersize=9, markeredgecolor=C_SPINE, label="GPT-2"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FAM_LLAMA,
               markersize=9, markeredgecolor=C_SPINE, label="Llama"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#718096",
               markersize=8, label="Dataset"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#718096",
               markersize=8, label="Dataset + prompt"),
        Line2D([0], [0], color=C_POS, lw=1.5, label="Frontera de Pareto"),
    ], ncol=5, y=0.01)
    finish_fig(fig, bottom=0.20, top=0.92, left=0.08, right=0.97)
    save_fig(fig, "ex07_pareto_MGP_dLMS")

def fig_subgrupos_profesion_heatmap(subgroups):
    sub = subgroups[subgroups["domain"] == "profession"].copy()
    sub["cfg"] = sub.apply(fmt_config_short, axis=1)
    pivot = sub.pivot_table(index="cfg", columns="subgroup",
                            values="delta_regard", aggfunc="mean")
    family_order_keys = []
    for cfg in pivot.index:
        if "GPT-2" in cfg:
            family_order_keys.append((0, cfg))
        else:
            family_order_keys.append((1, cfg))
    pivot = pivot.loc[[c for _, c in sorted(family_order_keys)]]
    occ_means = pivot.mean(axis=0).sort_values(ascending=False)
    pivot = pivot[occ_means.index]
    fig, ax = plt.subplots(figsize=(13, 12))
    vmax = max(abs(np.nanmin(pivot.values)), abs(np.nanmax(pivot.values)))
    _heatmap(ax, pivot.values,
             [c for c in pivot.index],
             [c.replace("_", " ") for c in pivot.columns],
             -vmax, vmax, "RdBu_r",
             "BOLD profesión: $\\Delta$regard por (configuración $\\times$ ocupación)",
             fmt="{:+.2f}")
    fig.colorbar(ax.get_images()[0], ax=ax, fraction=0.025, pad=0.02,
                 label="$\\Delta$regard (LoRA - baseline)")
    finish_fig(fig, bottom=0.20, top=0.96, left=0.22, right=0.95)
    save_fig(fig, "ex08_subgrupos_profesion_heatmap")


def fig_brecha_regard_gp(bold):
    df = bold.copy()
    df["cfg"] = df.apply(fmt_config_short, axis=1)
    df = df.sort_values(["family", "model_short", "lora_trainable_pct", "technique"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 11), sharey=True)
    for ax, dom, color in zip(axes, BIAS_TYPES, [C_GENDER, C_PROFESSION]):
        y = np.arange(len(df))
        vals = df[f"d_rggap_{dom}"].values
        bars = ax.barh(y, vals, color=color, alpha=0.85, edgecolor=C_EDGE)
        for bar, v in zip(bars, vals):
            w = bar.get_width()
            xtext = w + (0.005 if w >= 0 else -0.005)
            ha = "left" if w >= 0 else "right"
            ax.annotate(f"{v:+.3f}", (xtext, bar.get_y() + bar.get_height() / 2),
                        ha=ha, va="center", fontsize=FS_VALUE_DENSE,
                        fontweight="bold", color=C_TEXT, bbox=VALUE_BBOX)
        ax.axvline(0, color=C_SPINE, linewidth=0.7, linestyle="--")
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels(df["cfg"], fontsize=FS_TICK - 1)
        else:
            ax.set_yticklabels([])
        style_axis(ax, xlabel="$\\Delta$rg\\_gap = baseline $-$ LoRA",
                   title=f"BOLD {BIAS_LABEL[dom]}")
    fig.suptitle("Reducción de brecha de regard por configuración (mayor = más equitativo)",
                 fontsize=FS_SUPTITLE, fontweight="bold", y=1.0)
    finish_fig(fig, bottom=0.06, top=0.94, left=0.22, right=0.97)
    save_fig(fig, "ex09_brecha_regard_gp")


def fig_correlacion_benchmarks(rho_df, p_df):
    fig, ax = plt.subplots(figsize=(11, 9))
    n = len(rho_df)
    sig_mask = (p_df.values < 0.05) & (~np.eye(n, dtype=bool))
    _heatmap(ax, rho_df.values, list(rho_df.index), list(rho_df.columns),
             -1, 1, "RdBu_r",
             "Correlacion de Spearman entre metricas (* p<0,05)",
             fmt="{:+.2f}", sig_mask=sig_mask)
    fig.colorbar(ax.get_images()[0], ax=ax, fraction=0.04, pad=0.02,
                 label="$\\rho$")
    finish_fig(fig, bottom=0.18, top=0.94, left=0.22, right=0.92)
    save_fig(fig, "ex10_correlacion_benchmarks")


def fig_descomposicion_anova(anova_df):
    df = anova_df[anova_df["factor"] != "Total"].copy()
    if "Residual" in df["factor"].values:
        df = df[df["factor"] != "Residual"]
    df = df.sort_values("eta2", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(df["factor"], df["eta2"] * 100, color=C_TECH_DEBIAS,
                   edgecolor=C_EDGE, alpha=0.85)
    for bar, v, p in zip(bars, df["eta2"] * 100, df["p"]):
        sig = "*" if (pd.notna(p) and p < 0.05) else ""
        ax.annotate(f"{v:.1f}%{sig}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=FS_VALUE, fontweight="bold",
                    color=C_TEXT, bbox=VALUE_BBOX)
    style_axis(ax, xlabel="$\\eta^2$ (\\%) -- proporcion de varianza de $M_{GP}$",
               title="Descomposicion de varianza por factor (* significativo p<0,05)")
    finish_fig(fig, bottom=0.10, top=0.92, left=0.30, right=0.95)
    save_fig(fig, "ex11_descomposicion_anova")


def fig_distribucion_distancia_ideal(sset):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    pairs = [
        (axes[0, 0], "StereoSet inter -- género",
         "dist_ss_inter_gender_blm", "dist_ss_inter_gender_lrm"),
        (axes[0, 1], "StereoSet inter -- profesión",
         "dist_ss_inter_profession_blm", "dist_ss_inter_profession_lrm"),
        (axes[1, 0], "StereoSet intra -- género",
         "dist_ss_intra_gender_blm", "dist_ss_intra_gender_lrm"),
        (axes[1, 1], "StereoSet intra -- profesión",
         "dist_ss_intra_profession_blm", "dist_ss_intra_profession_lrm"),
    ]
    for ax, title, c_b, c_l in pairs:
        b = sset[c_b].dropna().values
        l = sset[c_l].dropna().values
        bins = np.linspace(0, max(b.max(), l.max()) + 2, 14)
        ax.hist(b, bins=bins, alpha=0.55, color=C_TECH_CDA,
                edgecolor=C_SPINE, label="Baseline", density=False)
        ax.hist(l, bins=bins, alpha=0.55, color=C_TECH_DEBIAS,
                edgecolor=C_SPINE, label="LoRA", density=False)
        ax.axvline(b.mean(), color=C_TECH_CDA, linestyle="--", linewidth=1.4)
        ax.axvline(l.mean(), color=C_TECH_DEBIAS, linestyle="-", linewidth=1.4)
        style_axis(ax, ylabel="N configs",
                   xlabel="|SS - 50| (p.p.)", title=title)
    fig.suptitle("Distribucion de la distancia al ideal SS=50 antes y despues de LoRA",
                 fontsize=FS_SUPTITLE, fontweight="bold", y=1.0)
    legend_below_fig(fig, baseline_lora_legend_handles(), ncol=2, y=0.01)
    finish_fig(fig, bottom=0.14, top=0.88, left=0.07, right=0.97)
    save_fig(fig, "ex12_distribucion_distancia_ideal")


def fig_ranking_MEI_top10(boot_per_config):
    df = boot_per_config.sort_values("MEI_gp", ascending=False).head(10).iloc[::-1]
    df["cfg"] = df.apply(fmt_config_short, axis=1)
    fig, ax = plt.subplots(figsize=(11, 7.5))
    y = np.arange(len(df))
    colors = [C_TECH_DEBIAS if t == "debias" else C_TECH_CDA for t in df["technique"]]
    bars = ax.barh(y, df["MEI_gp"], color=colors, alpha=0.88, edgecolor=C_EDGE,
                   height=0.7)
    for bar, val, lo, hi in zip(bars, df["MEI_gp"], df["M_gp_ic_lo"], df["M_gp_ic_hi"]):
        w = bar.get_width()
        ax.annotate(f"MEI={val:.2f} | M_GP IC=[{lo:.1f}; {hi:.1f}]",
                    (w, bar.get_y() + bar.get_height() / 2),
                    xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=FS_VALUE_DENSE, fontweight="bold",
                    color=C_TEXT, bbox=VALUE_BBOX)
    ax.set_yticks(y)
    ax.set_yticklabels(df["cfg"], fontsize=FS_TICK)
    ax.axvline(1.0, color=C_SPINE, linestyle="--", linewidth=1)
    style_axis(ax, xlabel="MEI$_{GP}$",
               title="Top-10 por MEI$_{GP}$ con IC95\\% bootstrap del $M_{GP}$")
    legend_below_fig(fig, tech_legend_handles(), ncol=2, y=0.01)
    finish_fig(fig, bottom=0.20, top=0.92, left=0.27, right=0.95)
    save_fig(fig, "ex13_ranking_MEI_top10")


def fig_panel_global_5modelos(sset, bold):
    fig, axes = plt.subplots(5, 4, figsize=(18, 16))
    titles = ["StereoSet inter", "StereoSet intra", "CrowS-Pairs", "BOLD"]
    for i, m in enumerate(MODELS):
        sub_s = sset[sset["model_short"] == m].sort_values(["lora_trainable_pct", "technique"])
        sub_b = bold[bold["model_short"] == m].sort_values(["lora_trainable_pct", "technique"])
        if sub_s.empty or sub_b.empty:
            continue
        x = np.arange(6)
        labels = [f"{int(r.lora_trainable_pct)}%\n{TECH_SHORT[r.technique]}" for r in sub_s.itertuples()]
        for j, title in enumerate(titles):
            ax = axes[i, j]
            if title == "StereoSet inter":
                vg = sub_s["delta_ss_inter_gender"].values
                vp = sub_s["delta_ss_inter_profession"].values
                bg = ax.bar(x - 0.18, vg, 0.36, color=C_GENDER, edgecolor=C_EDGE)
                bp = ax.bar(x + 0.18, vp, 0.36, color=C_PROFESSION, edgecolor=C_EDGE)
                add_bar_labels(ax, bg, fmt="{:+.1f}", fontsize=FS_VALUE_DENSE)
                add_bar_labels(ax, bp, fmt="{:+.1f}", fontsize=FS_VALUE_DENSE)
            elif title == "StereoSet intra":
                vg = sub_s["delta_ss_intra_gender"].values
                vp = sub_s["delta_ss_intra_profession"].values
                bg = ax.bar(x - 0.18, vg, 0.36, color=C_GENDER, edgecolor=C_EDGE)
                bp = ax.bar(x + 0.18, vp, 0.36, color=C_PROFESSION, edgecolor=C_EDGE)
                add_bar_labels(ax, bg, fmt="{:+.1f}", fontsize=FS_VALUE_DENSE)
                add_bar_labels(ax, bp, fmt="{:+.1f}", fontsize=FS_VALUE_DENSE)
            elif title == "CrowS-Pairs":
                vg = sub_s["delta_csp_gender"].values
                vs = sub_s["delta_csp_socioeconomic"].values
                bg = ax.bar(x - 0.18, vg, 0.36, color=C_GENDER, edgecolor=C_EDGE)
                bs = ax.bar(x + 0.18, vs, 0.36, color=C_SOCIO, edgecolor=C_EDGE)
                add_bar_labels(ax, bg, fmt="{:+.1f}", fontsize=FS_VALUE_DENSE)
                add_bar_labels(ax, bs, fmt="{:+.1f}", fontsize=FS_VALUE_DENSE)
            else:
                vg = sub_b["d_rggap_gender"].values
                vp = sub_b["d_rggap_profession"].values
                bg = ax.bar(x - 0.18, vg, 0.36, color=C_GENDER, edgecolor=C_EDGE)
                bp = ax.bar(x + 0.18, vp, 0.36, color=C_PROFESSION, edgecolor=C_EDGE)
                add_bar_labels(ax, bg, fmt="{:+.2f}", fontsize=FS_VALUE_DENSE)
                add_bar_labels(ax, bp, fmt="{:+.2f}", fontsize=FS_VALUE_DENSE)
            ax.axhline(0, color=C_SPINE, linestyle="--", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=FS_TICK - 1)
            patch_v = [p.get_height() for p in ax.patches]
            fix_ylim(ax, patch_v if patch_v else [0], label_pad=1.2)
            if i == 0:
                ax.set_title(title, fontsize=FS_TITLE - 1, pad=6)
            if j == 0:
                ax.set_ylabel(m, fontsize=FS_TITLE - 1, fontweight="bold")
    fig.suptitle("Panel global: GP por modelo y benchmark",
                 fontsize=FS_SUPTITLE, fontweight="bold", y=1.0)
    handles = [
        Patch(facecolor=C_GENDER, label="Género"),
        Patch(facecolor=C_PROFESSION, label="Profesión"),
        Patch(facecolor=C_SOCIO, label="Socio (repr. prof.)"),
    ]
    legend_below_fig(fig, handles, ncol=3, y=0.005)
    finish_fig(fig, bottom=0.10, top=0.95, left=0.06, right=0.99)
    save_fig(fig, "ex14_panel_global_5modelos")


def fig_diff_in_diff_tecnica(sset, bold):
    """Ventaja media de GPT-2 sobre Llama por componente de M_GP (dataset + prompt)."""
    metrics = [
        ("delta_ss_inter_gender", "SS inter\ngénero", C_GENDER),
        ("delta_ss_inter_profession", "SS inter\nprofesión", C_PROFESSION),
        ("delta_ss_intra_gender", "SS intra\ngénero", C_GENDER),
        ("delta_ss_intra_profession", "SS intra\nprofesión", C_PROFESSION),
        ("delta_csp_gender", "STS\ngénero", C_GENDER),
        ("delta_csp_socioeconomic", "STS\nsocio", C_SOCIO),
    ]
    diffs, colors, labels = [], [], []
    for col, label, color in metrics:
        g = sset[(sset["family"] == "gpt2") & (sset["technique"] == "debias")][col].mean()
        l = sset[(sset["family"] == "llama") & (sset["technique"] == "debias")][col].mean()
        diffs.append(float(g - l))
        colors.append(color)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(11.5, 7))
    x = np.arange(len(metrics))
    label_off = 0.30
    bars = ax.bar(x, diffs, 0.58, color=colors, edgecolor=C_EDGE, zorder=3)
    vmin, vmax = min(diffs), max(diffs)
    y_pad_top = max(0.55, vmax * 0.12)
    y_pad_bot = max(0.65, abs(vmin) * 0.18) + label_off
    ax.set_ylim(vmin - y_pad_bot, vmax + y_pad_top)
    add_bar_labels(ax, bars, fmt="{:+.1f}", fontsize=FS_VALUE_DENSE, offset=label_off)
    ax.axhline(0, color=C_SPINE, linestyle="--", linewidth=0.7, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    style_axis(
        ax,
        ylabel="GPT-2 $-$ Llama (p.p.)",
        title="Ventaja de GPT-2 sobre Llama por componente ($M_{GP}$, dataset + prompt)",
    )
    handles = [
        Patch(facecolor=C_GENDER, edgecolor=C_EDGE, label="Género"),
        Patch(facecolor=C_PROFESSION, edgecolor=C_EDGE, label="Profesión"),
        Patch(facecolor=C_SOCIO, edgecolor=C_EDGE, label="Socio (repr. prof.)"),
    ]
    legend_below_fig(fig, handles, ncol=3, y=0.01)
    finish_fig(fig, bottom=0.24, top=0.88, left=0.09, right=0.98)
    save_fig(fig, "ex15_diff_in_diff_tecnica")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 64)
    print("ANALISIS EXHAUSTIVO: GENERO + PROFESION (mitigacion LoRA)")
    print("=" * 64)
    sset, bold = load_and_compute()
    subgroups = build_bold_subgroups(bold)
    print(f"[1] Cargados: {len(sset)} configs StereoSet/CSP, "
          f"{len(bold)} configs BOLD, "
          f"{len(subgroups)} filas de subgrupos.")

    stat_df = stat_table_full(sset, bold)
    anova_df = anova_decomposition(sset)
    rho_df, p_df = correlation_benchmarks(sset, bold)
    boot_cells = bootstrap_by_cell(sset)
    boot_per_config = bootstrap_per_config(sset)
    print("[2] Estadistica inferencial calculada (Wilcoxon, bootstrap, ANOVA).")

    export_csvs(sset, bold, subgroups, stat_df, anova_df, rho_df, p_df,
                boot_cells, boot_per_config)
    print(f"[3] CSVs exportados a {DAT_DIR}")

    write_panorama_baseline(sset, bold)
    write_resumen_categoria(sset, bold)
    write_estadistica_inferencial(stat_df)
    write_factores_anova(anova_df)
    write_top10_mei(sset, boot_per_config)
    write_familia_x_tecnica(sset, bold, stat_df, boot_cells)
    write_subgrupos_profesion_top(subgroups)
    write_decision_recomendacion(sset, boot_per_config, stat_df)
    print(f"[4] Tablas LaTeX en {TAB_DIR}")

    fig_baseline_panorama(sset, bold)
    fig_heatmap_deltaSS(sset, "inter", "ex02_heatmap_deltaSS_inter")
    fig_heatmap_deltaSS(sset, "intra", "ex03_heatmap_deltaSS_intra")
    fig_forest_M_GP(boot_per_config)
    fig_box_familia_tecnica(sset)
    fig_evolucion_lora(sset)
    fig_pareto(sset)
    fig_subgrupos_profesion_heatmap(subgroups)
    fig_brecha_regard_gp(bold)
    fig_correlacion_benchmarks(rho_df, p_df)
    fig_descomposicion_anova(anova_df)
    fig_distribucion_distancia_ideal(sset)
    fig_ranking_MEI_top10(boot_per_config)
    fig_panel_global_5modelos(sset, bold)
    fig_diff_in_diff_tecnica(sset, bold)
    print(f"[5] Figuras 300 DPI en {FIG_DIR}")

    # Resumen ejecutivo en JSON
    best_m = sset.loc[sset["M_gp"].idxmax()]
    best_mei = sset.loc[sset["MEI_gp"].idxmax()]
    summary = {
        "n_configs": int(len(sset)),
        "M_gp_mean": float(sset["M_gp"].mean()),
        "M_gp_max_config": fmt_config_short(best_m),
        "M_gp_max_value": float(best_m["M_gp"]),
        "MEI_gp_max_config": fmt_config_short(best_mei),
        "MEI_gp_max_value": float(best_mei["MEI_gp"]),
        "M_stereo_gp_mean": float(sset["M_stereo_gp"].mean()),
        "delta_lms_avg_mean": float(sset["delta_lms_avg"].mean()),
        "anova_top_factor": str(anova_df.sort_values("eta2", ascending=False)["factor"].iloc[0]),
        "wilcoxon_global": [
            {"metrica": r["metrica"], "p": float(r["wilcoxon_p"]) if pd.notna(r["wilcoxon_p"]) else None,
             "n_mejoran": int(r["configs_mejoran"]), "n": int(r["n"])}
            for _, r in stat_df[stat_df["grupo"] == "Global"].iterrows()
        ],
    }
    (DAT_DIR / "resumen_ejecutivo.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[6] Resumen ejecutivo: {DAT_DIR / 'resumen_ejecutivo.json'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
