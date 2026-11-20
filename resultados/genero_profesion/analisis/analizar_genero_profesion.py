"""
Analisis focalizado en sesgo de GENERO y PROFESION (objetivos iniciales de la tesis).

Uso:
    python resultados/genero_profesion/analisis/analizar_genero_profesion.py

Salidas:
    resultados/genero_profesion/figuras/     -> PNG 300 DPI
    resultados/genero_profesion/analisis/datos/
    resultados/genero_profesion/analisis/tablas/
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
FS_VALUE = 12          # etiquetas numericas sobre barras / puntos
FS_VALUE_DENSE = 10    # paneles con muchas barras o heatmaps
FS_TICK = 12           # nombres de modelos, ticks de ejes
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
    "savefig.pad_inches": 0.08,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

VALUE_BBOX = dict(
    boxstyle="round,pad=0.28", facecolor="white", alpha=0.95,
    edgecolor="#CBD5E0", linewidth=0.6,
)

C_TEXT = "#2D3436"
C_SPINE = "#4A5568"
C_GRID = "#E2E8F0"
C_EDGE = "#FFFFFF"
C_GENDER = "#2C5282"
C_PROFESSION = "#A0826D"
C_SOCIO = "#6B5B4F"  # Proxy ocupacion / estatus socioeconomico (CrowS-Pairs)
C_TECH_CDA = "#B8A090"
C_TECH_DEBIAS = "#2C5282"
C_FAM_GPT2 = "#2C5282"
C_FAM_LLAMA = "#744C4E"

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
CSP_CATS = {
    "gender": ("delta_csp_gender", "Género"),
    "socioeconomic": ("delta_csp_socioeconomic", "Socioecon./ocupación"),
}


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
        ax.set_title(title, pad=12, loc="left")
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.7, color=C_GRID)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_SPINE)


def add_bar_labels(ax, bars, fmt="{:.1f}", fontsize=None, offset=None, fontweight="bold"):
    """Etiqueta cada barra (incluye valores pequeños y negativos)."""
    if fontsize is None:
        fontsize = FS_VALUE
    heights = [b.get_height() for b in bars]
    if not heights:
        return
    span = max(heights) - min(heights) if heights else 1.0
    pad = offset if offset is not None else max(0.45, span * 0.10, abs(max(heights, key=abs)) * 0.08)
    for bar in bars:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        y = h + pad if h >= 0 else h - pad
        ax.annotate(
            fmt.format(h), (bar.get_x() + bar.get_width() / 2, y),
            ha="center", va=va, fontsize=fontsize, fontweight=fontweight, color=C_TEXT,
            bbox=VALUE_BBOX, clip_on=False, zorder=5,
        )


def add_line_point_labels(ax, xs, ys, fmt="{:.1f}", fontsize=None, dy=12, offsets=None):
    if fontsize is None:
        fontsize = FS_VALUE
    for i, (x, y) in enumerate(zip(xs, ys)):
        off = offsets[i] if offsets and i < len(offsets) else dy
        ax.annotate(
            fmt.format(y), (x, y), textcoords="offset points", xytext=(0, off),
            ha="center", fontsize=fontsize, fontweight="bold", color=C_TEXT,
            bbox=VALUE_BBOX, clip_on=False, zorder=5,
        )


def tech_legend_handles():
    return [
        Patch(facecolor=C_TECH_CDA, edgecolor=C_EDGE, label="Dataset"),
        Patch(facecolor=C_TECH_DEBIAS, edgecolor=C_EDGE, label="Dataset + prompt"),
    ]


def category_gp_legend_handles():
    return [
        Patch(facecolor=C_GENDER, edgecolor=C_EDGE, label="Género"),
        Patch(facecolor=C_PROFESSION, edgecolor=C_EDGE, label="Profesión"),
    ]


def category_csp_legend_handles():
    return [
        Patch(facecolor=C_GENDER, edgecolor=C_EDGE, label="Género"),
        Patch(facecolor=C_SOCIO, edgecolor=C_EDGE, label="Socioecon./ocupación"),
    ]


def tech_linestyle_legend_handles():
    return [
        Line2D([0], [0], color=C_TEXT, ls="-", lw=2.2, marker="o", markersize=8,
               markeredgecolor=C_SPINE, label="Dataset + prompt"),
        Line2D([0], [0], color=C_TEXT, ls="--", lw=2.2, marker="o", markersize=8,
               markeredgecolor=C_SPINE, label="Dataset"),
    ]


def zero_ref_legend_handle():
    return Line2D([0], [0], color=C_SPINE, ls=":", lw=1.2, label=r"$\Delta=0$")


def legend_below_fig(fig, handles, ncol=2, y=0.01, fontsize=None):
    """Leyenda bajo la figura (nunca loc='best' dentro del eje: tapa barras/valores)."""
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
        ncol=ncol, frameon=True, fontsize=fontsize or FS_LEGEND, columnspacing=1.4,
    )


def finish_fig(fig, bottom=0.28, top=0.90, left=0.07, right=0.98):
    fig.subplots_adjust(bottom=bottom, top=top, left=left, right=right)


def fix_ylim(ax, values, label_pad=1.1):
    flat = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not flat:
        return
    vmin, vmax = min(flat), max(flat)
    extra = max(label_pad, abs(vmax) * 0.28, abs(vmin) * 0.28)
    if vmin < -0.05:
        ax.set_ylim(vmin - max(0.5, abs(vmin) * 0.14), vmax + extra)
        ax.axhline(0, color=C_SPINE, linestyle=":", linewidth=0.6, zorder=1)
    else:
        ax.set_ylim(0, vmax + extra)


def best_row_at_max(sset, col):
    return sset.loc[sset[col].idxmax()]


def fmt_config_tex(row):
    tech = r"Dataset + prompt" if row["technique"] == "debias" else r"Dataset"
    return f"{row['model_short']}, {tech}, {int(row['lora_trainable_pct'])}\\% LoRA"


def fmt_config_plain(row):
    tech = "Dataset + prompt" if row["technique"] == "debias" else "Dataset"
    pct = int(row["lora_trainable_pct"])
    return f"{row['model_short']}, {tech}, {pct}% LoRA"


def load_and_compute():
    df = pd.read_csv(DATA_PATH)
    df["model_short"] = df["model_base"].map(MODEL_SHORT)
    sset = df[df["suite_kind"] == "crowspairs_stereoset"].copy()
    bold = df[df["suite_kind"] == "bold"].copy()

    for bt in BIAS_TYPES:
        sset[f"delta_ss_inter_{bt}"] = (
            sset[f"sset_inter_blm_en_bbt_{bt}_ss"] - sset[f"sset_inter_lrm_en_bbt_{bt}_ss"]
        )
        sset[f"delta_ss_intra_{bt}"] = (
            sset[f"sset_intra_blm_en_bbt_{bt}_ss"] - sset[f"sset_intra_lrm_en_bbt_{bt}_ss"]
        )

    sset["delta_csp_gender"] = (
        sset["csp_blm_en_bbt_gender_sts"] - sset["csp_lrm_en_bbt_gender_sts"]
    )
    sset["delta_csp_socioeconomic"] = (
        sset["csp_blm_en_bbt_socioeconomic_sts"] - sset["csp_lrm_en_bbt_socioeconomic_sts"]
    )

    sset["M_gp"] = (
        sset["delta_ss_inter_gender"] + sset["delta_ss_intra_gender"]
        + sset["delta_ss_inter_profession"] + sset["delta_ss_intra_profession"]
        + sset["delta_csp_gender"] + sset["delta_csp_socioeconomic"]
    ) / 6.0

    sset["M_stereo_gp"] = (
        sset["delta_ss_inter_gender"] + sset["delta_ss_intra_gender"]
        + sset["delta_ss_inter_profession"] + sset["delta_ss_intra_profession"]
    ) / 4.0

    sset["delta_lms_inter"] = sset["sset_inter_blm_en_lms"] - sset["sset_inter_lrm_en_lms"]
    sset["delta_lms_intra"] = sset["sset_intra_blm_en_lms"] - sset["sset_intra_lrm_en_lms"]
    sset["delta_lms_avg"] = (sset["delta_lms_inter"] + sset["delta_lms_intra"]) / 2
    sset["MEI_gp"] = sset["M_gp"] / sset["delta_lms_avg"].abs().clip(lower=0.05)

    for dom in BIAS_TYPES:
        bold[f"d_rggap_{dom}"] = bold[f"bold_blm_{dom}_rg_gap"] - bold[f"bold_lrm_{dom}_rg_gap"]
        bold[f"d_regard_{dom}"] = bold[f"bold_lrm_{dom}_avg_regard"] - bold[f"bold_blm_{dom}_avg_regard"]

    return sset, bold


def export_data(sset, bold):
    cols = [
        "model_short", "family", "technique", "lora_trainable_pct",
        "delta_ss_inter_gender", "delta_ss_inter_profession",
        "delta_ss_intra_gender", "delta_ss_intra_profession",
        "delta_csp_gender", "delta_csp_socioeconomic",
        "M_gp", "M_stereo_gp", "delta_lms_avg", "MEI_gp",
    ]
    sset[cols].to_csv(DAT_DIR / "metricas_genero_profesion.csv", index=False)

    bold_cols = ["model_short", "family", "technique", "lora_trainable_pct"]
    for dom in BIAS_TYPES:
        for p in ("blm", "lrm"):
            for m in ("avg_regard", "rg_gap", "avg_length"):
                c = f"bold_{p}_{dom}_{m}"
                if c in bold.columns:
                    bold_cols.append(c)
        bold_cols.extend([f"d_rggap_{dom}", f"d_regard_{dom}"])
    bold[[c for c in bold_cols if c in bold.columns]].to_csv(
        DAT_DIR / "bold_genero_profesion.csv", index=False
    )

    resumen_rows = []
    for bt in BIAS_TYPES:
        for task, suffix in [("Inter", "inter"), ("Intra", "intra")]:
            col = f"delta_ss_{suffix}_{bt}"
            br = best_row_at_max(sset, col)
            resumen_rows.append({
                "categoria": BIAS_LABEL[bt],
                "benchmark": "StereoSet",
                "tarea": task,
                "media_pp": sset[col].mean(),
                "max_pp": sset[col].max(),
                "config_maximo": fmt_config_plain(br),
            })
    for _key, (col, label) in CSP_CATS.items():
        br = best_row_at_max(sset, col)
        resumen_rows.append({
            "categoria": label,
            "benchmark": "CrowS-Pairs",
            "tarea": "STS",
            "media_pp": sset[col].mean(),
            "max_pp": sset[col].max(),
            "config_maximo": fmt_config_plain(br),
        })
    pd.DataFrame(resumen_rows).to_csv(DAT_DIR / "resumen_categorias.csv", index=False)


def fig_stereoset_by_model(sset, task, col_tpl, fname, ylabel,
                           tick_fontsize=None, legend_fontsize=None):
    tick_fs = tick_fontsize or FS_TICK
    leg_fs = legend_fontsize or FS_LEGEND
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    all_vals = []
    for ax, bt in zip(axes, BIAS_TYPES):
        x = np.arange(len(MODELS))
        w = 0.34
        cda, deb = [], []
        for m in MODELS:
            sub = sset[sset["model_short"] == m]
            col = col_tpl.format(bt)
            cda.append(sub[sub["technique"] == "cda"][col].mean())
            deb.append(sub[sub["technique"] == "debias"][col].mean())
        b1 = ax.bar(x - w / 2, cda, w, color=C_TECH_CDA, edgecolor=C_EDGE)
        b2 = ax.bar(x + w / 2, deb, w, color=C_TECH_DEBIAS, edgecolor=C_EDGE)
        all_vals.extend(cda + deb)
        add_bar_labels(ax, b1)
        add_bar_labels(ax, b2)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=28, ha="right", fontsize=tick_fs)
        style_axis(ax, ylabel=ylabel if ax is axes[0] else "",
                   title=f"{BIAS_LABEL[bt]} — {task}")
        ax.axhline(0, color=C_SPINE, linewidth=0.4)
    for ax in axes:
        fix_ylim(ax, all_vals, label_pad=3.0)
    legend_below_fig(fig, tech_legend_handles(), ncol=2, y=0.01, fontsize=leg_fs)
    finish_fig(fig, bottom=0.38 if tick_fs > FS_TICK else 0.36, top=0.90)
    save_fig(fig, fname)


def fig_crowspairs_gender_socio(sset):
    """CrowS-Pairs: genero directo + socioeconomico como proxy de ocupacion."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    all_vals = []
    for ax, (cat, (col, title)) in zip(axes, CSP_CATS.items()):
        x = np.arange(len(MODELS))
        w = 0.34
        cda, deb = [], []
        for m in MODELS:
            sub = sset[sset["model_short"] == m]
            cda.append(sub[sub["technique"] == "cda"][col].mean())
            deb.append(sub[sub["technique"] == "debias"][col].mean())
        b1 = ax.bar(x - w / 2, cda, w, color=C_TECH_CDA, edgecolor=C_EDGE)
        b2 = ax.bar(x + w / 2, deb, w, color=C_TECH_DEBIAS, edgecolor=C_EDGE)
        all_vals.extend(cda + deb)
        add_bar_labels(ax, b1)
        add_bar_labels(ax, b2)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=28, ha="right", fontsize=FS_TICK)
        style_axis(ax, ylabel=r"$\Delta$STS (p.p.)" if ax is axes[0] else "",
                   title=f"CrowS-Pairs — {title}")
        ax.axhline(0, color=C_SPINE, linewidth=0.4)
    for ax in axes:
        fix_ylim(ax, all_vals, label_pad=3.0)
    legend_below_fig(fig, tech_legend_handles(), ncol=2, y=0.01)
    finish_fig(fig, bottom=0.36, top=0.90)
    save_fig(fig, "gp03_crowspairs_genero_socioeconomico")


def fig_crowspairs_lora(sset):
    fig, ax = plt.subplots(figsize=(11, 7.5))
    all_y = []
    offsets_cycle = [16, -18, 16, -18]
    idx = 0
    for cat, (col, label) in CSP_CATS.items():
        color = C_GENDER if cat == "gender" else C_SOCIO
        for tech, ls in [("cda", "--"), ("debias", "-")]:
            means = sset[sset["technique"] == tech].groupby("lora_trainable_pct")[col].mean()
            ax.plot(means.index, means.values, ls, color=color, marker="o", lw=2.2, markersize=9)
            add_line_point_labels(
                ax, means.index, means.values,
                offsets=[offsets_cycle[idx % 4]] * len(means),
            )
            idx += 1
            all_y.extend(means.values.tolist())
    fix_ylim(ax, all_y, label_pad=2.8)
    ax.set_xlim(0.7, 3.3)
    ax.set_xticks([1, 2, 3])
    style_axis(ax, ylabel=r"$\Delta$STS (p.p.)", xlabel="Porcentaje LoRA (%)",
               title="CrowS-Pairs: género y socioeconómico vs. LoRA")
    handles = (
        category_csp_legend_handles()
        + tech_linestyle_legend_handles()
        + ([zero_ref_legend_handle()] if min(all_y) < -0.05 else [])
    )
    legend_below_fig(fig, handles, ncol=3, y=0.01)
    finish_fig(fig, bottom=0.40, top=0.90)
    save_fig(fig, "gp14_crowspairs_lora_genero_socio")


def fig_lora_lines(sset, col, fname, title):
    fig, ax = plt.subplots(figsize=(11, 7.5))
    all_y = []
    offsets_cycle = [14, -16, 16, -14]
    k = 0
    for bt, color in zip(BIAS_TYPES, [C_GENDER, C_PROFESSION]):
        for tech, ls in [("cda", "--"), ("debias", "-")]:
            sub = sset[sset["technique"] == tech]
            means = sub.groupby("lora_trainable_pct")[col.format(bt)].mean()
            ax.plot(means.index, means.values, ls, color=color, marker="o", lw=2.2, markersize=9)
            add_line_point_labels(
                ax, means.index, means.values,
                offsets=[offsets_cycle[k % 4]] * len(means),
            )
            k += 1
            all_y.extend(means.values.tolist())
    fix_ylim(ax, all_y, label_pad=2.8)
    ax.set_xlim(0.7, 3.3)
    ax.set_xticks([1, 2, 3])
    style_axis(ax, ylabel=r"$\Delta$SS (p.p.)", xlabel="Porcentaje LoRA (%)", title=title)
    handles = (
        category_gp_legend_handles()
        + tech_linestyle_legend_handles()
        + ([zero_ref_legend_handle()] if min(all_y) < -0.05 else [])
    )
    legend_below_fig(fig, handles, ncol=3, y=0.01)
    finish_fig(fig, bottom=0.40, top=0.88)
    save_fig(fig, fname)


def fig_mgp_lora(sset):
    fig, ax = plt.subplots(figsize=(10, 7))
    all_y = []
    for tech, color, ls in [("cda", C_TECH_CDA, "--"), ("debias", C_TECH_DEBIAS, "-")]:
        means = sset[sset["technique"] == tech].groupby("lora_trainable_pct")["M_gp"].mean()
        ax.plot(means.index, means.values, ls, color=color, marker="o", lw=2.4, markersize=10,
                label="Dataset + prompt" if tech == "debias" else "Dataset")
        dy = 14 if tech == "cda" else -16
        add_line_point_labels(ax, means.index, means.values, dy=dy)
        all_y.extend(means.values.tolist())
    fix_ylim(ax, all_y, label_pad=1.8)
    ax.set_xlim(0.7, 3.3)
    ax.set_xticks([1, 2, 3])
    style_axis(ax, ylabel=r"$M_{GP}$ (p.p.)", xlabel="Porcentaje LoRA (%)",
               title="Mitigación compuesta género+profesión vs. LoRA")
    legend_below_fig(fig, tech_linestyle_legend_handles(), ncol=2, y=0.01)
    finish_fig(fig, bottom=0.32, top=0.90)
    save_fig(fig, "gp04_mgp_vs_lora")


def fig_bold_metric(bold, metric, fname, title, ylabel):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False)
    col_lrm = f"bold_lrm_{{}}_{metric}"
    for ax, dom in zip(axes, BIAS_TYPES):
        x = np.arange(len(MODELS))
        w = 0.34
        cda, deb = [], []
        for m in MODELS:
            sub = bold[bold["model_short"] == m]
            c = col_lrm.format(dom)
            cda.append(sub[sub["technique"] == "cda"][c].mean())
            deb.append(sub[sub["technique"] == "debias"][c].mean())
        b1 = ax.bar(x - w / 2, cda, w, color=C_TECH_CDA, edgecolor=C_EDGE)
        b2 = ax.bar(x + w / 2, deb, w, color=C_TECH_DEBIAS, edgecolor=C_EDGE)
        vals = cda + deb
        if metric == "avg_length":
            add_bar_labels(ax, b1, fmt="{:.0f}")
            add_bar_labels(ax, b2, fmt="{:.0f}")
            ax.axhline(5, color=C_SPINE, linestyle=":", linewidth=1)
            fix_ylim(ax, vals, label_pad=5.0)
        else:
            add_bar_labels(ax, b1, fmt="{:.2f}", offset=0.06)
            add_bar_labels(ax, b2, fmt="{:.2f}", offset=0.06)
            fix_ylim(ax, vals, label_pad=0.22)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=28, ha="right", fontsize=FS_TICK)
        style_axis(ax, ylabel=ylabel if ax is axes[0] else "", title=f"{BIAS_LABEL[dom]}")
    fig.suptitle(title, y=0.98, fontsize=FS_SUPTITLE, fontweight="bold")
    legend_below_fig(fig, tech_legend_handles(), ncol=2, y=0.01)
    finish_fig(fig, bottom=0.34, top=0.86)
    save_fig(fig, fname)


def fix_ylim_rggap(ax, values, label_pad=0.08):
    """Escala ajustada a delta RG: evita comprimir barras pequenas en un eje global."""
    flat = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not flat:
        return
    vmin, vmax = min(flat), max(flat)
    span = max(vmax - vmin, 0.04)
    pad = max(label_pad, span * 0.55)
    lo, hi = vmin - pad, vmax + pad
    if vmin <= 0 <= vmax or vmin < 0:
        lo = min(lo, -pad * 0.35)
    if vmax <= 0:
        hi = max(hi, pad * 0.35)
    ax.set_ylim(lo, hi)
    ax.axhline(0, color=C_SPINE, linestyle=":", linewidth=0.7, zorder=1)


def add_bar_labels_tight(ax, bars, fmt="{:.2f}", fontsize=None):
    """Etiquetas con padding proporcional al rango visible del eje."""
    if fontsize is None:
        fontsize = FS_VALUE
    y0, y1 = ax.get_ylim()
    pad = max(0.012, (y1 - y0) * 0.06)
    add_bar_labels(ax, bars, fmt=fmt, fontsize=fontsize, offset=pad)


def fig_bold_rggap(bold):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False)
    for ax, dom in zip(axes, BIAS_TYPES):
        x = np.arange(len(MODELS))
        w = 0.34
        cda, deb = [], []
        for m in MODELS:
            sub = bold[bold["model_short"] == m]
            col = f"d_rggap_{dom}"
            cda.append(sub[sub["technique"] == "cda"][col].mean())
            deb.append(sub[sub["technique"] == "debias"][col].mean())
        b1 = ax.bar(
            x - w / 2, cda, w, color=C_TECH_CDA, edgecolor=C_EDGE, linewidth=0.8,
        )
        b2 = ax.bar(
            x + w / 2, deb, w, color=C_TECH_DEBIAS, edgecolor=C_EDGE, linewidth=0.8,
        )
        vals = cda + deb
        fix_ylim_rggap(ax, vals)
        add_bar_labels_tight(ax, b1)
        add_bar_labels_tight(ax, b2)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=28, ha="right", fontsize=FS_TICK)
        style_axis(
            ax,
            ylabel="Reducción brecha Regard (BLM $-$ LoRA)" if ax is axes[0] else "",
            title=f"{BIAS_LABEL[dom]} — mayor = más equitativo",
        )
    fig.suptitle(
        "BOLD — Equidad entre subgrupos por dominio",
        y=0.98, fontsize=FS_SUPTITLE, fontweight="bold",
    )
    legend_below_fig(fig, tech_legend_handles(), ncol=2, y=0.01)
    finish_fig(fig, bottom=0.34, top=0.86)
    save_fig(fig, "gp11_bold_equidad_genero_profesion")


def fig_tradeoff(sset):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for _, row in sset.iterrows():
        color = C_FAM_GPT2 if row["family"] == "gpt2" else C_FAM_LLAMA
        marker = "D" if row["technique"] == "debias" else "o"
        size = 40 + row["lora_trainable_pct"] * 25
        ax.scatter(row["M_gp"], row["delta_lms_avg"], c=color, marker=marker,
                   s=size, alpha=0.78, edgecolors=C_SPINE, linewidth=0.5)
    ax.axhline(0, color=C_GRID, linewidth=0.6)
    style_axis(ax, ylabel=r"$\delta_{\mathrm{LMS}}$ (p.p.)",
               xlabel=r"$M_{GP}$ — Mitigación género+profesión (p.p.)",
               title="Trade-off: objetivos GP vs. coste lingüístico")
    legend_below_fig(fig, [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FAM_GPT2,
               markersize=8, markeredgecolor=C_SPINE, label="GPT-2"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FAM_LLAMA,
               markersize=8, markeredgecolor=C_SPINE, label="Llama"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#718096",
               markersize=7, label="Dataset"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#718096",
               markersize=7, label="Dataset + prompt"),
    ], ncol=4, y=0.01)
    finish_fig(fig, bottom=0.26, top=0.92)
    save_fig(fig, "gp12_tradeoff_mgp")


def fig_mei_ranking(sset):
    fig, ax = plt.subplots(figsize=(11, 12))
    sorted_df = sset.sort_values("MEI_gp")
    y = np.arange(len(sorted_df))
    colors = [C_TECH_DEBIAS if t == "debias" else C_TECH_CDA for t in sorted_df["technique"]]
    bars = ax.barh(y, sorted_df["MEI_gp"], color=colors, alpha=0.88, edgecolor=C_EDGE, height=0.75)
    for bar, val in zip(bars, sorted_df["MEI_gp"]):
        w = bar.get_width()
        ax.annotate(
            f"{val:.2f}", (w, bar.get_y() + bar.get_height() / 2),
            ha="left", va="center", fontsize=FS_VALUE, fontweight="bold",
            xytext=(6, 0), textcoords="offset points", color=C_TEXT, bbox=VALUE_BBOX,
        )
    labels = [
        f"{r.model_short} | {'D+p' if r.technique == 'debias' else 'D'} | {int(r.lora_trainable_pct)}%"
        for r in sorted_df.itertuples()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.axvline(1.0, color=C_SPINE, linestyle="--", linewidth=1)
    style_axis(ax, xlabel="MEI$_{GP}$", title="Ranking por eficiencia (género + profesión)")
    legend_below_fig(fig, tech_legend_handles(), ncol=2, y=0.01)
    finish_fig(fig, bottom=0.16, top=0.96, left=0.22, right=0.96)
    save_fig(fig, "gp13_mei_ranking_gp")


def fig_per_model(sset, model_key, slug):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    mname = MODEL_SHORT[model_key]
    sub = sset[sset["model_base"] == model_key]
    tasks = [
        ("delta_ss_inter_{}", "StereoSet inter"),
        ("delta_ss_intra_{}", "StereoSet intra"),
        ("delta_csp_gender", "C-Pairs género"),
        ("delta_csp_socioeconomic", "C-Pairs socioecon."),
    ]
    for ax, (col_tpl, ttitle) in zip(axes, tasks):
        x = np.arange(3)
        w = 0.2
        for i, bt in enumerate(BIAS_TYPES):
            if "{}" not in col_tpl:
                if bt == "profession":
                    continue
                col = col_tpl
            else:
                col = col_tpl.format(bt)
            offsets = []
            for pct in [1, 2, 3]:
                for tech in ["cda", "debias"]:
                    row = sub[(sub["lora_trainable_pct"] == pct) & (sub["technique"] == tech)]
                    val = row[col].values[0] if len(row) else 0
                    offsets.append(val)
            # simpler: 2 tech x 3 lora per bias type
        # Rebuild: grouped by LoRA, clusters gender/profession
        w = 0.12
        positions = []
        vals_g, vals_p = [], []
        labels = []
        for j, pct in enumerate([1, 2, 3]):
            for k, tech in enumerate(["cda", "debias"]):
                row = sub[(sub["lora_trainable_pct"] == pct) & (sub["technique"] == tech)]
                if len(row) == 0:
                    continue
                r = row.iloc[0]
                pos = j * 2.5 + k
                positions.append(pos)
                vals_g.append(r["delta_ss_inter_gender"] if "inter" in ttitle else (
                    r["delta_ss_intra_gender"] if "intra" in ttitle else r["delta_csp_gender"]))
                if "crowspairs" not in ttitle.lower():
                    vals_p.append(r["delta_ss_inter_profession"] if "inter" in ttitle else r["delta_ss_intra_profession"])
                labels.append(f"{int(pct)}%")
        if "c-pairs" in ttitle.lower() or "crowspairs" in ttitle.lower():
            xpos = np.arange(6)
            sub2 = sub.sort_values(["lora_trainable_pct", "technique"])
            col = "delta_csp_gender" if "genero" in ttitle.lower() else "delta_csp_socioeconomic"
            color = C_GENDER if "genero" in ttitle.lower() else C_SOCIO
            vals = list(sub2[col].values)
            bars = ax.bar(xpos, vals, 0.5, color=color, edgecolor=C_EDGE)
            add_bar_labels(ax, bars, fontsize=FS_VALUE)
            ax.set_xticks(xpos)
            ax.set_xticklabels(
                [f"{int(r.lora_trainable_pct)}%\n{'D+p' if r.technique == 'debias' else 'D'}"
                 for r in sub2.itertuples()], fontsize=FS_TICK
            )
        else:
            xpos = np.arange(6)
            sub2 = sub.sort_values(["lora_trainable_pct", "technique"])
            cg = "delta_ss_inter_gender" if "inter" in ttitle else "delta_ss_intra_gender"
            cp = "delta_ss_inter_profession" if "inter" in ttitle else "delta_ss_intra_profession"
            vg = [getattr(r, cg) for r in sub2.itertuples()]
            vp = [getattr(r, cp) for r in sub2.itertuples()]
            bg = ax.bar(xpos - 0.18, vg, 0.36, color=C_GENDER, edgecolor=C_EDGE)
            bp = ax.bar(xpos + 0.18, vp, 0.36, color=C_PROFESSION, edgecolor=C_EDGE)
            add_bar_labels(ax, bg, fontsize=FS_VALUE_DENSE)
            add_bar_labels(ax, bp, fontsize=FS_VALUE_DENSE)
            ax.set_xticks(xpos)
            ax.set_xticklabels(
                [f"{int(r.lora_trainable_pct)}%\n{'D+p' if r.technique == 'debias' else 'D'}"
                 for r in sub2.itertuples()], fontsize=FS_TICK
            )
        patch_vals = [p.get_height() for p in ax.patches]
        fix_ylim(ax, patch_vals if patch_vals else [0], label_pad=3.5)
        style_axis(ax, title=ttitle)
        if ax in (axes[0], axes[2]):
            ax.set_ylabel(r"$\Delta$ (p.p.)")
    fig.suptitle(mname, y=0.99, fontsize=FS_SUPTITLE, fontweight="bold")
    legend_below_fig(fig, [
        Patch(facecolor=C_GENDER, label="Género"),
        Patch(facecolor=C_PROFESSION, label="Profesión"),
    ], ncol=2, y=0.01)
    finish_fig(fig, bottom=0.22, top=0.90, left=0.08, right=0.98)
    save_fig(fig, f"gp_modelo_{slug}")


def write_tables(sset, bold):
    # Tabla resumen: media, maximo y configuracion que alcanza cada maximo
    def add_block(lines, title, row_specs):
        lines += [
            r"\midrule",
            rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{title}}}}} \\",
            r"\midrule",
        ]
        for cat, task, col in row_specs:
            br = best_row_at_max(sset, col)
            lines.append(
                f"{cat} & {task} & "
                f"{sset[col].mean():.2f} & {sset[col].max():.2f} & "
                f"{fmt_config_tex(br)} \\\\"
            )

    stereo_specs = []
    for bt in BIAS_TYPES:
        for task, suffix in [("Inter", "inter"), ("Intra", "intra")]:
            stereo_specs.append((BIAS_LABEL[bt], task, f"delta_ss_{suffix}_{bt}"))
    csp_specs = [
        ("Género", "STS", "delta_csp_gender"),
        ("Socioecon./ocupación", "STS", "delta_csp_socioeconomic"),
    ]

    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\caption{Resumen GP (30 configuraciones).}",
        r"\label{tab:gp_resumen}",
        r"\begin{tabular}{@{}llrrp{5.4cm}@{}}",
        r"\toprule",
        r"\textbf{Categoría} & \textbf{Tarea} & \textbf{Media} & \textbf{Máx.} & "
        r"\textbf{Config. del máximo} \\",
    ]
    add_block(lines, "StereoSet ($\Delta$SS, p.p.)", stereo_specs)
    add_block(lines, "CrowS-Pairs ($\Delta$STS, p.p.)", csp_specs)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "tab_resumen_gp.tex").write_text("\n".join(lines), encoding="utf-8")

    # Mejor config
    best = sset.loc[sset["M_gp"].idxmax()]
    (TAB_DIR / "tab_mejor_config_gp.tex").write_text(
        f"""\\begin{{table}}[H]
\\centering\\small
\\caption{{Mejor $M_{{GP}}$.}}
\\label{{tab:gp_mejor}}
\\begin{{tabular}}{{@{{}}llr@{{}}}}
\\toprule
\\textbf{{Campo}} & \\textbf{{Valor}} \\\\
\\midrule
Modelo & {best['model_short']} \\\\
T\'ecnica & {'Dataset + prompt' if best['technique'] == 'debias' else 'Dataset'} \\\\
LoRA & {int(best['lora_trainable_pct'])}\\% \\\\
$M_{{GP}}$ & {best['M_gp']:.2f} p.p. \\\\
$\\Delta$SS inter g\'enero & {best['delta_ss_inter_gender']:.2f} p.p. \\\\
$\\Delta$SS inter profesi\'on & {best['delta_ss_inter_profession']:.2f} p.p. \\\\
$\\Delta$SS intra g\'enero & {best['delta_ss_intra_gender']:.2f} p.p. \\\\
$\\Delta$SS intra profesi\'on & {best['delta_ss_intra_profession']:.2f} p.p. \\\\
$\\Delta$STS g\'enero & {best['delta_csp_gender']:.2f} p.p. \\\\
$\\Delta$STS socioecon. & {best['delta_csp_socioeconomic']:.2f} p.p. \\\\
MEI$_{{GP}}$ & {best['MEI_gp']:.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}""",
        encoding="utf-8",
    )

    # BOLD promedio
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\caption{BOLD por familia y técnica.}",
        r"\label{tab:gp_bold}",
        r"\begin{tabular}{@{}ll rrr rrr@{}}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{\textbf{Género}} & \multicolumn{3}{c}{\textbf{Profesión}} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"\textbf{Familia} & \textbf{Técnica} & Regard & Long. & Brecha & Regard & Long. & Brecha \\",
        r"\midrule",
    ]
    for fam, flab in [("gpt2", "GPT-2"), ("llama", "Llama")]:
        for tech, tlab in [("cda", "Solo dataset"), ("debias", "Dataset + prompt")]:
            sub = bold[(bold["family"] == fam) & (bold["technique"] == tech)]
            lines.append(
                f"{flab} & {tlab} & "
                f"{sub['bold_lrm_gender_avg_regard'].mean():.3f} & "
                f"{sub['bold_lrm_gender_avg_length'].mean():.1f} & "
                f"{sub['d_rggap_gender'].mean():.3f} & "
                f"{sub['bold_lrm_profession_avg_regard'].mean():.3f} & "
                f"{sub['bold_lrm_profession_avg_length'].mean():.1f} & "
                f"{sub['d_rggap_profession'].mean():.3f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "tab_bold_gp.tex").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=" * 60)
    print("ANALISIS FOCALIZADO: GENERO Y PROFESION")
    print("=" * 60)
    sset, bold = load_and_compute()
    export_data(sset, bold)
    print("[1] Datos exportados")

    fig_stereoset_by_model(sset, "Intersentence", "delta_ss_inter_{}",
                           "gp01_stereoset_inter_genero_profesion", r"$\Delta$SS (p.p.)",
                           tick_fontsize=FS_TICK + 1, legend_fontsize=FS_LEGEND + 1)
    fig_stereoset_by_model(sset, "Intrasentence", "delta_ss_intra_{}",
                           "gp02_stereoset_intra_genero_profesion", r"$\Delta$SS (p.p.)")
    fig_crowspairs_gender_socio(sset)
    fig_crowspairs_lora(sset)
    fig_lora_lines(sset, "delta_ss_inter_{}", "gp05_lora_inter_genero_profesion",
                   "StereoSet intersentence vs. LoRA")
    fig_lora_lines(sset, "delta_ss_intra_{}", "gp06_lora_intra_genero_profesion",
                   "StereoSet intrasentence vs. LoRA")
    fig_mgp_lora(sset)
    fig_bold_metric(bold, "avg_length", "gp07_bold_longitud",
                    "BOLD — Longitud de generacion", "Tokens")
    fig_bold_metric(bold, "avg_regard", "gp08_bold_regard",
                    "BOLD — Regard promedio (LoRA)", "Regard")
    fig_bold_rggap(bold)
    fig_tradeoff(sset)
    fig_mei_ranking(sset)

    for key, slug in [
        ("gpt2", "gpt2_base"),
        ("gpt2-medium", "gpt2_medium"),
        ("gpt2-xl", "gpt2_xl"),
        ("meta-llama/Llama-3.2-1B-Instruct", "llama_1b"),
        ("meta-llama/Llama-3.2-3B-Instruct", "llama_3b"),
    ]:
        fig_per_model(sset, key, slug)

    write_tables(sset, bold)
    print(f"[2] Figuras en {FIG_DIR}")
    print(f"[3] Tablas en {TAB_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
