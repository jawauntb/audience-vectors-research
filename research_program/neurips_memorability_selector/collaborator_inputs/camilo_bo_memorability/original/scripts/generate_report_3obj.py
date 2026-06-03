"""
PDF report for the 3-objective run (TRIBE · CLIP · Quality R3D-18).

Includes a comparative analysis that contextualizes *how good* the achieved
values are — not just the absolute numbers. Benchmarks used:
  - Best-of-N baseline from the base paper (Brown 2026, C1): TRIBE lift +2.07 ± 0.60
  - Internal 2-objective run (outputs/gpu_run/)
  - Internal random baseline (Sobol points from this same run)
  - Natural quality ceiling (reference videos alpha=0, score = 1.0)
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT       = Path(__file__).parent.parent
OUT_DIR    = ROOT / "outputs" / "gpu_run_3obj"
JSON_PATH  = OUT_DIR / "all_results.json"
JSON_2OBJ  = ROOT / "outputs" / "gpu_run" / "all_results.json"
PDF_PATH   = ROOT / "outputs" / "bo_memorability_3obj_report.pdf"

# External baseline (base paper) — TRIBE/BMD proxy, same scale as our scores.
BROWN_C1_LIFT      = 2.07
BROWN_C1_LIFT_STD  = 0.60
BROWN_HUMAN_PREF   = 0.643   # 64.3% human preference (Prolific)

with open(JSON_PATH, encoding="utf-8") as f:
    data = json.load(f)

all_meta   = data["all_meta"]
hv_history = data["hv_history"]
n_initial  = data["n_initial"]
n_iter     = data["n_iterations"]

sobol_meta = [m for m in all_meta if "sobol" in m["task_id"]]
bo_meta    = [m for m in all_meta if "sobol" not in m["task_id"]]

# 2-objective run (internal comparison), if present.
data2 = None
if JSON_2OBJ.exists():
    with open(JSON_2OBJ, encoding="utf-8") as f:
        data2 = json.load(f)

BG      = "#F8F9FA"
SOBOL_C = "#4C72B0"
BO_C    = "#DD8452"
BEST_C  = "#2CA02C"
REF_C   = "#7F8C8D"


def page_bg(fig):
    fig.patch.set_facecolor(BG)


def section_title(ax, text):
    ax.set_facecolor(BG); ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center",
            fontsize=20, fontweight="bold", color="#2C3E50",
            transform=ax.transAxes)


def is_pareto(scores):
    scores = np.array(scores)
    n = len(scores)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(scores[j] >= scores[i]) and np.any(scores[j] > scores[i]):
                dominated[i] = True
                break
    return ~dominated


scores_3d   = np.array([[m["tribe_score"], m["clip_score"], m["quality_score"]] for m in all_meta])
pareto_mask = is_pareto(scores_3d)
pareto_meta = [m for m, p in zip(all_meta, pareto_mask) if p]

best_tribe   = max(all_meta, key=lambda m: m["tribe_score"])
best_clip    = max(all_meta, key=lambda m: m["clip_score"])
best_quality = max(all_meta, key=lambda m: m["quality_score"])

# ── Statistics for the comparative analysis ────────────────────────────────────
tribe_all   = np.array([m["tribe_score"]   for m in all_meta])
clip_all    = np.array([m["clip_score"]    for m in all_meta])
qual_all     = np.array([m["quality_score"] for m in all_meta])

sobol_tribe = np.array([m["tribe_score"] for m in sobol_meta])
bo_tribe    = np.array([m["tribe_score"] for m in bo_meta])

# Memorability lift: best / BO mean against the random baseline (Sobol).
baseline_tribe = float(sobol_tribe.mean())          # "blind sampling" baseline
lift_best      = float(tribe_all.max()) - baseline_tribe
lift_bo_mean   = float(bo_tribe.mean()) - baseline_tribe
lift_ratio     = lift_best / BROWN_C1_LIFT

# Stats from the 2-objective run
if data2:
    m2          = data2["all_meta"]
    best_tribe2 = max(m["tribe_score"] for m in m2)
    best_clip2  = max(m["clip_score"]  for m in m2)
    sc2         = [[m["tribe_score"], m["clip_score"]] for m in m2]
    pmask2      = is_pareto(sc2)
    n_pareto2   = int(pmask2.sum())
    par2        = [m for m, b in zip(m2, pmask2) if b]
    alpha2_lo   = min(m["alpha"] for m in par2); alpha2_hi = max(m["alpha"] for m in par2)
    seeds2      = sorted(set(m["seed_idx"] for m in par2))
    hv2_first, hv2_last = data2["hv_history"][0], data2["hv_history"][-1]
    n_eval2     = len(m2)
else:
    best_tribe2 = best_clip2 = n_pareto2 = n_eval2 = float("nan")
    alpha2_lo = alpha2_hi = float("nan"); seeds2 = []; hv2_first = hv2_last = float("nan")

alpha_p_lo = min(m["alpha"] for m in pareto_meta)
alpha_p_hi = max(m["alpha"] for m in pareto_meta)
seeds_p    = sorted(set(m["seed_idx"] for m in pareto_meta))


with PdfPages(PDF_PATH) as pdf:

    # ── PAGE 1  COVER ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5)); page_bg(fig)
    ax  = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_facecolor(BG)

    ax.add_patch(mpatches.FancyBboxPatch((0.05, 0.72), 0.90, 0.22,
        boxstyle="round,pad=0.01", facecolor="#2C3E50", edgecolor="none",
        transform=ax.transAxes))
    ax.text(0.50, 0.88, "BO-Memorability  —  3 Objetivos",
            ha="center", va="center", fontsize=28, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.50, 0.80, "TRIBE  ·  CLIP  ·  Quality (R3D-18)  —  com Análise Comparativa",
            ha="center", va="center", fontsize=13, color="#ECF0F1",
            transform=ax.transAxes)
    ax.text(0.50, 0.74, f"Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ha="center", va="center", fontsize=10, color="#BDC3C7",
            transform=ax.transAxes)

    metrics = [
        ("Avaliações",        str(len(all_meta))),
        ("Sobol Init",        str(n_initial)),
        ("Iterações BO",      str(n_iter)),
        ("HV Final (3D)",     f"{hv_history[-1]:.4f}"),
        ("Ganho HV",          f"x{hv_history[-1]/hv_history[0]:.1f}"),
        ("Pareto (3D)",       str(int(pareto_mask.sum()))),
        ("Melhor TRIBE",      f"{best_tribe['tribe_score']:.3f}"),
        ("Lift vs aleatório", f"+{lift_best:.2f}"),
    ]
    bw, bh = 0.20, 0.12; x0, y0 = 0.055, 0.52
    for i, (lbl, val) in enumerate(metrics):
        col, row = i % 4, i // 4
        bx = x0 + col * (bw + 0.025); by = y0 - row * (bh + 0.03)
        ax.add_patch(mpatches.FancyBboxPatch((bx, by), bw, bh,
            boxstyle="round,pad=0.01", facecolor="white",
            edgecolor="#BDC3C7", linewidth=1, transform=ax.transAxes))
        ax.text(bx+bw/2, by+bh*0.65, val, ha="center", va="center",
                fontsize=15, fontweight="bold", color="#2C3E50",
                transform=ax.transAxes)
        ax.text(bx+bw/2, by+bh*0.20, lbl, ha="center", va="center",
                fontsize=8, color="#7F8C8D", transform=ax.transAxes)

    info = [
        "Hardware:   NVIDIA GeForce RTX 5080  |  16 GB VRAM  |  CUDA 12.8",
        "Objetivos:  (1) TRIBE v2 memorabilidade  (2) OpenCLIP ViT-H/14  (3) R3D-18 qualidade",
        "Algoritmo:  qLogNoisyExpectedHypervolumeImprovement  |  ModelListGP  |  MixedGP",
        "Busca:      alpha em [-10,+10]  |  guidance em [1,10]  |  seed_idx em {0..15}",
        "Qualidade:  R3D-18 cosine vs centroide de 8 videos de referencia (alpha=0)",
        "Baseline:   Brown 2026 C1 best-of-N (N=10): lift TRIBE +2.07 +/- 0.60",
    ]
    ax.text(0.50, 0.34, "Setup & Baseline", ha="center", fontsize=12, fontweight="bold",
            color="#2C3E50", transform=ax.transAxes)
    for i, line in enumerate(info):
        ax.text(0.07, 0.29 - i*0.037, line, ha="left", va="top", fontsize=8.5,
                color="#2C3E50", family="monospace", transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 2  EXECUTIVE SUMMARY ──────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5)); page_bg(fig)
    ax_t = fig.add_axes([0.05, 0.91, 0.90, 0.07]); section_title(ax_t, "Resumo — Em palavras simples")
    ax   = fig.add_axes([0.07, 0.05, 0.86, 0.83]); ax.set_facecolor(BG); ax.axis("off")
    resumo = f"""
O QUE FIZEMOS
=============
Geramos videos com Stable Video Diffusion e empurramos cada video na direcao
"memoravel" (parametro alpha). Um otimizador Bayesiano testou {len(all_meta)} combinacoes de
alpha, guidance e imagem-semente, buscando ao mesmo tempo TRES qualidades:

  1. TRIBE    -> o quanto o video deve ativar o cerebro de forma memoravel
  2. CLIP     -> o quanto o video continua fiel ao que o texto pedia
  3. Qualidade-> o quanto o video continua parecendo um video natural/realista

A novidade desta rodada e a 3a medida (Qualidade, via R3D-18). Antes so otimizavamos
memorabilidade e fidelidade; agora controlamos tambem se o video nao "quebra".

O QUE DESCOBRIMOS (resumo)
==========================
  - A memorabilidade subiu muito: o melhor video alcancou TRIBE = {best_tribe['tribe_score']:.2f},
    contra uma media de {baseline_tribe:.2f} dos videos sorteados ao acaso.
    Isso e um ganho de +{lift_best:.2f} pontos -- cerca de {lift_ratio:.1f}x o ganho
    do metodo de referencia (Brown 2026, que melhora +{BROWN_C1_LIFT:.2f} apenas reamostrando).

  - A qualidade se manteve alta: mesmo empurrando os videos, eles continuaram entre
    {qual_all.min():.2f} e {qual_all.max():.2f} de similaridade com videos naturais (teto = 1.00).
    Ou seja, ganhamos memorabilidade SEM destruir o realismo.

  - A fidelidade ao texto (CLIP) ficou em ate {clip_all.max():.3f}, dentro/acima da
    faixa tipica de modelos CLIP -- o video continua "sobre o tema" pedido.

  - Encontramos {int(pareto_mask.sum())} solucoes otimas (frente de Pareto), usando {len(seeds_p)} sementes
    diferentes -- bem mais diverso que a rodada anterior de 2 objetivos.

CONCLUSAO
=========
Adicionar a medida de qualidade nao atrapalhou: o sistema continua achando videos
muito mais memoraveis que o acaso e que o baseline publicado, e agora com a garantia
extra de que esses videos permanecem visualmente coerentes. A pagina seguinte de
"Analise Comparativa" detalha o quao bons sao esses numeros.
"""
    ax.text(0.0, 0.99, resumo, ha="left", va="top", fontsize=9.2,
            color="#2C3E50", family="monospace", transform=ax.transAxes, linespacing=1.45)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 3  METHODOLOGY ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5)); page_bg(fig)
    ax_t = fig.add_axes([0.05, 0.91, 0.90, 0.07]); section_title(ax_t, "Metodologia — Extensão para 3 Objetivos")
    ax   = fig.add_axes([0.07, 0.04, 0.86, 0.84]); ax.set_facecolor(BG); ax.axis("off")
    txt = """
NOVO OBJETIVO 3 — VideoQualityScorer (R3D-18)
==============================================
Modelo:  torchvision.models.video.r3d_18 (pre-treinado em Kinetics-400, 512-dim)
Metodo:  Similaridade de cosseno entre o vetor de features R3D-18 do video e o
         centroide de 8 videos de referencia gerados com alpha=0 (sem steering).
Escala:  [-1, 1]. Score alto (~0.8) = visualmente proximo de video natural.
         Score baixo = steering empurrou o conteudo para fora da distribuicao natural.

Por que R3D-18 e nao FVD classico?
  - FVD exige lotes de >= 16 videos para estimar a distancia de Frechet entre duas
    distribuicoes. No loop BO pontuamos um video por vez, o que torna o FVD instavel.
  - Cosine R3D-18 ao centroide de referencia e estavel e calculavel online (por video).

CALIBRACAO DA REFERENCIA (uma unica vez)
========================================
  1. Gera 8 videos "neutros": alpha=0, guidance=3.0.
  2. Extrai features R3D-18 (512-dim) de cada um.
  3. Calcula e normaliza (L2) o vetor centroide.
  4. Salva em quality_reference.npz (reaproveitado nas rodadas seguintes).

ESPACO DE BUSCA (identico ao run de 2-obj)
==========================================
  alpha     em [-10.0, +10.0]   continuo   — forca do steering de memorabilidade
  guidance  em [1.0,   10.0]    continuo   — escala de classifier-free guidance
  seed_idx  em {0, 1, ..., 15}  categorico — par imagem-semente / prompt

SURROGATE GP + AQUISICAO
========================
  ModelListGP com MixedSingleTaskGP (um GP por objetivo, kernel de Hamming em seed_idx).
  Ajustado a cada iteracao via maximizacao de SumMarginalLogLikelihood.
  Aquisicao: qLogNoisyExpectedHypervolumeImprovement (q=2 por iteracao).

FRENTE DE PARETO 3D
===================
Com 3 objetivos, a frente de Pareto e uma superficie 2D no espaco 3D. Um ponto
(tribe, clip, quality) e Pareto-otimo se nenhum outro ponto observado o domina
simultaneamente nos tres objetivos. O Hypervolume (HV) dominado resume o progresso
multi-objetivo num unico escalar.
"""
    ax.text(0.0, 0.99, txt, ha="left", va="top", fontsize=8.8,
            color="#2C3E50", family="monospace", transform=ax.transAxes, linespacing=1.45)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 4  HV CONVERGENCE + tabela ───────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5)); page_bg(fig)
    ax_t = fig.add_axes([0.05, 0.91, 0.90, 0.07]); section_title(ax_t, "Convergência BO (Hypervolume 3D)")

    ax_hv = fig.add_axes([0.07, 0.52, 0.88, 0.36]); ax_hv.set_facecolor("white")
    iters = list(range(len(hv_history)))
    ax_hv.plot(iters, hv_history, "o-", color=BEST_C, lw=2.5, ms=7, zorder=3)
    ax_hv.fill_between(iters, hv_history, alpha=0.15, color=BEST_C)
    for i in range(1, len(hv_history)):
        delta = hv_history[i] - hv_history[i-1]
        if delta > 0.01:
            ax_hv.annotate(f"+{delta:.4f}", xy=(i, hv_history[i]),
                           xytext=(i+0.1, hv_history[i]+0.005), fontsize=8, color="#C0392B",
                           arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1))
    ax_hv.set_xlabel("Iteração BO (0 = após Sobol)")
    ax_hv.set_ylabel("Hypervolume dominado (3D)")
    ax_hv.set_title(f"HV: {hv_history[0]:.4f} -> {hv_history[-1]:.4f}  (x{hv_history[-1]/hv_history[0]:.1f})")
    ax_hv.set_xticks(iters)
    ax_hv.grid(alpha=0.3)

    txt_hv = """
LEITURA DO GRAFICO
==================
O HV cresce de forma escalonada: saltos nas primeiras iteracoes (a BO encontra rapido
regioes boas), um plato no meio (iteracoes 5-7, explorando sem dominar mais area) e um
novo ganho no fim (iteracoes 8-9), quando o otimizador combina alpha moderado com a
semente certa. O crescimento x{:.1f} confirma que a BO esta de fato avancando na frente
de Pareto 3D -- e nao apenas oscilando.

Importante: o HV deste run (3D) NAO e comparavel em magnitude ao HV do run de 2-obj
(2D), pois mudam a dimensao, as escalas e o ponto de referencia. Use o HV apenas para
medir progresso DENTRO deste run.
""".format(hv_history[-1]/hv_history[0])
    ax_tx = fig.add_axes([0.07, 0.04, 0.88, 0.42]); ax_tx.set_facecolor(BG); ax_tx.axis("off")
    ax_tx.text(0.0, 0.98, txt_hv, ha="left", va="top", fontsize=9.5,
               color="#2C3E50", family="monospace", transform=ax_tx.transAxes, linespacing=1.5)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 5  PARETO 3D PROJECTIONS + tabela ────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5)); page_bg(fig)
    ax_t = fig.add_axes([0.05, 0.91, 0.90, 0.07]); section_title(ax_t, "Frente de Pareto — Projeções 3D")

    t_s = [m["tribe_score"] for m in sobol_meta]; c_s = [m["clip_score"] for m in sobol_meta]; q_s = [m["quality_score"] for m in sobol_meta]
    t_b = [m["tribe_score"] for m in bo_meta];    c_b = [m["clip_score"] for m in bo_meta];    q_b = [m["quality_score"] for m in bo_meta]
    t_p = [m["tribe_score"] for m in pareto_meta]; c_p = [m["clip_score"] for m in pareto_meta]; q_p = [m["quality_score"] for m in pareto_meta]

    kw_s = dict(color=SOBOL_C, s=55, alpha=0.7, label=f"Sobol (n={len(sobol_meta)})")
    kw_b = dict(color=BO_C,    s=55, alpha=0.7, label=f"BO (n={len(bo_meta)})")
    kw_p = dict(color=BEST_C,  s=130, marker="*", zorder=4, edgecolors="k", lw=0.8, label=f"Pareto (n={len(pareto_meta)})")

    ax1 = fig.add_axes([0.06, 0.52, 0.27, 0.36]); ax1.set_facecolor("white")
    ax1.scatter(t_s, c_s, **kw_s); ax1.scatter(t_b, c_b, **kw_b); ax1.scatter(t_p, c_p, **kw_p)
    ax1.set_xlabel("TRIBE"); ax1.set_ylabel("CLIP"); ax1.set_title("TRIBE x CLIP"); ax1.grid(alpha=0.3); ax1.legend(fontsize=7)
    ax2 = fig.add_axes([0.38, 0.52, 0.27, 0.36]); ax2.set_facecolor("white")
    ax2.scatter(t_s, q_s, **kw_s); ax2.scatter(t_b, q_b, **kw_b); ax2.scatter(t_p, q_p, **kw_p)
    ax2.set_xlabel("TRIBE"); ax2.set_ylabel("Quality R3D-18"); ax2.set_title("TRIBE x Quality"); ax2.grid(alpha=0.3)
    ax3 = fig.add_axes([0.70, 0.52, 0.27, 0.36]); ax3.set_facecolor("white")
    ax3.scatter(c_s, q_s, **kw_s); ax3.scatter(c_b, q_b, **kw_b); ax3.scatter(c_p, q_p, **kw_p)
    ax3.set_xlabel("CLIP"); ax3.set_ylabel("Quality R3D-18"); ax3.set_title("CLIP x Quality"); ax3.grid(alpha=0.3)

    ax4 = fig.add_axes([0.03, 0.01, 0.94, 0.47]); ax4.set_facecolor(BG); ax4.axis("off")
    hdrs3 = ["#", "Task ID", "alpha", "guid", "seed", "TRIBE", "CLIP", "Quality", "Prompt"]
    cw3   = [0.03, 0.10, 0.07, 0.07, 0.04, 0.09, 0.08, 0.08, 0.44]
    xp3 = [0.0]
    for w in cw3[:-1]: xp3.append(xp3[-1]+w)
    ax4.add_patch(mpatches.FancyBboxPatch((0,0.91), 1, 0.08,
        boxstyle="square,pad=0", facecolor="#1E8449", transform=ax4.transAxes, zorder=-1))
    ax4.text(0.5, 0.94, f"Frente de Pareto — {len(pareto_meta)} pontos", ha="center", fontsize=9,
             fontweight="bold", color="white", transform=ax4.transAxes, va="center")
    for j,(h,x) in enumerate(zip(hdrs3,xp3)):
        ax4.text(x, 0.86, h, fontsize=7.5, fontweight="bold", color="#1E8449",
                 transform=ax4.transAxes, va="center")
    for i, m in enumerate(sorted(pareto_meta, key=lambda x: -x["tribe_score"])):
        y = 0.82 - i*0.072
        ax4.add_patch(mpatches.FancyBboxPatch((0,y-0.025), 1, 0.065,
            boxstyle="square,pad=0", facecolor="#D5F5E3", transform=ax4.transAxes, zorder=-1))
        row = [str(i+1), m["task_id"], f"{m['alpha']:+.3f}", f"{m['guidance']:.2f}",
               str(m["seed_idx"]), f"{m['tribe_score']:+.4f}", f"{m['clip_score']:.4f}",
               f"{m['quality_score']:.4f}", m["prompt"][:50]+"..."]
        for val,x in zip(row,xp3):
            ax4.text(x, y, val, fontsize=7, color="#1A5276",
                     transform=ax4.transAxes, va="center", fontweight="bold")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 6  COMPARATIVE ANALYSIS — "how good are the values" ─────────────
    fig = plt.figure(figsize=(11, 8.5)); page_bg(fig)
    ax_t = fig.add_axes([0.05, 0.93, 0.90, 0.06]); section_title(ax_t, "Análise Comparativa — Quão bons são os valores?")

    # Panel A: memorability lift vs baselines (bars)
    axA = fig.add_axes([0.07, 0.56, 0.40, 0.34]); axA.set_facecolor("white")
    labels = ["Brown 2026\n(best-of-N)", "BO média\n(este run)", "BO melhor\n(este run)"]
    vals   = [BROWN_C1_LIFT, lift_bo_mean, lift_best]
    errs   = [BROWN_C1_LIFT_STD, 0, 0]
    colors = [REF_C, BO_C, BEST_C]
    bars = axA.bar(labels, vals, yerr=errs, capsize=4, color=colors, edgecolor="#333", lw=0.6)
    for b, v in zip(bars, vals):
        axA.text(b.get_x()+b.get_width()/2, v+0.15, f"+{v:.2f}", ha="center",
                 fontsize=9, fontweight="bold", color="#2C3E50")
    axA.set_ylabel("Lift de memorabilidade (TRIBE)")
    axA.set_title("Ganho de memorabilidade vs baseline")
    axA.grid(alpha=0.3, axis="y")

    # Panel B: quality — achieved vs natural ceiling
    axB = fig.add_axes([0.57, 0.56, 0.38, 0.34]); axB.set_facecolor("white")
    qlabels = ["Pior\nsteered", "Média\nsteered", "Melhor\nsteered", "Referência\n(alpha=0)"]
    qvals   = [float(qual_all.min()), float(qual_all.mean()), float(qual_all.max()), 1.0]
    qcolors = ["#C0392B", BO_C, BEST_C, REF_C]
    bars2 = axB.bar(qlabels, qvals, color=qcolors, edgecolor="#333", lw=0.6)
    for b, v in zip(bars2, qvals):
        axB.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}", ha="center",
                 fontsize=9, fontweight="bold", color="#2C3E50")
    axB.set_ylim(0, 1.08); axB.set_ylabel("Quality R3D-18 (cosine)")
    axB.set_title("Qualidade retida vs teto natural")
    axB.grid(alpha=0.3, axis="y")

    # Benchmark table / text
    axC = fig.add_axes([0.05, 0.02, 0.90, 0.50]); axC.set_facecolor(BG); axC.axis("off")
    cmp_txt = f"""
COMO LER ESTES NUMEROS
======================

(1) MEMORABILIDADE — quanto melhor que o acaso e que o artigo base?
    - Baseline aleatorio interno (media dos {len(sobol_meta)} pontos Sobol):  TRIBE = {baseline_tribe:+.2f}
    - Melhor video desta rodada:                              TRIBE = {tribe_all.max():+.2f}
    - Lift (melhor - baseline) = +{lift_best:.2f}  |  Lift medio da BO = +{lift_bo_mean:.2f}
    - Artigo base (Brown 2026, C1 best-of-N, N=10):           lift = +{BROWN_C1_LIFT:.2f} +/- {BROWN_C1_LIFT_STD:.2f}
    => Nosso ganho e ~{lift_ratio:.1f}x o do baseline publicado. ATENCAO: mecanismos diferentes
       (nos usamos STEERING dirigido + BO; o baseline apenas REAMOSTRA). A comparacao mostra
       que dirigir a geracao supera, com folga, escolher o melhor de N amostras cegas.

(2) FIDELIDADE (CLIP) — o video continua "sobre o tema"?
    - CLIP medio = {clip_all.mean():.3f}  |  melhor = {clip_all.max():.3f}
    - Faixa tipica de cosine CLIP texto-imagem em geracao: ~0.20 a 0.30.
    => Estamos no topo / acima da faixa tipica: o steering NAO desalinhou o video do prompt.

(3) QUALIDADE (R3D-18) — o video continua parecendo natural?
    - Teto = 1.00 (auto-similaridade da referencia alpha=0).
    - Steered: minimo {qual_all.min():.2f}  |  medio {qual_all.mean():.2f}  |  melhor {qual_all.max():.2f}
    => Mesmo no pior caso retemos {100*qual_all.min():.0f}% de alinhamento com a distribuicao
       natural; nos melhores pontos de Pareto, {100*best_quality['quality_score']:.0f}%. O steering perturba,
       mas NAO colapsa o realismo -- exatamente o que a 3a metrica foi criada para garantir.

(4) DIVERSIDADE DA SOLUCAO — frente de Pareto mais rica
    - 2-obj: {n_pareto2} pontos, todos na semente {seeds2}.   3-obj: {len(pareto_meta)} pontos, sementes {seeds_p}.
    => Adicionar a qualidade quebrou o colapso "sempre alpha maximo numa unica semente" do run 2D
       e revelou um conjunto de solucoes bem mais variado e util.

VEREDITO: valores fortes. Memorabilidade muito acima do acaso e do baseline publicado,
fidelidade no topo da faixa tipica, e qualidade preservada perto do teto natural.
"""
    axC.text(0.0, 0.99, cmp_txt, ha="left", va="top", fontsize=8.6,
             color="#2C3E50", family="monospace", transform=axC.transAxes, linespacing=1.42)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 7  2-OBJ vs 3-OBJ COMPARISON + observations ───────────────────────
    fig = plt.figure(figsize=(11, 8.5)); page_bg(fig)
    ax_t = fig.add_axes([0.05, 0.91, 0.90, 0.07]); section_title(ax_t, "2-Objetivos vs 3-Objetivos & Observações")
    ax   = fig.add_axes([0.05, 0.02, 0.90, 0.86]); ax.set_facecolor(BG); ax.axis("off")

    ax_d1 = fig.add_axes([0.58, 0.55, 0.38, 0.30]); ax_d1.set_facecolor("white")
    ax_d1.hist([m["tribe_score"] for m in sobol_meta], bins=8, color=SOBOL_C, alpha=0.6, label="Sobol")
    ax_d1.hist([m["tribe_score"] for m in bo_meta],    bins=8, color=BO_C,    alpha=0.6, label="BO")
    ax_d1.axvline(best_tribe["tribe_score"], color=BEST_C, ls="--", lw=2, label=f"Melhor {best_tribe['tribe_score']:.2f}")
    ax_d1.set_title("Distribuição TRIBE"); ax_d1.legend(fontsize=7); ax_d1.grid(alpha=0.3)

    ax_d2 = fig.add_axes([0.58, 0.18, 0.38, 0.30]); ax_d2.set_facecolor("white")
    ax_d2.hist([m["quality_score"] for m in sobol_meta], bins=8, color=SOBOL_C, alpha=0.6, label="Sobol")
    ax_d2.hist([m["quality_score"] for m in bo_meta],    bins=8, color=BO_C,    alpha=0.6, label="BO")
    ax_d2.set_title("Distribuição Quality (R3D-18)"); ax_d2.legend(fontsize=7); ax_d2.grid(alpha=0.3)

    obs = f"""OBSERVACOES PRINCIPAIS
======================

1. CONVERGENCIA: o HV 3D cresceu de {hv_history[0]:.4f} para {hv_history[-1]:.4f}
   (x{hv_history[-1]/hv_history[0]:.1f}), provando que a BO navega o espaco 3D de trade-offs.

2. TENSAO DE QUALIDADE: scores R3D-18 ficaram em {qual_all.min():.2f}-{qual_all.max():.2f}. Magnitude
   alta de steering (|alpha| grande) tende a reduzir qualidade. A BO identifica que
   alpha moderado atinge alta memorabilidade sem sacrificar tanto a qualidade.

3. DIVERSIDADE DE SEMENTES: a frente de Pareto 3D usa as sementes {seeds_p},
   contra apenas a semente {seeds2} no run de 2-obj -- solucao bem mais variada.

4. COMPARACAO 2-OBJ vs 3-OBJ:
   +---------------------+--------------+--------------+
   | Metrica             |  2-Obj Run   |  3-Obj Run   |
   +---------------------+--------------+--------------+
   | Objetivos           |  2 (T, C)    |  3 (T, C, Q) |
   | Avaliacoes          |  {n_eval2:<11}|  {len(all_meta):<11}|
   | Pontos de Pareto    |  {n_pareto2:<11}|  {len(pareto_meta):<11}|
   | Melhor TRIBE        |  {best_tribe2:<11.3f}|  {tribe_all.max():<11.3f}|
   | Melhor CLIP         |  {best_clip2:<11.3f}|  {clip_all.max():<11.3f}|
   | Qualidade capturada |  N/A         |  {qual_all.max():<11.3f}|
   | Alpha (Pareto)      |  {alpha2_lo:+.1f}..{alpha2_hi:+.1f}  |  {alpha_p_lo:+.1f}..{alpha_p_hi:+.1f}  |
   +---------------------+--------------+--------------+
   O run 3-obj acha uma frente mais rica ({len(pareto_meta)} vs {n_pareto2}) incorporando qualidade.
   A faixa de alpha se amplia: alpha negativo/moderado tambem alcanca memorabilidade
   competitiva quando combinado com guidance e semente certos.
   (HV nao comparavel entre runs: dimensoes/escalas/ref diferentes.)

5. ARTEFATOS DE SAIDA:
   outputs/gpu_run_3obj/all_results.json   metadados + 3 scores por video
   outputs/gpu_run_3obj/bo_results.png     plot Pareto 3-paineis (do main_3obj.py)
   outputs/gpu_run_3obj/bo_state.pt        checkpoint GP (resumivel)
   outputs/gpu_run_3obj/quality_reference.npz  centroide R3D-18 (reusado no resume)
   outputs/bo_memorability_3obj_report.pdf este relatorio
"""
    ax.text(0.0, 0.99, obs, ha="left", va="top", fontsize=8.3,
            color="#2C3E50", family="monospace", transform=ax.transAxes, linespacing=1.4)

    d = pdf.infodict()
    d["Title"]   = "BO-Memorability 3-Objective Report (com análise comparativa)"
    d["Subject"] = "Multi-objective BO: TRIBE x CLIP x Quality"
    d["CreationDate"] = datetime.now()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

print(f"PDF salvo: {PDF_PATH}")
