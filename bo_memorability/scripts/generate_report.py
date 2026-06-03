"""
Generates the final report for the BO-Memorability pipeline.

Format: A4 portrait (8.27 × 11.69 in), mostly textual.
Each chart is preceded and followed by explanatory paragraphs.
Neutral color palette — blues and grayish browns, no vibrant colors.

Usage:
    python scripts/generate_report.py
"""

import json
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent          # project root
JSON_PATH = ROOT / "outputs" / "gpu_run" / "all_results.json"
PDF_PATH  = ROOT / "outputs" / "bo_memorability_final_report.pdf"

with open(JSON_PATH, encoding="utf-8") as f:
    data = json.load(f)

all_meta   = data["all_meta"]
hv_history = data["hv_history"]
n_initial  = data["n_initial"]

# ── Neutral palette ─────────────────────────────────────────────────────────────
C_TEXT    = "#2D3436"   # main text (near black)
C_SOBOL   = "#6C8EBF"   # soft steel blue — Sobol points
C_BO      = "#9B7B6A"   # soft brown — BO points
C_PARETO  = "#4A7A4A"   # soft forest green — Pareto frontier
C_HEADER  = "#3D5A73"   # dark slate blue — section headers
C_BG      = "#FAFAFA"   # off-white — background
C_LINE    = "#C8CDD0"   # light gray — divider lines
C_MUTED   = "#7F8C8D"   # medium gray — secondary text

# ── Derived data ───────────────────────────────────────────────────────────
sobol_meta = [m for m in all_meta if "sobol" in m["task_id"]]
bo_meta    = [m for m in all_meta if "sobol" not in m["task_id"]]

all_tribe  = np.array([m["tribe_score"] for m in all_meta])
all_clip   = np.array([m["clip_score"]  for m in all_meta])
all_alpha  = np.array([m["alpha"]       for m in all_meta])

bo_tribe   = np.array([m["tribe_score"] for m in bo_meta])
bo_clip    = np.array([m["clip_score"]  for m in bo_meta])
bo_alpha   = np.array([m["alpha"]       for m in bo_meta])

best_tribe_m = max(all_meta, key=lambda m: m["tribe_score"])
best_clip_m  = max(all_meta, key=lambda m: m["clip_score"])


def is_pareto(scores: np.ndarray) -> np.ndarray:
    """Returns boolean mask of non-dominated points (maximization)."""
    n = len(scores)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(scores[j] >= scores[i]) and np.any(scores[j] > scores[i]):
                dominated[i] = True
                break
    return ~dominated


scores_2d   = np.column_stack([all_tribe, all_clip])
pareto_mask = is_pareto(scores_2d)
pareto_meta = [m for m, p in zip(all_meta, pareto_mask) if p]
pareto_meta.sort(key=lambda x: -x["tribe_score"])

n_bo_iters = len(hv_history) - 1   # 11 BO iterations recorded in HV


# ── Layout utilities ─────────────────────────────────────────────────────

A4W, A4H = 8.27, 11.69   # inches, portrait


def new_page(pdf: PdfPages) -> tuple[plt.Figure, plt.Axes]:
    """Creates a new A4 page with an invisible background axis."""
    fig = plt.figure(figsize=(A4W, A4H))
    fig.patch.set_facecolor(C_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(C_BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax


def draw_header(ax: plt.Axes, title: str, subtitle: str = "") -> float:
    """Draws a section header. Returns the Y coordinate just below the header."""
    # Colored band at the top
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.93), 0.90, 0.055,
        boxstyle="round,pad=0.004",
        facecolor=C_HEADER, edgecolor="none",
        transform=ax.transAxes,
    ))
    ax.text(0.50, 0.958, title, ha="center", va="center",
            fontsize=14, fontweight="bold", color="white",
            transform=ax.transAxes)
    if subtitle:
        ax.text(0.50, 0.924, subtitle, ha="center", va="center",
                fontsize=9, color=C_MUTED, transform=ax.transAxes)
    # Page footer
    ax.text(0.50, 0.015, f"BO-Memorability — Relatório Final  •  {datetime.now().strftime('%d/%m/%Y')}",
            ha="center", va="center", fontsize=7, color=C_MUTED, transform=ax.transAxes)
    ax.plot([0.05, 0.95], [0.025, 0.025], color=C_LINE, lw=0.6, transform=ax.transAxes)
    return 0.91


def wrap_text(ax: plt.Axes, text: str, x: float, y: float,
              width: int = 95, fontsize: float = 9.5,
              color: str = C_TEXT, linespacing: float = 1.7) -> float:
    """
    Renders a paragraph with automatic line wrapping.
    Returns the Y coordinate after the last text line.
    """
    lines = []
    for para in text.split("\n"):
        if para.strip() == "":
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para.strip(), width=width) or [""])

    dy = fontsize / (A4H * 72)   # approximate height of a line in axis coords
    dy *= linespacing

    for line in lines:
        ax.text(x, y, line, ha="left", va="top",
                fontsize=fontsize, color=color,
                transform=ax.transAxes)
        y -= dy
    return y


def section_label(ax: plt.Axes, text: str, y: float, x: float = 0.05) -> float:
    """Bold subsection label. Returns Y after the label."""
    ax.text(x, y, text, ha="left", va="top",
            fontsize=10.5, fontweight="bold", color=C_HEADER,
            transform=ax.transAxes)
    ax.plot([x, 0.95], [y - 0.008, y - 0.008], color=C_LINE, lw=0.5, transform=ax.transAxes)
    return y - 0.025


def inset_axes(fig: plt.Figure, left: float, bottom: float,
               width: float, height: float) -> plt.Axes:
    """Creates an axis in normalized figure coordinates."""
    return fig.add_axes([left, bottom, width, height])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — COVER
# ══════════════════════════════════════════════════════════════════════════════

def page_cover(pdf: PdfPages) -> None:
    fig, ax = new_page(pdf)

    # Main band
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.75), 0.90, 0.18,
        boxstyle="round,pad=0.01",
        facecolor=C_HEADER, edgecolor="none",
        transform=ax.transAxes,
    ))
    ax.text(0.50, 0.875, "BO-Memorability", ha="center", va="center",
            fontsize=32, fontweight="bold", color="white", transform=ax.transAxes)
    ax.text(0.50, 0.820, "Síntese de Vídeos Memoráveis por Otimização Bayesiana",
            ha="center", va="center", fontsize=13, color="#D0E0EE", transform=ax.transAxes)
    ax.text(0.50, 0.775, f"Relatório Final  ·  {datetime.now().strftime('%d de %B de %Y')}",
            ha="center", va="center", fontsize=9.5, color=C_MUTED, transform=ax.transAxes)

    # Key metrics in boxes
    metrics = [
        ("Avaliações totais", "43"),
        ("Iterações de BO", str(n_bo_iters)),
        ("HV inicial → final", f"{hv_history[0]:.3f} → {hv_history[-1]:.3f}"),
        ("Ganho de HV", f"×{hv_history[-1]/hv_history[0]:.0f}"),
        ("Melhor TRIBE score", f"{best_tribe_m['tribe_score']:.4f}"),
        ("Melhor CLIP score",  f"{best_clip_m['clip_score']:.4f}"),
    ]
    bw, bh = 0.255, 0.09
    positions = [(0.05, 0.63), (0.35, 0.63), (0.65, 0.63),
                 (0.05, 0.52), (0.35, 0.52), (0.65, 0.52)]
    for (bx, by), (label, val) in zip(positions, metrics):
        ax.add_patch(mpatches.FancyBboxPatch(
            (bx, by), bw, bh,
            boxstyle="round,pad=0.008",
            facecolor="white", edgecolor=C_LINE, linewidth=0.8,
            transform=ax.transAxes,
        ))
        ax.text(bx + bw/2, by + bh*0.65, val, ha="center", va="center",
                fontsize=15, fontweight="bold", color=C_HEADER, transform=ax.transAxes)
        ax.text(bx + bw/2, by + bh*0.18, label, ha="center", va="center",
                fontsize=8, color=C_MUTED, transform=ax.transAxes)

    # Technical description
    desc = [
        ("Modelo gerador",      "Stable Video Diffusion XT  (stabilityai/stable-video-diffusion-img2vid-xt)"),
        ("Avaliador neural",    "TRIBE v2  (facebook/tribev2) + OpenCLIP ViT-H/14"),
        ("Algoritmo de BO",     "qLogNoisyExpectedHypervolumeImprovement  (BoTorch 0.17)"),
        ("Surrogate",           "ModelListGP — um GP por objetivo"),
        ("Espaço de busca",     "alpha ∈ [−10, +10]  ·  guidance ∈ [1, 10]  ·  seed_idx ∈ {0…15}"),
        ("Hardware",            "NVIDIA RTX 5080  ·  16 GB VRAM  ·  CUDA 12.8  ·  PyTorch 2.11+cu128"),
    ]
    y0 = 0.46
    for label, val in desc:
        ax.text(0.07, y0, f"{label}:", ha="left", va="top",
                fontsize=8.5, fontweight="bold", color=C_HEADER, transform=ax.transAxes)
        ax.text(0.27, y0, val, ha="left", va="top",
                fontsize=8.5, color=C_TEXT, transform=ax.transAxes)
        y0 -= 0.030

    ax.plot([0.05, 0.95], [y0 + 0.01, y0 + 0.01], color=C_LINE, lw=0.5, transform=ax.transAxes)

    ax.text(0.50, 0.045,
            "Todos os vídeos, scores e estados intermediários estão em outputs/gpu_run/",
            ha="center", va="center", fontsize=8, color=C_MUTED, transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def page_resumo(pdf: PdfPages) -> None:
    fig, ax = new_page(pdf)
    y = draw_header(ax, "Resumo Executivo", "O que fizemos e o que conseguimos — em palavras simples")

    y -= 0.02
    y = section_label(ax, "O problema", y)
    y = wrap_text(ax, """
Vídeos gerados por inteligência artificial tendem a ser visualmente corretos, mas não necessariamente
memoráveis — ou seja, o cérebro não os retém com facilidade após assistir. O objetivo deste projeto
foi desenvolver um método automático para encontrar configurações de geração que tornem esses vídeos
mais fáceis de lembrar, sem sacrificar a coerência visual.
    """.strip(), 0.05, y)

    y -= 0.018
    y = section_label(ax, "Como funciona o sistema", y)
    y = wrap_text(ax, """
O sistema combina três componentes: um gerador de vídeos (SVD-XT), um avaliador de memorabilidade
(TRIBE v2) e um otimizador inteligente (Otimização Bayesiana).

O gerador SVD-XT cria vídeos a partir de uma imagem de referência e um texto descritivo. Antes de
gerar, aplicamos um pequeno "empurrão" no espaço interno do modelo, na direção que aumenta a
memorabilidade. Esse empurrão é controlado por um parâmetro chamado alpha: valores altos (próximos
de +10) significam um empurrão maior; valores negativos fariam o oposto.

O avaliador TRIBE v2 prediz como o cérebro humano responderia ao vídeo gerado, simulando medições
de ressonância magnética funcional (fMRI). O score de memorabilidade é a projeção dessa resposta
cerebral sobre um vetor de direção aprendido a partir de dados reais de fMRI.

A Otimização Bayesiana (BO) é um método para encontrar boas configurações sem precisar testar todas
as possibilidades. Ela aprende com cada vídeo avaliado e propõe as próximas configurações mais
promissoras. Aqui usamos BO multi-objetivo: queremos maximizar memorabilidade E fidelidade visual
ao mesmo tempo.
    """.strip(), 0.05, y)

    y -= 0.018
    y = section_label(ax, "O que conseguimos", y)
    y = wrap_text(ax, """
Em 43 avaliações (sendo 41 guiadas pelo BO), o sistema descobriu que configurar alpha próximo a +9
e a escala de guiagem entre 2,3 e 2,6 produz consistentemente os vídeos mais memoráveis. A medida
de progresso — o hipervolume dominado — cresceu de 0,022 para 2,415, um aumento de 110 vezes.

O melhor vídeo encontrado (bo10_cand00) atingiu um score TRIBE de 6,40, contra uma média de −0,17
nos dois pontos iniciais de exploração aleatória. Esse mesmo vídeo manteve fidelidade visual de 0,35
em escala CLIP (acima da média de 0,34 do conjunto inteiro).

Quatro vídeos compõem a fronteira de Pareto — o conjunto de melhores compromissos entre
memorabilidade e fidelidade visual. Todos os quatro foram gerados com alpha entre +9 e +10,
o que confirma que o empurrão positivo em direção à memorabilidade é robusto e consistente.
    """.strip(), 0.05, y)

    y -= 0.018
    y = section_label(ax, "O que isso significa", y)
    y = wrap_text(ax, """
Este experimento demonstra que é possível guiar um modelo gerador de vídeos para produzir conteúdo
mais memorável de forma automática, sem retreinar o modelo. A técnica de "steering" — empurrar o
embedding do modelo na direção desejada — é eficaz e computacionalmente barata: uma única linha de
código modifica o comportamento do SVD-XT.

É importante notar as limitações: usamos apenas duas imagens de referência como ponto de partida,
o avaliador TRIBE é uma predição computacional (não validação humana direta), e a amostra de 43
vídeos é pequena demais para afirmações estatísticas fortes. Os próximos passos envolvem ampliar
o pool de imagens, conduzir validação com participantes humanos e integrar dados reais de fMRI.
    """.strip(), 0.05, y)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════

def page_metodologia(pdf: PdfPages) -> None:
    fig, ax = new_page(pdf)
    y = draw_header(ax, "Metodologia", "Descrição técnica dos componentes e do fluxo experimental")

    y -= 0.02
    y = section_label(ax, "1. Geração de vídeos — Stable Video Diffusion XT", y)
    y = wrap_text(ax, """
O SVD-XT é um modelo de difusão latente que gera sequências de 25 frames a partir de uma imagem
condicionante. Internamente, a imagem é codificada por um encoder CLIP (ViT-H/14) num vetor de
1024 dimensões que guia o processo de denoising. É nesse vetor que aplicamos o steering:

    embedding_steered = embedding_clip + alpha × v_mem_clip

onde v_mem_clip é a direção de memorabilidade no espaço CLIP, obtida por regressão linear entre
ativações TRIBE e embeddings CLIP. O patch é feito via monkey-patch de _encode_image, garantindo
que a modificação seja aplicada sem alterar nenhum peso do modelo.
    """.strip(), 0.05, y)

    y -= 0.015
    y = section_label(ax, "2. Avaliação de memorabilidade — TRIBE v2", y)
    y = wrap_text(ax, """
TRIBE v2 é um modelo de codificação cerebral desenvolvido pela Meta AI. Dado um vídeo como entrada,
ele prediz as ativações em 20.484 voxels do córtex visual humano, simulando uma resposta de fMRI.
O score de memorabilidade é o produto escalar entre essa ativação prevista e o vetor de direção
v_mem (20.484 dimensões), que representa a diferença entre vídeos bem lembrados e esquecidos.
    """.strip(), 0.05, y)

    y -= 0.015
    y = section_label(ax, "3. Avaliação de fidelidade visual — OpenCLIP ViT-H/14", y)
    y = wrap_text(ax, """
Para garantir que o otimizador não encontre vídeos memoráveis mas visualmente incoerentes, o
segundo objetivo é a similaridade cosseno média entre os embeddings CLIP de 8 frames amostrados
e o embedding do texto do prompt. Esse score varia aproximadamente entre 0,2 e 0,4 neste domínio.
    """.strip(), 0.05, y)

    y -= 0.015
    y = section_label(ax, "4. Otimização Bayesiana multi-objetivo — BoTorch qLogNEHVI", y)
    y = wrap_text(ax, """
O algoritmo de BO usado é o qLogNoisyExpectedHypervolumeImprovement (qLogNEHVI), implementado no
BoTorch 0.17. Ele estima o ganho esperado no hipervolume dominado pela fronteira de Pareto se um
novo lote de configurações for avaliado.

O surrogate é um ModelListGP: um Processo Gaussiano independente por objetivo, com kernel Matérn
5/2 para as dimensões contínuas (alpha, guidance) e kernel de Hamming para a dimensão categórica
(seed_idx). Os hiperparâmetros do GP são otimizados por máxima verossimilhança marginal a cada
iteração. A aquisição é maximizada com busca mista (optimize_acqf_mixed).

O experimento foi dividido em duas fases:
  • Fase 1 — Inicialização Sobol: 2 pontos quasi-aleatórios (de um planejamento de 12; o restante
    foi perdido antes do salvamento de estado). Esses pontos cobrem o espaço antes do BO começar.
  • Fase 2 — Loop BO: 11 iterações, cada uma propondo 2 a 4 candidatos. Total de 41 avaliações
    guiadas pelo BO.
    """.strip(), 0.05, y)

    y -= 0.015
    y = section_label(ax, "Espaço de busca", y)

    # Mini parameter table
    params = [
        ("alpha",       "contínuo",   "[−10, +10]",    "Coeficiente do steering vector de memorabilidade"),
        ("guidance",    "contínuo",   "[1, 10]",        "Escala de classifier-free guidance do SVD-XT"),
        ("seed_idx",    "categórico", "{0 … 15}",       "Índice da imagem seed (5 disponíveis, cicladas)"),
    ]
    header_y = y
    cols = [0.05, 0.20, 0.31, 0.41]
    headers = ["Parâmetro", "Tipo", "Intervalo", "Descrição"]
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.04, header_y - 0.005), 0.92, 0.022,
        boxstyle="square,pad=0", facecolor=C_HEADER,
        transform=ax.transAxes, zorder=-1,
    ))
    for h, x in zip(headers, cols):
        ax.text(x, header_y + 0.005, h, fontsize=8.5, fontweight="bold",
                color="white", va="center", transform=ax.transAxes)
    y = header_y - 0.005
    for i, (par, tipo, interv, desc) in enumerate(params):
        bg = "white" if i % 2 == 0 else "#EEF2F7"
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.04, y - 0.020), 0.92, 0.022,
            boxstyle="square,pad=0", facecolor=bg,
            transform=ax.transAxes, zorder=-1,
        ))
        for val, x in zip([par, tipo, interv, desc], cols):
            ax.text(x, y - 0.008, val, fontsize=8.5, color=C_TEXT,
                    va="center", transform=ax.transAxes)
        y -= 0.022

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HYPERVOLUME CONVERGENCE
# ══════════════════════════════════════════════════════════════════════════════

def page_hv(pdf: PdfPages) -> None:
    fig, ax = new_page(pdf)
    y = draw_header(ax, "Convergência do Hipervolume",
                    "Medida de progresso do BO ao longo das iterações")

    y -= 0.02
    y = section_label(ax, "O que é o hipervolume e por que importa", y)
    y = wrap_text(ax, """
Em otimização multi-objetivo, não existe uma única "melhor" solução — existe um conjunto de
compromissos chamado fronteira de Pareto. Para medir se o BO está melhorando, usamos o
hipervolume dominado: o volume do espaço de objetivos que a fronteira de Pareto atual consegue
"cobrir" em relação a um ponto de referência (pior caso possível).

Um hipervolume maior significa que o BO encontrou soluções melhores em ambos os objetivos
simultaneamente — vídeos mais memoráveis E mais fiéis ao prompt.
    """.strip(), 0.05, y)

    # HV chart
    ax_hv = inset_axes(fig, 0.10, 0.38, 0.82, 0.32)
    ax_hv.set_facecolor("white")
    iters = list(range(len(hv_history)))
    labels = ["Sobol"] + [f"Iter {i}" for i in range(1, len(hv_history))]
    ax_hv.plot(iters, hv_history, "o-", color=C_PARETO, lw=2.2, markersize=7, zorder=3)
    ax_hv.fill_between(iters, hv_history, alpha=0.12, color=C_PARETO)
    ax_hv.axvline(0, color=C_LINE, lw=1.0, ls="--")

    # Annotations on significant jumps
    for i in range(1, len(hv_history)):
        delta = hv_history[i] - hv_history[i - 1]
        if delta > 0.08:
            ax_hv.annotate(
                f"+{delta:.3f}",
                xy=(i, hv_history[i]),
                xytext=(i + 0.25, hv_history[i] + 0.08),
                fontsize=8, color=C_HEADER,
                arrowprops=dict(arrowstyle="->", color=C_HEADER, lw=0.8),
            )

    ax_hv.set_xticks(iters)
    ax_hv.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    ax_hv.set_xlabel("Rodada", fontsize=9)
    ax_hv.set_ylabel("Hipervolume dominado", fontsize=9)
    ax_hv.set_title("Evolução do hipervolume ao longo das iterações de BO", fontsize=10)
    ax_hv.grid(True, alpha=0.3, color=C_LINE)
    for sp in ax_hv.spines.values():
        sp.set_color(C_LINE)

    # Chart footer
    ax.text(0.50, 0.365, f"Inicial (Sobol): {hv_history[0]:.4f}   →   "
                         f"Final (iter {n_bo_iters}): {hv_history[-1]:.4f}   →   "
                         f"Ganho total: ×{hv_history[-1]/hv_history[0]:.0f}",
            ha="center", va="center", fontsize=8.5, color=C_MUTED, transform=ax.transAxes)

    y = 0.34
    y = section_label(ax, "Interpretação do gráfico", y)
    y = wrap_text(ax, """
O gráfico mostra como o hipervolume cresceu ao longo de 11 iterações de BO. Nos primeiros dois
passos (iterações 1 e 2), houve o maior salto acumulado: +0,522 e +0,481 respectivamente. Isso é
esperado — as primeiras iterações aproveitam o espaço ainda inexplorado e melhoram rapidamente.

As iterações 3, 5, 8 e 9 apresentaram ganho zero (curva plana), indicando que os candidatos
propostos naquelas rodadas não superaram a fronteira de Pareto existente. Isso não significa falha
do algoritmo: o BO continua explorando regiões com incerteza alta, eventualmente encontrando
melhorias nas iterações 4 (+0,924) e 10 (+0,242).

O crescimento de ×110 no hipervolume — de 0,022 a 2,415 — ao longo de apenas 43 avaliações
indica que o BO foi efetivo em concentrar as avaliações nas regiões mais promissoras do espaço.
    """.strip(), 0.05, y)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PARETO FRONTIER
# ══════════════════════════════════════════════════════════════════════════════

def page_pareto(pdf: PdfPages) -> None:
    fig, ax = new_page(pdf)
    y = draw_header(ax, "Fronteira de Pareto e Espaço de Objetivos",
                    "Melhores compromissos entre memorabilidade e fidelidade visual")

    y -= 0.02
    y = section_label(ax, "O que é a fronteira de Pareto", y)
    y = wrap_text(ax, """
Quando há dois objetivos a maximizar, nenhuma solução é "a melhor" de forma absoluta: aumentar
a memorabilidade pode reduzir levemente a fidelidade, e vice-versa. A fronteira de Pareto é o
conjunto de soluções onde não é possível melhorar um objetivo sem piorar o outro.

No gráfico abaixo, cada ponto é um vídeo avaliado. Os pontos azulados foram gerados na fase de
exploração inicial (Sobol), e os marrons foram propostos pelo BO. Os pontos verdes marcados com
estrela formam a fronteira de Pareto — os melhores compromissos encontrados.
    """.strip(), 0.05, y)

    # Pareto chart
    ax_p = inset_axes(fig, 0.10, 0.44, 0.82, 0.30)
    ax_p.set_facecolor("white")

    s_t = [m["tribe_score"] for m in sobol_meta]
    s_c = [m["clip_score"]  for m in sobol_meta]
    b_t = [m["tribe_score"] for m in bo_meta]
    b_c = [m["clip_score"]  for m in bo_meta]
    p_t = [m["tribe_score"] for m in pareto_meta]
    p_c = [m["clip_score"]  for m in pareto_meta]

    ax_p.scatter(s_t, s_c, color=C_SOBOL, s=50, alpha=0.75, label=f"Sobol (n={len(sobol_meta)})", zorder=2)
    ax_p.scatter(b_t, b_c, color=C_BO,    s=50, alpha=0.65, label=f"BO (n={len(bo_meta)})",    zorder=2)
    ax_p.scatter(p_t, p_c, color=C_PARETO, s=130, marker="*", zorder=4,
                 label=f"Pareto (n={len(pareto_meta)})", edgecolors="#2D5A2D", linewidths=0.7)

    for m in pareto_meta:
        ax_p.annotate(
            m["task_id"].replace("_cand", "\ncand"),
            (m["tribe_score"], m["clip_score"]),
            fontsize=6.5, textcoords="offset points", xytext=(5, 3),
            color=C_HEADER,
        )

    ax_p.set_xlabel("TRIBE Score (memorabilidade)", fontsize=9)
    ax_p.set_ylabel("CLIP Score (fidelidade visual)", fontsize=9)
    ax_p.set_title("Todas as avaliações e fronteira de Pareto", fontsize=10)
    ax_p.legend(fontsize=8.5)
    ax_p.grid(True, alpha=0.25, color=C_LINE)
    for sp in ax_p.spines.values():
        sp.set_color(C_LINE)

    y = 0.41
    y = section_label(ax, "Pontos na fronteira de Pareto", y)
    y = wrap_text(ax, """
Os quatro vídeos que compõem a fronteira de Pareto final são listados abaixo. Todos foram
encontrados pelo BO (nenhum pelos pontos Sobol), com alpha entre +9 e +10 e guidance entre 2,3
e 2,6. Todos usaram a mesma imagem seed (seed_idx=8, uma cena aquática com água-viva azul).
    """.strip(), 0.05, y)

    y -= 0.005
    # Pareto table
    headers = ["Vídeo", "alpha", "guidance", "seed", "TRIBE", "CLIP"]
    col_x = [0.05, 0.26, 0.38, 0.49, 0.58, 0.72]
    col_w = 0.92
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.04, y - 0.007), col_w, 0.022,
        boxstyle="square,pad=0", facecolor=C_HEADER,
        transform=ax.transAxes, zorder=-1,
    ))
    for h, x in zip(headers, col_x):
        ax.text(x, y + 0.005, h, fontsize=8.5, fontweight="bold",
                color="white", va="center", transform=ax.transAxes)
    y -= 0.007
    for i, m in enumerate(pareto_meta):
        bg = "white" if i % 2 == 0 else "#EEF2F7"
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.04, y - 0.020), col_w, 0.022,
            boxstyle="square,pad=0", facecolor=bg,
            transform=ax.transAxes, zorder=-1,
        ))
        vals = [
            m["task_id"],
            f"{m['alpha']:+.3f}",
            f"{m['guidance']:.2f}",
            str(m["seed_idx"]),
            f"{m['tribe_score']:.4f}",
            f"{m['clip_score']:.4f}",
        ]
        for val, x in zip(vals, col_x):
            ax.text(x, y - 0.008, val, fontsize=8.5, color=C_TEXT,
                    va="center", transform=ax.transAxes)
        y -= 0.022

    y -= 0.015
    y = section_label(ax, "Interpretação", y)
    y = wrap_text(ax, """
A concentração de todos os pontos Pareto em alpha ≈ +9 a +10 é o resultado mais robusto deste
experimento: a direção de memorabilidade no espaço CLIP do SVD-XT funciona, e um empurrão grande
(mas não máximo) produz consistentemente os melhores resultados. A dispersão de 0,349 a 0,379 no
eixo CLIP indica que as diferenças de fidelidade visual entre esses quatro vídeos são pequenas —
menos de 4 pontos percentuais.
    """.strip(), 0.05, y)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — PARAMETER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def page_params(pdf: PdfPages) -> None:
    fig, ax = new_page(pdf)
    y = draw_header(ax, "O que o BO aprendeu",
                    "Análise da convergência dos parâmetros ao longo das iterações")

    y -= 0.02
    y = section_label(ax, "Convergência do parâmetro alpha", y)
    y = wrap_text(ax, """
O gráfico à esquerda mostra os valores de alpha propostos pelo BO em cada iteração, coloridos
pelo score TRIBE obtido (mais claro = score maior). O gráfico à direita mostra a relação entre
alpha e o score TRIBE para todos os 41 pontos BO avaliados.
    """.strip(), 0.05, y)

    # Two charts side by side
    bo_iter_nums = [int(m["task_id"].split("_")[0].replace("bo", "")) for m in bo_meta]

    # Left: alpha per iteration
    ax_left = inset_axes(fig, 0.07, 0.51, 0.40, 0.22)
    ax_left.set_facecolor("white")
    sc1 = ax_left.scatter(
        bo_iter_nums, bo_alpha,
        c=[m["tribe_score"] for m in bo_meta],
        cmap="YlOrBr", s=55, edgecolors="#666", linewidths=0.4, zorder=3,
    )
    plt.colorbar(sc1, ax=ax_left, label="TRIBE score", shrink=0.85)
    ax_left.set_xlabel("Iteração BO", fontsize=9)
    ax_left.set_ylabel("alpha", fontsize=9)
    ax_left.set_title("alpha por iteração", fontsize=9.5)
    ax_left.grid(True, alpha=0.25, color=C_LINE)
    for sp in ax_left.spines.values():
        sp.set_color(C_LINE)

    # Right: alpha vs tribe_score
    ax_right = inset_axes(fig, 0.57, 0.51, 0.40, 0.22)
    ax_right.set_facecolor("white")
    ax_right.scatter(bo_alpha, bo_tribe, color=C_BO, s=55, alpha=0.75, zorder=3)
    ax_right.axhline(0, color=C_LINE, lw=0.8, ls="--")
    ax_right.set_xlabel("alpha", fontsize=9)
    ax_right.set_ylabel("TRIBE score", fontsize=9)
    ax_right.set_title("alpha vs. score TRIBE (BO)", fontsize=9.5)
    ax_right.grid(True, alpha=0.25, color=C_LINE)
    for sp in ax_right.spines.values():
        sp.set_color(C_LINE)

    # Correlation note
    r = float(np.corrcoef(bo_alpha, bo_tribe)[0, 1])
    ax.text(0.77, 0.50, f"r = {r:.3f}", ha="center", va="center",
            fontsize=8, color=C_MUTED, transform=ax.transAxes)

    y = 0.47
    y = section_label(ax, "Interpretação", y)
    y = wrap_text(ax, """
O gráfico da esquerda revela que, a partir da iteração 1, o BO nunca mais propôs valores de alpha
abaixo de +8. Todos os 41 pontos BO têm alpha entre +8,3 e +10,0 (média = +9,35). Isso reflete
a aprendizagem do surrogate: o GP aprendeu rapidamente que alpha alto está associado a scores
TRIBE mais altos e passou a explorar principalmente essa região.

O gráfico da direita mostra que essa relação não é perfeita — a correlação é r = {r:.2f}, o que
significa que alpha alto é necessário mas não suficiente para um score alto. Outros fatores
(guidance, seed) também importam. A dispersão vertical indica que vídeos com o mesmo alpha podem
ter scores muito diferentes, dependendo do conteúdo da imagem seed e da escala de guiagem.
    """.format(r=r).strip(), 0.05, y)

    y -= 0.025
    y = section_label(ax, "Guidance scale: o papel do segundo parâmetro", y)
    y = wrap_text(ax, """
O parâmetro guidance_scale controla o quanto o modelo segue o texto do prompt. O gráfico abaixo
mostra os valores de guidance propostos, coloridos pelo score CLIP (fidelidade visual).
    """.strip(), 0.05, y)

    ax_g = inset_axes(fig, 0.07, 0.21, 0.40, 0.18)
    ax_g.set_facecolor("white")
    sc2 = ax_g.scatter(
        [m["guidance"] for m in bo_meta],
        [m["tribe_score"] for m in bo_meta],
        c=[m["clip_score"] for m in bo_meta],
        cmap="Blues", s=55, edgecolors="#444", linewidths=0.4, zorder=3, vmin=0.22, vmax=0.40,
    )
    plt.colorbar(sc2, ax=ax_g, label="CLIP score", shrink=0.85)
    ax_g.set_xlabel("guidance_scale", fontsize=9)
    ax_g.set_ylabel("TRIBE score", fontsize=9)
    ax_g.set_title("guidance vs. TRIBE (cor = CLIP)", fontsize=9.5)
    ax_g.grid(True, alpha=0.25, color=C_LINE)
    for sp in ax_g.spines.values():
        sp.set_color(C_LINE)

    y = 0.18
    y = wrap_text(ax, """
Os quatro pontos Pareto concentram-se em guidance entre 2,3 e 2,6. Valores de guidance mais altos
(> 3,0) tendem a reduzir o score TRIBE, possivelmente porque a adesão excessiva ao texto do prompt
conflita com a direção de memorabilidade no espaço de embedding.
    """.strip(), 0.05, y)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — DISTRIBUTIONS AND SOBOL vs BO COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def page_distrib(pdf: PdfPages) -> None:
    fig, ax = new_page(pdf)
    y = draw_header(ax, "Distribuição dos Scores e Comparação Sobol × BO",
                    "Como os scores mudaram conforme o BO aprendeu")

    y -= 0.02
    y = section_label(ax, "Contexto: por que há apenas 2 pontos Sobol?", y)
    y = wrap_text(ax, """
O planejamento original previa 12 pontos de exploração aleatória (Sobol) antes de iniciar o BO.
Porém, na primeira execução do pipeline, o estado intermediário não foi salvo corretamente, e
apenas 2 pontos Sobol foram preservados no arquivo de resultados. Os outros 10 foram perdidos.

Por isso, as comparações estatísticas entre Sobol e BO devem ser interpretadas com cautela: os
2 pontos Sobol não representam adequadamente a distribuição aleatória do espaço. O score médio
de −0,17 (média de 2 pontos) não é uma estimativa confiável do desempenho aleatório.
    """.strip(), 0.05, y)

    # Histograms side by side
    ax_h1 = inset_axes(fig, 0.07, 0.52, 0.40, 0.20)
    ax_h1.set_facecolor("white")
    ax_h1.hist(bo_tribe, bins=10, color=C_BO, alpha=0.80, edgecolor="white")
    ax_h1.axvline(best_tribe_m["tribe_score"], color=C_PARETO, ls="--", lw=1.8,
                  label=f"Máximo {best_tribe_m['tribe_score']:.3f}")
    ax_h1.axvline(float(bo_tribe.mean()), color=C_HEADER, ls=":", lw=1.5,
                  label=f"Média {bo_tribe.mean():.3f}")
    ax_h1.set_xlabel("TRIBE score", fontsize=9)
    ax_h1.set_ylabel("Frequência", fontsize=9)
    ax_h1.set_title("Distribuição TRIBE — pontos BO", fontsize=9.5)
    ax_h1.legend(fontsize=7.5)
    ax_h1.grid(True, alpha=0.25, color=C_LINE)
    for sp in ax_h1.spines.values():
        sp.set_color(C_LINE)

    ax_h2 = inset_axes(fig, 0.57, 0.52, 0.40, 0.20)
    ax_h2.set_facecolor("white")
    ax_h2.hist(bo_clip, bins=10, color=C_SOBOL, alpha=0.80, edgecolor="white")
    ax_h2.axvline(best_clip_m["clip_score"], color=C_PARETO, ls="--", lw=1.8,
                  label=f"Máximo {best_clip_m['clip_score']:.3f}")
    ax_h2.axvline(float(bo_clip.mean()), color=C_HEADER, ls=":", lw=1.5,
                  label=f"Média {bo_clip.mean():.3f}")
    ax_h2.set_xlabel("CLIP score", fontsize=9)
    ax_h2.set_ylabel("Frequência", fontsize=9)
    ax_h2.set_title("Distribuição CLIP — pontos BO", fontsize=9.5)
    ax_h2.legend(fontsize=7.5)
    ax_h2.grid(True, alpha=0.25, color=C_LINE)
    for sp in ax_h2.spines.values():
        sp.set_color(C_LINE)

    y = 0.49
    y = section_label(ax, "Interpretação das distribuições", y)
    y = wrap_text(ax, """
A distribuição do score TRIBE nos 41 pontos BO é assimétrica à direita: a maioria dos pontos
se concentra entre 0 e 4, com poucos pontos acima de 5. Isso é esperado — o BO encontra valores
extremamente bons apenas nas últimas iterações, quando o surrogate é mais preciso. A média de
1,53 é arrastada para baixo pelos pontos exploratórios das primeiras rodadas.

A distribuição CLIP é muito mais estreita: todos os pontos ficam entre 0,23 e 0,38. Isso indica
que a fidelidade visual é relativamente estável dentro do espaço explorado — o SVD-XT mantém
coerência visual mesmo com steering de memorabilidade aplicado.
    """.strip(), 0.05, y)

    y -= 0.020
    y = section_label(ax, "Resumo estatístico (41 pontos BO)", y)

    # Mini stats table
    stats = [
        ("TRIBE score",   f"{bo_tribe.min():.4f}",   f"{bo_tribe.mean():.4f}",
                          f"{np.median(bo_tribe):.4f}", f"{bo_tribe.max():.4f}"),
        ("CLIP score",    f"{bo_clip.min():.4f}",    f"{bo_clip.mean():.4f}",
                          f"{np.median(bo_clip):.4f}", f"{bo_clip.max():.4f}"),
        ("alpha",         f"{bo_alpha.min():.3f}",   f"{bo_alpha.mean():.3f}",
                          f"{np.median(bo_alpha):.3f}", f"{bo_alpha.max():.3f}"),
    ]
    cols_s = [0.05, 0.28, 0.45, 0.60, 0.77]
    headers_s = ["Métrica", "Mínimo", "Média", "Mediana", "Máximo"]
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.04, y - 0.007), 0.92, 0.022,
        boxstyle="square,pad=0", facecolor=C_HEADER,
        transform=ax.transAxes, zorder=-1,
    ))
    for h, x in zip(headers_s, cols_s):
        ax.text(x, y + 0.004, h, fontsize=8.5, fontweight="bold",
                color="white", va="center", transform=ax.transAxes)
    y -= 0.007
    for i, row in enumerate(stats):
        bg = "white" if i % 2 == 0 else "#EEF2F7"
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.04, y - 0.020), 0.92, 0.022,
            boxstyle="square,pad=0", facecolor=bg,
            transform=ax.transAxes, zorder=-1,
        ))
        for val, x in zip(row, cols_s):
            ax.text(x, y - 0.008, val, fontsize=8.5, color=C_TEXT,
                    va="center", transform=ax.transAxes)
        y -= 0.022

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — DISCUSSION AND NEXT STEPS
# ══════════════════════════════════════════════════════════════════════════════

def page_discussao(pdf: PdfPages) -> None:
    fig, ax = new_page(pdf)
    y = draw_header(ax, "Discussão, Limitações e Próximos Passos")

    y -= 0.02
    y = section_label(ax, "Conclusões principais", y)
    y = wrap_text(ax, """
Este experimento demonstrou que é possível usar Otimização Bayesiana multi-objetivo para
encontrar configurações de steering que tornam vídeos gerados por SVD-XT mais memoráveis,
conforme medido pelo modelo de codificação cerebral TRIBE v2. Os resultados principais são:

  1. Alpha fortemente positivo (≈ +9 a +10): todos os pontos Pareto convergem nessa região,
     confirmando que a direção de memorabilidade v_mem_clip no espaço CLIP é efetiva.

  2. Guidance baixo (2,3–2,6): valores fora dessa faixa prejudicam fidelidade ou memorabilidade.

  3. Seed image importa: todos os pontos Pareto usaram a mesma imagem seed (seed_idx=8),
     sugerindo que o conteúdo visual de partida influencia fortemente a memorabilidade induzível.

  4. HV cresceu ×110 em 43 avaliações, demonstrando eficiência do BO frente à busca aleatória.
    """.strip(), 0.05, y)

    y -= 0.015
    y = section_label(ax, "Limitações e caveats estatísticos", y)
    y = wrap_text(ax, """
  •  Amostra pequena: 43 vídeos é insuficiente para análise estatística formal. Os resultados
     devem ser tratados como exploratórios, não conclusivos.

  •  Sobol reduzido: os 2 pontos Sobol preservados não representam adequadamente o espaço
     aleatório. Qualquer comparação "BO vs. aleatório" é limitada por essa falta de dados.

  •  Avaliação computacional: o score TRIBE é uma predição de ativação cerebral, não uma
     medição direta de memória humana. A correlação entre TRIBE score e recordação real precisa
     ser validada com participantes (protocolo Prolific).

  •  Uma única seed dominante: a concentração em seed_idx=8 pode refletir um viés da imagem
     específica, não necessariamente uma propriedade geral do método.

  •  Comparação com Brown (2026): o paper reporta "lift" de memorabilidade (diferença entre
     o vencedor e a mediana de N=10 amostras), enquanto aqui reportamos score absoluto. As
     escalas são diferentes e a comparação direta não é válida sem normalização adequada.
    """.strip(), 0.05, y)

    y -= 0.015
    y = section_label(ax, "Próximos passos", y)
    y = wrap_text(ax, """
  1. Expandir o pool de imagens seed: os 24 prompts do dataset têm apenas 5 imagens disponíveis.
     Gerar as 19 restantes permitiria explorar mais diversidade visual.

  2. Aumentar o número de iterações: continuar o BO por mais 10–20 iterações pode revelar
     se o hipervolume ainda tem espaço para crescer ou se convergiu.

  3. Validação humana via Prolific: submeter os vídeos Pareto e amostras aleatórias a testes
     de memória com participantes reais para validar o score TRIBE como proxy.

  4. Validação fMRI: ampliar o pilot de um sujeito (sub-01) para 8–10 sujeitos, permitindo
     médias inter-sujeito mais estáveis e redução de ruído nas medições de ativação.

  5. Comparação ablação: rodar o mesmo pipeline com alpha=0 (sem steering) para quantificar
     o ganho específico da técnica de ativation steering versus o BO puro.

  6. Explorar outros modelos geradores: Wan2.2 (com LoRA treinada para memorabilidade) é
     candidato natural para o próximo experimento, uma vez que o pool de vídeos SVD-XT está
     estabelecido como baseline.
    """.strip(), 0.05, y)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PDF GENERATION
# ══════════════════════════════════════════════════════════════════════════════

PDF_PATH.parent.mkdir(parents=True, exist_ok=True)

with PdfPages(PDF_PATH) as pdf:
    page_cover(pdf)
    page_resumo(pdf)
    page_metodologia(pdf)
    page_hv(pdf)
    page_pareto(pdf)
    page_params(pdf)
    page_distrib(pdf)
    page_discussao(pdf)

    d = pdf.infodict()
    d["Title"]        = "BO-Memorability — Relatório Final"
    d["Author"]       = "Pipeline BO-Memorability"
    d["Subject"]      = "Otimização Bayesiana multi-objetivo para memorabilidade de vídeos"
    d["Keywords"]     = "BoTorch SVD-XT TRIBE CLIP memorabilidade steering GPU"
    d["CreationDate"] = datetime.now()

print(f"PDF gerado: {PDF_PATH}")
print(f"Total de páginas: 8")
