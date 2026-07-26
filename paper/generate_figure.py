import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.linewidth": 0.8})
fig = plt.figure(figsize=(8.5, 10), facecolor="white")
C1, C2, C3 = "#4472C4", "#ED7D31", "#70AD47"

def draw_box(ax, x, y, w, h, label, color="#4472C4", lw=1.5, alpha=0.15):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.05", facecolor=color, edgecolor="black", linewidth=lw, alpha=alpha)
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=8, fontweight="bold")

def draw_arrow(ax, x1, y1, x2, y2, lw=1.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="gray", lw=lw, shrinkA=3, shrinkB=3))

# Panel A
ax1 = fig.add_axes([0.04, 0.60, 0.93, 0.38])
ax1.set_xlim(0, 10); ax1.set_ylim(0, 6); ax1.axis("off")
ax1.text(0.05, 5.7, "a  Overall Architecture", fontsize=11, fontweight="bold")

for x, y, w, h, lbl, c in [
    [0.4, 3.5, 0.6, 0.6, "Input(x)", "white"],
    [0.4, 2.2, 0.6, 0.6, "Pert(p)", "white"],
]:
    draw_box(ax1, x, y, w, h, lbl, c, 1, 0)

ax1.add_patch(FancyBboxPatch([1.5, 1.5], 1.8, 3.5, boxstyle="round,pad=0.08", facecolor=C1, edgecolor=C1, linewidth=2, alpha=0.08))
ax1.text(2.4, 4.6, "1) Embedding Fusion", fontsize=8, fontweight="bold", color=C1, ha="center")
draw_box(ax1, 2.4, 3.2, 0.8, 0.6, "KGE+BioLLM", C1, 0)

ax1.add_patch(FancyBboxPatch([4.8, 1.5], 1.8, 3.5, boxstyle="round,pad=0.08", facecolor=C2, edgecolor=C2, linewidth=2, alpha=0.08))
ax1.text(5.7, 4.6, "2) Sparse Attention", fontsize=8, fontweight="bold", color=C2, ha="center")
ax1.text(5.7, 3.0, "Element-wise gate\ntop-25% sparse", ha="center", va="center", fontsize=6.5, color=C2)

draw_box(ax1, 3.7, 3.5, 0.8, 0.6, "Gene\nInteract", "#D9D9D9")
draw_arrow(ax1, 3.3, 3.5, 3.3, 3.5)

for i in range(4):
    draw_box(ax1, 6.9+i*0.3, 3.5, 0.22, 0.7, "" if i > 0 else "TFx8", "#D9D9D9", alpha=0.3)
draw_arrow(ax1, 6.6, 3.5, 6.8, 3.5)
draw_box(ax1, 8.5, 3.5, 0.6, 0.6, "Output(y)", "white", 1, 0)
draw_arrow(ax1, 8.0, 3.5, 8.2, 3.5)

ax1.add_patch(FancyBboxPatch([7.2, 1.5], 1.8, 1.2, boxstyle="round,pad=0.08", facecolor=C3, edgecolor=C3, linewidth=2, alpha=0.08))
ax1.text(8.1, 2.1, "3) Interaction\nGradient Penalty", fontsize=7, fontweight="bold", color=C3, ha="center")
draw_arrow(ax1, 0.7, 2.2, 1.5, 2.2)
draw_arrow(ax1, 0.7, 3.5, 1.5, 3.5)

# Panel B
ax2 = fig.add_axes([0.04, 0.02, 0.30, 0.55])
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")
ax2.text(0.3, 9.5, "b  Embedding Fusion Detail", fontsize=10, fontweight="bold")

for i, (lbl, c) in enumerate([["KGE(128d)", "#5B9BD5"], ["scGPT(512d)", "#4472C4"], ["BioLLM(384d)", "#2F5597"]]):
    draw_box(ax2, 1.9+i*2.5, 7.5, 1.5, 1.0, lbl, c)

ax2.add_patch(FancyBboxPatch([1.5, 2.5], 6.0, 1.5, boxstyle="round,pad=0.05", facecolor=C1, edgecolor=C1, linewidth=2, alpha=0.12))
ax2.text(4.5, 3.5, "Softmax attention weights", ha="center", va="center", fontsize=8, fontweight="bold", color=C1)
ax2.text(4.5, 3.0, "each gene learns own alpha", ha="center", va="center", fontsize=6.5, color=C1)

draw_box(ax2, 4.5, 1.0, 1.2, 0.7, "Sum x W_o", C1)
draw_box(ax2, 4.5, 0.2, 0.8, 0.4, "hidden(H)", "white", 0, 0)
for i in range(3):
    draw_arrow(ax2, 1.9+i*2.5, 6.5, 1.9+i*2.5, 4.0)
draw_arrow(ax2, 4.5, 2.5, 4.5, 1.7)

# Panel C
ax3 = fig.add_axes([0.36, 0.02, 0.30, 0.55])
ax3.set_xlim(0, 10); ax3.set_ylim(0, 10); ax3.axis("off")
ax3.text(0.3, 9.5, "c  DEG-Sparse Attention", fontsize=10, fontweight="bold")
draw_box(ax3, 1.5, 8.2, 2.0, 0.8, "Gene Emb E_g", C2)
draw_box(ax3, 6.5, 8.2, 2.0, 0.8, "Pert Emb E_p", C2)

ax3.add_patch(FancyBboxPatch([4.0, 4.5], 4.8, 2.5, boxstyle="round,pad=0.08", facecolor=C2, edgecolor=C2, linewidth=2, alpha=0.1))
ax3.text(5.5, 6.5, "Element-wise O(N)", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C2)
ax3.text(5.5, 6.0, "a = (Q*K)/sqrt(d)", ha="center", va="center", fontsize=7, color=C2)
ax3.text(5.5, 5.3, "Top-25% sparse gate", ha="center", va="center", fontsize=7, fontweight="bold", color=C2)
draw_box(ax3, 5.5, 1.5, 3.0, 0.8, "combined_emb", C2)
for i in range(3):
    draw_arrow(ax3, 3.5, 8.2-0.7*i, 4.0, 6.0)
draw_arrow(ax3, 5.5, 4.5, 5.5, 2.3)

# Panel D
ax4 = fig.add_axes([0.68, 0.02, 0.30, 0.55])
ax4.set_xlim(0, 10); ax4.set_ylim(0, 10); ax4.axis("off")
ax4.text(0.3, 9.5, "d  Interaction Penalty", fontsize=10, fontweight="bold")
draw_box(ax4, 1.5, 7.5, 1.8, 0.7, "Gene A pred", C3)
draw_box(ax4, 6.5, 7.5, 1.8, 0.7, "Gene B pred", C3)
ax4.add_patch(FancyBboxPatch([2.5, 5.0], 3.0, 0.7, boxstyle="round,pad=0.03", facecolor=C3, edgecolor=C3, linewidth=1.5, alpha=0.1))
ax4.text(4.0, 5.35, "additive = A+B-ctrl", ha="center", va="center", fontsize=7, fontweight="bold", color=C3)
draw_box(ax4, 7.5, 3.0, 1.8, 0.7, "Combo AB pred", C3)
ax4.add_patch(FancyBboxPatch([1.5, 0.8], 5.0, 1.2, boxstyle="round,pad=0.05", facecolor="white", edgecolor=C3, linewidth=2, alpha=0.15))
ax4.text(4.0, 1.6, "deviation = |combo - additive|", ha="center", va="center", fontsize=7, fontweight="bold", color=C3)
ax4.text(4.0, 1.0, "weight = sigmoid(deviation - margin)", ha="center", va="center", fontsize=7, color=C3)
draw_arrow(ax4, 2.5, 7.5, 3.5, 5.7)
draw_arrow(ax4, 7.5, 7.5, 4.5, 5.7)
draw_arrow(ax4, 7.5, 4.5, 4.0, 2.0)
draw_arrow(ax4, 4.0, 5.0, 4.0, 4.0)

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/Figure1_architecture.png", dpi=300, bbox_inches="tight")
plt.savefig("figures/Figure1_architecture.pdf", bbox_inches="tight")
print("Figure saved OK")
