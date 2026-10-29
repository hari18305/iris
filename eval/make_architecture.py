"""System architecture figure (enrollment / verification flow with the security layer)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7.5})
BLUE, GREY, RED, GREEN = "#dce6f5", "#eeeeee", "#f6dcd8", "#dcefdc"

fig, ax = plt.subplots(figsize=(7.1, 3.1))
ax.set_xlim(0, 100)
ax.set_ylim(0, 44)
ax.axis("off")


def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec="#333333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", linespacing=1.15)


def arrow(x0, y0, x1, y1, label=None, dy=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.9, color="#222222"))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center", fontsize=6.5, style="italic")


y = 30
box(1, y, 13, 9, "NIR eye\nimage", GREY)
box(19, y, 15, 9, "Iris\nsegmentation", BLUE)
box(39, y, 15, 9, "Rubber-sheet\nnormalization", BLUE)
box(59, y, 15, 9, "Log-Gabor\nencoding", BLUE)
box(79, y, 19, 9, "Iris code +\nvalidity mask", GREEN)
for a, b in ((14.3, 18.7), (34.3, 38.7), (54.3, 58.7), (74.3, 78.7)):
    arrow(a, y + 4.5, b, y + 4.5)

ax.text(1, 25.6, "Enrollment", fontweight="bold")
box(1, 14, 22, 8.5, "Duplicate-identity check\n(1:N masked Hamming)", RED)
box(29, 14, 22, 8.5, "Fernet encryption\n(AES-128-CBC + HMAC)", RED)
box(57, 14, 20, 8.5, "SQLite template\nstore (ciphertext)", GREY)
box(82, 14, 17, 8.5, "Hash-chained\naudit log", GREY)
arrow(88.5, 29.6, 12, 22.9)
ax.text(52, 27.4, "template", ha="center", fontsize=6.5, style="italic", bbox=dict(fc="white", ec="none", pad=0.6))
arrow(23.3, 18.2, 28.7, 18.2)
arrow(51.3, 18.2, 56.7, 18.2)
arrow(77.3, 18.2, 81.7, 18.2)

ax.text(1, 9.6, "Verification of a claimed identity (probe code from the same front end)", fontweight="bold")
box(1, 0.8, 22, 7.2, "Lockout check\n(5 failures / 300 s)", RED)
box(29, 0.8, 22, 7.2, "Decrypt + authenticate\nstored template", RED)
box(57, 0.8, 20, 7.2, "Masked Hamming\nmin over shifts", BLUE)
box(82, 0.8, 17, 7.2, "HD < T ?\ngrant / deny", GREEN)
for a, b in ((23.3, 28.7), (51.3, 56.7), (77.3, 81.7)):
    arrow(a, 4.4, b, 4.4)

os.makedirs("report/figures", exist_ok=True)
fig.savefig("report/figures/fig_architecture.png", dpi=300, bbox_inches="tight")
fig.savefig("report/figures/fig_architecture.pdf", bbox_inches="tight")
print("saved")
