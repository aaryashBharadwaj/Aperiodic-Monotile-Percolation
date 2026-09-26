"""Shared Matplotlib style for the manuscript figures.

One source of truth for the paper's figure font: Computer Modern via matplotlib's BUNDLED cm/cmr10
fonts (no LaTeX toolchain required), so every figure matches the REVTeX body font. Call
apply_paper_style() at the top of a figure module or generator script.
"""
import matplotlib.pyplot as plt

PAPER_RC = {
    "font.family": "cmr10",               # Computer Modern (bundled) -- matches the REVTeX body font
    "mathtext.fontset": "cm",             # render math in CM too, so labels and equations agree
    "axes.unicode_minus": False,          # cmr10 lacks U+2212; use the ASCII hyphen-minus instead
    "axes.formatter.use_mathtext": True,  # tick numbers via mathtext -> CM digits (also silences the cmr10 warning)
}


def apply_paper_style():
    """Apply the shared Computer Modern paper style to the global rcParams."""
    plt.rcParams.update(PAPER_RC)
