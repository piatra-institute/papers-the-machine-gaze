"""Figures for The Machine Gaze."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GRID, AMBER, GREEN, BLUE, GRAY, RED, PURPLE = ("#1a1a1a", "#d9d9d9", "#b45309", "#15803d",
                                                    "#2563eb", "#78716c", "#b3202c", "#7c3aed")


def _style(ax):
    ax.grid(True, color=GRID, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def _title(ax, t):
    ax.set_title(t, fontsize=9.5, loc="left")


LABEL = {"bubble_selection": "bubble + selection", "bubble_insertion": "bubble + insertion",
         "insertion_selection": "insertion + selection", "pure_bubble": "bubble", "pure_insertion": "insertion",
         "pure_selection": "selection"}


def plot_sorting(r, path):
    so = r["sorting"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))
    grid = np.linspace(0, 1, 101)

    ax = axes[0]
    for m, col in zip(so["mixes"], (BLUE, AMBER, GREEN)):
        ax.plot(grid, m["curve"], color=col, lw=1.8, label=LABEL[m["mix"]])
        ax.plot(grid, m["control_curve"], ":", color=col, lw=1.1)
    ax.plot([], [], ":", color=GRAY, label="same algorithm, two labels")
    ax.set_xlabel("fraction of the sorting run (swaps)", fontsize=9)
    ax.set_ylabel("aggregation by algotype", fontsize=9)
    ax.set_ylim(0.44, 0.7)
    _title(ax, "a. chimeric arrays cluster while they sort")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    _style(ax)

    ax = axes[1]
    tr = so["closure"]["trajectories"]
    for name, col in (("pure_bubble", BLUE), ("pure_insertion", AMBER), ("pure_selection", GREEN)):
        for k, t in enumerate(tr[name]):
            x = np.arange(len(t)) * 5
            ax.plot(x, t, color=col, lw=1.0, alpha=0.9, label=LABEL[name] if k == 0 else None)
    ax.axhline(so["closure"]["sortedness_level"], color=GRAY, ls="--", lw=0.8)
    ax.set_xlabel("swaps", fontsize=9)
    ax.set_ylabel("sortedness (ordered adjacent pairs)", fontsize=9)
    _title(ax, "b. one function, three dynamics")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    _style(ax)

    ax = axes[2]
    rows = so["closure"]["rows"]
    data = [row["remaining"] for row in rows]
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.55, showfliers=False)
    for patch, col in zip(bp["boxes"], (BLUE, AMBER, GREEN, PURPLE)):
        patch.set_facecolor(col)
        patch.set_alpha(0.35)
    ax.set_xticks(range(1, len(rows) + 1))
    ax.set_xticklabels([LABEL[row["composition"]].replace(" + ", "\n+ ") for row in rows], fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylabel(f"swaps remaining at sortedness {so['closure']['sortedness_level']}", fontsize=9)
    _title(ax, "c. same observed state, different futures")
    _style(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_spillover(r, path):
    sp = r["spillover"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))

    ax = axes[0]
    z = [row["zeta_residual"] for row in sp["rows"]]
    ax.plot(z, [row["full_max_re"] for row in sp["rows"]], "o-", color=RED, ms=4, label="controller on the full beam")
    ax.plot(z, [row["reduced_max_re"] for row in sp["rows"]], "s-", color=BLUE, ms=4, label="controller on its own model")
    ax.plot(z, [row["no_observation_spillover_max_re"] for row in sp["rows"]], "^--", color=GREEN, ms=4,
            label="full beam, sensor blind to residual modes")
    ax.axhline(0, color=INK, lw=0.7)
    ax.axvline(sp["critical_zeta_residual"], color=GRAY, ls=":", lw=1)
    ax.set_xlabel("damping ratio of the unmodelled modes", fontsize=9)
    ax.set_ylabel("largest real part of the closed-loop poles", fontsize=9)
    _title(ax, "a. a stable design that destabilizes its plant")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    _style(ax)

    ax = axes[1]
    for key, ls, lab in (("zeta_0.001", "-", "residual damping 0.001"), ("zeta_0.03", "--", "residual damping 0.03")):
        tr = sp["traces"][key]
        t = np.array(tr["t"])
        ax.semilogy(t, np.maximum(tr["controlled"], 1e-12), ls, color=BLUE, lw=1.4, label=f"modelled modes, {lab}")
        ax.semilogy(t, np.maximum(tr["residual"], 1e-12), ls, color=RED, lw=1.4, label=f"unmodelled modes, {lab}")
    ax.set_ylim(1e-12, 1e3)
    ax.set_xlim(0, 15)
    ax.set_xlabel("time", fontsize=9)
    ax.set_ylabel("modal amplitude", fontsize=9)
    _title(ax, "b. the modelled modes settle, then the residual takes over")
    ax.legend(fontsize=6.6, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.42))
    _style(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_slack(r, path):
    sl = r["slack"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))

    ax = axes[0]
    g = np.array(sl["grid"])
    ax.plot(g, sl["observed"], color=GRAY, lw=1.6, label="throughput while running (observed)")
    ax.plot(g, sl["Y"], color=BLUE, lw=1.8, label="long-run output")
    ax.axvline(sl["s_star"], color=GREEN, ls=":", lw=1, label=f"best maintenance share {sl['s_star']:.2f}")
    sims = sl["fixed_policy_checks"]
    ax.plot([c["s"] for c in sims], [c["simulated"] for c in sims], "o", color=RED, ms=5, label="simulation")
    ax.set_xlabel("share of time spent maintaining the repertoire, $s$", fontsize=9)
    ax.set_ylabel("output per unit time", fontsize=9)
    _title(ax, "a. what the gaze sees and what it gets")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    _style(ax)

    ax = axes[1]
    gz = sl["gaze"]
    ax.plot(gz["t"], gz["s"], color=AMBER, lw=1.8, label="maintenance share $s$")
    ax.plot(gz["t"], gz["coverage"], color=PURPLE, lw=1.4, label="disturbance types covered")
    ax.set_xlabel("time", fontsize=9)
    ax.set_ylabel("share", fontsize=9)
    ax.set_ylim(-0.03, 1.05)
    _title(ax, "b. ascent on observed throughput")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    _style(ax)

    ax = axes[2]
    ax.plot(g, sl["info"], color=INK, lw=1.8)
    ax.axvline(sl["s_star"], color=GREEN, ls=":", lw=1)
    ax.set_xlabel("maintenance share $s$", fontsize=9)
    ax.set_ylabel("bits", fontsize=9)
    _title(ax, "c. what the hidden repertoire adds to prediction")
    _style(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_all(results, outdir):
    outdir = Path(outdir)
    plot_sorting(results, str(outdir / "sorting.png"))
    plot_spillover(results, str(outdir / "spillover.png"))
    plot_slack(results, str(outdir / "slack.png"))
