"""Machinehood as closure under an observation: three computations and one identity.

Every number reported in the paper is computed here and written to output/results.json.
Sections:
  sorting   - self-sorting arrays with cell-level algorithms (after Zhang, Goldstein and Levin):
              chimeric aggregation against controls; the sortedness observable is not closed
  spillover - a controller designed on a reduced modal model of a beam, applied to the full beam
  slack     - an organisation whose throughput observable hides the repertoire that absorbs disturbances
  loop      - the performative loop between a model and the system it describes
"""
from __future__ import annotations

import math
import random

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigvals, expm, solve_continuous_are
from scipy.optimize import minimize_scalar
from scipy.stats import binom, ttest_ind, f_oneway


def _r(x, d=12):
    return float(np.round(float(x), d))


# ---------------------------------------------------------------------------------------------
# 1. Self-sorting arrays
# ---------------------------------------------------------------------------------------------

BUBBLE, INSERTION, SELECTION = 0, 1, 2
NAMES = {BUBBLE: "bubble", INSERTION: "insertion", SELECTION: "selection"}


def sort_run(algo, label, rng, N=100):
    """Cell-level sorting with random sequential updates.

    Each cell carries a value, an algorithm (algotype) and a label used only by the observer.
    Bubble: look left or right at random and swap if out of order with that neighbour.
    Insertion: move left past a larger left neighbour only if everything to its left is sorted.
    Selection: hold an ideal position (initially the left end); swap into it if the occupant is larger,
    otherwise advance the ideal position by one.
    Returns the aggregation trajectory, the sortedness trajectory (both recorded after every swap),
    the number of swaps and whether the final array equals the sorted input."""
    vals = list(range(N))
    rng.shuffle(vals)
    start = sorted(vals)
    ideal = [0] * N
    cid = list(range(N))
    pos = list(range(N))
    val = vals[:]
    lab = [label[c] for c in cid]

    def swap(i, j):
        a, b = cid[i], cid[j]
        cid[i], cid[j] = b, a
        pos[a], pos[b] = j, i
        val[i], val[j] = val[j], val[i]
        lab[i], lab[j] = lab[j], lab[i]

    def aggregation():
        # fraction of cells whose right neighbour carries the same label (Zhang et al.'s metric)
        return sum(1 for k in range(N - 1) if lab[k] == lab[k + 1]) / N

    def sortedness():
        return sum(1 for k in range(N - 1) if val[k] <= val[k + 1]) / (N - 1)

    agg = [aggregation()]
    srt = [sortedness()]
    nswap = 0
    while True:
        c = rng.randrange(N)
        p = pos[c]
        t = algo[c]
        moved = False
        if t == BUBBLE:
            if rng.random() < 0.5:
                if p < N - 1 and val[p] > val[p + 1]:
                    swap(p, p + 1)
                    moved = True
            elif p > 0 and val[p] < val[p - 1]:
                swap(p, p - 1)
                moved = True
        elif t == INSERTION:
            if p > 0 and val[p] < val[p - 1]:
                ok = all(val[k] <= val[k + 1] for k in range(p - 1))
                if ok:
                    swap(p, p - 1)
                    moved = True
        else:
            q = ideal[c]
            if q != p and 0 <= q < N:
                if val[p] >= val[q]:
                    ideal[c] = q + 1
                else:
                    swap(p, q)
                    moved = True
        if moved:
            nswap += 1
            agg.append(aggregation())
            srt.append(sortedness())
            if srt[-1] == 1.0:
                break
    return np.array(agg), np.array(srt), nswap, val == start


GRID = np.linspace(0, 1, 101)


def mix_experiment(a1, a2, rng, R=100, control=False, a3=None):
    """Chimeric arrays (half a1, half a2; or thirds with a3) or controls in which every cell runs a1
    but carries the same two labels."""
    curves, peaks, sorted_ok, swaps = [], [], 0, []
    for _ in range(R):
        if a3 is None:
            lab = [0] * 50 + [1] * 50
        else:
            lab = [0] * 33 + [1] * 33 + [2] * 34
        rng.shuffle(lab)
        if control:
            algo = [a1] * 100
        elif a3 is None:
            algo = [a1 if l == 0 else a2 for l in lab]
        else:
            algo = [(a1, a2, a3)[l] for l in lab]
        agg, _, ns, ok = sort_run(algo, lab, rng)
        frac = np.linspace(0, 1, len(agg))
        curves.append(np.interp(GRID, frac, agg))
        peaks.append(float(agg.max()))
        sorted_ok += int(ok)
        swaps.append(ns)
    curves = np.array(curves)
    mean = curves.mean(0)
    return {"mean_curve": mean, "curves": curves, "peaks": np.array(peaks), "sorted_fraction": sorted_ok / R,
            "mean_swaps": float(np.mean(swaps))}


def sorting_section(seed=20260930):
    rng = random.Random(seed)
    out = {"N": 100, "replicates": 100}
    mixes = [("bubble_selection", BUBBLE, SELECTION), ("bubble_insertion", BUBBLE, INSERTION),
             ("insertion_selection", INSERTION, SELECTION)]
    rows = []
    all_sorted = True
    for name, a1, a2 in mixes:
        ch = mix_experiment(a1, a2, rng)
        co = mix_experiment(a1, a2, rng, control=True)
        k = int(np.argmax(ch["mean_curve"]))
        t = ttest_ind(ch["curves"][:, k], co["curves"][:, k], equal_var=False)
        all_sorted &= ch["sorted_fraction"] == 1.0 and co["sorted_fraction"] == 1.0
        rows.append({"mix": name, "control_algorithm": NAMES[a1],
                     "peak_of_mean": _r(ch["mean_curve"].max(), 6), "peak_position": _r(GRID[k], 4),
                     "control_mean_at_peak_position": _r(co["mean_curve"][k], 6),
                     "control_max_of_mean": _r(co["mean_curve"].max(), 6),
                     "mean_run_peak": _r(ch["peaks"].mean(), 6), "control_mean_run_peak": _r(co["peaks"].mean(), 6),
                     "start": _r(ch["mean_curve"][0], 6), "end": _r(ch["mean_curve"][-1], 6),
                     "welch_t": _r(t.statistic, 4), "p_value": float(t.pvalue),
                     "mean_swaps": _r(ch["mean_swaps"], 2),
                     "curve": [_r(v, 5) for v in ch["mean_curve"]], "control_curve": [_r(v, 5) for v in co["mean_curve"]]})
    three = mix_experiment(BUBBLE, INSERTION, rng, a3=SELECTION)
    k3 = int(np.argmax(three["mean_curve"]))
    out["three_way"] = {"peak_of_mean": _r(three["mean_curve"].max(), 6), "peak_position": _r(GRID[k3], 4),
                        "mean_run_peak": _r(three["peaks"].mean(), 6), "start": _r(three["mean_curve"][0], 6)}
    all_sorted &= three["sorted_fraction"] == 1.0
    out["mixes"] = rows
    out["published"] = {"mean_swaps_bubble_selection": 1740.9, "mean_swaps_bubble_insertion": 2476.02,
                        "mean_swaps_insertion_selection": 1534.77, "peak_bubble_selection": 0.72, "peak_bubble_insertion": 0.65, "peak_insertion_selection": 0.69,
                        "peak_three_way": 0.62, "control_peaks": [0.61, 0.65, 0.57]}
    out["all_runs_sorted"] = bool(all_sorted)
    out["max_p_value"] = float(max(r["p_value"] for r in rows))
    out["min_chimeric_peak_of_mean"] = float(min(r["peak_of_mean"] for r in rows))
    out["max_control_max_of_mean"] = float(max(r["control_max_of_mean"] for r in rows))
    # closure of the sortedness observable: same function, same observed progress, different futures
    comps = [("pure_bubble", [BUBBLE] * 100), ("pure_insertion", [INSERTION] * 100),
             ("pure_selection", [SELECTION] * 100), ("bubble_selection", None)]
    level = 0.9
    clo = []
    samples = []
    traj_store = {}
    for name, algo in comps:
        rem, sw = [], []
        for _ in range(100):
            lab = [0] * 50 + [1] * 50
            rng.shuffle(lab)
            al = algo if algo is not None else [BUBBLE if l == 0 else SELECTION for l in lab]
            _, srt, ns, ok = sort_run(al, lab, rng)
            i = int(np.argmax(srt >= level))
            rem.append(ns - i)
            sw.append(ns)
            if len(traj_store.get(name, [])) < 3:
                traj_store.setdefault(name, []).append([_r(v, 4) for v in srt[::5]])
        rem = np.array(rem)
        samples.append(rem)
        clo.append({"composition": name, "mean_remaining_swaps": _r(rem.mean(), 2), "sd": _r(rem.std(ddof=1), 2),
                    "mean_total_swaps": _r(np.mean(sw), 2), "remaining": [int(v) for v in rem]})
    F = f_oneway(*samples)
    out["closure"] = {"sortedness_level": level, "rows": clo, "anova_F": _r(F.statistic, 3), "anova_p": float(F.pvalue),
                      "trajectories": traj_store,
                      "ratio_max_min_mean": _r(max(r["mean_remaining_swaps"] for r in clo) /
                                               min(r["mean_remaining_swaps"] for r in clo), 3)}
    return out


# ---------------------------------------------------------------------------------------------
# 2. Control spillover: a controller designed on a reduced modal model of a pinned beam
# ---------------------------------------------------------------------------------------------

SPILL = {"controlled_modes": 2, "residual_modes": 3, "x_actuator": 0.21, "x_sensor": 0.83,
         "zeta_controlled": 0.005, "lqr_q": 1.0, "lqr_r": 1e-3, "kf_q": 1.0, "kf_r": 1e-3}


def _phi(k, x):
    return math.sqrt(2) * math.sin(k * math.pi * x)


def beam_model(nm, zeta, xa, xs):
    w = np.array([(k * math.pi) ** 2 for k in range(1, nm + 1)])
    A = np.zeros((2 * nm, 2 * nm))
    B = np.zeros((2 * nm, 1))
    C = np.zeros((1, 2 * nm))
    for k in range(nm):
        A[2 * k, 2 * k + 1] = 1.0
        A[2 * k + 1, 2 * k] = -w[k] ** 2
        A[2 * k + 1, 2 * k + 1] = -2 * zeta[k] * w[k]
        B[2 * k + 1, 0] = _phi(k + 1, xa)
        C[0, 2 * k] = _phi(k + 1, xs)
    return A, B, C, w


def spill_loop(zeta_r, obs_spill=True, ctrl_spill=True, p=SPILL):
    m, r = p["controlled_modes"], p["residual_modes"]
    nm = m + r
    zeta = [p["zeta_controlled"]] * m + [zeta_r] * r
    A, B, C, w = beam_model(nm, zeta, p["x_actuator"], p["x_sensor"])
    Ac, Bc, Cc = A[:2 * m, :2 * m], B[:2 * m], C[:, :2 * m]
    P = solve_continuous_are(Ac, Bc, p["lqr_q"] * np.eye(2 * m), p["lqr_r"] * np.eye(1))
    K = (Bc.T @ P) / p["lqr_r"]
    Pf = solve_continuous_are(Ac.T, Cc.T, p["kf_q"] * np.eye(2 * m), p["kf_r"] * np.eye(1))
    L = (Pf @ Cc.T) / p["kf_r"]
    Bf, Cf = B.copy(), C.copy()
    if not ctrl_spill:
        Bf[2 * m:] = 0.0
    if not obs_spill:
        Cf[:, 2 * m:] = 0.0
    full = np.vstack([np.hstack([A, -Bf @ K]), np.hstack([L @ Cf, Ac - Bc @ K - L @ Cc])])
    red = np.vstack([np.hstack([Ac, -Bc @ K]), np.hstack([L @ Cc, Ac - Bc @ K - L @ Cc])])
    return full, red, w


def spillover_section():
    p = SPILL
    m = p["controlled_modes"]
    grid = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    rows = []
    for z in grid:
        full, red, w = spill_loop(z)
        f_obs, _, _ = spill_loop(z, obs_spill=False)
        f_ctl, _, _ = spill_loop(z, ctrl_spill=False)
        rows.append({"zeta_residual": z, "full_max_re": _r(max(eigvals(full).real), 8),
                     "reduced_max_re": _r(max(eigvals(red).real), 8),
                     "no_observation_spillover_max_re": _r(max(eigvals(f_obs).real), 8),
                     "no_control_spillover_max_re": _r(max(eigvals(f_ctl).real), 8)})
    lo, hi = 0.0, 0.05
    for _ in range(60):
        mid = (lo + hi) / 2
        if max(eigvals(spill_loop(mid)[0]).real) > 0:
            lo = mid
        else:
            hi = mid
    zc = (lo + hi) / 2
    # which mode goes unstable: the eigenvector's weight on residual coordinates at zeta_r = 0.001
    full, red, w = spill_loop(0.001)
    ev, V = np.linalg.eig(full)
    i = int(np.argmax(ev.real))
    v = np.abs(V[:, i])
    nm = m + p["residual_modes"]
    weights = [float(np.sum(v[2 * k:2 * k + 2] ** 2)) for k in range(nm)]
    dominant = int(np.argmax(weights)) + 1
    # time response from a small initial displacement of the first mode
    traces = {}
    for z in (0.001, 0.03):
        full, red, w = spill_loop(z)
        n = full.shape[0]
        x0 = np.zeros(n)
        x0[0] = 0.01
        ts = np.linspace(0, 30, 601)
        X = np.array([expm(full * t) @ x0 for t in ts])
        nmod = (n - 2 * m) // 2
        ctrl = np.sqrt(np.sum(X[:, :2 * m:2] ** 2, axis=1))
        resid = np.sqrt(np.sum(X[:, 2 * m:2 * nmod:2] ** 2, axis=1))
        est = np.sqrt(np.sum(X[:, 2 * nmod::2] ** 2, axis=1))
        traces[f"zeta_{z}"] = {"t": [_r(v, 4) for v in ts], "controlled": [_r(v, 10) for v in ctrl],
                               "residual": [_r(v, 10) for v in resid], "estimate": [_r(v, 10) for v in est]}
    sm = traces["zeta_0.001"]
    tt = np.array(sm["t"])
    cr = np.array(sm["controlled"])
    rs = np.array(sm["residual"])
    cross = next((float(tt[i]) for i in range(1, len(tt)) if rs[i] > cr[i]), float("nan"))
    early = tt <= 3
    min_ctrl_early = float(cr[early].min())
    k10 = int(np.argmin(np.abs(np.array(sm["t"]) - 10)))
    return {"params": p, "rows": rows, "critical_zeta_residual": _r(zc, 8),
            "reduced_max_re": rows[0]["reduced_max_re"],
            "full_max_re_at_zero_damping": rows[0]["full_max_re"],
            "full_max_re_at_0p001": rows[2]["full_max_re"],
            "unstable_mode_at_0p001": dominant, "unstable_frequency_at_0p001": _r(abs(ev[i].imag), 4),
            "mode_frequencies": [_r(v, 4) for v in w],
            "traces": traces,
            "controlled_amplitude_t10_low_damping": sm["controlled"][k10],
            "residual_overtakes_controlled_at": _r(cross, 4), "min_controlled_amplitude_first_3": _r(min_ctrl_early, 10),
            "residual_amplitude_t30_low_damping": sm["residual"][-1],
            "residual_amplitude_t30_high_damping": traces["zeta_0.03"]["residual"][-1]}


# ---------------------------------------------------------------------------------------------
# 3. Slack: throughput is observed, the repertoire that absorbs disturbances is not
# ---------------------------------------------------------------------------------------------

SLACK = {"N": 10, "K": 20, "rho": 1.0, "delta": 0.05, "nu": 0.005, "tau": 800.0, "W": 10.0}


def slack_q(s, p=SLACK):
    a = s * p["rho"] / p["K"]
    return a / (a + p["delta"]) if a > 0 else 0.0


def slack_cover(s, p=SLACK):
    return 1 - (1 - slack_q(s, p)) ** p["N"]


def slack_Y(s, p=SLACK):
    """Long-run output rate: throughput (1 - s) N while up; an unabsorbed disturbance stops output for tau."""
    return (1 - s) * p["N"] / (1 + p["nu"] * p["tau"] * (1 - slack_cover(s, p)))


def _h(x):
    x = np.clip(x, 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = -(x * np.log2(x) + (1 - x) * np.log2(1 - x))
    return np.nan_to_num(v)


def slack_info(s, p=SLACK):
    """I(H; A): information the hidden coverage carries about whether the next disturbance is absorbed."""
    c = slack_cover(s, p)
    K = p["K"]
    j = np.arange(K + 1)
    w = binom.pmf(j, K, c)
    return float(_h(np.array(c)) - np.sum(w * _h(j / K)))


def slack_simulate(policy, T, rng, p=SLACK, s0=0.3, eps=0.02, eta=0.01, record_every=200):
    """Event-level simulation. Chains: each (unit, type) is known or forgotten, with rates
    forgotten->known = s rho / K and known->forgotten = delta. Disturbances arrive at rate nu during uptime;
    a disturbance whose type no unit knows stops output for tau. Output accrues at (1 - s) N while up.
    policy: a float (fixed s) or "gaze" (finite-difference ascent on windowed throughput)."""
    N, K, W = p["N"], p["K"], p["W"]
    s = s0 if policy == "gaze" else float(policy)
    q0 = slack_q(s, p)
    known = rng.random((N, K)) < q0
    t, down_until, out = 0.0, 0.0, 0.0
    hist = []
    phase, y_plus, down_plus, used, skipped = 0, 0.0, False, 0, 0
    late_out, late_t, late_s_sum = 0.0, 0.0, 0.0
    next_dist = rng.exponential(1 / p["nu"])
    nwin = 0

    def evolve(known, s, dt):
        a = s * p["rho"] / K
        lam = a + p["delta"]
        if lam == 0:
            return known
        stay_known = (a + p["delta"] * math.exp(-lam * dt)) / lam
        become_known = (a - a * math.exp(-lam * dt)) / lam
        u = rng.random(known.shape)
        return np.where(known, u < stay_known, u < become_known)

    while t < T:
        s_eff = s
        if policy == "gaze":
            s_eff = min(1.0, s + eps) if phase == 0 else max(0.0, s - eps)
        w_end = t + W
        w_out = 0.0
        w_down = False
        while t < w_end:
            if t < down_until:
                dt = min(down_until, w_end) - t
                known = evolve(known, s_eff, dt)
                t += dt
                w_down = True
                continue
            dt = min(next_dist, w_end - t)
            known = evolve(known, s_eff, dt)
            w_out += (1 - s_eff) * N * dt
            t += dt
            next_dist -= dt
            if next_dist <= 1e-12:
                typ = rng.integers(K)
                if not known[:, typ].any():
                    down_until = t + p["tau"]
                    w_down = True
                next_dist = rng.exponential(1 / p["nu"])
        out += w_out
        nwin += 1
        if t > T / 2:
            late_out += w_out
            late_t += W
            late_s_sum += s_eff
        if policy == "gaze":
            # windows that contain an outage are classed as exogenous disruption and set aside
            if phase == 0:
                y_plus, down_plus, phase = w_out, w_down, 1
            else:
                if not (down_plus or w_down):
                    g = (y_plus - w_out) / (2 * eps * W * N)
                    s = float(min(1.0, max(0.0, s + eta * g)))
                    used += 1
                else:
                    skipped += 1
                phase = 0
        if nwin % record_every == 0:
            hist.append((t, s, float(known.any(axis=0).mean()), out / t))
    return {"rate": out / t, "final_s": s, "history": hist, "late_rate": late_out / late_t if late_t else float("nan"),
            "pairs_used": used, "pairs_skipped": skipped,
            "late_mean_s": late_s_sum * W / late_t if late_t else float("nan")}


def _first_zero(hist):
    return next((h[0] for h in hist if h[1] == 0.0), float("inf"))


def slack_section(seed=20260930):
    p = SLACK
    rng = np.random.default_rng(seed)
    res = minimize_scalar(lambda s: -slack_Y(s, p), bounds=(0, 1), method="bounded", options={"xatol": 1e-12})
    s_star = float(res.x)
    grid = np.linspace(0, 1, 201)
    out = {"params": p, "s_star": _r(s_star, 6), "Y_star": _r(slack_Y(s_star), 6), "Y_zero": _r(slack_Y(0.0), 6),
           "ratio": _r(slack_Y(s_star) / slack_Y(0.0), 6), "cover_star": _r(slack_cover(s_star), 6),
           "q_star": _r(slack_q(s_star), 6), "observed_at_star": _r((1 - s_star) * p["N"], 6),
           "grid": [_r(v, 4) for v in grid], "Y": [_r(slack_Y(v), 6) for v in grid],
           "observed": [_r((1 - v) * p["N"], 6) for v in grid], "cover": [_r(slack_cover(v), 6) for v in grid],
           "info": [_r(slack_info(v), 8) for v in grid]}
    out["info_at_zero"] = _r(slack_info(0.0), 12)
    out["info_at_star"] = _r(slack_info(s_star), 6)
    out["info_max"] = _r(max(out["info"]), 6)
    W = p["W"]
    out["p_down_start_in_window_zero"] = _r(1 - math.exp(-p["nu"] * W), 6)
    out["p_down_start_in_window_star"] = _r(1 - math.exp(-p["nu"] * W * (1 - slack_cover(s_star))), 6)
    out["downtime_fraction_zero"] = _r(1 - 1 / (1 + p["nu"] * p["tau"]), 6)
    # stochastic check of the long-run rate formula at fixed policies
    checks = []
    for s in (0.0, 0.1, s_star, 0.6):
        sim = slack_simulate(s, 2.0e7, rng)
        checks.append({"s": _r(s, 6), "simulated": _r(sim["rate"], 6), "formula": _r(slack_Y(s), 6),
                       "rel_err": _r(abs(sim["rate"] - slack_Y(s)) / slack_Y(s), 6)})
    out["fixed_policy_checks"] = checks
    out["max_rel_err_fixed"] = float(max(c["rel_err"] for c in checks))
    # the gaze: finite-difference ascent on windowed throughput, starting at the optimum
    gz = slack_simulate("gaze", 2.0e6, rng, s0=s_star, record_every=2)
    hist = gz["history"]
    out["gaze"] = {"final_s": _r(gz["final_s"], 6), "eta": 0.01, "eps": 0.02,
                   "t": [_r(h[0], 2) for h in hist if h[0] <= 4000], "s": [_r(h[1], 6) for h in hist if h[0] <= 4000],
                   "coverage": [_r(h[2], 6) for h in hist if h[0] <= 4000],
                   "time_to_zero": _r(next((h[0] for h in hist if h[1] == 0.0), float("nan")), 2),
                   "late_rate": _r(gz["late_rate"], 6), "late_over_star": _r(gz["late_rate"] / slack_Y(s_star), 6),
                   "late_over_zero": _r(gz["late_rate"] / slack_Y(0.0), 6),
                   "late_mean_s": _r(gz["late_mean_s"], 6), "Y_at_late_mean_s": _r(slack_Y(gz["late_mean_s"]), 6),
                   "late_over_Y_mean_s": _r(gz["late_rate"] / slack_Y(gz["late_mean_s"]), 6),
                   "pairs_used": gz["pairs_used"], "pairs_skipped": gz["pairs_skipped"],
                   "mean_coverage_late": _r(np.mean([h[2] for h in hist if h[0] > 1.0e6]), 6),
                   "cover_at_late_mean_s": _r(slack_cover(gz["late_mean_s"]), 6)}
    return out


# ---------------------------------------------------------------------------------------------
# 4. The performative loop
# ---------------------------------------------------------------------------------------------

def loop_section():
    """theta' = eta (x - theta) (the model learns), x' = kappa (theta - x) (the system is made to fit).
    kappa theta + eta x is conserved, so both converge to (kappa theta0 + eta x0) / (kappa + eta)."""
    rows = []
    for eta, kappa in ((1.0, 0.0), (1.0, 1.0), (1.0, 3.0), (1.0, 9.0), (0.0, 1.0)):
        th0, x0 = 0.0, 1.0
        f = lambda t, y: [eta * (y[1] - y[0]), kappa * (y[0] - y[1])]
        s = solve_ivp(f, [0, 60], [th0, x0], rtol=1e-11, atol=1e-13)
        th, x = s.y[0, -1], s.y[1, -1]
        pred = (kappa * th0 + eta * x0) / (kappa + eta)
        rows.append({"eta": eta, "kappa": kappa, "final_model": _r(th, 9), "final_system": _r(x, 9),
                     "predicted": _r(pred, 9), "share_by_system": _r(kappa / (kappa + eta), 6)})
    return {"rows": rows, "max_abs_error": float(max(abs(r["final_model"] - r["predicted"]) + abs(r["final_system"] - r["predicted"])
                                                     for r in rows))}


# ---------------------------------------------------------------------------------------------

def run():
    res = {"sorting": sorting_section(), "spillover": spillover_section(), "slack": slack_section(), "loop": loop_section()}
    res["checks"] = checks(res)
    res["n_checks"] = len(res["checks"])
    return res


def checks(r):
    so, sp, sl, lo = r["sorting"], r["spillover"], r["slack"], r["loop"]
    rows = sp["rows"]
    c = {
        "sorting_all_runs_sorted": so["all_runs_sorted"],
        "sorting_start_near_half": all(abs(m["start"] - 0.495) < 0.02 for m in so["mixes"]),
        "sorting_chimeric_above_control": so["max_p_value"] < 1e-6 and so["min_chimeric_peak_of_mean"] > so["max_control_max_of_mean"] + 0.05,
        "sorting_peak_mid_process": all(0.1 < m["peak_position"] < 0.6 for m in so["mixes"]),
        "sorting_swaps_near_published": all(abs(m["mean_swaps"] / pub - 1) < 0.05 for m, pub in
                                            zip(so["mixes"], (1740.9, 2476.02, 1534.77))),
        "sorting_runpeaks_near_published": all(abs(m["mean_run_peak"] - pub) < 0.06 for m, pub in
                                               zip(so["mixes"], (0.72, 0.65, 0.69))),
        "sorting_observable_not_closed": so["closure"]["anova_p"] < 1e-6 and so["closure"]["ratio_max_min_mean"] > 2,
        "spill_reduced_stable": all(row["reduced_max_re"] < 0 for row in rows),
        "spill_full_unstable_low_damping": rows[0]["full_max_re"] > 0 and rows[2]["full_max_re"] > 0,
        "spill_full_stable_high_damping": rows[-1]["full_max_re"] < 0,
        "spill_single_spillover_harmless": all(row["no_observation_spillover_max_re"] <= 1e-9 and row["no_control_spillover_max_re"] <= 1e-9
                                               for row in rows),
        "spill_critical_in_range": 0 < sp["critical_zeta_residual"] < 0.05,
        "spill_residual_grows": sp["residual_amplitude_t30_low_damping"] > 0.01 > sp["residual_amplitude_t30_high_damping"],
        "slack_interior_optimum": 0 < sl["s_star"] < 1 and sl["ratio"] > 1.5,
        "slack_formula_matches_simulation": sl["max_rel_err_fixed"] < 0.015,
        "slack_info_zero_at_zero": sl["info_at_zero"] == 0.0 and sl["info_at_star"] > 0,
        "slack_gaze_drives_to_zero": sl["gaze"]["final_s"] == 0.0 and abs(sl["gaze"]["late_over_Y_mean_s"] - 1) < 0.05
                                     and sl["gaze"]["late_over_star"] < 0.5,
        "loop_invariant": lo["max_abs_error"] < 1e-6,
    }
    return {k: bool(v) for k, v in c.items()}
