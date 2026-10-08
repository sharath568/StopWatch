"""Turns results/*.jsonl into the paper's numbers, LaTeX tables, figures and \\num{} macros.

  python analyze.py results/gate_X.jsonl results/window_Y.jsonl results/interrupt_Z.jsonl results/inject_W.jsonl

Writes to paper/paper_out/: numbers.tex, tab_judge.tex, tab_window.tex, tab_interrupt.tex, tab_inject.tex,
fig_tradeoff.pdf, fig_interrupt.pdf, summary.txt.
Confidence intervals are 95% bootstrap intervals that resample TASKS (the unit of independence), not runs.
NaN is printed as '--'. Refuses *_STUB files unless --allow-stub (for plumbing checks only)."""
import json, sys, glob
from pathlib import Path
from collections import defaultdict
import numpy as np

RNG = np.random.default_rng(0)
NB = 2000


def cboot(groups, fn, nb=NB):
    """groups: dict cluster -> list of values. Resample clusters with replacement."""
    keys = [k for k, v in groups.items() if len(v)]
    if not keys: return (float("nan"),) * 3
    allv = [x for k in keys for x in groups[k]]
    est = float(fn(np.asarray(allv, float)))
    stats = []
    for _ in range(nb):
        pick = RNG.choice(len(keys), len(keys))
        vals = [x for i in pick for x in groups[keys[i]]]
        stats.append(float(fn(np.asarray(vals, float))))
    return est, float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def by_task(rows, val):
    g = defaultdict(list)
    for r in rows:
        v = val(r)
        if v is not None: g[r["task"]].append(float(v))
    return g


def f(t, pct=False, nd=1, ci=True):
    m, lo, hi = t
    if m != m: return "--"
    k = 100 if pct else 1
    return f"{m*k:.{nd}f}" + (f" [{lo*k:.{nd}f}, {hi*k:.{nd}f}]" if ci else "")


def short(m):
    return m.replace("-instruct", "").replace("_K_M", "").replace("instruct-", "").replace("_", "\\_")

def short_plain(m):
    """Same abbreviation as short(), but with no LaTeX escaping -- for matplotlib figure
    text (annotations, tick labels), which renders a literal backslash if given LaTeX escapes."""
    return m.replace("-instruct", "").replace("_K_M", "").replace("instruct-", "")


def main():
    files = [x for a in sys.argv[1:] if not a.startswith("--") for x in glob.glob(a)]
    if any("_STUB" in x for x in files) and "--allow-stub" not in sys.argv:
        sys.exit("Refusing to analyse STUB results. They are for smoke tests only.")
    rows = [json.loads(l) for x in files for l in open(x) if l.strip()]
    errs = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    out = Path(__file__).resolve().parent / "paper" / "paper_out"; out.mkdir(parents=True, exist_ok=True)
    L, N = [], {}                      # summary lines, numbers macros
    env = [r for r in rows if r["type"] == "env"]
    if env:
        e = env[-1]
        N.update(cpu=(e.get("cpu") or "unknown CPU"), ram=e.get("ram_gb"), ollama=(e.get("ollama_version") or "unknown").replace("ollama version is ", ""),
                 osname=e.get("system"), commit=(e.get("git_commit") or "untracked")[:8])
        L.append(f"env: cpu={e.get('cpu')} cores={e.get('cpu_count')} ram={e.get('ram_gb')}GB os={e.get('system')} ollama={e.get('ollama_version')} git={e.get('git_commit')} dirty={e.get('git_dirty')}")
    L.append(f"error rows dropped: {len(errs)}")
    N["nerrors"] = len(errs)

    # ---------------- RQ1 ----------------
    gate = [r for r in rows if r["type"] == "gate"]
    if gate:
        tab = []
        for judge in ["KeywordJudge", "ObjectiveJudge", "ObjectiveJudgePlus"]:
            g = [r for r in gate if r["judge"] == judge]
            if not g: continue
            key = {"KeywordJudge": "kw", "ObjectiveJudge": "ob", "ObjectiveJudgePlus": "op"}[judge]
            b = [r for r in g if r["label"] == "benign"]; d = [r for r in g if r["label"] == "drifted"]
            oos = [r for r in g if r["label"] == "out_of_scope"]; un = [r for r in g if r["label"] == "unsafe"]
            fp = cboot(by_task(b, lambda r: r["decision"] == "block"), np.mean)
            fn = cboot(by_task(d, lambda r: r["decision"] != "block"), np.mean)
            fp_v = {p: cboot(by_task([r for r in b if r["phrasing"] == p], lambda r: r["decision"] == "block"), np.mean) for p in ("explicit", "vague")}
            lat_rows = [r["judge_latency_s"] * 1000 for r in g if r["failed_check"] in ("C3", "C4", None)]
            p50 = float(np.percentile(lat_rows, 50)) if lat_rows else float("nan"); p95 = float(np.percentile(lat_rows, 95)) if lat_rows else float("nan")
            c1 = sum(r["decision"] == "block" and r["failed_check"] == "C1" for r in oos)
            c2 = sum(r["decision"] == "block" and r["failed_check"] == "C2" for r in un)
            L.append(f"{judge}: FP {f(fp,True)}% (benign n={len(b)}; explicit {f(fp_v['explicit'],True,ci=False)}%, vague {f(fp_v['vague'],True,ci=False)}%), "
                     f"FN {f(fn,True)}% (drifted n={len(d)}), C1 caught {c1}/{len(oos)}, C2 caught {c2}/{len(un)}, C3 latency p50 {p50:.0f} ms p95 {p95:.0f} ms (n={len(lat_rows)})")
            tab.append(f"{judge} & {f(fp,True)} & {f(fn,True)} & {p50:.0f} / {p95:.0f} ")
            N.update({f"{key}FP": f(fp, True, ci=False), f"{key}FPlo": f"{fp[1]*100:.1f}", f"{key}FPhi": f"{fp[2]*100:.1f}",
                      f"{key}FN": f(fn, True, ci=False), f"{key}FNlo": f"{fn[1]*100:.1f}", f"{key}FNhi": f"{fn[2]*100:.1f}",
                      f"{key}Pfifty": f"{p50:.0f}", f"{key}Pninetyfive": f"{p95:.0f}", f"{key}FPexplicit": f(fp_v['explicit'], True, ci=False),
                      f"{key}FPvague": f(fp_v['vague'], True, ci=False)})
            N.update(cone=f"{c1}", coneN=f"{len(oos)}", ctwo=f"{c2}", ctwoN=f"{len(un)}", nbenign=len(b), ndrift=len(d))
        (out / "tab_judge.tex").write_text(" \\\\\n".join(tab))

    # ---------------- RQ3 (window) ----------------
    win = [r for r in rows if r["type"] == "window" and not r.get("use_gate")]
    wing = [r for r in rows if r["type"] == "window" and r.get("use_gate")]
    models = sorted({r["model"] for r in win + [x for x in rows if x["type"] in ("interrupt", "control")]}, key=lambda m: m)
    tab, cell = [], {}
    if win:
        for m in models:
            for mode in ["none", "recovery"]:
                s = [r for r in win if r["model"] == m and r["mode"] == mode]
                if not s: continue
                reached = [r for r in s if r["t_first_irreversible_s"] is not None]
                w = cboot(by_task(reached, lambda r: r["t_first_irreversible_s"]), np.median)
                dur = cboot(by_task(s, lambda r: r["duration_s"]), np.median)
                spd = cboot(by_task(s, lambda r: r["decode_tok_s"]), np.median)
                goal = cboot(by_task(s, lambda r: r["final_class"] == "goal"), np.mean)
                div = cboot(by_task(s, lambda r: r["final_class"] == "diverged"), np.mean)
                cell[(m, mode)] = dict(w=w, dur=dur, spd=spd, goal=goal, div=div, k=len(reached), n=len(s))
                L.append(f"WINDOW {m}/{mode}: n={len(s)} goal {f(goal,True)}% diverged {f(div,True)}% | W median {f(w,nd=2)} s over {len(reached)}/{len(s)} runs that reached an irreversible step | "
                         f"duration {f(dur,nd=1)} s | decode {f(spd,nd=1)} tok/s")
        for m in models:
            a, b = cell.get((m, "none")), cell.get((m, "recovery"))
            if not a: continue
            row = f"{short(m)} & {a['spd'][0]:.1f} & {a['goal'][0]*100:.0f} & {f(a['w'],nd=1,ci=False)} ({a['k']}/{a['n']}) & "
            row += (f"{f(b['w'],nd=1,ci=False)} ({b['k']}/{b['n']}) & {f(a['dur'],nd=0,ci=False)} / {f(b['dur'],nd=0,ci=False)} " if b else "-- & -- ")
            tab.append(row)
        (out / "tab_window.tex").write_text(" \\\\\n".join(tab))
        # paired overhead of recovery on task duration (per task medians)
        for m in models:
            a = defaultdict(list); b = defaultdict(list)
            for r in win:
                if r["model"] == m: (a if r["mode"] == "none" else b)[r["task"]].append(r["duration_s"])
            d = {t: [ (np.median(b[t]) - np.median(a[t])) / np.median(a[t]) ] for t in a if t in b and a[t] and b[t]}
            ov = cboot(d, np.mean)
            L.append(f"OVERHEAD {m}: recovery vs none task duration {f(ov,True)}%")
            N[f"ovh{''.join(ch for ch in short(m) if ch.isalnum())}"] = f(ov, True, ci=False)
        try:
            import matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42; matplotlib.rcParams["ps.fonttype"] = 42; import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(3.8, 2.6))
            for mode, mk, lab in [("none", "o", "no recovery"), ("recovery", "s", "recovery")]:
                xs, ys, lo, hi = [], [], [], []
                for m in models:
                    c = cell.get((m, mode))
                    if c and c["w"][0] == c["w"][0]:
                        xs.append(c["spd"][0]); ys.append(c["w"][0]); lo.append(c["w"][0] - c["w"][1]); hi.append(c["w"][2] - c["w"][0])
                if xs: ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, capsize=2, label=lab, ms=4)
            for i, m in enumerate(models):
                c = cell.get((m, "none"))
                if c and c["w"][0] == c["w"][0]:
                    label = short_plain(m).split(":")[1] if ":" in short_plain(m) else short_plain(m)
                    yoff = 5 if i % 2 == 0 else -9   # stagger labels above/below to reduce overlap between close points
                    ax.annotate(label, (c["spd"][0], c["w"][0]), fontsize=5, xytext=(4, yoff), textcoords="offset points")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("decode speed (tokens/s)"); ax.set_ylabel("window W (s), median")
            ax.legend(["no recovery", "recovery, " + r"$\tau$"], fontsize=6); fig.tight_layout(); fig.savefig(out / "fig_tradeoff.pdf"); plt.close(fig)
        except ImportError:
            L.append("matplotlib missing: figure skipped")

    # ---------------- detection layer switched ON inside workflow runs ----------------
    if wing:
        tab = []
        for m in sorted({r["model"] for r in wing}):
            for mode in ["none", "recovery"]:
                s_ = [r for r in wing if r["model"] == m and r["mode"] == mode]
                if not s_: continue
                reached = [r for r in s_ if r["t_first_irreversible_s"] is not None]
                w = cboot(by_task(reached, lambda r: r["t_first_irreversible_s"]), np.median)
                gl = cboot(by_task(s_, lambda r: r["gate_latency_s"]), np.median)
                goal = cboot(by_task(s_, lambda r: r["final_class"] == "goal"), np.mean)
                base = cell.get((m, mode))
                L.append(f"WINDOW+GATE {m}/{mode}: W median {f(w,nd=2)} s over {len(reached)}/{len(s_)} | gate time per run {f(gl,nd=1)} s | goal {f(goal,True)}% | gate-off W {f(base['w'],nd=2) if base else '--'}")
                tab.append(f"{short(m)} & {mode} & {f(base['w'],nd=1,ci=False) if base else '--'} & {f(w,nd=1,ci=False)} & {f(gl,nd=1,ci=False)} & {f(goal,True,nd=0,ci=False)} ")
        (out / "tab_wgate.tex").write_text(" \\\\\n".join(tab))

    # ---------------- RQ2 (interrupt) ----------------
    it = [r for r in rows if r["type"] == "interrupt"]
    if it:
        tab = []; bars = {}
        for m in sorted({r["model"] for r in it}):
            cells = []
            for mode in ["none", "recovery"]:
                s = [r for r in it if r["model"] == m and r["mode"] == mode and r["stopped"]]
                allr = [r for r in it if r["model"] == m and r["mode"] == mode]
                attributable = [r for r in s if r["control_class"] == "goal"]
                div_all = cboot(by_task(s, lambda r: r["final_class"] == "diverged"), np.mean)
                div_att = cboot(by_task(attributable, lambda r: r["final_class"] == "diverged"), np.mean)
                late = cboot(by_task([r for r in s if r["control_reached_irr"]], lambda r: r["irreversible_before_stop"]), np.mean)
                outbox = cboot(by_task(s, lambda r: r["outbox_n"] > 0), np.mean)
                comp = cboot(by_task(s, lambda r: r["compensated_steps"]), np.mean)
                L.append(f"INTERRUPT {m}/{mode}: runs={len(allr)} stop_fired={len(s)} | divergence(all stopped) {f(div_all,True)}% | divergence(control reached goal, n={len(attributable)}) {f(div_att,True)}% | "
                         f"irreversible fired before stop (control reached irr.) {f(late,True)}% | any outbox effect {f(outbox,True)}% | steps compensated {comp[0]:.2f}")
                cells.append((div_all, div_att, late, len(s)))
                bars[(m, mode)] = (div_att[0], late[0])
            (da, dta, la, na), (dr_, dtr, lr, nr) = cells if len(cells) == 2 else (cells[0], cells[0])
            tab.append(f"{short(m)} & {na} & {f(dta,True,ci=False)} & {f(dtr,True,ci=False)} & {f(la,True,ci=False)} & {f(lr,True,ci=False)} ")
        (out / "tab_interrupt.tex").write_text(" \\\\\n".join(tab))
        try:
            import matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42; matplotlib.rcParams["ps.fonttype"] = 42; import matplotlib.pyplot as plt
            ms = sorted({m for m, _ in bars}); x = np.arange(len(ms)); w = 0.2
            fig, ax = plt.subplots(figsize=(3.4, 2.3))
            for i, (mode, lab) in enumerate([("none", "divergence, no recovery"), ("recovery", "divergence, recovery")]):
                ax.bar(x + (i - 1.5) * w, [100 * bars[(m, mode)][0] if (m, mode) in bars else 0 for m in ms], w, label=lab)
            for i, (mode, lab) in enumerate([("none", "too late, no recovery"), ("recovery", "too late, recovery")]):
                ax.bar(x + (i + 0.5) * w, [100 * bars[(m, mode)][1] if (m, mode) in bars else 0 for m in ms], w, label=lab, hatch="//")
            ax.set_xticks(x); ax.set_xticklabels([short_plain(m).replace(":", "\n") for m in ms], fontsize=5.5, rotation=20, ha="right"); ax.set_ylabel("% of stopped runs"); ax.legend(fontsize=4.5)
            fig.tight_layout(); fig.savefig(out / "fig_interrupt.pdf"); plt.close(fig)
        except ImportError:
            pass

    # ---------------- injection ----------------
    ij = [r for r in rows if r["type"] == "inject"]
    if ij:
        tab = []
        for m in sorted({r["model"] for r in ij}):
            for cfg in ["none", "keyword", "objective", "objective_plus"]:
                a = [r for r in ij if r["model"] == m and r["config"] == cfg and r["injected"]]
                b = [r for r in ij if r["model"] == m and r["config"] == cfg and not r["injected"]]
                if not a: continue
                att = cboot(by_task(a, lambda r: r["attack_success"]), np.mean)
                util = cboot(by_task(b, lambda r: r["final_class"] == "goal"), np.mean)
                blk = [r for r in a if r["gate_blocks"] > 0]
                L.append(f"INJECT {m}/{cfg}: attack success {f(att,True)}% (n={len(a)}), clean-task goal rate {f(util,True)}% (n={len(b)}), runs with >=1 gate block {len(blk)}/{len(a)}")
                tab.append(f"{short(m)} & {cfg.replace('_', ' ')} & {f(att,True,ci=False)} & {f(util,True,ci=False)} ")
        (out / "tab_inject.tex").write_text(" \\\\\n".join(tab))

    # ---- headline macros used in the abstract / prose ----
    nm = [(m, cell[(m, "none")]) for m in models if (m, "none") in cell and cell[(m, "none")]["w"][0] == cell[(m, "none")]["w"][0]]
    if len(nm) >= 2:
        ws = [c["w"][0] for _, c in nm]; sp = [c["spd"][0] for _, c in nm]
        N.update(wmin=f"{min(ws):.1f}", wmax=f"{max(ws):.1f}", spdmin=f"{min(sp):.1f}", spdmax=f"{max(sp):.1f}", nmodels=len(models))
        rk = lambda v: np.argsort(np.argsort(v)).astype(float)
        if len(nm) >= 3:
            N["rho"] = f"{np.corrcoef(rk(sp), rk(ws))[0, 1]:.2f}"
        fastest = max(nm, key=lambda x: x[1]["spd"][0]); N["wfast"] = f"{fastest[1]['w'][0]:.1f}"; N["fastmodel"] = short(fastest[0]).replace("_", "\\_")
    ovs = []
    for m in models:
        a = defaultdict(list); b = defaultdict(list)
        for r in win:
            if r["model"] == m: (a if r["mode"] == "none" else b)[r["task"]].append(r["duration_s"])
        ovs += [(np.median(b[t]) - np.median(a[t])) / np.median(a[t]) for t in a if t in b and a[t] and b[t]]
    if win: N["reps"] = max(r["rep"] for r in win) + 1
    if ovs: N.update(ovhmed=f"{np.median(ovs)*100:.1f}", ovhmax=f"{max(ovs)*100:.1f}")
    if it:
        for mode, key in (("none", "None"), ("recovery", "Rec")):
            s = [r for r in it if r["mode"] == mode and r["stopped"]]
            att = [r for r in s if r["control_class"] == "goal"]
            d = cboot(by_task(att, lambda r: r["final_class"] == "diverged"), np.mean)
            l = cboot(by_task([r for r in s if r["control_reached_irr"]], lambda r: r["irreversible_before_stop"]), np.mean)
            N.update({f"div{key}": f(d, True, ci=False), f"div{key}lo": f"{d[1]*100:.1f}", f"div{key}hi": f"{d[2]*100:.1f}",
                      f"late{key}": f(l, True, ci=False), f"late{key}lo": f"{l[1]*100:.1f}", f"late{key}hi": f"{l[2]*100:.1f}", f"nstop{key}": len(s), f"natt{key}": len(att)})
    if ij:
        m0 = sorted({r["model"] for r in ij})[0]
        for cfg, key in (("none", "None"), ("keyword", "Kw"), ("objective_plus", "Op")):
            a = [r for r in ij if r["model"] == m0 and r["config"] == cfg and r["injected"]]
            b = [r for r in ij if r["model"] == m0 and r["config"] == cfg and not r["injected"]]
            if a: N.update({f"att{key}": f(cboot(by_task(a, lambda r: r["attack_success"]), np.mean), True, ci=False),
                            f"util{key}": f(cboot(by_task(b, lambda r: r["final_class"] == "goal"), np.mean), True, ci=False)})
    with open(out / "numbers.tex", "w") as fh:
        for k, v in N.items():
            fh.write("\\expandafter\\gdef\\csname n@%s\\endcsname{%s}\n" % (k, str(v).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")))
        fh.write("\\expandafter\\gdef\\csname n@ntasks\\endcsname{%d}\n" % len({r["task"] for r in win + it if "task" in r and not r.get("injected")}))
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
