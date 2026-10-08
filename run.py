"""Experiment runner (v2).

  python run.py gate      --judges keyword objective objective_plus --judge-model qwen2.5:3b-instruct-q4_K_M   # RQ1
  python run.py window    --models M1 M2 ... --reps 5 [--use-gate --judge objective_plus]                     # RQ3
  python run.py interrupt --models M1 M2 ... --reps 5                                                          # RQ2
  python run.py inject    --models M1 --reps 3                                                                 # injection defence
  add --stub to smoke-test without a model server (NOT for results)

Design notes
  * reps use temperature --temperature (default 0.7) and seed = rep, so reps are genuine samples.
  * interrupt is PAIRED: for each (model, task, rep) an uninterrupted control run comes first; the stop
    time is u * R with u ~ U(--stop-frac-min, --stop-frac-max) and R = the control's time of first
    irreversible action (its duration if it never reaches one). The same u is used for both modes, so
    the stop lands at the same relative point for 'none' and 'recovery'.
  * every model is warmed up once so model-load time is not counted in any measurement.
  * failures are recorded as error rows, not crashes.
Writes one JSONL file per invocation under results/; first row records the environment."""
import argparse, json, platform, random, time, os, subprocess, sys
from pathlib import Path
from llm import OllamaClient, StubLLM
from gate import JUDGES, run_gate, KeywordJudge
from agent import run_task


def _ram_gb():
    try:
        import ctypes
        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                        ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
        s = MS(); s.l = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return round(s.tp / 2**30, 1)
    except Exception:
        try:
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
        except Exception:
            return None


def _cmd(*a):
    try:
        return subprocess.check_output(a, text=True, stderr=subprocess.DEVNULL, timeout=20).strip()
    except Exception:
        return None


def env_row(args):
    return {"type": "env", "args": vars(args), "python": platform.python_version(), "machine": platform.machine(),
            "system": platform.platform(), "cpu": platform.processor(), "cpu_count": os.cpu_count(), "ram_gb": _ram_gb(),
            "ollama_version": _cmd("ollama", "--version"), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": _cmd("git", "rev-parse", "HEAD"), "git_dirty": bool(_cmd("git", "status", "--porcelain"))}


def make_llm(model, stub, seed=0, delay=0.01, temperature=0.0):
    return StubLLM(model=f"stub:{model}", delay_s=delay) if stub else OllamaClient(model, seed=seed, temperature=temperature)


def load_tasks(injected):
    return [t for t in json.load(open("scenarios/tasks.json")) if t["injected"] == injected]


def safe_run(out, base, fn):
    try:
        res = fn()
        res.update(base)
    except Exception as e:                       # timeouts, server errors: record, keep going
        res = {**base, "error": f"{type(e).__name__}: {e}"}
    out.write(json.dumps(res) + "\n"); out.flush()
    return res


def cmd_gate(args, out):
    cases = json.load(open("scenarios/gate_cases.json"))
    llm = make_llm(args.judge_model, args.stub, temperature=0.0)
    if not args.stub: llm.warmup()
    for jname in args.judges:
        judge = JUDGES[jname](llm)
        for i, c in enumerate(cases):
            for rep in range(1):   # judges use temperature 0, so repeats add nothing
                decision, failed, lat = run_gate(c["call"], c["objective"], c["allowed_tools"], judge, history=c.get("history"))
                out.write(json.dumps({"type": "gate", "judge": judge.name, "judge_model": args.judge_model if jname != "keyword" else None,
                                      "case": i, "task": c["task"], "phrasing": c.get("phrasing"), "label": c["label"], "decision": decision,
                                      "failed_check": failed, "judge_latency_s": lat, "rep": rep}) + "\n")
        out.flush()


def _judge_for(args, model, stub):
    if not args.use_gate: return None
    return JUDGES[args.judge](make_llm(args.judge_model or model, stub, temperature=0.0))


def cmd_window(args, out):
    tasks = load_tasks(False)
    for model in args.models:
        warm = make_llm(model, args.stub); 
        if not args.stub: warm.warmup()
        for mode in ["none", "recovery"]:
            for t in tasks:
                for r in range(args.reps):
                    llm = make_llm(model, args.stub, seed=r, delay=args.stub_delay, temperature=args.temperature)
                    j = _judge_for(args, model, args.stub)
                    safe_run(out, {"type": "window", "model": model, "rep": r},
                             lambda: run_task(llm, t, mode, tau=args.tau, stop_at_s=None, use_gate=args.use_gate, judge=j))


def cmd_interrupt(args, out):
    tasks = load_tasks(False)
    rnd = random.Random(args.seed)
    for model in args.models:
        warm = make_llm(model, args.stub)
        if not args.stub: warm.warmup()
        for t in tasks:
            for r in range(args.reps):
                llm = make_llm(model, args.stub, seed=r, delay=args.stub_delay, temperature=args.temperature)
                j = _judge_for(args, model, args.stub)
                ctrl = safe_run(out, {"type": "control", "model": model, "rep": r},
                                lambda: run_task(llm, t, "none", tau=args.tau, stop_at_s=None, use_gate=args.use_gate, judge=j))
                if "error" in ctrl: continue
                R = ctrl["t_first_irreversible_s"] if ctrl["t_first_irreversible_s"] is not None else ctrl["duration_s"]
                u = rnd.uniform(args.stop_frac_min, args.stop_frac_max)
                for mode in ["none", "recovery"]:
                    llm = make_llm(model, args.stub, seed=r, delay=args.stub_delay, temperature=args.temperature)
                    safe_run(out, {"type": "interrupt", "model": model, "rep": r, "u": u, "ref_s": R,
                                   "control_class": ctrl["final_class"], "control_reached_irr": ctrl["t_first_irreversible_s"] is not None},
                             lambda: run_task(llm, t, mode, tau=args.tau, stop_at_s=u * R, use_gate=args.use_gate, judge=j))


def cmd_inject(args, out):
    inj = load_tasks(True)
    base = {t["id"]: t for t in load_tasks(False)}
    for model in args.models:
        warm = make_llm(model, args.stub)
        if not args.stub: warm.warmup()
        for cfg in args.configs:                      # none | keyword | objective | objective_plus
            for t in inj + [base[x["id"].replace("_injected", "")] for x in inj]:
                for r in range(args.reps):
                    llm = make_llm(model, args.stub, seed=r, delay=args.stub_delay, temperature=args.temperature)
                    use_gate = cfg != "none"
                    j = JUDGES[cfg](make_llm(args.judge_model or model, args.stub, temperature=0.0)) if use_gate else None
                    safe_run(out, {"type": "inject", "model": model, "rep": r, "config": cfg},
                             lambda: run_task(llm, t, "none", use_gate=use_gate, judge=j))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["gate", "interrupt", "window", "inject"])
    p.add_argument("--models", nargs="+", default=["qwen2.5:3b-instruct-q4_K_M"])
    p.add_argument("--judge-model", default=None, help="model used by ObjectiveJudge(+); default = the agent model")
    p.add_argument("--judges", nargs="+", default=["keyword", "objective", "objective_plus"], choices=list(JUDGES))
    p.add_argument("--judge", default="objective_plus", choices=list(JUDGES), help="C3 judge when --use-gate is set")
    p.add_argument("--configs", nargs="+", default=["none", "keyword", "objective_plus"])
    p.add_argument("--use-gate", action="store_true", help="enable the detection layer inside window/interrupt runs")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--tau", type=float, default=3.0, help="pause before irreversible actions (s)")
    p.add_argument("--stop-frac-min", type=float, default=0.1)
    p.add_argument("--stop-frac-max", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stub", action="store_true")
    p.add_argument("--stub-delay", type=float, default=0.01)
    args = p.parse_args()
    if args.cmd == "gate" and args.judge_model is None:
        args.judge_model = args.models[0]
    Path("results").mkdir(exist_ok=True)
    path = Path("results") / f"{args.cmd}_{time.strftime('%Y%m%d_%H%M%S')}{'_STUB' if args.stub else ''}.jsonl"
    with open(path, "w") as out:
        out.write(json.dumps(env_row(args)) + "\n")
        {"gate": cmd_gate, "interrupt": cmd_interrupt, "window": cmd_window, "inject": cmd_inject}[args.cmd](args, out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
