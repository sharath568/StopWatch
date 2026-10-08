"""Pilot check: run this on a SMALL window run before spending overnight runs.

  python run.py window --models llama3.2:1b-instruct-q4_K_M qwen2.5:3b-instruct-q4_K_M --reps 1
  python check_pilot.py results/window_<timestamp>.jsonl

What you should see if the v2 benchmark fix worked: the 3B model reaches the goal on a clear share of
tasks (v1 was near 0% on tasks with unstated values). If goal% is still ~0 for every task and model,
STOP and look at one run with demo.py before doing anything else."""
import json, sys
from collections import defaultdict
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f) if l.strip()]
rows = [r for r in rows if r.get("type") == "window" and "error" not in r]
errs = sum(1 for f in sys.argv[1:] for l in open(f) if '"error"' in l)
per = defaultdict(lambda: defaultdict(int))
for r in rows: per[(r["model"], r["task"])][r["final_class"]] += 1
print(f"{len(rows)} usable rows, {errs} error rows\n")
print(f"{'model':30} {'task':14} goal  init  div")
for (m, t), c in sorted(per.items()):
    print(f"{m:30} {t:14} {c['goal']:4}  {c['initial']:4}  {c['diverged']:3}")
tot = defaultdict(lambda: [0, 0])
for r in rows: tot[r["model"]][1] += 1; tot[r["model"]][0] += r["final_class"] == "goal"
print("\ngoal rate per model:", {m: f"{g}/{n} = {100*g/n:.0f}%" for m, (g, n) in tot.items()})
zero = [t for (m, t), c in per.items() if c["goal"] == 0]
allz = defaultdict(lambda: True)
for (m, t), c in per.items(): allz[t] &= (c["goal"] == 0)
bad = [t for t, z in allz.items() if z]
print("tasks that NO model completed even once:", bad or "none")
