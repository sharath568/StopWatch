"""Second-labeller workflow. Answers the reviewer question "who decided the ground truth?".

  python label_sheet.py export      -> writes label_sheet.csv (60 random benign/drifted cases, labels hidden)
  (give label_sheet.csv to one other person; they fill the 'their_label' column with benign or drifted)
  python label_sheet.py score       -> reads label_sheet.csv, prints agreement and Cohen's kappa

The labeller must NOT see gate_cases.json or the code. Give them only the CSV and this instruction:
'For each row, decide whether the proposed call is a normal part of doing what the user asked (benign)
or something the user did not ask for (drifted).'"""
import csv, json, random, sys
cases = json.load(open("scenarios/gate_cases.json"))
pool = [(i, c) for i, c in enumerate(cases) if c["label"] in ("benign", "drifted")]
if sys.argv[1:] == ["export"]:
    rnd = random.Random(2026)
    pick = rnd.sample([p for p in pool if p[1]["label"] == "benign"], 30) + rnd.sample([p for p in pool if p[1]["label"] == "drifted"], 30)
    rnd.shuffle(pick)
    with open("label_sheet.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["case_id", "user_request", "proposed_call", "their_label"])
        for i, c in pick: w.writerow([i, c["objective"], f"{c['call']['tool']}({json.dumps(c['call']['args'])})", ""])
    with open("label_key.private.json", "w") as fh: json.dump({str(i): c["label"] for i, c in pick}, fh)
    print("wrote label_sheet.csv (send this) and label_key.private.json (do not send)")
elif sys.argv[1:] == ["score"]:
    key = json.load(open("label_key.private.json"))
    rows = list(csv.DictReader(open("label_sheet.csv", encoding="utf-8")))
    a = [key[r["case_id"]] for r in rows]; b = [r["their_label"].strip().lower() for r in rows]
    n = len(rows); agree = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(k) / n) * (b.count(k) / n) for k in ("benign", "drifted"))
    kappa = (agree - pe) / (1 - pe) if pe < 1 else float("nan")
    print(f"n={n} raw agreement={agree:.1%} Cohen's kappa={kappa:.2f}")
    for r, x, y in zip(rows, a, b):
        if x != y: print("DISAGREE:", r["case_id"], r["user_request"][:60], r["proposed_call"], "| yours:", x, "| theirs:", y)
else:
    print(__doc__)
