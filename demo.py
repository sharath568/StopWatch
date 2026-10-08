"""Live demo: watch the agent act on the simulated home and press Enter to say "stop".

  python demo.py --model qwen2.5:3b-instruct-q4_K_M --task movie_night --mode none
  python demo.py --model qwen2.5:3b-instruct-q4_K_M --task movie_night --mode recovery --tau 5
Run both modes side by side for the talk: without recovery the message is already sent by the time
you react; with recovery you see the announcement, press Enter, and the completed steps are undone."""
import argparse, json, sys, threading, time
from llm import OllamaClient
from agent import run_task

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--task", default="movie_night")
ap.add_argument("--mode", default="recovery", choices=["none", "recovery"]); ap.add_argument("--tau", type=float, default=5.0)
a = ap.parse_args()
task = {t["id"]: t for t in json.load(open("scenarios/tasks.json"))}[a.task]
stop = threading.Event()
print(f"\nTask: {task['objective']}\nMode: {a.mode}   (press Enter at any time to say STOP)\n")
threading.Thread(target=lambda: (sys.stdin.readline(), stop.set(), print(">>> STOP")), daemon=True).start()

def show(e):
    extra = f"   (waiting {a.tau:.0f}s; press Enter to cancel)" if e["event"] == "announced" else ""
    print(f"{e['t']:6.1f}s  {e['event']:<10} {e['call']['tool']}({e['call']['args']}){extra}", flush=True)

res = run_task(OllamaClient(a.model), task, a.mode, tau=a.tau, stop_event=stop, on_event=show)
print(f"\nFinal state: {res['final_class']}   steps undone: {res['compensated_steps']}   "
      f"first irreversible action at: {res['t_first_irreversible_s'] and round(res['t_first_irreversible_s'], 1)} s")
