"""LLM clients. OllamaClient talks to a local Ollama server (no data leaves the machine).
StubLLM is deterministic and exists ONLY to smoke-test the harness plumbing;
its outputs must never be reported as results."""
from __future__ import annotations
import json, time, re
import urllib.request


class OllamaClient:
    def __init__(self, model: str, host: str = "http://localhost:11434", seed: int = 0, temperature: float = 0.0):
        assert host.startswith("http://localhost") or host.startswith("http://127.0.0.1"), \
            "privacy guard: only a local model server is allowed"
        self.model, self.host, self.seed, self.temperature = model, host, seed, temperature

    def chat_json(self, system: str, user: str) -> tuple[dict, dict]:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature, "seed": self.seed},
        }
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=900) as r:
            resp = json.loads(r.read())
        wall = time.perf_counter() - t0
        stats = {
            "wall_s": wall,
            "eval_count": resp.get("eval_count"),
            "eval_s": (resp.get("eval_duration") or 0) / 1e9,
            "prompt_eval_s": (resp.get("prompt_eval_duration") or 0) / 1e9,
        }
        stats["decode_tok_s"] = (stats["eval_count"] / stats["eval_s"]) if stats["eval_s"] else None
        return parse_json(resp["message"]["content"]), stats

    def warmup(self):
        """Load the model into memory so the first measured run does not include load time."""
        self.chat_json("Reply with JSON.", '{"ping": true}')


def parse_json(text: str) -> dict:
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}


class StubLLM:
    """Replays each task's reference plan with an artificial delay. SMOKE TEST ONLY."""
    def __init__(self, model: str = "stub", delay_s: float = 0.01):
        self.model, self.delay_s, self._plan, self._i = model, delay_s, [], 0

    def load_plan(self, plan):
        self._plan, self._i = list(plan), 0

    def warmup(self):
        pass

    def chat_json(self, system: str, user: str):
        time.sleep(self.delay_s)
        stats = {"wall_s": self.delay_s, "eval_count": 20, "eval_s": self.delay_s,
                 "prompt_eval_s": 0.0, "decode_tok_s": 20 / self.delay_s}
        if "serves_objective" in system or "verdict" in system:        # judge call
            return {"serves_objective": True, "verdict": "serves", "reason": "stub"}, stats
        if self._i >= len(self._plan):
            return {"done": True}, stats
        step = self._plan[self._i]; self._i += 1
        return {"tool": step["tool"], "args": step["args"]}, stats
