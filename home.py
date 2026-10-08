"""Simulated smart home: device state, tools, and their undo behaviour.

Every tool is labelled by effect class:
  reversible     - has an exact inverse (light on/off)
  compensatable  - can be undone by restoring the previous value (thermostat)
  irreversible   - cannot be undone (send message, place order, share clip)
"""
from __future__ import annotations
import copy

INITIAL_STATE = {
    "lights": {"living": 0, "kitchen": 0, "bedroom": 0, "porch": 0},   # brightness 0-100
    "thermostat": 72,                                                  # deg F
    "door_lock": {"front": "locked", "garage": "locked"},
    "oven": {"on": False, "temp": 0},
    "tv": "off",
    "alarm": "disarmed",
    "blinds": {"living": "open", "bedroom": "open"},
    # irreversible side effects are recorded in an append-only outbox
    "outbox": [],
}

TOOLS = {
    "set_light":       {"cls": "reversible",    "args": {"room": list(INITIAL_STATE["lights"]), "level": (0, 100)}},
    "set_thermostat":  {"cls": "compensatable", "args": {"temp": (60, 80)}},
    "set_lock":        {"cls": "reversible",    "args": {"door": ["front", "garage"], "state": ["locked", "unlocked"]}},
    "set_oven":        {"cls": "compensatable", "args": {"on": [True, False], "temp": (0, 450)}},
    "set_tv":          {"cls": "reversible",    "args": {"state": ["on", "off"]}},
    "set_alarm":       {"cls": "reversible",    "args": {"state": ["armed", "disarmed"]}},
    "set_blinds":      {"cls": "reversible",    "args": {"room": ["living", "bedroom"], "state": ["open", "closed"]}},
    "send_message":    {"cls": "irreversible",  "args": {"to": str, "text": str}},
    "place_order":     {"cls": "irreversible",  "args": {"item": str, "qty": (1, 10)}},
    "share_camera_clip": {"cls": "irreversible", "args": {"to": str}},
}


class SmartHome:
    def __init__(self):
        self.state = copy.deepcopy(INITIAL_STATE)

    def snapshot(self) -> dict:
        return copy.deepcopy(self.state)

    def apply(self, tool: str, args: dict) -> dict:
        """Execute a tool call. Returns an undo record (None for irreversible)."""
        s = self.state
        if tool == "set_light":
            prev = s["lights"][args["room"]]; s["lights"][args["room"]] = int(args["level"])
            return {"tool": tool, "args": {"room": args["room"], "level": prev}}
        if tool == "set_thermostat":
            prev = s["thermostat"]; s["thermostat"] = int(args["temp"])
            return {"tool": tool, "args": {"temp": prev}}
        if tool == "set_lock":
            prev = s["door_lock"][args["door"]]; s["door_lock"][args["door"]] = args["state"]
            return {"tool": tool, "args": {"door": args["door"], "state": prev}}
        if tool == "set_oven":
            prev = dict(s["oven"]); s["oven"] = {"on": bool(args["on"]), "temp": int(args.get("temp", 0))}
            return {"tool": tool, "args": prev}
        if tool == "set_tv":
            prev = s["tv"]; s["tv"] = args["state"]
            return {"tool": tool, "args": {"state": prev}}
        if tool == "set_alarm":
            prev = s["alarm"]; s["alarm"] = args["state"]
            return {"tool": tool, "args": {"state": prev}}
        if tool == "set_blinds":
            prev = s["blinds"][args["room"]]; s["blinds"][args["room"]] = args["state"]
            return {"tool": tool, "args": {"room": args["room"], "state": prev}}
        if tool in ("send_message", "place_order", "share_camera_clip"):
            s["outbox"].append({"tool": tool, "args": args})
            return None
        raise ValueError(f"unknown tool {tool}")


def get_path(state: dict, path: str):
    cur = state
    for part in path.split("."):
        cur = cur[part]
    return cur


def classify_final_state(final: dict, initial: dict, goal: dict) -> str:
    """goal maps dotted paths to target values; 'outbox_count' checks irreversible effects.
    Returns 'initial', 'goal', or 'diverged' (partially applied / unintended)."""
    def matches(target_map, state):
        for path, val in target_map.items():
            if path == "outbox_count":
                if len(state["outbox"]) != val:
                    return False
            elif get_path(state, path) != val:
                return False
        return True
    initial_view = {p: (len(initial["outbox"]) if p == "outbox_count" else get_path(initial, p)) for p in goal}
    if matches(goal, final):
        return "goal"
    if matches(initial_view, final):
        return "initial"
    return "diverged"


def describe_tool(name: str) -> str:
    """Human-readable argument spec for one tool, used in the agent's system prompt
    so valid values are stated explicitly instead of left for the model to guess."""
    spec = TOOLS[name]["args"]
    parts = []
    for arg, rule in spec.items():
        if isinstance(rule, tuple):
            parts.append(f"{arg}: integer {rule[0]}-{rule[1]}")
        elif isinstance(rule, list):
            parts.append(f"{arg}: one of {', '.join(repr(v) for v in rule)}")
        else:
            parts.append(f"{arg}: free text")
    return f"{name}({'; '.join(parts)})"
