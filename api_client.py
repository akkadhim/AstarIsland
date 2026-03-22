"""
Astar Island API client.
Set your JWT token in the TOKEN variable or pass it directly.
"""

import requests
import json
import time
from typing import Optional

BASE = "https://api.ainm.no"

class AstarClient:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"
        self.session.headers["Content-Type"] = "application/json"

    def get_rounds(self):
        r = self.session.get(f"{BASE}/astar-island/rounds")
        r.raise_for_status()
        return r.json()

    def get_active_round(self):
        rounds = self.get_rounds()
        for r in rounds:
            if r["status"] == "active":
                return r
        return None

    def get_round_detail(self, round_id: str):
        r = self.session.get(f"{BASE}/astar-island/rounds/{round_id}")
        r.raise_for_status()
        return r.json()

    def get_budget(self):
        r = self.session.get(f"{BASE}/astar-island/budget")
        r.raise_for_status()
        return r.json()

    def simulate(self, round_id: str, seed_index: int,
                 viewport_x: int, viewport_y: int,
                 viewport_w: int = 15, viewport_h: int = 15,
                 retry_on_429: bool = True) -> Optional[dict]:
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": viewport_x,
            "viewport_y": viewport_y,
            "viewport_w": viewport_w,
            "viewport_h": viewport_h,
        }
        for attempt in range(5):
            r = self.session.post(f"{BASE}/astar-island/simulate", json=payload)
            if r.status_code == 429 and retry_on_429:
                wait = 0.5 * (2 ** attempt)
                print(f"  Rate limited, waiting {wait:.1f}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("Exceeded retry limit for simulate")

    def submit(self, round_id: str, seed_index: int, prediction: list,
               retry_on_429: bool = True) -> dict:
        """prediction: H×W×6 nested list of floats."""
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        }
        for attempt in range(5):
            r = self.session.post(f"{BASE}/astar-island/submit", json=payload)
            if r.status_code == 429 and retry_on_429:
                wait = 0.5 * (2 ** attempt)
                print(f"  Rate limited, waiting {wait:.1f}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("Exceeded retry limit for submit")

    def get_my_rounds(self):
        r = self.session.get(f"{BASE}/astar-island/my-rounds")
        r.raise_for_status()
        return r.json()

    def get_leaderboard(self):
        r = self.session.get(f"{BASE}/astar-island/leaderboard")
        r.raise_for_status()
        return r.json()
