from __future__ import annotations

import json
import requests


class CloudTeacher78B:
    """HTTP client for cloud teacher inference."""

    def __init__(self, api_url: str = "http://localhost:8000/v1/inference"):
        self.api_url = api_url

    def analyse_motion(self, description: str):
        payload = {
            "prompt": f"""
You are a world-class badminton professional coach.
Given the motion description below, produce:
1. Identify mistakes
2. Correction suggestions
3. Professional scoring 0~100
4. Key biomechanical insights

Return JSON.

Motion:
{description}
"""
        }
        response = requests.post(self.api_url, json=payload)
        try:
            return response.json()
        except Exception:
            return {"error": "Invalid response", "raw": response.text}
