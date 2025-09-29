#!/usr/bin/env python3
"""
Live TTS (no file) for SASL-AI
- Reads 05_OUTPUT_GENERATED/session_transcripts.json
- Lets user choose a sentence
- Speaks it immediately using pyttsx3
- Automatically selects a female / more natural voice if available
"""

import os, json, sys
from pathlib import Path
from typing import List, Dict, Any

# ----------------------------------------------------------------------
# Utility paths
# ----------------------------------------------------------------------
def base_dir() -> Path:
    return Path(__file__).resolve().parent.parent

def transcripts_path() -> Path:
    return base_dir() / "05_OUTPUT_GENERATED" / "session_transcripts.json"

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_sessions(fp: Path) -> List[Dict[str, Any]]:
    if not fp.exists():
        raise FileNotFoundError(f"Transcript not found: {fp}")
    data = json.loads(fp.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Transcript JSON must be a list of session objects.")
    sessions = [s for s in data if isinstance(s, dict) and "final_sentence" in s]
    if not sessions:
        raise ValueError("No sessions with 'final_sentence' found.")
    try:
        # newest first
        sessions.sort(key=lambda x: x.get("created_utc",""), reverse=True)
    except Exception:
        pass
    return sessions

def truncate(s: str, n: int = 70) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n-1] + "…"

def choose_sentence(sessions: List[Dict[str, Any]]) -> str:
    print("\nAvailable transcript sentences (newest first):")
    print("=" * 60)
    for i, s in enumerate(sessions, 1):
        sid = s.get("session_id","(no-id)")
        when = s.get("created_utc","")
        sent = s.get("final_sentence","").strip()
        print(f"{i:>2}. [{sid}] {truncate(sent)}  ({when})")
    print(" 0. Cancel")
    while True:
        choice = input("\nSelect a number to speak: ").strip()
        if choice == "0":
            return ""
        if choice.isdigit() and 1 <= int(choice) <= len(sessions):
            return sessions[int(choice)-1]["final_sentence"].strip()
        print("Invalid choice. Try again.")

# ----------------------------------------------------------------------
# Speech
# ----------------------------------------------------------------------
def speak(text: str) -> None:
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)   # slightly slower for natural speech
    engine.setProperty("volume", 0.9) # slight headroom

    # -------- try to select a female / natural voice --------
    female_keywords = {"female", "zira", "aria", "samantha", "ava", "jenny"}
    for v in engine.getProperty('voices'):
        name_lower = v.name.lower()
        gender = getattr(v, 'gender', '').lower()
        if any(k in name_lower for k in female_keywords) or "female" in gender:
            engine.setProperty('voice', v.id)
            break
    # --------------------------------------------------------

    print("\nSpeaking…")
    engine.say(text)
    engine.runAndWait()  # blocks until finished
    print("Done.")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def run_interactive() -> None:
    print("\nText-to-Speech (pyttsx3 – live)")
    print("=" * 40)
    try:
        sessions = load_sessions(transcripts_path())
    except Exception as e:
        print(f"Error loading transcripts: {e}")
        return

    text = choose_sentence(sessions)
    if not text:
        print("Cancelled.")
        return

    speak(text)

if __name__ == "__main__":
    run_interactive()