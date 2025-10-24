#!/usr/bin/env python3
"""
Live TTS (no file) for SASL-AI
- Reads 05_OUTPUT_GENERATED/session_transcripts.json
- Lets user choose a sentence
- Speaks it immediately using pyttsx3
- Automatically selects a female / more natural voice if available
"""

import os, json, sys, time, argparse, hashlib
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
def configure_engine(engine, voice_pref: str, rate: int, volume: float):
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)

    if voice_pref:
        pref = voice_pref.lower()
        female_keywords = {"female", "zira", "aria", "samantha", "ava", "jenny"}
        male_keywords   = {"male", "david", "mark", "alex", "george", "daniel"}
        for v in engine.getProperty('voices'):
            name_lower = v.name.lower()
            gender = getattr(v, 'gender', '').lower()
            if pref.startswith('f'):
                if any(k in name_lower for k in female_keywords) or 'female' in gender:
                    engine.setProperty('voice', v.id)
                    break
            elif pref.startswith('m'):
                if any(k in name_lower for k in male_keywords) or 'male' in gender:
                    engine.setProperty('voice', v.id)
                    break

def speak(text: str):
    import pyttsx3
    engine = pyttsx3.init()
    configure_engine(engine, 'female', 160, 0.9)
    engine.say(text)
    engine.runAndWait()

def synth_to_file(text: str, out_path: Path, voice: str, rate: int, volume: float) -> Path:
    import pyttsx3
    out_path.parent.mkdir(parents=True, exist_ok=True)
    engine = pyttsx3.init()
    configure_engine(engine, voice, rate, volume)
    engine.save_to_file(text, str(out_path))
    engine.runAndWait()  # ensure file is written
    return out_path

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
    parser = argparse.ArgumentParser(description="TTS utility (interactive or CLI)")
    parser.add_argument('--text', help='Text to synthesize to WAV (non-interactive).')
    parser.add_argument('--out', help='Output WAV path (non-interactive).')
    parser.add_argument('--voice', default='female', help='Preferred voice (female|male).')
    parser.add_argument('--rate', type=int, default=160, help='Speech rate (default 160).')
    parser.add_argument('--volume', type=float, default=0.9, help='Volume 0..1 (default 0.9).')
    parser.add_argument('--hash-name', action='store_true', help='Derive filename from SHA256(text). Overrides --out basename.')
    args = parser.parse_args()

    if not args.text:
        # fallback to legacy interactive mode
        run_interactive()
        sys.exit(0)

    text = args.text.strip()
    if not text:
        print(json.dumps({"ok": False, "error": "Empty text"}))
        sys.exit(1)

    # Decide output path
    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_dir = Path(os.getenv('TTS_OUTPUT_DIR', Path(__file__).resolve().parent / 'tts_audio'))
        out_dir.mkdir(parents=True, exist_ok=True)
        base = 'speech.wav'
        out_path = out_dir / base

    if args.hash_name:
        h = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        out_path = out_path.with_name(f"tts_{h}.wav")

    start = time.time()
    try:
        synth_to_file(text, out_path, args.voice, args.rate, args.volume)
        size = out_path.stat().st_size if out_path.exists() else 0
        elapsed = int((time.time() - start) * 1000)
        print(json.dumps({
            "ok": True,
            "path": str(out_path),
            "bytes": size,
            "elapsed_ms": elapsed
        }))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        sys.exit(2)