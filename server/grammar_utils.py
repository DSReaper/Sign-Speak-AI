
from __future__ import annotations
import os, json
from datetime import datetime
from typing import List, Dict, Tuple

# ========================= 
# # COMPREHENSIVE GRAMMAR # 
# =========================

V_TIME_FUTURE = {"tomorrow","next","later","soon","tonight","this_evening","this_afternoon"}
V_TIME_PAST   = {"yesterday","ago","earlier","last"}
TIME_WORDS    = {"today","now","morning","afternoon","evening","tonight","week","month","year",
                 "monday","tuesday","wednesday","thursday","friday","saturday","sunday",
                 "tomorrow","yesterday","later","soon","next","last","ago","earlier"}
NEG_WORDS     = {"not","no","dont","don't","cant","can't","wont","won't","never","nothing","none"}
Q_WORDS       = {"who","what","when","where","why","how","whom","which"}

DEST_NOUNS_TO = {"school","work","home","office","shop","store","church","bank","clinic","hospital",
                 "class","lecture","campus","gym","library"}

PLURALS       = {"books":"book","cars":"car","hands":"hand","days":"day","weeks":"week",
                 "months":"month","years":"year"}

PRONOUNS = {"i":"I","you":"you","he":"he","she":"she","they":"they","we":"we",
            "me":"I","him":"he","her":"she","them":"they","us":"we"}

A_AN_SET = {"coffee","apple","idea","appointment","car","meeting","book","message","question","lunch","dinner"}

# ---- Small verb lexicon (expand as needed)
STATIVE_VERBS = {"want","like","need","know","believe","prefer","love","hate","think"}
MOVE_VERBS    = {"go","come","leave","arrive","walk","run","drive","travel"}
COMMON_VERBS  = STATIVE_VERBS | MOVE_VERBS | {
    "work","study","eat","drink","call","meet","see","buy","bring","teach","make","do",
    "get","write","read","have","say","sleep","wait","visit","help","start","finish"
}

IRREG_PAST = {
    "go":"went","see":"saw","be":"was","eat":"ate","meet":"met","have":"had","do":"did",
    "make":"made","come":"came","say":"said","get":"got","buy":"bought","teach":"taught",
    "leave":"left","bring":"brought","run":"ran","write":"wrote","read":"read"
}

# -------------------------- Helper functions ----------------------------- #

def _is_vowel(x:str)->bool: return x[:1].lower() in "aeiou"

def a_an(word:str)->str:
    """Return the word with a/an prefix depending on initial vowel sound.
    (Utility not currently used in pipeline but provided for completeness.)"""
    if not word: return word
    return ("an " if _is_vowel(word) else "a ") + word

def progressive(verb:str)->str:
    v = verb.lower()
    if v.endswith("ie"): return v[:-2]+"ying"
    if v.endswith("e") and v not in {"be","see","flee","knee","tie"}: return v[:-1]+"ing"
    if len(v)>=3 and v[-1] not in "aeiou" and v[-2] in "aeiou" and v[-3] not in "aeiou":
        return v + v[-1] + "ing"
    return v + "ing"

def past_simple(verb:str)->str:
    v = verb.lower()
    if v in IRREG_PAST: return IRREG_PAST[v]
    if v.endswith("e"): return v + "d"
    if len(v)>=3 and v[-1] not in "aeiou" and v[-2] in "aeiou" and v[-3] not in "aeiou":
        return v + v[-1] + "ed"
    return v + "ed"

def present_3rd_s(verb:str)->str:
    v = verb.lower()
    if v in {"have"}: return "has"
    if v.endswith(("s","sh","ch","x","z","o")): return v + "es"
    if v.endswith("y") and (len(v)>=2 and v[-2] not in "aeiou"): return v[:-1] + "ies"
    return v + "s"

def choose_subject(tokens: List[str])->str:
    for t in tokens:
        lt = t.lower()
        if lt in PRONOUNS: return PRONOUNS[lt]
    return "I"

def detect_question(tokens: List[str])->bool:
    return any(t.lower() in Q_WORDS for t in tokens)

def detect_negation(tokens: List[str])->bool:
    return any(t.lower() in NEG_WORDS for t in tokens)

def split_time_tokens(tokens: List[str])->Tuple[List[str], List[str]]:
    times, rest = [], []
    for t in tokens:
        if t.lower() in TIME_WORDS: times.append(t.lower())
        else: rest.append(t)
    return times, rest

def plural_to_singular(w:str)->str:
    return PLURALS.get(w.lower(), w.lower())

def insert_articles(words: List[str])->List[str]:
    out=[]
    determiners = {"a","an","the","my","your","his","her","their","our","this","that"}
    for i,w in enumerate(words):
        lw = w.lower()
        if lw in A_AN_SET:
            if i>0 and words[i-1].lower() in determiners:
                out.append(lw)
            else:
                out.extend(["an" if _is_vowel(lw) else "a", lw])
        else:
            out.append(lw)
    return out

def map_destinations(words: List[str])->List[str]:
    out=[]
    i=0
    while i < len(words):
        lw = words[i].lower()
        if lw in DEST_NOUNS_TO:
            if lw in {"home","work"}:
                out.append(lw)   # “go home/work” (no “to”)
            else:
                 # ensure single preceding "to" only if not already "to"
                if out and out[-1] == "to":
                    out.append(lw)
                else:
                    out.extend(["to", lw])
        else:
            out.append(lw)
        i += 1
    return out

def pick_verb_and_rest(core: List[str]):
    """
    Try to pick a sensible main verb from the core tokens.
    If the first token is a known verb, use it; otherwise if a later token is a verb, pull it forward.
    """
    if not core: return "", []
      # find first verb
    for idx, tok in enumerate(core):
        if tok.lower() in COMMON_VERBS:
            verb = tok.lower()
            rest = core[:idx] + core[idx+1:]
            return verb, rest
    # fallback: treat first token as verb if it looks like one (very naive), else default to 'want'
    first = core[0].lower()
    if first in {"go","be","have","do"}:
        return first, core[1:]
    return "want", core

def aux_for(subject:str, tense:str)->str:
    subj = subject
    if tense == "present":
        return "am" if subj=="I" else ("are" if subj in {"you","we","they"} else "is")
    if tense == "past":
        return "was" if subj in {"I","he","she"} else "were"
    return "will"

def needs_progressive(verb:str, tense:str, time_tokens:List[str])->bool:
    if verb in STATIVE_VERBS: return False
    if tense == "future": return False
    if "now" in time_tokens or "today" in time_tokens or "tonight" in time_tokens:
        return True
    return True  # prefer progressive for present in this pipeline

def tidy(s:str)->str:
    s = " ".join(s.split()).strip()
    if not s: return s
    s = s[0].upper()+s[1:]
    if s[-1] not in ".?!": s += "."
    return s

def grammar_fix(words_in: List[str]) -> str:
    """Improved Rule-based SASL→English:
    - better verb selection & conjugation
    - consistent AUX placement
    - negation & questions
    - destinations & articles
    - natural time placement
    """
    if not words_in: return ""
      # normalize tokens
    toks = [w.strip().lower().replace("_"," ") for w in words_in if w and w.strip()]
    toks = [("this_evening" if t=="this evening" else "this_afternoon" if t=="this afternoon" else t) for t in toks]

    subj = choose_subject(toks)
    is_question = detect_question(toks)
    is_neg = detect_negation(toks)
    time_tokens, core = split_time_tokens(toks)

    # light destination & articles
    core = map_destinations(core)
    core = [plural_to_singular(w) for w in core]
    core = insert_articles(core)

    # pick verb + rest
    verb, rest = pick_verb_and_rest(core)

    # short, high-confidence shortcuts
    set_toks = set(toks)
    if {"work","tomorrow"} <= set_toks:
        return tidy(f"{subj} will work tomorrow")
    if {"go","home"} <= set_toks:
        return tidy(f"{subj} am {progressive('go')} home")
    if {"meet","you","tomorrow"} <= set_toks:
        return tidy(f"{subj} will meet you tomorrow")

    # tense decision
    tense = "present"
    if any(t in V_TIME_PAST for t in time_tokens): tense = "past"
    elif any(t in V_TIME_FUTURE for t in time_tokens) or "tomorrow" in time_tokens: tense = "future"

    phrase = ""

    # form the predicate
    if is_question:
        wh = next((t for t in toks if t in Q_WORDS), None)
        if wh:
             # WH + be/do/will + subject + verb(+ing/base) + rest
            if tense == "past":
                aux = "was" if subj in {"I","he","she"} else "were"
                v = progressive(verb) if verb not in STATIVE_VERBS else verb
                phrase = f"{wh.capitalize()} {aux} {subj} {v}"
            elif tense == "future":
                phrase = f"{wh.capitalize()} will {subj} {verb}"
            else:
                 # present: prefer progressive unless stative
                if needs_progressive(verb, tense, time_tokens) and verb not in STATIVE_VERBS:
                    aux = aux_for(subj,"present")
                    phrase = f"{wh.capitalize()} {aux} {subj} {progressive(verb)}"
                else:
                     # simple present with do/does
                    aux = "do" if subj in {"I","you","we","they"} else "does"
                    phrase = f"{wh.capitalize()} {aux} {subj} {verb}"
        else:
             # yes/no: Did/Do/Does/Will + subject + base
            if tense == "past":
                phrase = f"Did {subj} {verb}"
            elif tense == "future":
                phrase = f"Will {subj} {verb}"
            else:
                aux = "Do" if subj in {"I","you","we","they"} else "Does"
                phrase = f"{aux} {subj} {verb}"
    else:
        if tense == "past":
            if is_neg:
                predicate = f"did not {verb}"
            else:
                predicate = past_simple(verb)
            phrase = f"{subj} {predicate}"
        elif tense == "future":
            predicate = f"will not {verb}" if is_neg else f"will {verb}"
            phrase = f"{subj} {predicate}"
        else:
            if verb == "be":
                aux = aux_for(subj,"present")
                predicate = f"{aux} not" if is_neg else aux
                phrase = f"{subj} {predicate}"
            elif needs_progressive(verb, tense, time_tokens) and verb not in STATIVE_VERBS:
                aux = aux_for(subj,"present")
                predicate = f"{aux} not {progressive(verb)}" if is_neg else f"{aux} {progressive(verb)}"
                phrase = f"{subj} {predicate}"
            else:
                 # simple present with 3rd person -s
                base = verb
                if is_neg:
                    aux = "do not" if subj in {"I","you","we","they"} else "does not"
                    predicate = f"{aux} {base}"
                else:
                    main = base if subj in {"I","you","we","they"} else present_3rd_s(base)
                    predicate = main
                phrase = f"{subj} {predicate}"

    # attach rest (clean determiners like "... a a ...")
    if rest:
        cleaned=[]
        for tok in rest:
            if cleaned and tok in {"a","an","the"} and cleaned[-1] in {"a","an","the"}:
                continue
            cleaned.append(tok)
        # avoid duplicate “to to”
        dedup=[]
        for tok in cleaned:
            if dedup and tok=="to" and dedup[-1]=="to": continue
            dedup.append(tok)
        if dedup:
            phrase += " " + " ".join(dedup)
            
     # put time last, normalized
    if time_tokens:
        pretty=[]
        for t in time_tokens:
            if t == "this_evening": pretty.append("this evening")
            elif t == "this_afternoon": pretty.append("this afternoon")
            else: pretty.append(t)
        phrase += " " + " ".join(pretty)

    return tidy(phrase)

# =========================
#   JSON SAVE
# =========================
def save_session_json(out_dir: str, words: List[Dict[str, float]], sentence: str, meta: Dict[str, str]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "session_transcripts.json")
    session_obj = {
        "session_id": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "model_type": meta.get("model_type", ""),
        "words": words,           # [{"text": "...", "confidence": 0.93, "t_utc": "..."}]
        "final_sentence": sentence
    }
    try:
        data = json.load(open(path, "r", encoding="utf-8")) if os.path.exists(path) else []
        if not isinstance(data, list): data = []
    except Exception:
        data = []
    data.append(session_obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

__all__ = ["grammar_fix", "save_session_json"]
