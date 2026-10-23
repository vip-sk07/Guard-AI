from __future__ import annotations
import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

@dataclass
class NLPNormalizationResult:
    original_text: str
    normalized_text: str
    has_corrections: bool
    corrections: List[Dict[str, any]] = field(default_factory=list)
    detected_hazard_terms: List[str] = field(default_factory=list)
    techniques_detected: List[str] = field(default_factory=list)

class NLPPerturbationDefense:

    SAFETY_VOCABULARY: Dict[str, str] = {

        "violence": "violent_harm",
        "violent": "violent_harm",
        "bloodshed": "violent_harm",
        "bloody": "violent_harm",
        "murder": "violent_harm",
        "murdered": "violent_harm",
        "murdering": "violent_harm",
        "killing": "violent_harm",
        "killed": "violent_harm",
        "killer": "violent_harm",
        "torture": "violent_harm",
        "tortured": "violent_harm",
        "torturing": "violent_harm",
        "execution": "violent_harm",
        "executed": "violent_harm",
        "massacre": "violent_harm",
        "assassination": "violent_harm",
        "assassinate": "violent_harm",
        "beheading": "violent_harm",
        "behead": "violent_harm",
        "slaughter": "violent_harm",
        "slaughtered": "violent_harm",
        "gore": "violent_harm",
        "snuff": "violent_harm",
        "stabbing": "violent_harm",
        "shooting": "violent_harm",
        "suicide": "self_harm",

        "weapon": "weapons",
        "weapons": "weapons",
        "weopen": "weapons",
        "weopens": "weapons",
        "wepon": "weapons",
        "wepons": "weapons",
        "weapn": "weapons",
        "weapns": "weapons",
        "firearm": "weapons",
        "firearms": "weapons",
        "gun": "weapons",
        "guns": "weapons",
        "rifle": "weapons",
        "rifles": "weapons",
        "missile": "weapons",
        "missiles": "weapons",
        "artillery": "weapons",
        "grenade": "weapons",
        "grenades": "weapons",
        "ammunition": "weapons",
        "ammo": "weapons",
        "bullet": "weapons",
        "bullets": "weapons",
        "explosive": "weapons",
        "explosives": "weapons",
        "bomb": "weapons",
        "bombs": "weapons",
        "poison": "weapons",
        "chemical": "cbrn",
        "biological": "cbrn",

        "malware": "cyberattack",
        "ransomware": "cyberattack",
        "keylogger": "cyberattack",
        "exploit": "cyberattack",
        "exploits": "cyberattack",
        "payload": "cyberattack",
        "botnet": "cyberattack",
        "phishing": "cyberattack",
        "backdoor": "cyberattack",
        "trojan": "cyberattack",
        "rootkit": "cyberattack",

        "jailbreak": "evasion",
        "override": "evasion",
        "unrestricted": "evasion",
        "unaligned": "evasion",
        "developer": "evasion",
        "bypass": "evasion",

        "video": "media_context",
        "videos": "media_context",
        "movie": "media_context",
        "movies": "media_context",
        "footage": "media_context",
        "scene": "media_context",
        "scenes": "media_context",
        "film": "media_context",
        "films": "media_context",
        "animation": "media_context",
        "generate": "intent_context",
        "resolution": "media_context",
    }

    LEET_MAP = {
        "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
        "7": "t", "8": "b", "@": "a", "$": "s", "!": "i",
        "+": "t", "|": "l", "#": "h"
    }

    HOMOGLYPH_MAP = {
        "а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "х": "x", "у": "y",
        "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "ԛ": "q", "ԝ": "w", "А": "A",
        "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I", "Ј": "J", "К": "K",
        "М": "M", "О": "O", "Р": "P", "Т": "T", "Х": "X", "Ү": "Y"
    }

    CANONICAL_TYPOS: Dict[str, str] = {
        "weopen": "weapon",
        "weopens": "weapons",
        "wepon": "weapon",
        "wepons": "weapons",
        "weapn": "weapon",
        "weapns": "weapons",
        "violenve": "violence",
        "bloodsheet": "bloodshed",
        "movice": "movie",
        "kll": "kill",
        "klling": "killing",
        "murd3r": "murder",
        "b0mb": "bomb",
    }

    BENIGN_COMMON_WORDS: Set[str] = {
        "explain", "explore", "expert", "expansion", "example", "express",
        "starting", "saving", "waiting", "talking", "eating", "standing",
        "learning", "reading", "writing", "studying", "asking", "looking",
        "thinking", "playing", "working", "living", "helping", "teaching",
        "going", "coming", "running", "walking", "giving", "taking",
        "start", "save", "talk", "eat", "stand", "learn", "read", "write",
        "study", "ask", "look", "think", "play", "work", "live", "help"
    }

    def __init__(self, fuzzy_threshold: float = 0.83):
        self.fuzzy_threshold = fuzzy_threshold
        self._vocab_words = list(self.SAFETY_VOCABULARY.keys())

    def decode_encoded_payloads(self, text: str) -> Tuple[str, bool]:
        import base64
        import binascii
        decoded_segments = []
        has_encoded = False
        words = text.split()
        for w in words:

            clean_token = w.strip(".,;:\"'()[]{}")
            if len(clean_token) >= 12 and re.match(r"^[A-Za-z0-9+/]{12,}={0,2}$", clean_token):
                try:
                    dec = base64.b64decode(clean_token).decode("utf-8", errors="ignore")
                    if any(c.isalpha() for c in dec) and all(31 < ord(c) < 127 for c in dec):
                        decoded_segments.append(dec)
                        has_encoded = True
                except Exception:
                    pass

            if re.match(r"^(?:\\x[0-9a-fA-F]{2}){4,}$", clean_token):
                try:
                    raw_bytes = bytes.fromhex(clean_token.replace("\\x", ""))
                    dec = raw_bytes.decode("utf-8", errors="ignore")
                    if any(c.isalpha() for c in dec):
                        decoded_segments.append(dec)
                        has_encoded = True
                except Exception:
                    pass
        if has_encoded:
            return f"{text} {' '.join(decoded_segments)}", True
        return text, False

    def normalize_homoglyphs(self, text: str) -> Tuple[str, bool]:
        has_homo = False
        res = []
        for ch in text:
            if ch in self.HOMOGLYPH_MAP:
                res.append(self.HOMOGLYPH_MAP[ch])
                has_homo = True
            else:
                res.append(ch)
        return "".join(res), has_homo

    def deobfuscate_leetspeak(self, text: str) -> Tuple[str, bool]:
        has_leet = False
        words = text.split()
        normalized_words = []
        for w in words:

            if re.match(r"^\d+[a-zA-Z]{1,2}$", w) or re.match(r"^\d+$", w):
                normalized_words.append(w)
                continue

            w_chars = []
            word_has_leet = False
            for ch in w:
                if ch in self.LEET_MAP and any(c.isalpha() for c in w):
                    w_chars.append(self.LEET_MAP[ch])
                    word_has_leet = True
                else:
                    w_chars.append(ch)
            if word_has_leet:
                has_leet = True
                normalized_words.append("".join(w_chars))
            else:
                normalized_words.append(w)
        return " ".join(normalized_words), has_leet

    def collapse_repetitions(self, text: str) -> Tuple[str, bool]:

        collapsed = re.sub(r"(.)\1{2,}", r"\1\1", text)
        return collapsed, (collapsed != text)

    def remove_interspersed_separators(self, text: str) -> Tuple[str, bool]:

        pattern = r"\b([a-zA-Z])[\.\-_ ]([a-zA-Z])[\.\-_ ]([a-zA-Z])[\.\-_ ]([a-zA-Z])(?:\b|[\.\-_ ])"
        has_sep = bool(re.search(pattern, text))
        cleaned = re.sub(r"(?<=\b[a-zA-Z])[\.\-_ ](?=[a-zA-Z]\b)", "", text)
        return cleaned, has_sep

    def normalize(self, prompt: str) -> NLPNormalizationResult:
        techniques = []

        clean_text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", prompt)

        clean_text, has_homo = self.normalize_homoglyphs(clean_text)
        if has_homo:
            techniques.append("homoglyph_obfuscation")

        clean_text, has_enc = self.decode_encoded_payloads(clean_text)
        if has_enc:
            techniques.append("encoded_payload_injection")

        clean_text, has_leet = self.deobfuscate_leetspeak(clean_text)
        if has_leet:
            techniques.append("leetspeak_substitution")

        clean_text, has_rep = self.collapse_repetitions(clean_text)
        if has_rep:
            techniques.append("character_elongation")

        clean_text, has_sep = self.remove_interspersed_separators(clean_text)
        if has_sep:
            techniques.append("punctuation_segmentation")

        words = re.findall(r"\b[a-zA-Z]+\b", clean_text)
        normalized_words = list(words)
        corrections = []
        detected_hazards = []

        for idx, word in enumerate(words):
            w_lower = word.lower()

            if w_lower in self.CANONICAL_TYPOS:
                canonical = self.CANONICAL_TYPOS[w_lower]
                category = self.SAFETY_VOCABULARY.get(canonical, "weapons")
                normalized_words[idx] = canonical
                corrections.append({
                    "original": word,
                    "normalized": canonical,
                    "similarity": 0.95,
                    "category": category
                })
                detected_hazards.append(canonical)
                techniques.append(f"typo_squatting('{word}'->'{canonical}')")
                continue

            if w_lower in self.SAFETY_VOCABULARY:
                category = self.SAFETY_VOCABULARY[w_lower]
                if category not in ["media_context", "intent_context"]:
                    detected_hazards.append(w_lower)
                continue

            if w_lower in self.BENIGN_COMMON_WORDS or len(w_lower) < 4:
                continue

            cutoff = self.fuzzy_threshold if len(w_lower) >= 6 else max(self.fuzzy_threshold, 0.85)
            matches = difflib.get_close_matches(w_lower, self._vocab_words, n=1, cutoff=cutoff)

            if matches:
                canonical = matches[0]
                similarity = difflib.SequenceMatcher(None, w_lower, canonical).ratio()
                category = self.SAFETY_VOCABULARY[canonical]

                normalized_words[idx] = canonical
                corrections.append({
                    "original": word,
                    "normalized": canonical,
                    "similarity": round(similarity, 3),
                    "category": category
                })

                if category not in ["media_context", "intent_context"]:
                    detected_hazards.append(canonical)
                    techniques.append(f"typo_squatting('{word}'->'{canonical}')")

        reconstructed = clean_text
        for c in corrections:

            pattern = re.compile(rf"\b{re.escape(c['original'])}\b", re.I)
            reconstructed = pattern.sub(c["normalized"], reconstructed)

        return NLPNormalizationResult(
            original_text=prompt,
            normalized_text=reconstructed,
            has_corrections=len(corrections) > 0,
            corrections=corrections,
            detected_hazard_terms=list(set(detected_hazards)),
            techniques_detected=list(set(techniques))
        )
