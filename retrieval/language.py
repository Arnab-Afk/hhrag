"""Map Sarvam BCP-47 codes onto MSMARCO-XI / our index shard names."""

from __future__ import annotations

SARVAM_TO_SHARD: dict[str, str] = {
    "hi-IN": "hi",
    "en-IN": "en",
    "en-US": "en",
    "en-GB": "en",
    "bn-IN": "bn",
    "ta-IN": "ta",
    "te-IN": "te",
    "mr-IN": "mr",
    "gu-IN": "gu",
    "kn-IN": "kn",
    "ml-IN": "ml",
    "pa-IN": "pa",
    "ur-IN": "ur",
    "as-IN": "as",
    "or-IN": "or",
    "od-IN": "or",
    "ne-IN": "ne",
    "sa-IN": "sa",
}

SHARD_TO_NAME: dict[str, str] = {
    "as": "Assamese",
    "bn": "Bengali",
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}


def shard_for_language_code(code: str | None) -> str | None:
    if not code:
        return None
    if code in SARVAM_TO_SHARD:
        return SARVAM_TO_SHARD[code]
    if "-" in code:
        prefix = code.split("-", 1)[0].lower()
        if prefix == "od":
            return "or"
        if prefix in SHARD_TO_NAME:
            return prefix
    lowered = code.lower()
    if lowered in SHARD_TO_NAME:
        return lowered
    return None
