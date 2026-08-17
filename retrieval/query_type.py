from __future__ import annotations

import re

NUMERIC_HINTS = re.compile(
    r"\d|कितना|कितनी|कितने|how many|how much|when|year|price|cost|"
    r"எத்தனை|কত|किती",
    re.IGNORECASE,
)
PERSON_HINTS = re.compile(r"\bwho\b|कौन|किसने|யார்|কে ", re.IGNORECASE)
LOCATION_HINTS = re.compile(r"\bwhere\b|कहाँ|कहां|எங்கே|কোথায়", re.IGNORECASE)


def guess_query_type(query: str) -> str:
    if NUMERIC_HINTS.search(query):
        return "NUMERIC"
    if PERSON_HINTS.search(query):
        return "PERSON"
    if LOCATION_HINTS.search(query):
        return "LOCATION"
    return "DESCRIPTION"
