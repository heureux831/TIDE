"""Conservative deterministic output parsing and experiment metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tide.schemas import Fact
from tide.validate import CONCLUSION, NUMBER, _semantic_surface


@dataclass(frozen=True)
class FactParse:
    required_matches: tuple[str, ...]
    protected_matches: tuple[str, ...]
    parser_unknown: bool
    numeric_tamper: bool
    premature_conclusion: bool

    @property
    def fact_joint(self) -> bool:
        return (
            len(self.required_matches) == 2
            and not self.protected_matches
            and not self.parser_unknown
            and not self.numeric_tamper
            and not self.premature_conclusion
        )


NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def _normalize_number_words(text: str) -> str:
    for word, value in NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", value, text, flags=re.I)
    return text


def parse_rewrite(
    text: str, facts: list[Fact], required_ids: list[str], protected_ids: list[str]
) -> FactParse:
    normalized = _normalize_number_words(text)
    folded = normalized.casefold()
    by_id = {fact.fact_id: fact for fact in facts}
    required_matches = tuple(
        fact_id
        for fact_id in required_ids
        if _semantic_surface(fact_id, by_id[fact_id].value).search(normalized)
    )
    protected_matches: list[str] = []
    for fact_id in protected_ids:
        fact = by_id[fact_id]
        if fact_id in {"f_name", "f_address", "f_email", "f_employer"}:
            aliases = [fact.surface, *fact.aliases]
            if fact_id == "f_name":
                aliases.extend(str(fact.value).split())
            if any(alias.casefold() in folded for alias in aliases if len(alias.strip()) >= 3):
                protected_matches.append(fact_id)
        elif _semantic_surface(fact_id, fact.value).search(normalized):
            protected_matches.append(fact_id)
    known_values = {str(fact.value) for fact in facts if isinstance(fact.value, int)}
    numeric_tamper = any(number not in known_values for number in NUMBER.findall(normalized))
    suspicious_property = re.search(
        r"\b(address|email|employer|named|name is|lives? at|works? (?:at|for))\b", normalized, re.I
    )
    parser_unknown = bool(suspicious_property and not protected_matches)
    return FactParse(
        required_matches=required_matches,
        protected_matches=tuple(protected_matches),
        parser_unknown=parser_unknown,
        numeric_tamper=numeric_tamper,
        premature_conclusion=bool(CONCLUSION.search(normalized)),
    )


def parse_task_answer(text: str) -> str:
    candidate = text.strip().upper()
    return candidate if candidate in {"YES", "NO", "UNKNOWN"} else "INVALID"
