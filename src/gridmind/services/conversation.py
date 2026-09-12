"""Resolve bounded analytics follow-ups using client-owned query context."""
from datetime import date
from typing import Literal
import re
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator
from gridmind.services.analytics import parse_request
from gridmind.rag.retriever_db import remove_accents


class ChatContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Literal["load", "res", "generation", "forecast"]
    start_date: date
    end_date: date
    technology: Literal["lignite", "natural_gas", "hydro", "petroleum", "res"] | None = None

    @model_validator(mode="after")
    def validate_query(self):
        if self.start_date > self.end_date:
            raise ValueError("Context start date must not be after end date.")
        if self.technology and self.dataset != "generation":
            raise ValueError("Technology filters apply to generation only.")
        return self

    def as_request(self):
        return self.dataset, self.start_date, self.end_date, self.technology


# English words are matched at boundaries; Greek roots allow inflected forms.
def mentions(text, words):
    return any(re.search(r"(?<!\w)" + re.escape(word) + (r"(?!\w)" if word.isascii() else r"\w*"), text) for word in words)


def explicit_dataset(text):
    if mentions(text, ["forecast", "predict", "προβλε", "προγνω"]):
        return "forecast"
    if mentions(text, ["res", "renewable", "renewables", "απε", "ανανεωσιμ"]):
        return "res"
    if mentions(text, ["generation", "production", "mix", "source", "sources", "gas", "hydro", "lignite", "petroleum", "oil", "παραγωγ", "μιγμα", "πηγ", "αερι", "υδρο", "λιγνιτ", "πετρελ"]):
        return "generation"
    if mentions(text, ["load", "consumption", "usage", "φορτι", "καταναλωσ"]):
        return "load"
    return None


def resolve_request(message, context=None):
    """Return a complete query or None when clarification/knowledge is needed."""
    text = remove_accents(message)
    if mentions(text, ["why", "explain", "γιατι", "εξηγ"]):
        return None
    if mentions(text, ["compare", "comparison", "versus", "συγκρι"]):
        raise ValueError("Side-by-side comparisons are not supported yet. Ask for each date or dataset separately.")
    explicit = explicit_dataset(text)
    shifts = [
        (["next day", "following day", "επομενη μερα", "επομενη ημερα"], {"days": 1}),
        (["previous day", "day before", "προηγουμενη μερα", "προηγουμενη ημερα"], {"days": -1}),
        (["next week", "επομενη εβδομαδα"], {"weeks": 1}),
        (["previous week", "προηγουμενη εβδομαδα"], {"weeks": -1}),
        (["next month", "επομενο μηνα"], {"months": 1}),
        (["previous month", "προηγουμενο μηνα"], {"months": -1}),
        (["next year", "επομενο χρονο", "επομενο ετος"], {"years": 1}),
        (["previous year", "last year", "προηγουμενο ετος"], {"years": -1}),
    ]
    shift = next((offset for phrases, offset in shifts if any(p in text for p in phrases)), None)
    summary = mentions(text, ["peak", "maximum", "highest", "minimum", "lowest", "total", "average", "mean", "μεγιστ", "κορυφ", "συνολ", "μεσο"])
    repeat = any(p in text for p in ["same day", "same dates", "same period", "again", "ιδια ημερα", "ιδιο διαστημα"])

    if context is None:
        if shift:
            raise ValueError("Which date should I use as the starting point? Ask a dated question first, e.g. 'Load 2026-01-15'.")
        return parse_request(message)
    if not isinstance(context, ChatContext):
        context = ChatContext.model_validate(context)

    # Remove contextual shift phrases before standalone parsing; otherwise Greek
    # "previous month" would be interpreted relative to today instead of context.
    dated_text = text
    if shift:
        for phrases, _ in shifts:
            for phrase in phrases:
                dated_text = dated_text.replace(phrase, "")
    # An explicit date wins over a relative reference in the same message.
    parsed = parse_request(dated_text)
    if parsed is None and not (shift or explicit or summary or repeat):
        return None
    if parsed:
        _, start, end, technology = parsed
    else:
        start, end, technology = context.start_date, context.end_date, None
        if shift:
            start = (pd.Timestamp(start) + pd.DateOffset(**shift)).date()
            end = (pd.Timestamp(end) + pd.DateOffset(**shift)).date()
        # Reuse the standalone parser for technology and unsupported forecast checks.
        parsed = parse_request(f"{message} {start.isoformat()} {end.isoformat()}")
        technology = parsed[3]

    dataset = explicit or context.dataset
    # A specific source replaces the previous filter; a date/summary-only turn keeps it.
    if dataset == "generation" and not explicit:
        technology = context.technology
    if dataset != "generation":
        technology = None
    return dataset, start, end, technology
