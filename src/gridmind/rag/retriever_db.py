import re
import logging
import unicodedata
import sqlalchemy as sa
from gridmind.data.database import get_engine

logger = logging.getLogger(__name__)

# Χαρτογράφηση ριζών ελληνικών μηνών σε αριθμούς
GREEK_MONTH_ROOTS = {
    "ιανουαρ": 1, "ιαν": 1,
    "φεβρουαρ": 2, "φεβ": 2,
    "μαρτ": 3, "μαρ": 3,
    "απριλ": 4, "απρ": 4,
    "μαι": 5, "μαϊ": 5,
    "ιουν": 6,
    "ιουλ": 7,
    "αυγουστ": 8, "αυγ": 8,
    "σεπτεμβρ": 9, "σεπτ": 9, "σεπ": 9,
    "οκτωβρ": 10, "οκτωβ": 10, "οκτ": 10,
    "νοεμβρ": 11, "νοεμβ": 11, "νοε": 11,
    "δεκεμβρ": 12, "δεκεμβ": 12, "δεκ": 12
}

def remove_accents(input_str: str) -> str:
    """Μετατρέπει τα πάντα σε πεζά και αφαιρεί τόνους/διακριτικά."""
    nfkd_form = unicodedata.normalize('NFD', input_str.lower())
    return "".join([c for c in nfkd_form if unicodedata.category(c) != 'Mn'])

class DatabaseRetriever:
    def __init__(self):
        self.engine = get_engine()

    def _extract_date(self, text: str) -> str | None:
        if not text:
            return None
            
        clean_text = remove_accents(text)

        # 1. ISO format: YYYY-MM-DD (π.χ. 2026-02-03)
        iso_match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", clean_text)
        if iso_match:
            year, month, day = iso_match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"

        # 2. Ελληνικό κείμενο με μήνα (π.χ. "3 φεβρουαριου 2026", "03 φεβ 2026")
        for root, month_num in GREEK_MONTH_ROOTS.items():
            if root in clean_text:
                # Ψάχνουμε 1-2 ψηφία (ημέρα) πριν τη ρίζα και 4 ψηφία (έτος) μετά
                pattern = r"(\d{1,2})\s+[a-zα-ω]*" + re.escape(root) + r"[a-zα-ω]*\s+(20\d{2})"
                match = re.search(pattern, clean_text)
                if match:
                    day, year = match.group(1), match.group(2)
                    return f"{year}-{int(month_num):02d}-{int(day):02d}"

        # 3. European format: DD/MM/YYYY (π.χ. 03/02/2026)
        eu_match = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](20\d{2})\b", clean_text)
        if eu_match:
            day, month, year = eu_match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"

        # 4. European format διψήφιο: DD/MM/YY (π.χ. 03/02/26)
        eu_short = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{2})\b", clean_text)
        if eu_short:
            day, month, year = eu_short.groups()
            return f"20{year}-{int(month):02d}-{int(day):02d}"

        return None

    def get_consumption_by_date(self, query: str) -> tuple[str, list[str]]:
        date_str = self._extract_date(query)
        logger.info(f"[DB Retriever] Query: '{query}' -> Parsed Date: {date_str}")
        
        if not date_str:
            return "", []

        try:
            sql = sa.text("""
                SELECT date, SUM(net_load_mwh) as total_load
                FROM system_load
                WHERE date = :date_val AND period BETWEEN 1 AND 24
                GROUP BY date;
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(sql, {"date_val": date_str}).fetchone()
                
            if result and result[1] is not None:
                context = f"Ημερομηνία: {result[0]} | Συνολικό Φορτίο Συστήματος (Net Load): {result[1]:.2f} MWh"
                source = f"PostgreSQL Table: system_load ({result[0]})"
                return context, [source]
            else:
                logger.warning(f"[DB Retriever] No records found in DB for date: {date_str}")
                
        except Exception as e:
            logger.error(f"[DB Retriever] Database query error: {e}")

        return "", []

    def get_context(self, query: str) -> tuple[str, list[str]]:
        return self.get_consumption_by_date(query)
