import logging
import os
from typing import Optional
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


class DatabaseRetriever:
    """Service to retrieve raw energy consumption and system load metrics from PostgreSQL."""

    def __init__(self):
        user = os.getenv("POSTGRES_USER", "gridmind")
        password = os.getenv("POSTGRES_PASSWORD", "gridmind")
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        db_name = os.getenv("POSTGRES_DB", "gridmind")

        db_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
        self.engine = create_engine(db_url)

    def get_consumption_by_date(self, target_date: str) -> Optional[str]:
        """Queries total system load metrics for a target date (YYYY-MM-DD)."""
        query = text("""
            SELECT date, SUM(net_load_mwh) as total_load_mwh, COUNT(period) as periods_count
            FROM system_load 
            WHERE date = :target_date
            GROUP BY date;
        """)

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {"target_date": target_date}).fetchone()
                if result:
                    return (
                        f"Ημερομηνία: {result.date}, "
                        f"Συνολικό Φορτίο Συστήματος (Net Load): {result.total_load_mwh:.2f} MWh "
                        f"(Καταγράφηκαν {result.periods_count} περίοδοι/μετρήσεις)."
                    )
                return None
        except Exception as e:
            logger.error(f"Error querying PostgreSQL database: {e}")
            return None