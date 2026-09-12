import sqlalchemy as sa
from gridmind.data.database import get_engine

engine = get_engine()

with engine.connect() as conn:
    result = conn.execute(sa.text("SELECT date, COUNT(*) FROM system_load GROUP BY date ORDER BY date ASC LIMIT 10;")).fetchall()
    print("--- ΗΜΕΡΟΜΗΝΙΕΣ ΚΑΙ ΠΛΗΘΟΣ ΕΓΓΡΑΦΩΝ ΣΤΟ SYSTEM_LOAD ---")
    for row in result:
        print(f"Ημερομηνία: {row[0]} | Εγγραφές: {row[1]}")
