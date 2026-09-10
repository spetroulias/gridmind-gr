import os
import sqlalchemy as sa
from dotenv import load_dotenv

load_dotenv()

user = os.getenv("POSTGRES_USER", "gridmind")
password = os.getenv("POSTGRES_PASSWORD", "gridmind")
host = os.getenv("POSTGRES_HOST", "localhost")
port = os.getenv("POSTGRES_PORT", "5432")
db_name = os.getenv("POSTGRES_DB", "gridmind")

engine = sa.create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db_name}")

with engine.connect() as conn:
    result = conn.execute(sa.text("SELECT date, COUNT(*) FROM system_load GROUP BY date ORDER BY date ASC LIMIT 10;")).fetchall()
    print("--- ΗΜΕΡΟΜΗΝΙΕΣ ΚΑΙ ΠΛΗΘΟΣ ΕΓΓΡΑΦΩΝ ΣΤΟ SYSTEM_LOAD ---")
    for row in result:
        print(f"Ημερομηνία: {row[0]} | Εγγραφές: {row[1]}")