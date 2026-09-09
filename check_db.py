import os
import sqlalchemy as sa
from dotenv import load_dotenv

# Φόρτωση μεταβλητών από το .env
load_dotenv()

user = os.getenv("POSTGRES_USER", "postgres")
password = os.getenv("POSTGRES_PASSWORD", "postgres")
host = os.getenv("POSTGRES_HOST", "localhost")
port = os.getenv("POSTGRES_PORT", "5432")
db_name = os.getenv("POSTGRES_DB", "gridmind_db")

db_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"

try:
    engine = sa.create_engine(db_url)
    inspector = sa.inspect(engine)
    tables = inspector.get_table_names()
    
    print("--- ΑΠΟΤΕΛΕΣΜΑΤΑ ΒΑΣΗΣ ΔΕΔΟΜΕΝΩΝ ---")
    print(f"ΠΙΝΑΚΕΣ: {tables}\n")
    
    for t in tables:
        columns = [c['name'] for c in inspector.get_columns(t)]
        print(f"ΣΤΗΛΕΣ στον πίνακα '{t}': {columns}")

except Exception as e:
    print(f"Σφάλμα σύνδεσης: {e}")