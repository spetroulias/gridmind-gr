import sqlalchemy as sa
from gridmind.data.database import get_engine

engine = get_engine()

try:
    inspector = sa.inspect(engine)
    tables = inspector.get_table_names()
    
    print("--- ΑΠΟΤΕΛΕΣΜΑΤΑ ΒΑΣΗΣ ΔΕΔΟΜΕΝΩΝ ---")
    print(f"ΠΙΝΑΚΕΣ: {tables}\n")
    
    for t in tables:
        columns = [c['name'] for c in inspector.get_columns(t)]
        print(f"ΣΤΗΛΕΣ στον πίνακα '{t}': {columns}")

except Exception as e:
    print(f"Σφάλμα σύνδεσης: {e}")
