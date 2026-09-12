import sqlalchemy as sa
from gridmind.data.database import get_engine

engine = get_engine()

tables = ['system_load', 'res_production', 'generation_actual', 'load_forecasts', 'res_forecasts']

with engine.connect() as conn:
    print("--- ΑΡΙΘΜΟΣ ΕΓΓΡΑΦΩΝ ΑΝΑ ΠΙΝΑΚΑ ---")
    for table in tables:
        count = conn.execute(sa.text(f"SELECT COUNT(*) FROM {table};")).scalar()
        print(f"Πίνακας '{table}': {count} εγγραφές")
