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

tables = ['system_load', 'res_production', 'generation_actual', 'load_forecasts', 'res_forecasts']

with engine.connect() as conn:
    print("--- ΑΡΙΘΜΟΣ ΕΓΓΡΑΦΩΝ ΑΝΑ ΠΙΝΑΚΑ ---")
    for table in tables:
        count = conn.execute(sa.text(f"SELECT COUNT(*) FROM {table};")).scalar()
        print(f"Πίνακας '{table}': {count} εγγραφές")