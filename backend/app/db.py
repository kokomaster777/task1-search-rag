"""Полнотекстовый поиск в Postgres: русская морфология (tsvector) + pg_trgm для опечаток."""
import os
import psycopg

DATABASE_URL = os.environ["DATABASE_URL"]

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE TABLE IF NOT EXISTS products (
    id INT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    brand TEXT,
    price NUMERIC,
    description TEXT,
    fts tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('russian', coalesce(name,'')), 'A') ||
        setweight(to_tsvector('russian', coalesce(description,'')), 'B') ||
        setweight(to_tsvector('russian', coalesce(category,'')), 'C')
    ) STORED
);
CREATE INDEX IF NOT EXISTS idx_products_fts ON products USING GIN (fts);
CREATE INDEX IF NOT EXISTS idx_products_trgm ON products USING GIN (name gin_trgm_ops);
"""


def connect():
    return psycopg.connect(DATABASE_URL)


def init_and_ingest(products: list[dict]):
    with connect() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA)
        cur.execute("SELECT count(*) FROM products")
        if cur.fetchone()[0] >= len(products):
            return False
        cur.executemany(
            "INSERT INTO products (id, name, category, brand, price, description) "
            "VALUES (%(id)s, %(name)s, %(category)s, %(brand)s, %(price)s, %(description)s) "
            "ON CONFLICT (id) DO NOTHING",
            products,
        )
    return True


def fulltext_search(query: str, limit: int = 5) -> list[dict]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT id, name, category, price, description,
                      ts_rank(fts, websearch_to_tsquery('russian', %s)) AS score
               FROM products
               WHERE fts @@ websearch_to_tsquery('russian', %s)
               ORDER BY score DESC LIMIT %s""",
            (query, query, limit),
        )
        rows = cur.fetchall()
        if not rows:  # fallback на триграммы — ловит опечатки
            cur.execute(
                """SELECT id, name, category, price, description,
                          similarity(name, %s) AS score
                   FROM products
                   WHERE similarity(name, %s) > 0.15
                   ORDER BY score DESC LIMIT %s""",
                (query, query, limit),
            )
            rows = cur.fetchall()
    cols = ["id", "name", "category", "price", "description", "score"]
    return [dict(zip(cols, r)) for r in rows]
