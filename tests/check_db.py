import os
import psycopg

with psycopg.connect(os.environ["DATABASE_URL"]) as c:
    print(c.execute("select version()").fetchone())
    print(c.execute("select extname from pg_extension where extname='vector'").fetchone())