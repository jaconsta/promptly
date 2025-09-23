import os
from typing import Tuple

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import connection, cursor


def connect_to_pg() -> Tuple[connection, cursor]:
    """
    Connects to postgres DB
    """
    connection_string = os.environ["PG_CONNECTION_URL"]

    conn = psycopg2.connect(connection_string)
    cur = conn.cursor()

    return (conn, cur)


def create_db_and_tables(conn: connection):
    """
    Initializes extensions and creates embeddings table
    """
    cur = conn.cursor()
    # Install Extensions
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;")
    conn.commit()

    register_vector(conn)

    # Create table to store embeddings and metadata
    # Embeddings size according to chatgpt
    table_create_command = """
        CREATE TABLE embeddings (
            id bigserial primary key,
            act text,
            prompt text,
            tokens integer,
            embedding vector(1536)
        );
    """

    cur.execute(table_create_command)
    cur.close()
    conn.commit()


if __name__ == "__main__":
    print("Creating tables")
    (conn, cur) = connect_to_pg()
    create_db_and_tables(conn)
    print("Done")
