##
# Calculate costs of embedding data
# Get embeddings from text
##

import os

import openai
from typing import Any, List
import pandas as pd
import numpy as np
import tiktoken
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

from create_db import connect_to_pg


def num_tokens_from_string(string: Any | str, encoding_name="cl100k_base") -> int:
    """
    Calculate number of tokens
    """
    if not string:
        return 0

    # Number of tokens in a text string
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_embedding_cost(num_tokens: int):
    """
    Calculat cost of embedding num_token
    Price based on openai text-embedding-3-small
    """
    return num_tokens / 1000 * 0.0002


def get_total_embedding_cost(prompt_df: pd.DataFrame):
    """
    Calculate total cost of embedding all content
    """
    total_tokens = 0

    for i in range(len(prompt_df.index)):
        text = prompt_df["prompt"][i]
        token_len = num_tokens_from_string(text)
        total_tokens += token_len

    total_cost = get_embedding_cost(total_tokens)
    return total_cost


def chunk_content(prompt_df: pd.DataFrame) -> List[pd.Series]:
    """
    Create content chunks to stay bellow max token limits.
    note: MAx number of embedding tokens for a single request: 8191
    """
    checked_content = []
    # Split text token size -> aprox 512 token
    ideal_token_size = 512
    for i in range(len(prompt_df.index)):
        # Hardcoding a transformation
        if "app" in prompt_df:
            act = prompt_df["app"][i]
        else:
            act = prompt_df["act"][i]

        text = prompt_df["prompt"][i]
        token_len = num_tokens_from_string(text)
        if token_len <= ideal_token_size:
            checked_content.append([act, text, token_len])
            continue

        # Add content in chunks
        start = 0
        # 1 token 3/4 a word
        ideal_size = int(ideal_token_size // 4 / 3)
        end = ideal_size

        # Split in words
        words = str(text).split()
        # Remove empty spaces
        words = [w for w in words if w != " "]
        total_words = len(words)

        # Calculate iterations
        chunks = total_words
        if total_words / ideal_size != 0:
            chunks += 1

        new_content = []
        for _ in range(chunks):
            if end > total_words:
                end = total_words
            new_content = words[start:end]
            new_content_string = " ".join(new_content)
            new_content_token_len = num_tokens_from_string(new_content_string)
            if new_content_token_len > 0:
                checked_content.append(
                    [act[i], new_content_string, new_content_token_len]
                )

            start += ideal_size
            end += ideal_size

    return checked_content


def get_embeddings(text, oai: openai.OpenAI):
    """
    Get embeddings for a text.
    """
    embeddings_model = os.environ["OPENAI_EMBEDDINGS"]
    response = oai.embeddings.create(
        model=embeddings_model, input=text.replace("\n", " ")
    )
    return response.data[0].embedding


def create_embeddings_for_prompt_content(
    chunked_content: List[pd.Series], oai: openai.OpenAI
) -> pd.DataFrame:
    """
    Create embeddings for each piece of content.
    It mutates chunked_content. Appends an extra element
    """
    for i in range(len(chunked_content)):
        text = chunked_content[i][1]
        embedding = get_embeddings(text, oai)
        chunked_content[i].append(embedding)

    df = pd.DataFrame(chunked_content, columns=["act", "prompt", "tokens", "embedding"])

    return df


def df_to_csv(df: pd.DataFrame, filename="prompt_data_and_embeddings.csv"):
    df.to_csv(f"../datasets/{filename}")


def df_to_datalist(df: pd.DataFrame):
    data_list = [
        (
            row["act"],
            row["prompt"],
            int(row["tokens"]),
            np.array(row["embedding"]),
        )
        for _, row in df.iterrows()
    ]

    return data_list


def csv_with_embeddings_to_list(
    filename="prompt_data_and_embeddings.csv",
):
    df = pd.read_csv(f"../datasets/{filename}")
    data_list = [
        (
            row["act"],
            row["prompt"],
            int(row["tokens"]),
            # Need parsing due to arr->string conversion on df_to_csv
            np.array(row["embedding"][1:-1].split(", ")),
        )
        for _, row in df.iterrows()
    ]
    return data_list


def insert_embeddings_in_db(content):  # List[pd.Series]):
    """
    Insert the prompt content with embeddings into the database
    """
    print(content[0][-1])
    (conn, _) = connect_to_pg()
    cur = conn.cursor()
    register_vector(conn)

    execute_values(
        cur,
        "INSERT INTO embeddings (act, prompt, tokens, embedding) VALUES %s",
        content,
    )
    conn.commit()


def run_insert():
    """
    Perform the necessary first steps to load and insert the prompts to the database with embeddings.
    """
    openai.api_key = os.environ["OPENAI_API_KEY"]
    oai = openai.OpenAI()

    print("../datasets/awesome-chatgpt-prompts.csv")
    df = pd.read_csv("../datasets/awesome-chatgpt-prompts.csv")
    total_cost = get_total_embedding_cost(df)
    print("Estimated price to embed the content => $", str(total_cost))
    chunked_df = chunk_content(df)
    df_with_embeddings = create_embeddings_for_prompt_content(chunked_df, oai)
    df_as_datalist = df_to_datalist(df_with_embeddings)
    insert_embeddings_in_db(df_as_datalist)

    print("../datasets/awesome-chatgpt-vibeprompts.csv")
    df = pd.read_csv("../datasets/awesome-chatgpt-vibeprompts.csv")
    total_cost = get_total_embedding_cost(df)
    print("Extimated price to embed the content => $", str(total_cost))
    chunked_df = chunk_content(df)
    df_with_embeddings = create_embeddings_for_prompt_content(chunked_df, oai)
    df_as_datalist = df_to_datalist(df_with_embeddings)
    insert_embeddings_in_db(df_as_datalist)


if __name__ == "__main__":
    run_insert()
