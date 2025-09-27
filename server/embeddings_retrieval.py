import os
from typing import Optional, Tuple
import openai
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import connection, cursor
import numpy as np


def connect_to_pg() -> Tuple[connection, cursor]:
    """
    Connects to postgres DB
    # from create_db import connect_to_pg
    """
    connection_string = os.environ["PG_CONNECTION_URL"]

    conn = psycopg2.connect(connection_string)
    cur = conn.cursor()

    return (conn, cur)


def get_embeddings(text, oai: openai.OpenAI):
    """
    Get embeddings for a text.
    # from create_embeddings import get_embeddings
    """
    embeddings_model = os.environ["OPENAI_EMBEDDINGS"]
    response = oai.embeddings.create(
        model=embeddings_model, input=text.replace("\n", " ")
    )
    return response.data[0].embedding


class PromptEmbeddings:
    conn: connection
    oai: openai.OpenAI
    gpt_model: str = os.environ["OPENAI_GPT_MODEL"]
    gpt_key: str = os.environ["OPENAI_API_KEY"]
    gpt_embeddings: str = os.environ["OPENAI_EMBEDDINGS"]
    temperature = 1
    max_tokens = 1000
    delimiter = "```"

    system_message = os.environ.get(
        "SYSTEM_MESSAGE_PROMPT",
        """
        You are a friendly prompt engineer advisor. \
        You can answer questions about creating prompts for LLMs, with possible application and use cases. \
        You respond in a concise, technically credible tone. \
        """,
    )

    def __init__(self, conn, oai: Optional[openai.OpenAI]):
        self.conn = conn
        if oai is not None:
            self.oai = oai
        else:
            openai.api_key = self.gpt_key
            self.oai = openai.OpenAI()

        register_vector(self.conn)

    def process_input_with_retrieval(self, user_input):
        """
        Process input and get most appropriate prompt
        """
        input_embeddings = get_embeddings(user_input, self.oai)
        related_docs = self._get_top_3_similar_prompts(input_embeddings)

        # Prepare message to pass to the model
        # The delimter helps the model understand where the user input starts and ends.
        messages = [
            {"role": "system", "content": self.system_message},
            {
                "role": "user",
                "content": f"{self.delimiter}{user_input}{self.delimiter}",
            },
            {
                "role": "assistant",
                "content": f"Relevant prompt templates information: {related_docs[0][0]} \n {related_docs[1][0]} \n {related_docs[2][0]}",
            },
        ]

        final_response = self._get_completion_from_messages(messages)
        return final_response

    def _get_top_3_similar_prompts(self, query_embeddings):
        """
        Gets the top 3 most similar queries from the database
        """
        embedding_array = np.array(query_embeddings)

        cur = self.conn.cursor()
        cur.execute(
            "SELECT prompt from embeddings ORDER BY embedding <=> %s limit 3",
            (embedding_array,),
        )

        top_3_prompts = cur.fetchall()
        return top_3_prompts

    def _get_completion_from_messages(self, messages):
        """
        Get text completion from OpenAI API
        """
        response = self.oai.chat.completions.create(
            model=self.gpt_model,
            messages=messages,
            temperature=self.temperature,
            # max_tokens=self.max_tokens,
            # max_completion_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    (conn, cur) = connect_to_pg()
    prompt_searcher = PromptEmbeddings(conn, None)

    question = "What would be a good prompt to create a smart contract?"

    print(prompt_searcher.process_input_with_retrieval(question))
