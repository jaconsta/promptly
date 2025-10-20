# Promptly

Give it an input. Get the most appropriate LLM prompt from it.

This is a RAG system fed with Prompts on different topics (software development, project management, etc)

You can also feed with your own prompts if you follow the file pattern in csv format.

There are 2 folders

- promptIngest: Here the prompt ETL takes places and embeddings are generated.
- server: The user facing application.

It uses OpenAI to generate embeddings, and to improve the prompts suggestion for the queries.

Yo need a Postgres database that supports vector fields. For that I suggest Timescale DB.

To run it simply install python 3.13 and uv. Install dependencies. The server is very easy to deploy, few files only.

To feed the database, you can run promptIngest locally, get the connection url and run it.
