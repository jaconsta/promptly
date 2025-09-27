from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from psycopg2.extensions import connection
from pydantic import BaseModel
from fastapi import Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Template
import json
import time
import os
from embeddings_retrieval import PromptEmbeddings, connect_to_pg
import mistune

app = FastAPI()

# CORS should not be needed because stuff is rendered on the same server
# Allow CORS for local development
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

INITIAL_POST_COUNT = 3


def get_db():
    (conn, _) = connect_to_pg()
    try:
        yield conn
    finally:
        conn.close()


app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "./assets")),
    name="static",
)


class QuestionRequest(BaseModel):
    question: str


class CookieManager:
    def __init__(self, request: Request) -> None:
        self.request = request

    def get_cookie(self, name="post_limit"):
        return self.request.cookies.get(name)

    def get_posts_left(
        self,
    ):
        post_lefts_cookie = self.get_cookie()
        now = int(time.time())
        reset_time = now + 3600
        posts_left = INITIAL_POST_COUNT
        if post_lefts_cookie:
            try:
                obj = json.loads(post_lefts_cookie)
                if now > obj.get("reset", 0):
                    posts_left = INITIAL_POST_COUNT
                    reset_time = now + 3600
                else:
                    posts_left = obj.get("count", INITIAL_POST_COUNT)
                    reset_time = obj.get("reset", reset_time)
            except Exception:
                posts_left = INITIAL_POST_COUNT
                reset_time = now + 3600

        return (posts_left, reset_time)

    def update_cookie(self, response, value, name="post_limit"):
        response.set_cookie(
            name,
            json.dumps(value),
            max_age=3600,
            path="/",
        )


def render_form(posts_left=INITIAL_POST_COUNT, error_msg=None):
    has_post = posts_left < INITIAL_POST_COUNT
    return Template("""
        <div class="container" id="app">
            <img src="/static/promptly525x545_logo.png" alt="Logo" style="display:block;margin:0 auto 1rem auto;width:120px;height:auto;" />
            <div id="form-area">
                <form id="searchForm" hx-post="/ask" hx-target="#app" hx-swap="outerHTML" hx-disabled-elt="find input[type='text'], find button" hx-indicator="#spinner">
                    <div class="input-group">
                        <input type="text" name="question"  id="questionInput" hx-disabled-elt="this" placeholder="What kind of prompt do you want..." maxlength="300" autocomplete="off" required />
                        <button class="button-text" hx-disabled-elt="this" {%- if posts_left == 0 %} disabled {%- endif %} type="submit">Submit</button>
                    </div>
                    <svg class="spinner animate-spin" id="spinner" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"></path></svg>

                    {%- if posts_left < max_posts %}
                    <div style="color:#666;font-size:0.95rem;margin-bottom:0.5rem;">Posts left this hour: <b id="postsLeft">{{ posts_left }}</b></div>
                    {%- endif %}
                    <div class="error" id="errorMsg" style="display:{"block" if error_msg else "none"};">{{error_msg or ""}}</div>
                </form>
            </div>
        </div>
        """).render(
        posts_left=posts_left,
        has_post=has_post,
        error_msg=error_msg,
        max_posts=INITIAL_POST_COUNT,
    )


def render_result(question, message, posts_left=3):
    message_html = mistune.html(message)

    return Template("""
         <div class="container" id="app">
             <img src="/static/promptly525x545_logo.png" alt="Logo" style="display:block;margin:0 auto 1rem auto;width:60px;height:auto;" />
             <div class="question" style="color:#888;">{{question}}</div>
             <div class="message-box" id="messageBox">{{message}}</div>
             <div style="color:#666;font-size:0.95rem;margin-bottom:0.5rem;">Posts left this hour: <b id="postsLeft">{{posts_left}}</b></div>
             <div class="actions">
                 <button onclick="navigator.clipboard.writeText(document.getElementById('messageBox').innerText)">Copy</button>
                 <button hx-get="form" hx-trigger="click" hx-target="#app" hx-swap="outerHTML">Reset</button>
             </div>
         </div>
         """).render(question=question, message=message_html, posts_left=posts_left)


@app.get("/form", response_class=HTMLResponse)
async def get_form(request: Request):
    cookie_manager = CookieManager(request)
    (posts_left, reset_time) = cookie_manager.get_posts_left()

    response = HTMLResponse(render_form(posts_left=posts_left))
    response.set_cookie(
        "post_limit",
        json.dumps({"count": posts_left, "reset": reset_time}),
        max_age=3600,
        path="/",
    )
    return response


@app.post("/ask", response_class=HTMLResponse)
async def ask(
    request: Request, question: str = Form(...), db: connection = Depends(get_db)
):
    cookie_manager = CookieManager(request)
    (posts_left, reset_time) = cookie_manager.get_posts_left()

    if not question.strip():
        response = HTMLResponse(
            render_form(posts_left=posts_left, error_msg="Input cannot be empty.")
        )

        # cookie_manager.update_cookie(
        #     response, {"count": posts_left, "reset": reset_time}
        # )
        return response
    if len(question) > 300:
        response = HTMLResponse(
            render_form(
                posts_left=posts_left, error_msg="Input must be 300 characters or less."
            )
        )

        # cookie_manager.update_cookie(
        #     response, {"count": posts_left, "reset": reset_time}
        # )
        return response
    # Enforce post limit
    if posts_left <= 0:
        response = HTMLResponse(
            render_form(posts_left=0, error_msg="Post limit reached. Try again later.")
        )

        cookie_manager.update_cookie(response, {"count": 0, "reset": reset_time})
        return response

    message = PromptEmbeddings(db, None).process_input_with_retrieval(question)

    posts_left -= 1
    # message =  f"You asked: {question}"
    response = HTMLResponse(render_result(question, message, posts_left=posts_left))
    cookie_manager.update_cookie(response, {"count": posts_left, "reset": reset_time})
    return response


@app.get("/", response_class=HTMLResponse)
async def root():
    with open(
        os.path.join(os.path.dirname(__file__), "./assets/index.html"),
        "r",
        encoding="utf-8",
    ) as f:
        return HTMLResponse(f.read())
