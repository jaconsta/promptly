from fastapi import Request

def get_post_limit_cookie(request: Request):
    cookie = request.cookies.get('post_limit')
    return cookie
