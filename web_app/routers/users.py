from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from web_app.database import get_db
from web_app.crud import create_user, get_user
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates

router = APIRouter(prefix="/users", tags=["Users"])
templates = Jinja2Templates(directory="web_app/templates")


class UserCreate(BaseModel):
    username: str
    password: str


@router.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@router.post("/register")
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user(db, user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already taken")

    new_user = create_user(db, user.username, user.password)
    return {"message": "User created", "user": new_user.username}
