from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.auth import AuthUser, Role, create_access_token, get_current_user

router = APIRouter(prefix="/api/auth", tags=["Auth"])


class LoginRequest(BaseModel):
    user_id: str
    password: str


USER_STORE: dict[str, dict[str, str | Role]] = {
    "admin.user": {"name": "Admin", "password": "ordertocash", "role": "admin"},
    "finance.user": {"name": "Finance Controller", "password": "ordertocash", "role": "finance"},
    "collections.user": {"name": "Collections Manager", "password": "ordertocash", "role": "collections"},
    "exec.user": {"name": "Executive", "password": "ordertocash", "role": "exec"},
}


@router.post("/login")
def login(body: LoginRequest):
    user_id = body.user_id.strip().lower()
    user = USER_STORE.get(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unknown user ID. Use admin.user, finance.user, collections.user, or exec.user",
        )
    if body.password.strip() != user["password"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password. Demo password is: ordertocash",
        )
    role = user["role"]
    name = str(user["name"])
    token = create_access_token(name=name, role=role)  # type: ignore[arg-type]
    return {"access_token": token, "token_type": "bearer", "user": {"name": name, "role": role}}


@router.get("/me")
def me(user: AuthUser = Depends(get_current_user)):
    return {"name": user.name, "role": user.role}
