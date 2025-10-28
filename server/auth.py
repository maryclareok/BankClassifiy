# server/auth.py
from typing import AsyncGenerator
from uuid import UUID

from fastapi import Depends
from fastapi_users import FastAPIUsers, BaseUserManager, UUIDIDMixin
from fastapi_users.authentication import AuthenticationBackend, BearerTransport, JWTStrategy
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import EmailStr

# Use FastAPI-Users provided schemas (Pydantic v2 ready)
from fastapi_users.schemas import BaseUser, BaseUserCreate, BaseUserUpdate

from server.db import get_db
from server.models import User
from server.config import SECRET


# --- DB adapter dependency
async def get_user_db(session: AsyncSession = Depends(get_db)) -> AsyncGenerator:
    yield SQLAlchemyUserDatabase(session, User)


# --- JWT auth backend
bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")

def get_jwt_strategy() -> JWTStrategy:
    # 30 days validity
    return JWTStrategy(secret=SECRET, lifetime_seconds=60 * 60 * 24 * 30)

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)


# --- User manager
class UserManager(UUIDIDMixin, BaseUserManager[User, UUID]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)


# --- Schemas (Pydantic v2, aligned with fastapi-users)
class UserRead(BaseUser[UUID]):
    """Read model returned by the API."""
    pass

class UserCreate(BaseUserCreate):
    """Registration payload: {email, password} (+optional flags)."""
    pass

class UserUpdate(BaseUserUpdate):
    """Update payload for /users routes."""
    pass


# --- FastAPI Users instance
fastapi_users = FastAPIUsers[User, UUID](
    get_user_manager,
    [auth_backend],
)

# Dependency you can use in routes
current_active_user = fastapi_users.current_user(active=True)
