# server/models.py
from sqlalchemy.orm import DeclarativeBase
from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTableUUID

# Base for SQLAlchemy models
class Base(DeclarativeBase):
    pass

# FastAPI-Users UUID user table
class User(SQLAlchemyBaseUserTableUUID, Base):
    __tablename__ = "users"
    # Columns provided:
    # id (UUID), email, hashed_password, is_active, is_superuser, is_verified
