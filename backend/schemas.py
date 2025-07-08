# schemas.py

from pydantic import BaseModel
from datetime import datetime


# Schemas for File model
class FileBase(BaseModel):
    file_uid: str
    file_name: str  # <-- Добавляем здесь


class FileCreate(FileBase):
    pass


class File(FileBase):
    id: int
    user_id: int
    datetime_created: datetime

    class Config:
        from_attributes = True


# Schemas for User model
class UserBase(BaseModel):
    email: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    is_active: bool
    # Теперь список файлов будет содержать и file_name
    files: list[File] = []

    class Config:
        from_attributes = True
