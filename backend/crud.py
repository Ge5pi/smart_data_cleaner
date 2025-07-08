# crud.py
from sqlalchemy.orm import Session
import models, schemas
from passlib.context import CryptContext
from datetime import datetime, timezone

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = models.User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_files_by_user_id(db: Session, user_id: int):
    """Возвращает все файлы, принадлежащие пользователю."""
    return db.query(models.File).filter(models.File.user_id == user_id).order_by(models.File.datetime_created.desc()) \
        .all()


def create_user_file(db: Session, user_id: int, file_uid: str, file_name: str) -> models.File:
    db_file = models.File(
        user_id=user_id,
        file_uid=file_uid,
        file_name=file_name,  # <-- Добавляем имя файла
        datetime_created=datetime.now(timezone.utc)
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    return db_file
