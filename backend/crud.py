from sqlalchemy.orm import Session
import models
import schemas
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def create_user(db: Session, user: schemas.UserCreate):
    print("--- 4a. Внутри crud.create_user ---")

    print("--- 4b. Хешируем пароль ---")
    hashed_password = get_password_hash(user.password)

    print("--- 4c. Создаем объект модели User ---")
    db_user = models.User(email=user.email, hashed_password=hashed_password)

    print("--- 4d. Добавляем в сессию (db.add) ---")
    db.add(db_user)

    print("--- 4e. *** ПЫТАЕМСЯ СОХРАНИТЬ В БД (db.commit) *** ---")
    db.commit()
    print("--- 4f. *** COMMIT УСПЕШЕН *** ---")

    print("--- 4g. Обновляем объект (db.refresh) ---")
    db.refresh(db_user)

    print("--- 4h. Возвращаем созданного пользователя из crud ---")
    return db_user
