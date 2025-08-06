import bcrypt
import jwt
from datetime import timedelta, datetime
from app.core.config import settings
from fastapi import HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def hash_password(password: str) -> str:
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(),
    ).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))


def generate_token(user_id: str, expires_delta: timedelta = timedelta(hours=24)) -> str:
    expiration = datetime.utcnow() + expires_delta  # Calculate expiration time
    return jwt.encode(
        {"user_id": user_id, "exp": expiration},  # Add expiration to payload
        settings.SECRET_KEY,
        algorithm="HS256",
    )


def verify_token(token: str) -> str:
    return jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])


def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = verify_token(token)
        user_id: str = payload.get("user_id")
        if user_id is None:
            raise credentials_exception
        return user_id
    except Exception as e:
        raise credentials_exception
