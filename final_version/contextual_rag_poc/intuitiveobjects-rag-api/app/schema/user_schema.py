from pydantic import BaseModel, EmailStr


class SignupSchema(BaseModel):
    email: EmailStr
    password: str


class SigninSchema(BaseModel):
    email: EmailStr
    password: str
