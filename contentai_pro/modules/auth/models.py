from pydantic import BaseModel


class UserCredentials(BaseModel):
    email: str
    password: str


class UserInfo(BaseModel):
    email: str
    credits: int
