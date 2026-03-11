from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from datetime import datetime, timezone
import os

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./contentai.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    credits = Column(Integer, default=15)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Contact(Base):
    __tablename__ = "contacts"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    subject = Column(String)
    message = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class GeneratedContent(Base):
    __tablename__ = "generated_content"

    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, index=True)
    topic = Column(String)
    content = Column(Text)
    content_type = Column(String, default="blog")
    keywords_used = Column(JSON)
    metrics = Column(JSON)
    generation_method = Column(String, default="standard")  # standard, stream, crew, dspy, refine
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Template(Base):
    __tablename__ = "templates"

    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, index=True, nullable=False)
    name = Column(String, nullable=False)
    topic = Column(String, nullable=False)
    keywords = Column(JSON, default=list)
    content_type = Column(String, default="blog")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    from auth import hash_password

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == "demo@test.com").first()
        if not existing:
            db.add(User(
                email="demo@test.com",
                password_hash=hash_password("demo123"),
                credits=15,
            ))
            db.commit()
    finally:
        db.close()
