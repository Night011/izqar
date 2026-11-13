"""Database engine and session lifecycle."""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool
from .config import DATABASE_URL

if DATABASE_URL in ('sqlite://', 'sqlite:///:memory:'):
    kwargs = {'connect_args': {'check_same_thread': False}, 'poolclass': StaticPool}
elif DATABASE_URL.startswith('sqlite'):
    kwargs = {'connect_args': {'check_same_thread': False}}
else:
    kwargs = {'pool_pre_ping': True}
engine = create_engine(DATABASE_URL, **kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
