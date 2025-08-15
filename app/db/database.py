# app/db/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

#DB_URL = "postgresql://postgres:1234@localhost:5432/postgres"
DB_URL = "postgresql://postgres.mllnojrfmcxvzfrzbeqe:4492Madras!1992@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
