import os
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from .models import Base
from dotenv import load_dotenv

load_dotenv()

class DatabaseClient:
    def __init__(self, database_url: str = None):
        if not database_url:
            postgres_user = os.getenv("POSTGRES_USER", "chainlens_user")
            postgres_password = os.getenv("POSTGRES_PASSWORD", "chainlens_password")
            postgres_db = os.getenv("POSTGRES_DB", "chainlens_db")
            postgres_host = os.getenv("POSTGRES_HOST", "localhost")
            postgres_port = os.getenv("POSTGRES_PORT", "5432")
            
            # Allow fallback to SQLite if specifically requested or if running in test mode without postgres
            fallback = os.getenv("USE_SQLITE_FALLBACK", "false").lower() == "true"
            
            if fallback:
                self.database_url = "sqlite:///chainlens.db"
            else:
                self.database_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
        else:
            self.database_url = database_url

        self.engine = create_engine(self.database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def init_db(self):
        """Create all tables."""
        Base.metadata.create_all(bind=self.engine)
        
    def drop_all(self):
        """Drop all tables."""
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def get_session(self) -> Session:
        """Provide a transactional scope around a series of operations."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_table_counts(self):
        from sqlalchemy import select, func
        counts = {}
        with self.get_session() as session:
            for table in Base.metadata.sorted_tables:
                counts[table.name] = session.execute(select(func.count()).select_from(table)).scalar()
        return counts
