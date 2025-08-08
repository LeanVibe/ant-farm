"""Database migration and setup utilities."""

import asyncio
import logging
from pathlib import Path

import alembic.command
import alembic.config
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from .async_db import AsyncDatabaseManager, DatabaseConnectionError
from .config import settings
from .models import Base

logger = structlog.get_logger()


class DatabaseSetup:
    """Database setup and migration utilities."""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or settings.database_url
        if "postgresql://" in self.database_url:
            self.database_url = self.database_url.replace(
                "postgresql://", "postgresql+asyncpg://"
            )

    async def check_database_exists(self) -> bool:
        """Check if the database exists."""
        try:
            # Extract database name from URL
            db_parts = self.database_url.split("/")
            db_name = db_parts[-1]
            server_url = "/".join(db_parts[:-1]) + "/postgres"

            engine = create_async_engine(server_url)

            async with engine.connect() as conn:
                result = await conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                    {"db_name": db_name},
                )
                exists = result.scalar() is not None

            await engine.dispose()
            return exists

        except Exception as e:
            logger.warning("Could not check if database exists", error=str(e))
            return False

    async def create_database(self) -> bool:
        """Create the database if it doesn't exist."""
        try:
            # Extract database name from URL
            db_parts = self.database_url.split("/")
            db_name = db_parts[-1]
            server_url = "/".join(db_parts[:-1]) + "/postgres"

            engine = create_async_engine(server_url, isolation_level="AUTOCOMMIT")

            async with engine.connect() as conn:
                await conn.execute(text(f"CREATE DATABASE {db_name}"))
                logger.info("Database created successfully", database=db_name)

            await engine.dispose()
            return True

        except Exception as e:
            logger.error("Failed to create database", error=str(e))
            return False

    async def check_connection(self) -> bool:
        """Check database connection."""
        try:
            db_manager = AsyncDatabaseManager(self.database_url)
            return await db_manager.health_check()
        except Exception as e:
            logger.error("Database connection failed", error=str(e))
            return False

    async def create_tables(self) -> bool:
        """Create database tables."""
        try:
            db_manager = AsyncDatabaseManager(self.database_url)
            await db_manager.create_tables()
            return True
        except Exception as e:
            logger.error("Failed to create tables", error=str(e))
            return False

    async def check_pgvector_extension(self) -> bool:
        """Check if pgvector extension is available."""
        try:
            engine = create_async_engine(self.database_url)

            async with engine.connect() as conn:
                # Check if extension exists
                result = await conn.execute(
                    text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                )
                exists = result.scalar() is not None

            await engine.dispose()
            return exists

        except Exception as e:
            logger.warning("Could not check pgvector extension", error=str(e))
            return False

    async def install_pgvector_extension(self) -> bool:
        """Install pgvector extension."""
        try:
            engine = create_async_engine(self.database_url)

            async with engine.connect() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.commit()
                logger.info("pgvector extension installed")

            await engine.dispose()
            return True

        except Exception as e:
            logger.error("Failed to install pgvector extension", error=str(e))
            return False

    def run_migrations(self, upgrade: bool = True) -> bool:
        """Run Alembic migrations."""
        try:
            # Find alembic.ini
            alembic_ini_path = Path("alembic.ini")
            if not alembic_ini_path.exists():
                logger.error("alembic.ini not found")
                return False

            # Create Alembic config
            alembic_cfg = alembic.config.Config(str(alembic_ini_path))

            # Override database URL in config
            alembic_cfg.set_main_option(
                "sqlalchemy.url", self.database_url.replace("+asyncpg", "")
            )

            if upgrade:
                alembic.command.upgrade(alembic_cfg, "head")
                logger.info("Database migrations applied successfully")
            else:
                alembic.command.current(alembic_cfg)
                logger.info("Current migration status checked")

            return True

        except Exception as e:
            logger.error("Migration failed", error=str(e))
            return False

    def generate_migration(self, message: str) -> bool:
        """Generate a new migration."""
        try:
            alembic_ini_path = Path("alembic.ini")
            if not alembic_ini_path.exists():
                logger.error("alembic.ini not found")
                return False

            alembic_cfg = alembic.config.Config(str(alembic_ini_path))
            alembic_cfg.set_main_option(
                "sqlalchemy.url", self.database_url.replace("+asyncpg", "")
            )

            alembic.command.revision(alembic_cfg, message=message, autogenerate=True)
            logger.info("Migration generated successfully", message=message)
            return True

        except Exception as e:
            logger.error("Failed to generate migration", error=str(e))
            return False

    async def setup_database(self, create_if_missing: bool = True) -> bool:
        """Complete database setup process."""
        logger.info("Starting database setup")

        # Step 1: Check if database exists
        if create_if_missing and not await self.check_database_exists():
            logger.info("Database does not exist, creating...")
            if not await self.create_database():
                return False

        # Step 2: Check connection
        if not await self.check_connection():
            logger.error("Cannot connect to database")
            return False

        logger.info("Database connection successful")

        # Step 3: Install pgvector extension if available
        if not await self.check_pgvector_extension():
            logger.info("Installing pgvector extension...")
            await self.install_pgvector_extension()

        # Step 4: Run migrations
        logger.info("Running database migrations...")
        if not self.run_migrations():
            # If migrations fail, try creating tables directly
            logger.warning("Migrations failed, creating tables directly...")
            if not await self.create_tables():
                return False

        logger.info("Database setup completed successfully")
        return True

    async def get_database_info(self) -> dict:
        """Get database information and statistics."""
        try:
            db_manager = AsyncDatabaseManager(self.database_url)

            # Basic connection check
            if not await db_manager.health_check():
                return {"status": "disconnected", "error": "Health check failed"}

            # Get database stats
            stats = await db_manager.get_database_stats()

            # Check extensions
            pgvector_available = await self.check_pgvector_extension()

            return {
                "status": "connected",
                "database_url": self.database_url.split("@")[-1],  # Hide credentials
                "pgvector_available": pgvector_available,
                "stats": stats,
                "connection_pool": {
                    "pool_size": getattr(settings, "database_pool_size", 5),
                    "max_overflow": getattr(settings, "database_max_overflow", 10),
                },
            }

        except Exception as e:
            logger.error("Failed to get database info", error=str(e))
            return {"status": "error", "error": str(e)}


async def main():
    """CLI for database setup."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.core.db_setup <command>")
        print("Commands: setup, migrate, info, create-migration")
        return

    command = sys.argv[1]
    db_setup = DatabaseSetup()

    if command == "setup":
        success = await db_setup.setup_database()
        if success:
            print("Database setup completed successfully")
        else:
            print("Database setup failed")
            sys.exit(1)

    elif command == "migrate":
        success = db_setup.run_migrations()
        if success:
            print("Migrations completed successfully")
        else:
            print("Migrations failed")
            sys.exit(1)

    elif command == "info":
        info = await db_setup.get_database_info()
        print("Database Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    elif command == "create-migration":
        if len(sys.argv) < 3:
            print("Usage: python -m src.core.db_setup create-migration <message>")
            return

        message = " ".join(sys.argv[2:])
        success = db_setup.generate_migration(message)
        if success:
            print(f"Migration '{message}' created successfully")
        else:
            print("Failed to create migration")
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
