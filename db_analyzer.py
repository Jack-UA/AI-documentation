#!/usr/bin/env python3
"""
Extensible Database Analyzer for Text-to-SQL AI Systems
=======================================================

This module refactors the original monolithic PostgreSQL analysis script into
an adapter-driven, extensible architecture using SQLAlchemy and Pydantic.

Key Features:
- Adapter Pattern: Abstract DBAdapter with concrete implementations (Postgres, MySQL example)
- Secure Configuration: Credentials loaded via environment variables and validated by Pydantic
- Rich Metadata: Schema, relationships (including composite keys), row counts, column statistics
- AI-Ready Output: JSON report similar in spirit to the original Working Example
- Comments and Guidance: Extensive inline comments for easy extension by AI/engineers

Install dependencies:
    pip install sqlalchemy psycopg2-binary pymysql pydantic pandas numpy

Environment variables (retains Code Examples credentials style):
    export DB_HOST=localhost
    export DB_PORT=5432
    export DB_NAME=fund_synthetic_db
    export DB_USER=postgres
    export DB_PASS=

Usage:
    python db_analyzer.py postgres --output db_analysis_for_ai.json

To add a new database (e.g., Oracle):
- Implement a new adapter subclass inheriting DBAdapter
- Override _create_engine and dialect-specific behaviors if needed
- Register in AdapterFactory
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, SecretStr, ValidationError
from sqlalchemy import (
    create_engine,
    inspect,
    func,
    select,
    MetaData,
    Table,
    distinct,
    text
)
from sqlalchemy.engine import Engine, Inspector
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import expression
from sqlalchemy.sql.sqltypes import (
    Integer, Float, Numeric, DECIMAL, BigInteger, SmallInteger,
    Date, DateTime, Time
)

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("db_analyzer")


# -----------------------------------------------------------------------------
# Secure Configuration via Pydantic
# -----------------------------------------------------------------------------
class DBConfig(BaseModel):
    """
    Secure database configuration model.

    Uses Pydantic for type validation and SecretStr for password safety.
    Defaults mirror the Code Examples to keep credentials consistent.
    """
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(..., gt=0, le=65535, description="Database port")
    database: str = Field(..., min_length=1, description="Database name")
    username: str = Field(..., min_length=1, description="Database username")
    password: SecretStr = Field(default=SecretStr(""), description="Database password")

    class Config:
        json_encoders = {SecretStr: lambda v: "***REDACTED***"}


def load_config_from_env(db_type: str) -> DBConfig:
    """
    Load DBConfig from environment variables with sensible defaults.

    - DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS
    - Defaults align with the Code Examples and Working Example

    Args:
        db_type: Used to pick default port (e.g., postgres → 5432)

    Returns:
        Validated DBConfig
    """
    default_ports = {
        "postgres": 5432,
        "postgresql": 5432,
        "mysql": 3306,
        "mariadb": 3306,
    }
    try:
        return DBConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", str(default_ports.get(db_type.lower(), 5432)))),
            database=os.getenv("DB_NAME", "fund_synthetic_db"),
            username=os.getenv("DB_USER", "postgres"),
            password=SecretStr(os.getenv("DB_PASS", "")),
        )
    except (ValidationError, ValueError, TypeError) as e:
        logger.error("Invalid DB configuration: %s", e)
        raise


# -----------------------------------------------------------------------------
# Abstract Adapter
# -----------------------------------------------------------------------------
class DBAdapter(ABC):
    """
    Abstract adapter for database interactions.

    Responsibilities:
    - Engine creation (dialect-specific)
    - Connection testing
    - Schema extraction via SQLAlchemy inspector
    - Column statistics via SQLAlchemy Core
    - SQL example generation

    Extensibility:
    - To support new DBs, subclass and implement _create_engine()
    - Override _get_random_function() if the dialect differs (e.g., MySQL uses RAND)

    Analysis thresholds (retained from Working Example):
    - LOW_CARDINALITY_THRESHOLD: columns with <= N distinct values get samples
    - SAMPLE_SIZE_LOW: number of sample values for low-cardinality columns
    - HIGH_CARDINALITY_SAMPLE: random sample size for high-cardinality stats
    """
    LOW_CARDINALITY_THRESHOLD = 20
    SAMPLE_SIZE_LOW = 10
    HIGH_CARDINALITY_SAMPLE = 1000

    def __init__(self, config: DBConfig):
        self.config = config
        self.engine: Engine = self._create_engine()
        self._inspector: Optional[Inspector] = None
        logger.info(
            "%s initialized for %s@%s:%s/%s",
            self.__class__.__name__,
            config.username,
            config.host,
            config.port,
            config.database,
        )

    # --------------------- Abstract methods ---------------------
    @abstractmethod
    def _create_engine(self) -> Engine:
        """Create the SQLAlchemy engine for this adapter."""
        raise NotImplementedError

    # --------------------- Overridable helpers ------------------
    def _get_random_function(self) -> expression.Function:
        """
        Return dialect-specific random function used for random sampling.
        - Postgres: func.random()
        - MySQL: adapter should override to func.rand()
        """
        return func.random()

    # --------------------- Lazy-loaded properties ---------------
    @property
    def inspector(self) -> Inspector:
        """Lazy inspector to introspect schema via SQLAlchemy."""
        if self._inspector is None:
            self._inspector = inspect(self.engine)
        return self._inspector

    # --------------------- Public API ---------------------------
    def connect(self) -> None:
        """Health check: ensure we can execute a trivial query."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.debug("Connection test succeeded")
        except SQLAlchemyError as e:
            logger.error("Database connection failed: %s", e)
            raise

    def analyze(self) -> Dict[str, Any]:
        """
        Run full analysis pipeline:
        - Test connection
        - Discover tables
        - Extract table schema and relationships
        - Compute per-column statistics
        - Generate example SQL queries

        Returns:
            Analysis dictionary ready for JSON serialization.
        """
        self.connect()

        tables = self._get_table_names()
        if not tables:
            logger.warning("No tables found in database %s", self.config.database)
            return {
                "database": self.config.database,
                "generated_at": datetime.utcnow().isoformat(),
                "schema": [],
                "relationships": [],
                "column_statistics": [],
                "sql_examples": [],
                "summary": {"total_tables": 0},
            }

        analysis: Dict[str, Any] = {
            "database": self.config.database,
            "generated_at": datetime.utcnow().isoformat(),
            "schema": [],
            "relationships": [],
            "column_statistics": [],
            "sql_examples": [],
            "summary": {},
        }

        # 1) Extract schema and relationships
        for table_name in tables:
            try:
                schema_info = self._extract_table_schema(table_name)
                analysis["schema"].append(schema_info)

                # Build relationships list from FKs. Handles composite keys correctly.
                for fk in schema_info.get("foreign_keys", []):
                    analysis["relationships"].append({
                        "from_table": table_name,
                        "from_columns": fk["constrained_columns"],
                        "to_table": fk["referred_table"],
                        "to_columns": fk["referred_columns"],
                    })
            except Exception as e:
                logger.error("Failed to extract schema for table %s: %s", table_name, e)

        # 2) Calculate column statistics
        for schema_info in analysis["schema"]:
            table_name = schema_info["table"]
            for col in schema_info.get("columns", []):
                col_name = col["name"]
                try:
                    stats = self._calculate_column_statistics(table_name, col_name)
                    stats["table"] = table_name
                    analysis["column_statistics"].append(stats)
                except Exception as e:
                    logger.warning(
                        "Failed to calculate statistics for %s.%s: %s",
                        table_name, col_name, e
                    )

        # 3) Generate SQL examples
        try:
            analysis["sql_examples"] = self._generate_sql_examples(analysis["schema"])
        except Exception as e:
            logger.error("Failed to generate SQL examples: %s", e)

        # 4) Summary
        analysis["summary"] = {
            "total_tables": len(analysis["schema"]),
            "total_columns": sum(len(s.get("columns", [])) for s in analysis["schema"]),
            "total_relationships": len(analysis["relationships"]),
            "total_column_stats": len(analysis["column_statistics"]),
        }

        return analysis

    # --------------------- Internal helpers ---------------------
    def _get_table_names(self) -> List[str]:
        """Retrieve table names via inspector."""
        try:
            return self.inspector.get_table_names()
        except SQLAlchemyError as e:
            logger.error("Could not retrieve table names: %s", e)
            return []

    def _reflect_table(self, table_name: str) -> Table:
        """
        Reflect a single table into a SQLAlchemy Table object.
        Reflection enables safe identifier handling and SQL building.
        """
        # Using a fresh MetaData for targeted reflection, reduces overhead
        md = MetaData()
        return Table(table_name, md, autoload_with=self.engine)

    def _extract_table_schema(self, table_name: str) -> Dict[str, Any]:
        """
        Extract table schema using SQLAlchemy inspector.

        Returns:
            {
                "table": str,
                "columns": [{name, type, nullable, default, max_length}],
                "primary_key": [col1, ...],
                "foreign_keys": [{constrained_columns, referred_table, referred_columns}],
                "row_count": int
            }
        """
        inspector = self.inspector

        # Columns
        columns = []
        try:
            cols = inspector.get_columns(table_name)
            for c in cols:
                # Convert SQLAlchemy type to string for JSON output
                col_type_str = str(c.get("type"))
                columns.append({
                    "name": c.get("name"),
                    "type": col_type_str,
                    "nullable": c.get("nullable", True),
                    "default": c.get("default"),
                    "max_length": getattr(c.get("type"), "length", None),
                })
        except SQLAlchemyError as e:
            logger.error("Error retrieving columns for %s: %s", table_name, e)

        # Primary key
        pk = []
        try:
            pk_info = inspector.get_pk_constraint(table_name)
            pk = pk_info.get("constrained_columns", []) if pk_info else []
        except SQLAlchemyError as e:
            logger.error("Error retrieving PK for %s: %s", table_name, e)

        # Foreign keys (handles composite keys)
        foreign_keys: List[Dict[str, Any]] = []
        try:
            fk_info = inspector.get_foreign_keys(table_name)
            for fk in fk_info or []:
                foreign_keys.append({
                    "constrained_columns": fk.get("constrained_columns", []),
                    "referred_table": fk.get("referred_table"),
                    "referred_columns": fk.get("referred_columns", [])
                })
        except SQLAlchemyError as e:
            logger.error("Error retrieving FKs for %s: %s", table_name, e)

        # Row count
        row_count = None
        try:
            tbl = self._reflect_table(table_name)
            with self.engine.connect() as conn:
                # SELECT COUNT(*) FROM table
                result = conn.execute(select(func.count()).select_from(tbl))
                row_count = int(result.scalar() or 0)
        except SQLAlchemyError as e:
            logger.warning("Error retrieving row count for %s: %s", table_name, e)
            row_count = -1 # Indicate failure

        return {
            "table": table_name,
            "columns": columns,
            "primary_key": pk,
            "foreign_keys": foreign_keys,
            "row_count": row_count,
        }
    
    def _is_numeric_type(self, col_proxy: Any) -> bool:
        """Check if a SQLAlchemy column proxy represents a numeric type."""
        numeric_types = (Integer, Float, Numeric, DECIMAL, BigInteger, SmallInteger)
        return isinstance(col_proxy.type, numeric_types)

    def _calculate_column_statistics(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """
        Calculate statistics for a single column.

        - Determines cardinality.
        - Low cardinality: Gets value counts.
        - High cardinality: Samples for distribution stats (numeric) or samples (text).
        - Safely uses SQLAlchemy Core for all queries to prevent SQL injection.
        """
        tbl = self._reflect_table(table_name)
        col = tbl.c[column_name]
        stats: Dict[str, Any] = {"column": column_name}

        # Calculate unique count
        unique_count = 0
        try:
            with self.engine.connect() as conn:
                q_unique = select(func.count(distinct(col)))
                unique_count = conn.execute(q_unique).scalar() or 0
                stats["unique_count"] = int(unique_count)
        except SQLAlchemyError as e:
            logger.warning("Unique count failed for %s.%s: %s", table_name, column_name, e)
            stats["unique_count"] = -1
            return stats # Abort if we can't even count

        # Low-cardinality path: get value frequencies
        if unique_count <= self.LOW_CARDINALITY_THRESHOLD:
            try:
                with self.engine.connect() as conn:
                    q_low = (
                        select(col, func.count().label("freq"))
                        .where(col.isnot(None))
                        .group_by(col)
                        .order_by(func.count().desc(), col)
                        .limit(min(unique_count, self.SAMPLE_SIZE_LOW))
                    )
                    rows = conn.execute(q_low).fetchall()
                    stats["type"] = "low_cardinality"
                    stats["sample_values"] = [
                        {"value": r[0], "frequency": int(r[1])} for r in rows
                    ]
            except SQLAlchemyError as e:
                logger.warning("Low-cardinality sample failed for %s.%s: %s", table_name, column_name, e)

            return stats

        # High-cardinality path: sample values randomly
        values: List[Any] = []
        try:
            with self.engine.connect() as conn:
                q_sample = (
                    select(col)
                    .where(col.isnot(None))
                    .order_by(self._get_random_function())
                    .limit(self.HIGH_CARDINALITY_SAMPLE)
                )
                rows = conn.execute(q_sample).fetchall()
                values = [r[0] for r in rows]
        except SQLAlchemyError as e:
            logger.warning("Sampling failed for %s.%s: %s", table_name, column_name, e)
            values = []

        # Handle empty sample gracefully
        if not values:
            stats["type"] = "high_cardinality_empty_sample"
            stats["sample_values"] = []
            return stats

        # Determine numeric vs text vs other using column type
        tbl_col_is_numeric = self._is_numeric_type(col)

        if tbl_col_is_numeric:
            # Convert to floats where possible for numpy stats
            try:
                arr = np.array([float(v) for v in values if v is not None], dtype=float)
                if arr.size == 0:
                    raise ValueError("No numeric values sampled")
                stats["type"] = "high_cardinality_numeric"
                stats["stats"] = {
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "p25": float(np.percentile(arr, 25)),
                    "p50": float(np.percentile(arr, 50)),
                    "p75": float(np.percentile(arr, 75)),
                }
            except Exception as e:
                logger.warning("Numeric stats failed for %s.%s: %s", table_name, column_name, e)
                stats["type"] = "high_cardinality_numeric"
                stats["stats"] = {} # Report failure but keep type
        else:
            # Treat as text/mixed; provide samples and average length
            stats["type"] = "high_cardinality_text"
            sample_size = min(10, len(values))
            stats["sample_values"] = [str(v) for v in values[:sample_size]]
            try:
                avg_len = float(np.mean([len(str(v)) for v in values])) if values else 0.0
            except Exception:
                avg_len = 0.0
            stats["avg_length"] = avg_len

        return stats

    def _generate_sql_examples(self, schema_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Generate example SQL queries based on schema and relationships.

        Examples include:
        - SELECT * with LIMIT
        - SELECT specific columns
        - WHERE on primary key
        - ORDER BY date/timestamp column if present
        - JOIN example using first available FK (handles composite keys)
        - Aggregation (COUNT, AVG) for numeric columns
        """
        examples: List[Dict[str, str]] = []
        table_names = [s["table"] for s in schema_data]

        # Helper to detect date/timestamp columns from schema rows
        def find_datetime_column(schema: Dict[str, Any]) -> Optional[str]:
            for c in schema.get("columns", []):
                t = (c.get("type") or "").lower()
                if "date" in t or "timestamp" in t or "time" in t:
                    return c.get("name")
            return None

        for schema in schema_data:
            table = schema["table"]
            cols = [c["name"] for c in schema.get("columns", [])]
            pk_cols = schema.get("primary_key", [])
            pk = pk_cols[0] if pk_cols else (cols[0] if cols else "id")

            # SELECT all
            examples.append({
                "purpose": f"List all records from {table}",
                "sql": f"SELECT * FROM {table} LIMIT 5;"
            })

            # SELECT specific columns (first two if present)
            if len(cols) >= 2:
                examples.append({
                    "purpose": f"Select {cols[0]} and {cols[1]} from {table}",
                    "sql": f"SELECT {cols[0]}, {cols[1]} FROM {table} LIMIT 5;"
                })

            # WHERE on PK (example)
            if pk_cols: # Only generate if PK is known
                examples.append({
                    "purpose": f"Find record in {table} by primary key",
                    "sql": f"SELECT * FROM {table} WHERE {pk} = 1;"
                })

            # ORDER BY date/timestamp if exists
            dt_col = find_datetime_column(schema)
            if dt_col:
                examples.append({
                    "purpose": f"Recent rows from {table}",
                    "sql": f"SELECT * FROM {table} ORDER BY {dt_col} DESC LIMIT 5;"
                })

        # JOIN example: use first table that has a FK
        for schema in schema_data:
            if schema.get("foreign_keys"):
                fk = schema["foreign_keys"][0]
                child = schema["table"]
                parent = fk.get("referred_table")
                child_cols = fk.get("constrained_columns", [])
                parent_cols = fk.get("referred_columns", [])
                
                if parent in table_names and child_cols and parent_cols and len(child_cols) == len(parent_cols):
                    join_conditions = " AND ".join(
                        [f"c.{c_col} = p.{p_col}" for c_col, p_col in zip(child_cols, parent_cols)]
                    )
                    examples.append({
                        "purpose": f"Join {child} with {parent} on foreign key",
                        "sql": (
                            f"SELECT c.*, p.*\n"
                            f"FROM {child} AS c\n"
                            f"JOIN {parent} AS p ON {join_conditions}\n"
                            f"LIMIT 5;"
                        )
                    })
                    break # Add only one join example for brevity

        # Aggregation: for each table that has numeric columns
        for schema in schema_data:
            table = schema["table"]
            # Naive detection based on type string for example generation
            numeric_cols = [
                c["name"] for c in schema.get("columns", [])
                if any(tok in (c.get("type") or "").lower() for tok in ["int", "decimal", "numeric", "float", "double"])
            ]
            if numeric_cols:
                examples.append({
                    "purpose": f"Aggregate count and average of {numeric_cols[0]} in {table}",
                    "sql": f"SELECT COUNT(*) AS total_rows, AVG({numeric_cols[0]}) AS avg_value FROM {table};"
                })

        return examples

# -----------------------------------------------------------------------------
# Concrete Adapters
# -----------------------------------------------------------------------------
class PostgresAdapter(DBAdapter):
    """PostgreSQL adapter implementation."""
    def _create_engine(self) -> Engine:
        url = (
            f"postgresql://{self.config.username}:{self.config.password.get_secret_value()}@"
            f"{self.config.host}:{self.config.port}/{self.config.database}"
        )
        return create_engine(url)

    def _get_random_function(self) -> expression.Function:
        return func.random()  # PostgreSQL uses RANDOM()


class MySQLAdapter(DBAdapter):
    """MySQL adapter implementation (example extension)."""
    def _create_engine(self) -> Engine:
        url = (
            f"mysql+pymysql://{self.config.username}:{self.config.password.get_secret_value()}@"
            f"{self.config.host}:{self.config.port}/{self.config.database}"
        )
        return create_engine(url)

    def _get_random_function(self) -> expression.Function:
        return func.rand()  # MySQL uses RAND()


# -----------------------------------------------------------------------------
# Adapter Factory
# -----------------------------------------------------------------------------
class AdapterFactory:
    """
    Factory for creating DB adapters based on db_type.
    Extend by adding elif branches or a registry dict.
    """
    @staticmethod
    def get_adapter(db_type: str, config: DBConfig) -> DBAdapter:
        t = db_type.lower()
        if t in ("postgres", "postgresql"):
            return PostgresAdapter(config)
        elif t in ("mysql", "mariadb"):
            return MySQLAdapter(config)
        # Add more adapters here...
        raise ValueError(f"Unsupported DB type: {db_type}")


# -----------------------------------------------------------------------------
# JSON Encoder for Complex Types
# -----------------------------------------------------------------------------
class EnhancedJSONEncoder(json.JSONEncoder):
    """Ensure Decimal, datetime, date are serializable."""
    def default(self, obj: Any):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extensible Database Analyzer for Text-to-SQL AI Systems"
    )
    parser.add_argument(
        "db_type",
        type=str,
        nargs="?",
        default="postgres",
        help="Database type (e.g., postgres, mysql)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="db_analysis_for_ai.json",
        help="Output JSON file path"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    db_type = args.db_type

    # Load configuration from environment variables (retaining Code Examples style)
    try:
        config = load_config_from_env(db_type)
    except Exception: # Catch broad exception from load_config_from_env
        logger.error("Configuration failed. Please check environment variables (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS).")
        return

    # Create adapter via factory
    try:
        adapter = AdapterFactory.get_adapter(db_type, config)
    except ValueError as e:
        logger.error(e)
        return

    # Run analysis
    try:
        analysis = adapter.analyze()
    except SQLAlchemyError as e:
        logger.error("Analysis failed due to DB error: %s", e)
        return
    except Exception as e:
        logger.error("An unexpected error occurred during analysis: %s", e)
        return

    # Write JSON output
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(analysis, f, cls=EnhancedJSONEncoder, indent=2)
        logger.info("Analysis complete! Report saved to: %s", args.output)
        summary = analysis.get("summary", {})
        logger.info(
            "Summary: tables=%d, columns=%d, relationships=%d, column_stats=%d",
            summary.get("total_tables", 0),
            summary.get("total_columns", 0),
            summary.get("total_relationships", 0),
            summary.get("total_column_stats", 0),
        )
    except Exception as e:
        logger.error("Failed to write JSON output: %s", e)


if __name__ == "__main__":
    main()
