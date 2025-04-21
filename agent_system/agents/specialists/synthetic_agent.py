import logging
from typing import Dict, Any
from sqlalchemy import text
import pandas as pd
import numpy as np

from config import get_engine
from agents.specialists.sql_agent import SQLAgent

logger = logging.getLogger(__name__)

class SyntheticAgent:
    """
    Synthetic Data Generator that:
    - Reads a spec with tables/fields/relationships/constraints + temp_prefix
    - Creates temp tables named {temp_prefix}_{Table}
    - Populates them with synthetic data
    - Returns how many rows were inserted per table
    """

    def __init__(self):
        self.engine = get_engine()
        self.schema_info = SQLAgent().schema_info  # raw CREATE statements for production tables

    def __call__(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        temp_prefix = spec["temp_prefix"]
        tables = spec.get("tables", {})
        fields_spec = spec.get("fields", {})
        relationships = spec.get("relationships", {})
        constraints = spec.get("constraints", [])

        created_counts = {}

        for table_name, row_count in tables.items():
            temp_table = f"{temp_prefix}_{table_name}"
            logger.info(f"Creating temp table {temp_table} for {row_count} rows")

            # 1. Drop & recreate temp table based on production schema
            ddl = self._extract_create_statement(table_name, temp_table)
            with self.engine.begin() as conn:
                conn.execute(text(f'DROP TABLE IF EXISTS "{temp_table}";'))
                conn.execute(text(ddl))

            # 2. Generate a DataFrame of synthetic rows
            df = self._generate_table_rows(table_name, row_count, fields_spec, relationships)

            # 3. Bulk‐insert via pandas to_sql
            df.to_sql(
                name=temp_table,
                con=self.engine,
                if_exists="append",
                index=False,
                method="multi"
            )

            created_counts[table_name] = row_count
            logger.info(f"Inserted {row_count} rows into {temp_table}")

        return {
            "created": created_counts,
            "temp_prefix": temp_prefix
        }

    def _extract_create_statement(self, source_table: str, temp_table: str) -> str:
        """
        Locate the CREATE TABLE for source_table in self.schema_info,
        then rewrite it to CREATE TABLE "{temp_table}" (…) with identical columns.
        """
        # naive split on blank line
        chunks = self.schema_info.split("\n\n")
        for chunk in chunks:
            if chunk.startswith(f'CREATE TABLE "{source_table}"'):
                # replace table name
                return chunk.replace(f'CREATE TABLE "{source_table}"',
                                     f'CREATE TABLE "{temp_table}"')
        raise ValueError(f"Schema for table {source_table} not found")

    def _generate_table_rows(
        self,
        table: str,
        n: int,
        fields_spec: Dict[str, Any],
        relationships: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Build a DataFrame with n rows for 'table', applying:
        - fields_spec: e.g. { "Person.GPA": {"distribution":"normal","mean":3.2,"std":0.4}, ... }
        - relationships: e.g. { "Enrollment.PersonId": "Person.PersonId", ... }
        """
        # 1. Inspect production schema for column names/types
        # Here we assume SQLAgent can give us a list of columns
        sql_agent = SQLAgent()
        # hack: ask SQLAgent for zero‐row SELECT to get column order
        probe = sql_agent(f"SELECT * FROM \"{table}\" LIMIT 0;")
        cols = probe.get("column_names", [])

        data = {}
        for col in cols:
            fq = f"{table}.{col}"
            spec = fields_spec.get(fq, {})

            # Numeric with normal distribution
            if spec.get("distribution") == "normal":
                mean = spec.get("mean", 0)
                std = spec.get("std", 1)
                data[col] = np.random.normal(loc=mean, scale=std, size=n).round(2)

            # Categorical choices
            elif "choices" in spec:
                choices = spec["choices"]
                data[col] = np.random.choice(choices, size=n)

            # If this column is a foreign‑key in relationships, just fill later
            elif fq in relationships:
                data[col] = [None] * n  # placeholder

            # Fallback: numeric→uniform; strings→empty or sequential
            else:
                # numeric?
                sample_type = probe["results"][0].get(col) if probe.get("results") else None
                if isinstance(sample_type, (int, float)):
                    data[col] = np.random.randint(0, 100, size=n)
                else:
                    data[col] = ["" for _ in range(n)]

        df = pd.DataFrame(data)

        # 2. Fill relationships by sampling keys from parent temp tables
        for child_col, parent_ref in relationships.items():
            # parent_ref = "ParentTable.ParentKey"
            parent_table, parent_key = parent_ref.split(".")
            parent_temp = f"{spec['temp_prefix']}_{parent_table}"
            parent_rows = self.engine.execute(
                text(f'SELECT "{parent_key}" FROM "{parent_temp}"')
            ).fetchall()
            parent_keys = [r[0] for r in parent_rows]

            df_col = child_col.split(".")[1]
            df[df_col] = np.random.choice(parent_keys, size=n)

        return df
