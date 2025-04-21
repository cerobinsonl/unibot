from __future__ import annotations

"""SyntheticAgent – second‑generation synthetic data generator that uses RowFactory
for row‑level generation and bulk COPY loading instead of INSERT … SELECT.

Key features
============
* **RowFactory integration** – simple rule strings (``fake.name``, ``choice(A,B)``,
  ``normal(70,10)`` …) handled in Python; no heavy SQL random functions.
* **Fast bulk load** – generates a CSV in‑memory and sends it through `COPY FROM
  STDIN` for 10×‑100× speed‑up versus row‑at‑a‑time INSERT.
* **Referential integrity** – supports ``relationships`` spec by sampling keys
  from parent tables already generated in the same temp schema.
* **Reproducibility** – optional deterministic seed so runs are repeatable.

This agent is designed to be called by **SyntheticDataCoordinator** with an
LLM‑produced spec::

    {
      "tables": {"Student": 1000, "Enrollment": 3000},
      "fields": {
        "Student.FirstName": "fake.first_name",
        "Student.GPA": "normal(3.2,0.4)",
        "Enrollment.Grade": "choice(A,B,C,D,F)"
      },
      "relationships": {"Enrollment.StudentId": "Student.StudentId"},
      "constraints": [],
      "temp_prefix": "temp_synth_20250420_ab12"
    }
"""

import io
import logging
import random
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from psycopg2 import sql  # type: ignore
from sqlalchemy import text

from config import get_engine
from agents.specialists.sql_agent import SQLAgent
from agents.specialists.row_factory import RowFactory
from contextlib import closing

logger = logging.getLogger(__name__)


class SyntheticAgent:
    """Generates and bulk‑loads synthetic data into temporary Postgres tables."""

    def __init__(self, *, seed: int | None = None):
        self.engine = get_engine()
        self.sql_agent = SQLAgent()
        self.row_factory = RowFactory(seed=seed or random.randrange(1 << 30))
        self.schema_info = self.sql_agent.schema_info  # raw CREATE statements

    # ------------------------------------------------------------------
    def __call__(self, spec: Dict[str, Any]) -> Dict[str, int]:
        """Execute the full run defined by *spec*.

        Returns a mapping ``table -> rows_inserted`` so the Coordinator can
        summarise progress.
        """
        temp_prefix = spec["temp_prefix"]
        generated_counts: Dict[str, int] = {}

        for table, tbl_cfg in spec.get("tables", {}).items():
            n_rows = tbl_cfg["rows"] if isinstance(tbl_cfg, dict) else tbl_cfg
            if n_rows <= 0:
                continue

            temp_table = f"{temp_prefix}_{table}"
            logger.info("Generating %s rows for table %s (temp table %s)", n_rows, table, temp_table)

            # ------------------------------------------------------------------
            # 1. Create TEMP TABLE with identical structure
            ddl = f"DROP TABLE IF EXISTS \"{temp_table}\";"
            ddl += f"CREATE TABLE \"{temp_table}\" AS TABLE \"{table}\" WITH NO DATA;"
            with self.engine.begin() as conn:
                conn.execute(text(f'DROP TABLE IF EXISTS "{temp_table}"'))
                conn.execute(text(ddl))

            # ------------------------------------------------------------------
            # 2. Generate synthetic DataFrame via RowFactory
            df = self._generate_table_rows(
                base_table=table,
                temp_table=temp_table,
                n=n_rows,
                fields_spec=spec.get("fields", {}),
                relationships=spec.get("relationships", {}),
            )

            # ------------------------------------------------------------------
            # 3. Bulk COPY into Postgres
            self._copy_dataframe_to_table(df, temp_table)
            generated_counts[table] = len(df)
            logger.info("Loaded %s rows into %s", len(df), temp_table)

        return generated_counts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_temp_table_ddl(self, source_table: str, temp_table: str) -> str:
        """
        Create a temp table that clones the production table’s structure
        without copying any rows.
        """
        return (
            f'CREATE TEMPORARY TABLE "{temp_table}" '
            f'(LIKE "{source_table}" INCLUDING ALL);'
        )
        raise ValueError(f"Cannot find DDL for {source_table}")

    # .................................................................
    def _generate_table_rows(
        self,
        *,
        base_table: str,
        temp_table: str,
        n: int,
        fields_spec: Dict[str, Any],
        relationships: Dict[str, str],
    ) -> pd.DataFrame:
        """Generate a DataFrame for *base_table* with *n* rows using RowFactory."""
        # 1. Get column order from information_schema via SQLAgent (LIMIT 0)
        probe = self.sql_agent(f'SELECT * FROM "{base_table}" LIMIT 0;')
        columns: List[str] = probe.get("column_names", [])

        rows = self.row_factory.generate_rows(
            table=base_table,
            columns=columns,
            n=n,
            fields_spec=fields_spec,
            relationships=relationships,
        )
        df = pd.DataFrame(rows, columns=columns)
        return df

    # .................................................................
    def _sample_parent_keys(
        self,
        parent_table: str,
        parent_key: str,
        relationships: Dict[str, str],
        child_temp_table: str,
    ) -> List[Any]:
        """Fetch existing parent key values from the already‑generated temp table."""
        # infer the same temp prefix from child_table name
        prefix = child_temp_table.split("_", maxsplit=2)[0]  # crude but effective
        parent_temp = f"{prefix}_{parent_table}"
        with self.engine.connect() as conn:
            result = conn.execute(text(f'SELECT "{parent_key}" FROM "{parent_temp}"'))
            keys = [r[0] for r in result]
        return keys

    def _copy_dataframe_to_table(self, df: pd.DataFrame, table_name: str) -> None:
        if df.empty:
            return
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False)
        buf.seek(0)

        raw = self.engine.raw_connection()          # ← no “with”
        try:
            with raw.cursor() as cur:
                stmt = sql.SQL("COPY {} FROM STDIN WITH CSV").format(
                    sql.Identifier(table_name)
                )
                cur.copy_expert(stmt.as_string(cur), buf)
            raw.commit()
        finally:
            raw.close()
