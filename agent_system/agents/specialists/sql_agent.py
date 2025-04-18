import logging
from typing import Dict, List, Any, Tuple, Optional
import json
import re
import os

# Import configuration
from config import settings, AGENT_CONFIGS, get_llm

# Configure logging
logger = logging.getLogger(__name__)

class SQLAgent:
    """
    SQL Agent is responsible for translating natural language queries into SQL
    and executing them directly against the university database.
    """
    
    def __init__(self):
        """Initialize the SQL Agent with dynamic schema retrieval"""
        # Create the LLM using the helper function
        self.llm = get_llm("sql_agent")
        self.engine = None  # Initialize engine to None
        self.db_initialized = False # Initialize db_initialized to False
        self.schema_info = "Database connection not yet initialized" # Default schema info
        
        # Try to set up the database connection
        try:
            import sqlalchemy
            from sqlalchemy import create_engine, text
            from decimal import Decimal
            from config import get_engine
            
            self.engine = get_engine()
            self.db_initialized = True
            logger.info("SQL Agent DB connection initialized successfully")

            # Dynamically fetch the database schema on initialization
            self.schema_info = self._get_enhanced_schema_info()
            schema_size = len(self.schema_info)
            table_count = self.schema_info.count('Table:')
            
        except Exception as e:
            logger.error(f"Error initializing SQL database connection: {e}", exc_info=True)
            self.db_initialized = False
            self.schema_info = f"Error: Could not retrieve database schema - {e}"
        
        logger.info(f"SQL Agent initialization complete. DB Initialized: {self.db_initialized}")

        # Create the code generation prompt
        self.code_prompt = """
        You need to generate a valid PostgreSQL statement (SELECT, INSERT, UPDATE, or DELETE)
        based on a natural-language request for our university database.

        CURRENT DATABASE SCHEMA INFORMATION:
        {schema_info}
        USER'S REQUEST:
        {task}

        IMPORTANT GUIDELINES:
        1. Use ONLY the tables and columns in the schema above.
        2. Always quote identifiers with double quotes: "TableName"."ColumnName".
        3. For INSERT/UPDATE/DELETE, ensure the correct syntax and handle NULLs appropriately.
        4. Do not include extraneous comments—output only the pure SQL statement.
        5. If the operation cannot be performed, return an error explanation instead of SQL.
        6. Pay close attention to the actual table and column names in the schema.
        7. When filtering data, ONLY use values that make sense for the column based on its data type and the statistics provided.
        8. The database is PostgreSQL.
        9. For date/time filters, ensure the format matches what's expected by PostgreSQL.
        10. For numeric filters, ensure values are within appropriate ranges based on column statistics.
        
        Reply with a JSON object **only**:
        {{ "sql": "<your SQL here>", "type": "<select|insert|update|delete>" }}
        """
    
    def _get_enhanced_schema_info(self) -> str:
        """
        Get enhanced schema information including statistics and sample values
        
        Returns:
            Formatted schema information with statistics
        """
        try:
            from sqlalchemy import text
            
            schema_info = []
            
            # Get all tables
            with self.engine.connect() as connection:
                tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
                """
                result = connection.execute(text(tables_query))
                tables = [row[0] for row in result]
                
                # For each table
                for table in tables:
                    # Get table structure
                    table_info = [f"Table: {table}"]
                    
                    # Get columns and their types
                    columns_query = f"""
                    SELECT 
                        column_name, 
                        data_type, 
                        is_nullable
                    FROM information_schema.columns
                    WHERE table_name = '{table}'
                    ORDER BY ordinal_position;
                    """
                    result = connection.execute(text(columns_query))
                    columns = []
                    
                    for row in result:
                        col_name, data_type, nullable = row
                        nullable_str = "NULL" if nullable == "YES" else "NOT NULL"
                        columns.append(f"  - {col_name} ({data_type}, {nullable_str})")
                    
                    table_info.extend(columns)
                    
                    # Get row count
                    try:
                        count_query = f'SELECT COUNT(*) FROM "{table}"'
                        count_result = connection.execute(text(count_query))
                        count = count_result.scalar()
                        table_info.append(f"  - Row count: {count}")
                    except:
                        pass
                    
                    # For each column, get statistics
                    column_stats = []
                    
                    for row in connection.execute(text(columns_query)):
                        col_name, data_type, _ = row
                        
                        # For numeric columns
                        if data_type in ('integer', 'numeric', 'real', 'double precision'):
                            try:
                                stats_query = f"""
                                SELECT 
                                    MIN("{col_name}"), 
                                    MAX("{col_name}"), 
                                    AVG("{col_name}"),
                                    COUNT(*) FILTER (WHERE "{col_name}" IS NOT NULL)
                                FROM "{table}";
                                """
                                stats_result = connection.execute(text(stats_query))
                                min_val, max_val, avg_val, non_null_count = stats_result.fetchone()
                                
                                if min_val is not None:
                                    column_stats.append(f"  - {col_name}: Range [{min_val} to {max_val}], Avg: {avg_val:.2f}, Non-null: {non_null_count}")
                            except:
                                pass
                        
                        # For all columns, get distinct values if few
                        try:
                            distinct_query = f"""
                            SELECT COUNT(DISTINCT "{col_name}") 
                            FROM "{table}";
                            """
                            distinct_result = connection.execute(text(distinct_query))
                            distinct_count = distinct_result.scalar()
                            
                            if distinct_count is not None and distinct_count < 15:
                                sample_query = f"""
                                SELECT DISTINCT "{col_name}" 
                                FROM "{table}" 
                                WHERE "{col_name}" IS NOT NULL 
                                LIMIT 10;
                                """
                                sample_result = connection.execute(text(sample_query))
                                sample_values = [str(row[0]) for row in sample_result]
                                
                                column_stats.append(f"  - {col_name}: {distinct_count} distinct values. Samples: {', '.join(sample_values)}")
                        except:
                            pass
                    
                    if column_stats:
                        table_info.append("  Column Statistics:")
                        table_info.extend(column_stats)
                    
                    schema_info.append("\n".join(table_info))
                
                return "\n\n".join(schema_info)
        
        except Exception as e:
            logger.error(f"Error getting enhanced schema info: {e}")
            return f"Error retrieving schema information: {e}"
    
    def __call__(self, task: str) -> Dict[str, Any]:
        """
        Translate a natural-language request into SQL (SELECT, INSERT, UPDATE, DELETE),
        execute it, and return structured results or error information.

        Args:
            task: Natural-language description of the query or DML operation.

        Returns:
            A dict with one of these shapes:
            - SELECT/WITH:
                {
                "is_error": False,
                "query": "<generated SQL>",
                "results": [ {col: val, …}, … ],
                "column_names": [ … ],
                "row_count": N
                }
            - INSERT/UPDATE/DELETE:
                {
                "is_error": False,
                "query": "<generated SQL>",
                "affected_rows": N
                }
            - On error:
                {
                "is_error": True,
                "error": "<error message>",
                "query": "<generated SQL, if any>",
                // For SELECT errors:
                "results": [{ "error_message": "<db error>" }],
                "column_names": ["error_message"],
                "row_count": 1
                }
        """
        try:
            # 0. Ensure DB is ready
            if not self.db_initialized:
                raise ValueError("SQL database connection was not properly initialized")

            logger.info(f"Processing SQL task: {task}")

            # 1. Generate SQL (+ type) via LLM
            formatted_prompt = self.code_prompt.format(
                schema_info=self.schema_info,
                task=task
            )
            llm_resp = self.llm.invoke(formatted_prompt).content

            # 2. Clean markdown fences
            import re, json
            sql_text = llm_resp.strip()
            sql_text = re.sub(r"^```(?:sql|json)?\s*", "", sql_text, flags=re.MULTILINE)
            sql_text = re.sub(r"\s*```$", "", sql_text, flags=re.MULTILINE)
            # remove any inline comments
            sql_text = re.sub(r"--.*?\n", "\n", sql_text)
            sql_text = re.sub(r"/\*.*?\*/", "", sql_text, flags=re.DOTALL)

            # 3. Detect JSON wrapper for type+sql or assume pure SQL
            op_type = "select"
            try:
                payload = json.loads(sql_text)
                sql = payload.get("sql", "").strip()
                op_type = payload.get("type", "select").lower()
            except json.JSONDecodeError:
                sql = sql_text

            sql = sql.strip()
            logger.info(f"Cleaned SQL ({op_type}): {sql}")

            # 4. Safety check for SELECT path
            is_select = op_type == "select"
            if is_select:
                head = sql[:10].upper()
                if not (head.startswith("SELECT") or head.startswith("WITH")):
                    raise ValueError(f"Only SELECT/WITH queries allowed for read operations; got: {sql[:20]}")

            # 5. Execute via SQLAlchemy
            from sqlalchemy import text
            from decimal import Decimal

            if op_type == "select":
                with self.engine.connect() as conn:
                    result = conn.execute(text(sql))
                    col_names = list(result.keys())
                    rows = []
                    for r in result:
                        row = {}
                        for idx, col in enumerate(col_names):
                            val = r[idx]
                            if isinstance(val, Decimal):
                                row[col] = float(val)
                            elif hasattr(val, "isoformat"):
                                row[col] = val.isoformat()
                            elif isinstance(val, bytes):
                                row[col] = val.decode("utf-8", errors="replace")
                            else:
                                row[col] = val
                        rows.append(row)
                    if not rows:
                        logger.info("Query executed successfully but returned no results")
                        return {
                            "is_error": False,
                            "query": sql,
                            "results": [],
                            "column_names": col_names,
                            "row_count": 0,
                            "message": "The query executed successfully but returned no results."
                        }
                    return {
                        "is_error": False,
                        "query": sql,
                        "results": rows,
                        "column_names": col_names,
                        "row_count": len(rows)
                    }

            else:
                # DML: INSERT, UPDATE, DELETE
                # use transactional execution for commit
                with self.engine.begin() as conn:
                    result = conn.execute(text(sql))
                    affected = result.rowcount
                    return {
                        "is_error": False,
                        "query": sql,
                        "affected_rows": affected
                    }

        except Exception as e:
            logger.error(f"Error in SQLAgent __call__: {e}", exc_info=True)
            # On failure, return structured error
            if 'sql' in locals():
                # if SQL was generated, include it
                return {
                    "is_error": True,
                    "error": str(e),
                    "query": sql,
                    "results": [{"error_message": str(e)}],
                    "column_names": ["error_message"],
                    "row_count": 1
                }
            else:
                return {
                    "is_error": True,
                    "error": str(e),
                    "results": [{"error_message": str(e)}],
                    "column_names": ["error_message"],
                    "row_count": 1
                }