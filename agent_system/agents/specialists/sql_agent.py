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
            logger.info(f"Retrieved database schema with {table_count} tables, schema size: {schema_size} chars")
            
        except Exception as e:
            logger.error(f"Error initializing SQL database connection: {e}", exc_info=True)
            self.db_initialized = False
            self.schema_info = f"Error: Could not retrieve database schema - {e}"
        
        logger.info(f"SQL Agent initialization complete. DB Initialized: {self.db_initialized}")

        # Create the code generation prompt
        self.code_prompt = """
You need to generate a SQL query based on a natural language request for a university database.

CURRENT DATABASE SCHEMA INFORMATION:
{schema_info}

USER'S REQUEST:
{task}

IMPORTANT GUIDELINES:
1. This is the ACTUAL schema from the database - use ONLY these tables and columns.
2. DO NOT include comments in your SQL query, just the pure SQL.
3. Always use double quotes around table and column names: "TableName"."ColumnName".
4. Only query tables that exist in the schema provided.
5. If you cannot answer a query with the available schema, explain what's missing.
6. Never invent or assume tables or columns that aren't in the schema.
7. The database is PostgreSQL.
8. Pay close attention to the actual table and column names in the schema.
9. NEVER use fallback queries - if you can't find the right tables, indicate that clearly.
10. When filtering data, ONLY use values that make sense for the column based on its data type and the statistics provided.
11. If a query needs to filter by a certain column value, check if sample values are provided in the schema, and use only those exact values.
12. For date/time filters, ensure the format matches what's expected by PostgreSQL.
13. For numeric filters, ensure values are within appropriate ranges based on column statistics.

Based on the schema provided, generate a single SELECT SQL query that will answer this request.
Make sure to use only tables and columns that actually exist in the schema above.

Reply with ONLY the SQL query, nothing else.
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
        Process a natural language query by generating and executing SQL

        Args:
            task: Natural language description of the data to retrieve

        Returns:
            Dictionary containing query results
        """
        try:
            if not self.db_initialized:
                raise ValueError("SQL database connection was not properly initialized")

            # Log the query task
            logger.info(f"Processing SQL query task: {task}")

            # Generate the SQL query using the LLM
            formatted_prompt = self.code_prompt.format(
                schema_info=self.schema_info,
                task=task
            )

            query_response = self.llm.invoke(formatted_prompt)
            sql_query = query_response.content.strip()

            # Clean up the query
            # Remove any markdown formatting
            sql_query = re.sub(r'^```sql\s*', '', sql_query)
            sql_query = re.sub(r'\s*```$', '', sql_query)

            # Remove any comments that might cause issues with the SELECT check
            sql_query = re.sub(r'--.*?\n', '\n', sql_query)
            sql_query = re.sub(r'/\*.*?\*/', '', sql_query, flags=re.DOTALL)

            # Make sure it starts with SELECT or WITH (after cleaning)
            normalized_query = sql_query.strip().upper()
            if not normalized_query.startswith("SELECT") and not normalized_query.startswith("WITH"):
                raise ValueError(f"Only SELECT or WITH queries are allowed for safety. Query starts with: {sql_query[:20]}")

            # Log the generated query
            logger.info(f"Cleaned SQL query: {sql_query}")

            # Check if the response indicates the query can't be answered with available schema
            if "cannot" in sql_query.lower() or "missing" in sql_query.lower() or "don't have" in sql_query.lower():
                logger.warning(f"LLM indicated schema limitations: {sql_query}")
                return {
                    "error": "Cannot execute query with available schema",
                    "message": sql_query,
                    "results": [],
                    "column_names": ["message"],
                    "row_count": 0,
                    "is_error": True
                }

            # Import necessary modules here to avoid issues
            from sqlalchemy import text
            from decimal import Decimal

            # Execute the query
            with self.engine.connect() as connection:
                try:
                    result = connection.execute(text(sql_query))

                    # Get column names (convert to list to avoid RMKeyView issues)
                    column_names = list(result.keys())

                    # Fetch all rows
                    rows = []
                    for row in result:
                        # Convert row to dictionary
                        row_dict = {}
                        for i, col in enumerate(column_names):
                            value = row[i]
                            # Convert non-serializable types
                            if isinstance(value, Decimal):
                                row_dict[col] = float(value)
                            elif hasattr(value, 'isoformat') and callable(getattr(value, 'isoformat')):
                                row_dict[col] = value.isoformat()
                            elif isinstance(value, bytes):
                                row_dict[col] = value.decode('utf-8', errors='replace')
                            else:
                                row_dict[col] = value
                        rows.append(row_dict)

                    # If no results found, provide clear feedback
                    if len(rows) == 0:
                        logger.info("Query returned zero results")
                        return {
                            "query": sql_query,
                            "results": [],
                            "column_names": column_names,
                            "row_count": 0,
                            "message": "The query executed successfully but returned no results."
                        }

                    # Return the results
                    return {
                        "query": sql_query,
                        "results": rows,
                        "column_names": column_names,
                        "row_count": len(rows)
                    }

                except Exception as db_error:
                    logger.error(f"Database error executing query: {db_error}")

                    # Return with specific database error
                    return {
                        "error": f"Database error: {str(db_error)}",
                        "query": sql_query,
                        "results": [{"error_message": str(db_error)}],
                        "column_names": ["error_message"],
                        "row_count": 1,
                        "is_error": True
                    }

        except Exception as e:
            logger.error(f"Error in SQL Agent: {e}", exc_info=True)

            # Return error information
            return {
                "error": str(e),
                "results": [{"error_message": str(e)}],
                "column_names": ["error_message"],
                "row_count": 1,
                "is_error": True
            }