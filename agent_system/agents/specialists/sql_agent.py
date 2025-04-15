import logging
from typing import Dict, List, Any, Optional
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
        
        # Try to set up the database connection
        try:
            import sqlalchemy
            from sqlalchemy import create_engine, text
            from decimal import Decimal
            
            # Connect to the database using environment variables or settings
            conn_string = settings.DATABASE_URL
            self.engine = create_engine(conn_string)
            logger.info("SQL Agent DB connection initialized successfully")
            self.db_initialized = True
            
            # Dynamically fetch the database schema on initialization
            self.schema_info = self._get_database_schema()
            schema_size = len(self.schema_info)
            table_count = self.schema_info.count('CREATE TABLE')
            logger.info(f"Retrieved database schema with {table_count} tables, schema size: {schema_size} chars")
            
        except Exception as e:
            logger.error(f"Error initializing SQL database connection: {e}", exc_info=True)
            self.db_initialized = False
            self.schema_info = "Error: Could not retrieve database schema"
        
        # Create the code generation prompt
        self.code_prompt = """
You need to generate a SQL query based on a natural language request for a university database.

University Database Schema:
{schema_info}

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

The request is: {task}

Based on the schema provided, generate a single SELECT SQL query that will answer this request.
Make sure to use only tables and columns that actually exist in the schema above.

Reply with ONLY the SQL query, nothing else.
"""
    
    def _get_database_schema(self) -> str:
        """
        Dynamically retrieve and format the database schema
        
        Returns:
            Formatted database schema as a string
        """
        schema_info = []
        try:
            from sqlalchemy import text
            
            # Get all tables in the database
            with self.engine.connect() as connection:
                # Get list of tables
                tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
                """
                result = connection.execute(text(tables_query))
                tables = [row[0] for row in result]
                
                # For each table, get its schema definition
                for table in tables:
                    columns_query = f"""
                    SELECT 
                        column_name, 
                        data_type, 
                        is_nullable,
                        column_default,
                        character_maximum_length
                    FROM information_schema.columns
                    WHERE table_name = '{table}'
                    ORDER BY ordinal_position;
                    """
                    result = connection.execute(text(columns_query))
                    columns = []
                    
                    for row in result:
                        col_name, data_type, nullable, default, max_length = row
                        
                        # Format column type with length if applicable
                        if max_length and data_type == 'character varying':
                            data_type = f"VARCHAR({max_length})"
                        
                        # Format nullable constraint
                        null_constraint = "NULL" if nullable == "YES" else "NOT NULL"
                        
                        # Format default if exists
                        default_str = f" DEFAULT {default}" if default else ""
                        
                        # Add to columns list
                        columns.append(f'    "{col_name}" {data_type} {null_constraint}{default_str}')
                    
                    # Get primary key information
                    pk_query = f"""
                    SELECT a.attname
                    FROM   pg_index i
                    JOIN   pg_attribute a ON a.attrelid = i.indrelid
                                        AND a.attnum = ANY(i.indkey)
                    WHERE  i.indrelid = '"{table}"'::regclass
                    AND    i.indisprimary;
                    """
                    try:
                        pk_result = connection.execute(text(pk_query))
                        pk_columns = [row[0] for row in pk_result]
                        if pk_columns:
                            pk_constraint = f'    PRIMARY KEY ("{pk_columns[0]}")'
                            columns.append(pk_constraint)
                    except:
                        # Skip primary key if query fails
                        pass
                    
                    # Try to get foreign key information
                    fk_query = f"""
                    SELECT
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM
                        information_schema.table_constraints AS tc
                        JOIN information_schema.key_column_usage AS kcu
                          ON tc.constraint_name = kcu.constraint_name
                          AND tc.table_schema = kcu.table_schema
                        JOIN information_schema.constraint_column_usage AS ccu
                          ON ccu.constraint_name = tc.constraint_name
                          AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name='{table}';
                    """
                    try:
                        fk_result = connection.execute(text(fk_query))
                        for row in fk_result:
                            col_name, foreign_table, foreign_col = row
                            fk_constraint = f'    FOREIGN KEY ("{col_name}") REFERENCES "{foreign_table}"("{foreign_col}")'
                            columns.append(fk_constraint)
                    except:
                        # Skip foreign keys if query fails
                        pass
                    
                    # Format as CREATE TABLE statement
                    table_def = f'CREATE TABLE "{table}" (\n'
                    table_def += ",\n".join(columns)
                    table_def += "\n);"
                    
                    schema_info.append(table_def)
                
                # Get row counts for each table to help with query planning
                row_counts = []
                for table in tables:
                    try:
                        count_query = f'SELECT COUNT(*) FROM "{table}"'
                        count_result = connection.execute(text(count_query))
                        count = count_result.scalar()
                        row_counts.append(f'-- "{table}" has {count} rows')
                    except:
                        # Skip if count query fails
                        pass
                
                if row_counts:
                    schema_info.append("\n-- Table Row Counts:")
                    schema_info.extend(row_counts)
                
                # Add explicit notes about important tables
                important_tables = [t for t in tables if ('Student' in t or 'Person' in t or 'Enrollment' in t)]
                if important_tables:
                    schema_info.append("\n-- Important Tables for Student Queries:")
                    for table in important_tables:
                        schema_info.append(f'-- Use "{table}" for student-related information')
                
                return "\n\n".join(schema_info)
        
        except Exception as e:
            logger.error(f"Error retrieving database schema: {e}")
            return "Error retrieving database schema: " + str(e)
    
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
            
            # Make sure it starts with SELECT (after removing comments)
            sql_query = sql_query.strip()
            
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
                # Check if it's a SELECT query
                if not sql_query.strip().upper().startswith("SELECT"):
                    raise ValueError(f"Only SELECT queries are allowed for safety. Query starts with: {sql_query[:20]}")
                
                # Execute the query
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
                            elif hasattr(value, 'isoformat'):
                                row_dict[col] = value.isoformat()
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