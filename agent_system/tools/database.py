import logging
from typing import List, Dict, Any, Tuple, Optional
import sqlalchemy
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import os
import re
from decimal import Decimal

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseConnection:
    """
    Utility class for database operations.
    Handles connections to the PostgreSQL database and query execution.
    """
    
    def __init__(self, connection_string: str):
        """
        Initialize the database connection
        
        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string
        self.engine = None
        self.connected = False
        
        # Connect to the database
        self._connect()
    
    def _connect(self) -> None:
        """
        Establish connection to the database
        """
        try:
            self.engine = create_engine(self.connection_string)
            self.connected = True
            logger.info("Successfully connected to the database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.connected = False
    
    def execute_query(self, query: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Execute a SQL query and return the results
        
        Args:
            query: SQL query to execute
            
        Returns:
            Tuple of (rows as dictionaries, column names)
        """
        # Check if connected
        if not self.connected or not self.engine:
            try:
                self._connect()
                if not self.connected:
                    raise Exception("Not connected to database")
            except Exception as e:
                logger.error(f"Connection error: {e}")
                raise e
        
        try:
            # Process the query to remove escaped quotes which cause issues
            # This is a workaround for the specific issue with quotes in identifiers
            query = query.replace('\\"', '"')
            
            # Execute the query
            with self.engine.connect() as connection:
                # Check if it's a SELECT query (for safety)
                # Use a more robust way to detect SELECT queries by removing leading whitespace and comments
                cleaned_query = re.sub(r'^\s*(--.*?\n)*\s*', '', query, flags=re.DOTALL).strip().upper()
                is_select = cleaned_query.startswith("SELECT")
                
                if not is_select and not os.getenv("ALLOW_NON_SELECT", "false").lower() == "true":
                    # Log the detected query type for debugging
                    logger.warning(f"Non-SELECT query detected: {cleaned_query[:20]}...")
                    raise ValueError("Only SELECT queries are allowed for safety")
                
                # Execute the query
                result = connection.execute(text(query))
                
                # Get column names
                column_names = result.keys()
                
                # Fetch all rows - using the _asdict() method for safety
                try:
                    rows = []
                    for row in result:
                        try:
                            # First try to convert using built-in _asdict() method
                            rows.append(row._asdict())
                        except (AttributeError, TypeError):
                            # Fallback to manual conversion
                            row_dict = {}
                            for i, column in enumerate(column_names):
                                try:
                                    row_dict[column] = row[i]
                                except (IndexError, TypeError):
                                    row_dict[column] = None
                            rows.append(row_dict)
                except Exception as e:
                    logger.error(f"Error converting rows to dictionaries: {e}")
                    # If all else fails, return empty result
                    rows = []
                
                # Clean up non-serializable data types
                rows = self._clean_data_types(rows)
                
                return rows, list(column_names)
                
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise e
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get the schema information for a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column information dictionaries
        """
        try:
            # Query to get table schema
            query = f"""
            SELECT column_name, data_type, character_maximum_length, is_nullable
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position;
            """
            
            # Execute the query
            rows, _ = self.execute_query(query)
            
            return rows
            
        except Exception as e:
            logger.error(f"Error getting table schema: {e}")
            raise e
    
    def get_tables(self) -> List[str]:
        """
        Get list of all tables in the database
        
        Returns:
            List of table names
        """
        try:
            # Query to get all tables
            query = """
            SELECT table_name 
            FROM information_schema.tables
            WHERE table_schema = 'public';
            """
            
            # Execute the query
            rows, _ = self.execute_query(query)
            
            # Extract table names
            table_names = [row["table_name"] for row in rows]
            
            return table_names
            
        except Exception as e:
            logger.error(f"Error getting tables: {e}")
            raise e
    
    def _clean_data_types(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean non-serializable data types in query results
        
        Args:
            rows: List of result rows
            
        Returns:
            List of rows with serializable data types
        """
        clean_rows = []
        
        for row in rows:
            clean_row = {}
            
            for key, value in row.items():
                # Convert non-serializable types
                if isinstance(value, Decimal):
                    clean_row[key] = float(value)
                elif hasattr(value, 'isoformat') and callable(getattr(value, 'isoformat')):
                    clean_row[key] = value.isoformat()
                elif isinstance(value, bytes):
                    clean_row[key] = value.decode('utf-8', errors='replace')
                else:
                    clean_row[key] = value
            
            clean_rows.append(clean_row)
        
        return clean_rows