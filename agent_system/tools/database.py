import logging
from typing import List, Dict, Any, Tuple, Optional
import sqlalchemy
from sqlalchemy import create_engine, text
import pandas as pd
import os

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
            # Execute the query
            with self.engine.connect() as connection:
                # Check if it's a SELECT query (for safety)
                is_select = query.strip().upper().startswith("SELECT")
                
                if not is_select and not os.getenv("ALLOW_NON_SELECT", "false").lower() == "true":
                    raise ValueError("Only SELECT queries are allowed for safety")
                
                # Execute the query
                result = connection.execute(text(query))
                
                # Get column names
                column_names = result.keys()
                
                # Fetch all rows
                rows = [dict(row) for row in result]
                
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
                if isinstance(value, (sqlalchemy.Decimal, pd._libs.tslibs.timestamps.Timestamp)):
                    clean_row[key] = float(value)
                elif isinstance(value, pd._libs.tslibs.timestamps.Timestamp):
                    clean_row[key] = value.isoformat()
                elif isinstance(value, bytes):
                    clean_row[key] = value.decode('utf-8', errors='replace')
                else:
                    clean_row[key] = value
            
            clean_rows.append(clean_row)
        
        return clean_rows
    
    # Additional methods specifically for working with the provided schema
    
    def get_person_by_id(self, person_id: int) -> Dict[str, Any]:
        """
        Retrieve a person by ID
        
        Args:
            person_id: The PersonId to look for
            
        Returns:
            Person data as dictionary
        """
        query = f'SELECT * FROM "Person" WHERE "PersonId" = {person_id}'
        rows, _ = self.execute_query(query)
        return rows[0] if rows else None
    
    def get_person_by_email(self, email: str) -> Dict[str, Any]:
        """
        Retrieve a person by email address
        
        Args:
            email: The email address to look for
            
        Returns:
            Person data as dictionary
        """
        query = f'SELECT * FROM "Person" WHERE "EmailAddress" = \'{email}\''
        rows, _ = self.execute_query(query)
        return rows[0] if rows else None
    
    def get_financial_aid_by_person(self, person_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve financial aid records for a person
        
        Args:
            person_id: The PersonId to look for
            
        Returns:
            List of financial aid records
        """
        query = f'SELECT * FROM "FinancialAid" WHERE "PersonId" = {person_id}'
        rows, _ = self.execute_query(query)
        return rows
    
    def get_academic_record_by_person(self, person_id: int) -> Dict[str, Any]:
        """
        Retrieve academic record for a person
        
        Args:
            person_id: The PersonId to look for
            
        Returns:
            Academic record data as dictionary
        """
        query = f'SELECT * FROM "PsStudentAcademicRecord" WHERE "PersonId" = {person_id}'
        rows, _ = self.execute_query(query)
        return rows[0] if rows else None
    
    def get_enrollments_by_person(self, person_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve enrollment records for a person
        
        Args:
            person_id: The PersonId to look for
            
        Returns:
            List of enrollment records
        """
        query = f'SELECT * FROM "PsStudentEnrollment" WHERE "PersonId" = {person_id}'
        rows, _ = self.execute_query(query)
        return rows
    
    def get_class_sections_for_enrollment(self, enrollment_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve class sections for a specific enrollment
        
        Args:
            enrollment_id: The StudentEnrollmentId to look for
            
        Returns:
            List of class section records
        """
        query = f'''
        SELECT cs.* FROM "ClassSection" cs
        JOIN "PsStudentClassSection" scs ON cs."ClassSectionId" = scs."ClassSectionId"
        WHERE scs."StudentEnrollmentId" = {enrollment_id}
        '''
        rows, _ = self.execute_query(query)
        return rows
    
    def get_person_roles(self, person_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve roles for a person
        
        Args:
            person_id: The PersonId to look for
            
        Returns:
            List of role records
        """
        query = f'SELECT * FROM "OperationPersonRole" WHERE "PersonId" = {person_id}'
        rows, _ = self.execute_query(query)
        return rows