#!/usr/bin/env python3
"""
Simplified Database Backup and Restore Script for PostgreSQL
===========================================================
This script provides local backup and restore functionality for PostgreSQL databases.
Uses only pip-installable packages (psycopg2-binary).

Usage:
    # Create a backup
    python label_pizza/backup_restore.py backup --database-url-name DBURL --backup-dir ./backups --output mybackup.sql
    
    # Restore from backup (full path)
    python label_pizza/backup_restore.py restore --database-url-name DBURL --input ./backups/mybackup.sql
    
    # Restore from backup (filename + backup dir)
    python label_pizza/backup_restore.py restore --database-url-name DBURL --backup-dir ./backups --input mybackup.sql.gz
    
    # Backup with compression
    python label_pizza/backup_restore.py backup --database-url-name DBURL --backup-dir ./backups --output mybackup.sql.gz --compress
    
    # List available backups
    python label_pizza/backup_restore.py list --backup-dir ./backups

Dependencies:
    pip install psycopg2-binary python-dotenv
"""

import argparse
import os
import sys
import gzip
import datetime
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")


try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("❌ psycopg2 not available. Install with: pip install psycopg2-binary")
    sys.exit(1)


class DatabaseBackupRestore:
    """Handle database backup and restore operations using psycopg2"""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.parsed_url = urlparse(db_url)
    
    def create_backup(self, output_file: str, compress: bool = False, schema_only: bool = False, verbose: bool = False) -> bool:
        """Create a database backup"""
        try:
            # Handle compression
            if compress and not output_file.endswith('.gz'):
                output_file += '.gz'
                
            print(f"🔄 Creating backup: {output_file}")
            if verbose:
                print(f"   Database: {self.parsed_url.hostname}")
                print(f"   Schema only: {schema_only}")
                print(f"   Compressed: {compress}")
            
            # Connect to database
            conn = psycopg2.connect(self.db_url)
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            
            # Choose file handler based on compression
            if compress:
                file_handle = gzip.open(output_file, 'wt', encoding='utf-8')
            else:
                file_handle = open(output_file, 'w', encoding='utf-8')
            
            try:
                # Write header
                file_handle.write("-- PostgreSQL database backup\n")
                file_handle.write(f"-- Created: {datetime.datetime.now().isoformat()}\n")
                file_handle.write(f"-- Database: {self.parsed_url.hostname}\n")
                file_handle.write(f"-- Schema only: {schema_only}\n")
                file_handle.write("-- Format: JSON data with parameterized queries\n\n")
                
                # Backup schema
                if verbose:
                    print("   📝 Starting schema backup...")
                self._backup_schema(conn, file_handle, verbose)
                if verbose:
                    print("   ✅ Schema backup completed")
                
                # Backup data (unless schema-only)
                if not schema_only:
                    if verbose:
                        print("   💾 Starting data backup...")
                    self._backup_data(conn, file_handle, verbose)
                    if verbose:
                        print("   ✅ Data backup completed")
                
                file_handle.write("\n-- Backup completed\n")
                
            finally:
                file_handle.close()
                conn.close()
            
            # Get file size
            file_size = os.path.getsize(output_file)
            print(f"✅ Backup completed: {file_size:,} bytes")
            
            # Create metadata file
            self._create_backup_metadata(output_file, schema_only, compress)
            
            return True
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            if verbose:
                import traceback
                print(f"   Full error: {traceback.format_exc()}")
            return False

    def restore_backup(self, input_file: str, force: bool = False, verbose: bool = False) -> bool:
        """Restore a database backup with optimized performance"""
        try:
            if not os.path.exists(input_file):
                print(f"❌ Backup file not found: {input_file}")
                return False
                
            # Check if file is compressed
            is_compressed = input_file.endswith('.gz')
            
            print(f"🔄 Restoring backup: {input_file}")
            if verbose:
                print(f"   Database: {self.parsed_url.hostname}")
                print(f"   Compressed: {is_compressed}")
                print(f"   Mode: FAST RESTORE (optimized for speed)")
            
            if not force:
                response = input("⚠️  This will overwrite existing data. Continue? (y/N): ")
                if response.lower() not in ['y', 'yes']:
                    print("❌ Restore cancelled")
                    return False
            
            # Connect to database
            conn = psycopg2.connect(self.db_url)
            
            try:
                # Use a transaction for better performance and atomicity
                with conn:
                    with conn.cursor() as cursor:
                        # Read backup file
                        if verbose:
                            print("   📖 Reading backup file...")
                        if is_compressed:
                            file_handle = gzip.open(input_file, 'rt', encoding='utf-8')
                        else:
                            file_handle = open(input_file, 'r', encoding='utf-8')
                        
                        try:
                            content = file_handle.read()
                            if verbose:
                                print(f"   📄 Processing backup content ({len(content)} characters)")
                            
                            # Parse and execute backup with optimizations
                            self._restore_from_content(cursor, content, verbose)
                            
                        finally:
                            file_handle.close()
                        
                        if verbose:
                            print(f"   ✅ Transaction committed - all changes saved")
                
            finally:
                conn.close()
            
            print("🎉 Restore completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Restore failed: {e}")
            if verbose:
                import traceback
                print(f"   Full error: {traceback.format_exc()}")
            return False

    def _backup_schema(self, conn, file_handle, verbose: bool = False):
        """Backup database schema"""
        with conn.cursor() as cursor:
            # Get all tables
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            if verbose:
                print(f"   📋 Found {len(tables)} tables to backup")
            
            # Drop existing tables (in reverse order for dependencies)
            file_handle.write("-- Drop existing tables and sequences\n")
            for table in reversed(tables):
                file_handle.write(f"DROP TABLE IF EXISTS \"{table}\" CASCADE;\n")
            file_handle.write("\n")
            
            # Drop and create sequences
            file_handle.write("-- Create sequences\n")
            try:
                # Try the newer column names first
                cursor.execute("""
                    SELECT sequence_name, start_value, increment, maximum_value, minimum_value, cycle_option
                    FROM information_schema.sequences 
                    WHERE sequence_schema = 'public'
                    ORDER BY sequence_name;
                """)
                
                sequences = cursor.fetchall()
                for seq_name, start_val, increment, max_val, min_val, cycle in sequences:
                    file_handle.write(f"DROP SEQUENCE IF EXISTS \"{seq_name}\" CASCADE;\n")
                    
                    # Create sequence with proper parameters
                    create_seq = f'CREATE SEQUENCE "{seq_name}"'
                    if start_val is not None:
                        create_seq += f' START WITH {start_val}'
                    if increment is not None:
                        create_seq += f' INCREMENT BY {increment}'
                    if max_val is not None:
                        create_seq += f' MAXVALUE {max_val}'
                    if min_val is not None:
                        create_seq += f' MINVALUE {min_val}'
                    if cycle == 'YES':
                        create_seq += ' CYCLE'
                    else:
                        create_seq += ' NO CYCLE'
                    
                    file_handle.write(create_seq + ";\n")
                
                file_handle.write("\n")
                if verbose:
                    print(f"   📝 Found {len(sequences)} sequences")
                
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Warning: Could not backup sequences with detailed info: {e}")
                # Try the simpler pg_sequences approach
                try:
                    cursor.execute("""
                        SELECT schemaname, sequencename 
                        FROM pg_sequences 
                        WHERE schemaname = 'public'
                        ORDER BY sequencename;
                    """)
                    
                    simple_sequences = cursor.fetchall()
                    for schema, seq_name in simple_sequences:
                        file_handle.write(f"DROP SEQUENCE IF EXISTS \"{seq_name}\" CASCADE;\n")
                        file_handle.write(f"CREATE SEQUENCE \"{seq_name}\";\n")
                    
                    file_handle.write("\n")
                    if verbose:
                        print(f"   📝 Found {len(simple_sequences)} sequences (simple mode)")
                    
                except Exception as e2:
                    if verbose:
                        print(f"   ⚠️  Could not backup sequences at all: {e2}")
                    # Final fallback - try to detect sequences from table defaults
                    try:
                        cursor.execute("""
                            SELECT DISTINCT 
                                substring(pg_get_expr(d.adbin, d.adrelid) from 'nextval\\(''([^'']+)')
                            FROM pg_attrdef d
                            JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum
                            JOIN pg_class c ON c.oid = d.adrelid
                            JOIN pg_namespace n ON n.oid = c.relnamespace
                            WHERE n.nspname = 'public'
                            AND pg_get_expr(d.adbin, d.adrelid) LIKE 'nextval%'
                            ORDER BY 1;
                        """)
                        
                        detected_sequences = cursor.fetchall()
                        for (seq_name,) in detected_sequences:
                            if seq_name:  # Make sure it's not None
                                clean_seq_name = seq_name.replace('public.', '')
                                file_handle.write(f"DROP SEQUENCE IF EXISTS \"{clean_seq_name}\" CASCADE;\n")
                                file_handle.write(f"CREATE SEQUENCE \"{clean_seq_name}\";\n")
                        
                        file_handle.write("\n")
                        if verbose:
                            print(f"   📝 Detected {len(detected_sequences)} sequences from table defaults")
                        
                    except Exception as e3:
                        if verbose:
                            print(f"   ⚠️  No sequences could be detected: {e3}")
                        file_handle.write("-- No sequences detected\n\n")
            
            # Create tables
            file_handle.write("-- Create tables\n")
            for table in tables:
                if verbose:
                    print(f"   📝 Backing up schema for: {table}")
                
                try:
                    # Get table definition
                    cursor.execute("""
                        SELECT 
                            a.attname as column_name,
                            format_type(a.atttypid, a.atttypmod) as data_type,
                            a.attnotnull as not_null,
                            pg_get_expr(d.adbin, d.adrelid) as default_value
                        FROM pg_attribute a
                        LEFT JOIN pg_attrdef d ON a.attrelid = d.adrelid AND a.attnum = d.adnum
                        WHERE a.attrelid = %s::regclass
                        AND a.attnum > 0 
                        AND NOT a.attisdropped
                        ORDER BY a.attnum;
                    """, (f'public.{table}',))
                    
                    columns = cursor.fetchall()
                    
                    if not columns:
                        if verbose:
                            print(f"   ⚠️  No columns found for {table}, skipping")
                        continue
                    
                    file_handle.write(f"CREATE TABLE \"{table}\" (\n")
                    
                    column_defs = []
                    for col_name, data_type, not_null, default_value in columns:
                        col_def = f"    \"{col_name}\" {data_type}"
                        
                        if not_null:
                            col_def += " NOT NULL"
                        
                        if default_value and default_value.strip():
                            col_def += f" DEFAULT {default_value}"
                        
                        column_defs.append(col_def)
                    
                    file_handle.write(",\n".join(column_defs))
                    
                    # Add primary key constraint
                    try:
                        cursor.execute("""
                            SELECT string_agg(quote_ident(a.attname), ', ' ORDER BY c.conkey[s.i]) as pk_columns
                            FROM pg_constraint c
                            CROSS JOIN generate_series(1, array_length(c.conkey, 1)) AS s(i)
                            JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = c.conkey[s.i]
                            WHERE c.conrelid = %s::regclass AND c.contype = 'p'
                            GROUP BY c.conname;
                        """, (f'public.{table}',))
                        
                        pk_result = cursor.fetchone()
                        if pk_result and pk_result[0]:
                            file_handle.write(f",\n    PRIMARY KEY ({pk_result[0]})")
                    except Exception as e:
                        if verbose:
                            print(f"   ⚠️  Could not get primary key for {table}: {e}")
                    
                    file_handle.write("\n);\n\n")
                    
                except Exception as e:
                    if verbose:
                        print(f"   ❌ Error backing up schema for {table}: {e}")
                    continue
            
            # After creating tables, update sequence values to current max values
            file_handle.write("-- Update sequence values\n")
            try:
                cursor.execute("""
                    SELECT schemaname, sequencename 
                    FROM pg_sequences 
                    WHERE schemaname = 'public'
                    ORDER BY sequencename;
                """)
                
                sequences = cursor.fetchall()
                for schema, seq_name in sequences:
                    # Try to find the table and column that uses this sequence
                    table_name = seq_name.replace('_id_seq', '').replace('_seq', '')
                    if table_name in [t for t in tables]:
                        file_handle.write(f"-- Update {seq_name} will be done after data insert\n")
                        
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Warning: Could not prepare sequence updates: {e}")

    def _backup_data(self, conn, file_handle, verbose: bool = False):
        """Backup table data using JSON format for safety"""
        # First get tables with regular cursor
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
        
        file_handle.write("-- Insert data\n")
        
        total_rows = 0
        tables_with_data = 0
        
        for table in tables:
            try:
                if verbose:
                    print(f"   💾 Backing up data for: {table}")
                
                # Use regular cursor for metadata queries
                with conn.cursor() as meta_cursor:
                    # Get row count
                    meta_cursor.execute(f'SELECT COUNT(*) FROM "{table}";')
                    row_count = meta_cursor.fetchone()[0]
                    
                    if row_count == 0:
                        file_handle.write(f"-- No data in table {table}\n")
                        continue
                    
                    total_rows += row_count
                    tables_with_data += 1
                    
                    if verbose:
                        print(f"     💾 {row_count} rows")
                    
                    # Get column names
                    meta_cursor.execute("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_schema = 'public' AND table_name = %s
                        ORDER BY ordinal_position;
                    """, (table,))
                    
                    columns = [row[0] for row in meta_cursor.fetchall()]
                    
                    if not columns:
                        if verbose:
                            print(f"     ⚠️  No columns found for {table}")
                        continue
                
                # Write data section header
                file_handle.write(f"-- DATA_START:{table}\n")
                file_handle.write(f"-- COLUMNS:{json.dumps(columns)}\n")
                
                # Use RealDictCursor for data queries
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as data_cursor:
                    # Fetch and write data in batches
                    data_cursor.execute(f'SELECT * FROM "{table}";')
                    
                    batch_size = 1000
                    rows_processed = 0
                    
                    while True:
                        rows = data_cursor.fetchmany(batch_size)
                        if not rows:
                            break
                        
                        for row in rows:
                            try:
                                # Convert row to JSON-safe format
                                json_row = {}
                                for col in columns:
                                    value = row[col]
                                    
                                    if value is None:
                                        json_row[col] = None
                                    elif isinstance(value, datetime.datetime):
                                        json_row[col] = value.isoformat()
                                    elif isinstance(value, datetime.date):
                                        json_row[col] = value.isoformat()
                                    elif isinstance(value, (dict, list)):
                                        # Already JSON-serializable
                                        json_row[col] = value
                                    elif isinstance(value, (int, float, bool)):
                                        json_row[col] = value
                                    else:
                                        # Convert to string for safety
                                        json_row[col] = str(value)
                                
                                # Write as JSON line
                                file_handle.write(f"-- ROW:{json.dumps(json_row, ensure_ascii=False)}\n")
                                rows_processed += 1
                                
                            except Exception as e:
                                if verbose:
                                    print(f"     ⚠️  Error processing row {rows_processed} in {table}: {e}")
                                continue
                
                file_handle.write(f"-- DATA_END:{table}\n\n")
                if verbose:
                    print(f"     ✅ Processed {rows_processed} rows")
                
            except Exception as e:
                if verbose:
                    print(f"   ❌ Error backing up data for {table}: {e}")
                file_handle.write(f"-- ERROR backing up {table}: {e}\n\n")
                continue
        
        if not verbose:
            print(f"   📊 Backed up {total_rows:,} rows from {tables_with_data} tables")
        
        # Add sequence value updates at the end
        file_handle.write("-- Update sequence values to current maximums\n")
        with conn.cursor() as cursor:
            try:
                cursor.execute("""
                    SELECT schemaname, sequencename 
                    FROM pg_sequences 
                    WHERE schemaname = 'public'
                    ORDER BY sequencename;
                """)
                
                sequences = cursor.fetchall()
                for schema, seq_name in sequences:
                    # Try to find the table and column that uses this sequence
                    table_name = seq_name.replace('_id_seq', '').replace('_seq', '')
                    
                    # Check if this table exists in our backup
                    if table_name in tables:
                        try:
                            # Get the current maximum value for the sequence
                            cursor.execute(f'SELECT COALESCE(MAX(id), 0) + 1 FROM "{table_name}";')
                            max_val = cursor.fetchone()[0]
                            file_handle.write(f"SELECT setval('\"{seq_name}\"', {max_val}, false);\n")
                        except Exception as e:
                            if verbose:
                                print(f"   ⚠️  Could not get max value for {table_name}: {e}")
                            file_handle.write(f"-- Could not update {seq_name}: {e}\n")
                
                file_handle.write("\n")
                
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Warning: Could not update sequence values: {e}")

    def _restore_from_content(self, cursor, content: str, verbose: bool = False):
        """Restore from backup content with optimized bulk inserts"""
        lines = content.split('\n')
        
        schema_statements = []
        data_sections = []
        current_table = None
        current_columns = []
        current_rows = []
        current_sql = ""
        
        # Parse content
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            if not line:
                continue
            
            if line.startswith('--'):
                # If we have a pending SQL statement, save it
                if current_sql.strip():
                    schema_statements.append(current_sql.strip())
                    current_sql = ""
                
                if line.startswith('-- DATA_START:'):
                    current_table = line.split(':', 1)[1]
                    current_columns = []
                    current_rows = []
                elif line.startswith('-- COLUMNS:'):
                    try:
                        columns_json = line.split(':', 1)[1]
                        current_columns = json.loads(columns_json)
                    except Exception as e:
                        if verbose:
                            print(f"   ⚠️  Error parsing columns on line {line_num}: {e}")
                elif line.startswith('-- ROW:'):
                    try:
                        row_json = line.split(':', 1)[1]
                        current_rows.append(json.loads(row_json))
                    except Exception as e:
                        if verbose:
                            print(f"   ⚠️  Error parsing row on line {line_num}: {e}")
                elif line.startswith('-- DATA_END:'):
                    if current_table and current_columns:
                        data_sections.append({
                            'table': current_table,
                            'columns': current_columns,
                            'rows': current_rows
                        })
                    current_table = None
                    current_columns = []
                    current_rows = []
                # Skip other comments
            else:
                # SQL statement
                if current_sql:
                    current_sql += " " + line
                else:
                    current_sql = line
                
                # If statement ends with semicolon, it's complete
                if line.endswith(';'):
                    schema_statements.append(current_sql[:-1])  # Remove semicolon
                    current_sql = ""
        
        # Don't forget the last statement if it doesn't end with semicolon
        if current_sql.strip():
            schema_statements.append(current_sql.strip())
        
        # Separate sequence updates from regular schema statements
        sequence_updates = []
        regular_schema = []
        
        for stmt in schema_statements:
            if 'setval(' in stmt.lower():
                sequence_updates.append(stmt)
            else:
                regular_schema.append(stmt)
        
        # Execute regular schema statements first
        if verbose:
            print(f"   📝 Executing {len(regular_schema)} schema statements")
        schema_errors = 0
        
        for i, sql in enumerate(regular_schema):
            if not sql.strip():
                continue
            
            try:
                cursor.execute(sql)
            except Exception as e:
                schema_errors += 1
                if verbose:
                    error_msg = str(e).replace('\n', ' ')[:100]
                    print(f"   ⚠️  Schema error #{schema_errors}: {error_msg}...")
                    
                    # For debugging, show the problematic SQL
                    if schema_errors <= 3:  # Only show first few errors
                        print(f"        SQL: {sql[:200]}...")
        
        if schema_errors > 0 and verbose:
            print(f"   ⚠️  Had {schema_errors} schema errors (this may be normal)")
        
        # Execute data inserts with FAST bulk loading
        total_rows_restored = sum(len(section['rows']) for section in data_sections)
        if verbose:
            print(f"   📊 Restoring data for {len(data_sections)} tables (FAST MODE)")
        else:
            print(f"   📊 Restoring {total_rows_restored:,} rows to {len(data_sections)} tables")
        
        for section in data_sections:
            table = section['table']
            columns = section['columns']
            rows = section['rows']
            
            if not rows:
                if verbose:
                    print(f"   📋 No data for table {table}")
                continue
            
            if verbose:
                print(f"   💾 FAST restoring {len(rows)} rows to {table}")
            
            try:
                # Method 1: Try COPY FROM (fastest possible)
                if self._try_copy_from_restore(cursor, table, columns, rows, verbose):
                    if verbose:
                        print(f"     ⚡ Used COPY FROM - maximum speed!")
                    continue
                    
                # Method 2: Fallback to batch inserts (still much faster than individual)
                self._batch_insert_restore(cursor, table, columns, rows, verbose)
                    
            except Exception as e:
                if verbose:
                    print(f"   ❌ Failed to restore {table}: {e}")
        
        # Finally, execute sequence updates
        if sequence_updates:
            if verbose:
                print(f"   🔄 Updating {len(sequence_updates)} sequences")
            seq_errors = 0
            
            for seq_sql in sequence_updates:
                try:
                    cursor.execute(seq_sql)
                except Exception as e:
                    seq_errors += 1
                    if verbose and seq_errors <= 3:
                        error_msg = str(e).replace('\n', ' ')[:50]
                        print(f"   ⚠️  Sequence update error: {error_msg}...")
            
            if seq_errors == 0:
                if verbose:
                    print(f"   ✅ All sequences updated successfully")
            elif verbose:
                print(f"   ⚠️  {seq_errors} sequence update errors")

    def _try_copy_from_restore(self, cursor, table: str, columns: List[str], rows: List[dict], verbose: bool = False) -> bool:
        """Try to use COPY FROM for maximum speed (10-100x faster than INSERT)"""
        try:
            import io
            
            # Create a CSV-like string buffer
            csv_data = io.StringIO()
            
            for row_data in rows:
                csv_row = []
                for col in columns:
                    value = row_data.get(col)
                    
                    if value is None:
                        csv_row.append('\\N')  # NULL in COPY format
                    elif isinstance(value, dict):
                        # Escape JSON for COPY
                        json_str = json.dumps(value).replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r')
                        csv_row.append(json_str)
                    elif isinstance(value, list):
                        # Escape JSON for COPY
                        json_str = json.dumps(value).replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r')
                        csv_row.append(json_str)
                    elif isinstance(value, str):
                        # Escape string for COPY
                        if col.endswith('_at') or col.endswith('_time') or 'date' in col.lower():
                            # Try to parse datetime
                            try:
                                if 'T' in value:
                                    parsed_dt = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
                                    csv_row.append(str(parsed_dt))
                                else:
                                    csv_row.append(value)
                            except:
                                csv_row.append(value)
                        else:
                            escaped = value.replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r')
                            csv_row.append(escaped)
                    elif isinstance(value, bool):
                        csv_row.append('t' if value else 'f')
                    else:
                        csv_row.append(str(value))
                
                csv_data.write('\t'.join(csv_row) + '\n')
            
            # Reset buffer position
            csv_data.seek(0)
            
            # Use COPY FROM for super fast bulk insert
            columns_str = ', '.join(f'"{col}"' for col in columns)
            copy_sql = f'COPY public."{table}" ({columns_str}) FROM STDIN WITH (FORMAT text, DELIMITER E\'\\t\', NULL \'\\N\')'
            
            cursor.copy_expert(copy_sql, csv_data)
            if verbose:
                print(f"     ⚡ COPY FROM: {len(rows)} rows in milliseconds!")
            return True
            
        except Exception as e:
            if verbose:
                print(f"     ⚠️  COPY FROM failed: {e}")
            return False

    def _batch_insert_restore(self, cursor, table: str, columns: List[str], rows: List[dict], verbose: bool = False):
        """Use batch inserts (much faster than individual INSERTs)"""
        # Build parameterized INSERT statement
        columns_str = ', '.join(f'"{col}"' for col in columns)
        placeholders = ', '.join(['%s'] * len(columns))
        insert_sql = f'INSERT INTO "{table}" ({columns_str}) VALUES ({placeholders})'
        
        # Prepare all data for batch insert
        batch_data = []
        error_count = 0
        
        for row_idx, row_data in enumerate(rows):
            try:
                values = []
                for col in columns:
                    value = row_data.get(col)
                    
                    if value is None:
                        values.append(None)
                    elif isinstance(value, dict):
                        # Convert dict to JSON string for PostgreSQL
                        values.append(json.dumps(value))
                    elif isinstance(value, list):
                        # Convert list to JSON string for PostgreSQL
                        values.append(json.dumps(value))
                    elif isinstance(value, str) and (col.endswith('_at') or col.endswith('_time') or 'date' in col.lower()):
                        # Try to parse datetime
                        try:
                            if 'T' in value:
                                parsed_dt = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
                                values.append(parsed_dt)
                            else:
                                # Might be just a date
                                parsed_date = datetime.datetime.strptime(value, '%Y-%m-%d').date()
                                values.append(parsed_date)
                        except:
                            values.append(value)
                    else:
                        values.append(value)
                
                batch_data.append(values)
                
            except Exception as e:
                error_count += 1
                if verbose and error_count <= 3:  # Only show first few errors
                    error_msg = str(e).replace('\n', ' ')[:100]
                    print(f"     ⚠️  Row {row_idx} prep error: {error_msg}...")
        
        # Insert in batches (much faster than individual inserts)
        batch_size = 1000  # Adjust based on memory
        total_inserted = 0
        
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i + batch_size]
            try:
                # Use execute_many for batch insert (10-50x faster than individual execute calls)
                cursor.executemany(insert_sql, batch)
                total_inserted += len(batch)
                
                # Show progress for large tables
                if verbose and len(batch_data) > 5000 and i % (batch_size * 10) == 0:
                    progress = (i + len(batch)) / len(batch_data) * 100
                    print(f"     📈 Progress: {progress:.1f}% ({total_inserted}/{len(batch_data)} rows)")
                    
            except Exception as e:
                if verbose:
                    error_msg = str(e).replace('\n', ' ')[:100]
                    print(f"     ⚠️  Batch insert error: {error_msg}...")
                
                # Fallback: try individual inserts for this batch
                for row_values in batch:
                    try:
                        cursor.execute(insert_sql, row_values)
                        total_inserted += 1
                    except:
                        pass  # Skip problematic rows
        
        if verbose:
            print(f"     ✅ Batch insert: {total_inserted} rows inserted, {len(rows) - total_inserted} errors")

    def _create_backup_metadata(self, backup_file: str, schema_only: bool, compressed: bool):
        """Create metadata file alongside backup"""
        metadata = {
            'created_at': datetime.datetime.now().isoformat(),
            'database_host': self.parsed_url.hostname,
            'database_name': self.parsed_url.path.lstrip('/'),
            'schema_only': schema_only,
            'compressed': compressed,
            'file_size': os.path.getsize(backup_file),
            'backup_method': 'python-psycopg2-simplified',
        }
        
        metadata_file = backup_file + '.meta.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


def create_timestamped_filename(prefix: str = "backup", extension: str = ".sql") -> str:
    """Create a timestamped filename"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{extension}"


def list_backups(backup_dir: str = "./backups"):
    """List available backups with metadata"""
    backup_path = Path(backup_dir)
    if not backup_path.exists():
        print(f"📁 Backup directory doesn't exist: {backup_dir}")
        return
    
    backups = []
    for file in backup_path.glob("*.sql*"):
        if file.suffix in ['.sql', '.gz'] and not file.name.endswith('.meta.json'):
            metadata_file = file.with_suffix(file.suffix + '.meta.json')
            metadata = {}
            
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                except:
                    pass
            
            backups.append({
                'file': file,
                'metadata': metadata,
                'size': file.stat().st_size,
                'modified': datetime.datetime.fromtimestamp(file.stat().st_mtime)
            })
    
    if not backups:
        print(f"📁 No backups found in {backup_dir}")
        return
    
    print(f"📁 Backups in {backup_dir}:")
    print()
    for backup in sorted(backups, key=lambda x: x['modified'], reverse=True):
        file = backup['file']
        metadata = backup['metadata']
        
        print(f"📄 {file.name}")
        print(f"   📅 Modified: {backup['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   📏 Size: {backup['size']:,} bytes")
        
        if metadata:
            print(f"   🗄️  Database: {metadata.get('database_host', 'unknown')}")
            print(f"   📋 Type: {'Schema only' if metadata.get('schema_only') else 'Full backup'}")
            print(f"   🗜️  Compressed: {metadata.get('compressed', False)}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Simplified database backup and restore utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a backup (full path)
  python label_pizza/backup_restore.py backup --database-url-name DBURL --output ./backups/mybackup.sql
  
  # Create a backup (filename + backup dir)
  python label_pizza/backup_restore.py backup --database-url-name DBURL --backup-dir ./backups --output mybackup.sql.gz --compress
  
  # Create compressed backup with timestamp
  python label_pizza/backup_restore.py backup --database-url-name DBURL --backup-dir ./backups --compress --auto-name
  
  # Restore from backup (full path)
  python label_pizza/backup_restore.py restore --database-url-name DBURL --input ./backups/mybackup.sql --force
  
  # Restore from backup (filename + backup dir)
  python label_pizza/backup_restore.py restore --database-url-name DBURL --backup-dir ./backups --input mybackup.sql.gz --force
  
  # Schema-only backup
  python label_pizza/backup_restore.py backup --database-url-name DBURL --backup-dir ./backups --output schema.sql --schema-only
  
  # List backups
  python label_pizza/backup_restore.py list --backup-dir ./backups

Dependencies:
  pip install psycopg2-binary python-dotenv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create a database backup')
    backup_parser.add_argument('--database-url-name', default='DBURL', help='Environment variable name for database URL')
    backup_parser.add_argument('--output', help='Output file path')
    backup_parser.add_argument('--auto-name', action='store_true', help='Auto-generate timestamped filename')
    backup_parser.add_argument('--compress', action='store_true', help='Compress backup with gzip')
    backup_parser.add_argument('--schema-only', action='store_true', help='Backup schema only (no data)')
    backup_parser.add_argument('--backup-dir', default='./backups', help='Backup directory for auto-named files')
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore from a database backup')
    restore_parser.add_argument('--database-url-name', default='DBURL', help='Environment variable name for database URL')
    restore_parser.add_argument('--input', required=True, help='Input backup file path (filename or full path)')
    restore_parser.add_argument('--backup-dir', default='./backups', help='Backup directory (if --input is just filename)')
    restore_parser.add_argument('--force', action='store_true', help='Skip confirmation prompts')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available backups')
    list_parser.add_argument('--backup-dir', default='./backups', help='Backup directory to scan')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'list':
        list_backups(args.backup_dir)
        return
    
    # Get database URL from environment
    db_url = os.getenv(args.database_url_name)
    if not db_url:
        print(f"❌ Environment variable {args.database_url_name} not found")
        print("Make sure you have set your database URL in the .env file")
        sys.exit(1)
    
    # Initialize backup/restore handler
    handler = DatabaseBackupRestore(db_url)
    
    if args.command == 'backup':
        # Handle output filename
        if args.auto_name:
            backup_dir = Path(args.backup_dir)
            backup_dir.mkdir(exist_ok=True)
            
            prefix = "schema" if args.schema_only else "backup"
            extension = ".sql.gz" if args.compress else ".sql"
            filename = create_timestamped_filename(prefix, extension)
            output_file = str(backup_dir / filename)
        elif args.output:
            output_file = args.output
            
            # If output is just a filename (no path separator), combine with backup_dir
            if not os.path.sep in output_file and not os.path.isabs(output_file):
                backup_dir = Path(args.backup_dir)
                backup_dir.mkdir(exist_ok=True)
                output_file = str(backup_dir / output_file)
        else:
            print("❌ Either --output or --auto-name must be specified")
            sys.exit(1)
        
        success = handler.create_backup(
            output_file=output_file,
            compress=args.compress,
            schema_only=args.schema_only
        )
        sys.exit(0 if success else 1)
    
    elif args.command == 'restore':
        # Handle input file path
        input_file = args.input
        
        # If input is just a filename (no path separator), combine with backup_dir
        if not os.path.sep in input_file and not os.path.isabs(input_file):
            input_file = os.path.join(args.backup_dir, input_file)
        
        success = handler.restore_backup(
            input_file=input_file,
            force=args.force
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()