from typing import Optional
from typing_extensions import Self

from ..data import SQLAdapter


class MySQL(SQLAdapter):

    alias = ['mysql']

    driver = 'mysql+pymysql'
    default_port = 3306
    quote = '`'
    sessions = True
    data_types = {
        'varchar': 'VARCHAR',
        'text': 'TEXT',
        'date': 'DATE',
        'datetime': 'DATETIME',
        'numeric': 'DECIMAL',
        'tinyint': 'TINYINT',
        'smallint': 'SMALLINT',
        'mediumint': 'MEDIUMINT',
        'integer': 'INTEGER',
        'bigint': 'BIGINT',
        'boolean': 'TINYINT'
    }
    bool_as_int = True
    autokey_suffix = 'AUTO_INCREMENT'
    date_groupings = {
        'year': "CONCAT(YEAR({col}), '-01-01')",
        'month': "STR_TO_DATE(CONCAT(EXTRACT(YEAR_MONTH FROM {col}), '01'), '%Y%m%d')",
        'week': "STR_TO_DATE(CONCAT(YEARWEEK({col}), ' Monday'), '%X%V %W')",
        'day': "DATE({col})"
    }
    on_conflict_suffix = 'ON DUPLICATE KEY UPDATE'
    on_conflict_update = '{col} = VALUES({col})'
    create_suffix = 'ENGINE=InnoDB DEFAULT CHARSET=utf8'
    clone_suffix = None

    def build_delete_query(
        self,
        table: str,
        other: Optional[str] = None,
        key: Optional[str | list[str]] = None,
        filters: Optional[list[str] | dict] = None
    ) -> str:
        query = f'DELETE'

        if other is not None:
            query += f' {table} FROM {table} INNER JOIN {other} ON '

            if isinstance(key, (list, tuple)):
                query += ' AND '.join([
                    f'{table}."{i}" = {other}."{i}"'
                    for i in key
                ])
            else:
                query += f'{table}."{key}" = {other}."{key}"'
        else:
            query += f' FROM {table}'

        if isinstance(filters, (list, tuple)) and len(filters) > 0:
            filters = '\n    AND '.join(filters)
            query += f'\nWHERE {filters}'

        return query

    def get_version(
        self
    ) -> str:
        return self.get_var(query='SELECT VERSION()')
    
    def get_tables(
        self,
        schema: Optional[str] = None
    ) -> list[str]:
        if schema is None:
            schema = self.database

        query = f'SHOW TABLES IN {schema}'
        
        tables = self.get_results(query=query)[f'Tables_in_{schema or self.database}'].tolist()

        return tables
    
    def get_columns(
        self,
        table: str
    ) -> list[str]:
        query = f'SHOW COLUMNS FROM {table}'
        
        columns = self.get_results(query=query)['Field'].tolist()

        return columns
    
    def get_meta(
        self,
        table: str
    ) -> dict[str, dict]:
        meta = {
            col: {} for col in self.get_columns(table)
        }

        return meta
