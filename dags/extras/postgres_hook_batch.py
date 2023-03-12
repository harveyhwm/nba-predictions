from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.hooks.postgres_hook import PostgresHook
from airflow.contrib.sensors.file_sensor import FileSensor
from airflow.hooks.dbapi import DbApiHook
from airflow.models.connection import Connection
from datetime import datetime
import numpy as np
import os,csv,itertools

from psycopg2.extensions import connection
from typing import Iterable, List, Optional, Tuple, Union
from psycopg2.extras import DictCursor, NamedTupleCursor, RealDictCursor
CursorType = Union[DictCursor, RealDictCursor, NamedTupleCursor]

class PostgresHookBatching(DbApiHook):

    conn_name_attr = 'postgres_conn_id'
    default_conn_name = 'postgres_default'
    conn_type = 'postgres'
    hook_name = 'Postgres'
    supports_autocommit = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.connection: Optional[Connection] = kwargs.pop("connection", None)
        self.conn: connection = None
        self.schema: Optional[str] = kwargs.pop("schema", None)

    def _get_cursor(self, raw_cursor: str) -> CursorType:
        _cursor = raw_cursor.lower()
        if _cursor == 'dictcursor':
            return psycopg2.extras.DictCursor
        if _cursor == 'realdictcursor':
            return psycopg2.extras.RealDictCursor
        if _cursor == 'namedtuplecursor':
            return psycopg2.extras.NamedTupleCursor
        raise ValueError(f'Invalid cursor passed {_cursor}')

    def get_conn(self) -> connection:
        """Establishes a connection to a postgres database."""
        conn_id = getattr(self, self.conn_name_attr)
        conn = deepcopy(self.connection or self.get_connection(conn_id))

        # check for authentication via AWS IAM
        if conn.extra_dejson.get('iam', False):
            conn.login, conn.password, conn.port = self.get_iam_token(conn)

        conn_args = dict(
            host=conn.host,
            user=conn.login,
            password=conn.password,
            dbname=self.schema or conn.schema,
            port=conn.port,
        )
        raw_cursor = conn.extra_dejson.get('cursor', False)
        if raw_cursor:
            conn_args['cursor_factory'] = self._get_cursor(raw_cursor)

        for arg_name, arg_val in conn.extra_dejson.items():
            if arg_name not in [
                'iam',
                'redshift',
                'cursor',
                'cluster-identifier',
                'aws_conn_id',
            ]:
                conn_args[arg_name] = arg_val

        self.conn = psycopg2.connect(**conn_args)
        return self.conn

    def copy_batch(self, sql: str, filename: str, batch: int) -> None:
        if not os.path.isfile(filename):
            print('Sexy file not found!')
            pass
        else:
            self.log.info("Running copy batch: %s, filename: %s", sql, filename)
            f = csv.reader(open(filename),delimiter=';')
            with closing(self.get_conn()) as conn:
                rows = np.arange(batch)
                with closing(conn.cursor()) as cur:
                    i = 0
                    while len(rows) > 0:
                        print('I is '+str(i))
                        # itertools will automatically slice off those you process, so no need to increment inside islice
                        rows = [r for r in itertools.islice(f, max(0,1-i),batch)]
                        if (len(rows) == 0) or (i > 5):
                            return
                        psycopg2.extras.execute_values(cur,sql,rows,template=None,page_size=batch)
                        conn.commit()
                        i = i + 1