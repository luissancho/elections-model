import sqlalchemy as db

from typing_extensions import Self

from ..data import SQLAdapter


class BigQuery(SQLAdapter):

    alias = ['bq', 'bigquery']

    driver = 'bigquery'
    quote = ''

    def _build_client(self) -> Self:
        self._client = db.create_engine(
            f'{self.driver}://{self.database}',
            credentials_info=self.params,
            pool_pre_ping=True
        )
