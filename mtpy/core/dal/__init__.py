from importlib import import_module
import pkgutil
import sys


__all__ = [
    name for _, name, _ in pkgutil.iter_modules(sys.modules[__name__].__path__)
]


adapters = {
    name: name for name in __all__
} | {
    'bq': 'BigQuery',
    'bigquery': 'BigQuery',
    'dynamo': 'DynamoDB',
    'dynamodb': 'DynamoDB',
    'mongo': 'Mongo',
    'mongodb': 'Mongo',
    'mysql': 'MySQL',
    'pgsql': 'PostgreSQL',
    'postgresql': 'PostgreSQL',
    'postgre': 'PostgreSQL',
    'redshift': 'Redshift',
    'salesforce': 'Salesforce',
    'sf': 'Salesforce'
}


def __getattr__(name):
    package = sys.modules[__name__]
    module = import_module('.' + name, __name__)
    attribute = getattr(module, name)

    setattr(package, name, attribute)

    return getattr(package, name)
