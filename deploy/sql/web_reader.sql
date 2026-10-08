-- Rol de solo lectura para el contenedor web. Lo ejecuta Luis con el usuario maestro de la RDS.
-- La contraseña no puede contener comillas dobles ni barras invertidas (config.json se rellena por sustitución de texto).
CREATE ROLE web_reader LOGIN PASSWORD '<cambiar>';
GRANT CONNECT ON DATABASE manythings TO web_reader;
GRANT USAGE ON SCHEMA elections TO web_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA elections TO web_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA elections GRANT SELECT ON TABLES TO web_reader;
ALTER ROLE web_reader SET statement_timeout = '5s';
