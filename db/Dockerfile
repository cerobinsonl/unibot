FROM postgres:14

# Copy initialization scripts
COPY init.sql /docker-entrypoint-initdb.d/init.sql
COPY seeds.sql /docker-entrypoint-initdb.d/seeds.sql

# PostgreSQL configuration will be set through environment variables
# when deploying to Cloud Run

# Default port
EXPOSE 5432

# Healthcheck
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
  CMD pg_isready -U $POSTGRES_USER -d $POSTGRES_DB || exit 1
