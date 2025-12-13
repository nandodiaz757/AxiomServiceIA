import os
from dotenv import load_dotenv
import psycopg2
import asyncpg
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager

# Cargar variables del .env
load_dotenv()

DB_NAME = os.getenv("DB_NAME", "accessibility")
DB_USER = os.getenv("DB_USER", "axiom")
DB_PASS = os.getenv("DB_PASS", "axiom123")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# Conexión pool
pool = SimpleConnectionPool(
    1, 20,  # min/max conexiones
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS,
    host=DB_HOST,
    port=DB_PORT
)

@contextmanager
def get_conn_cm():
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)

def init_db():
    # conn = get_conn_cm()
    # c = conn.cursor()

    try:
        with get_conn_cm() as conn:
            with conn.cursor() as c:
                c.execute("CREATE SCHEMA IF NOT EXISTS public;")

                c.execute("""
                    CREATE TABLE IF NOT EXISTS accessibility_data (
                        id SERIAL PRIMARY KEY,
                        tester_id TEXT,
                        build_id TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        event_type TEXT,
                        event_type_name TEXT,
                        package_name TEXT,
                        class_name TEXT,
                        text TEXT,
                        content_description TEXT,
                        screens_id TEXT,
                        screens_id_short TEXT,
                        screen_names TEXT,
                        header_text TEXT,
                        actual_device TEXT,
                        global_signature TEXT,
                        partial_signature TEXT,
                        scroll_type TEXT,
                        signature TEXT,
                        version TEXT,
                        collect_node_tree TEXT,
                        additional_info TEXT,
                        tree_data TEXT,
                        is_baseline BOOLEAN DEFAULT FALSE,
                        enriched_vector TEXT,
                        cluster_id INTEGER,
                        is_stable BOOLEAN DEFAULT FALSE,
                        anomaly_score DOUBLE PRECISION,
                        session_key TEXT,
                        embedding_vector  TEXT, 
                        trained_incremental BOOLEAN DEFAULT FALSE,
                        trained_general BOOLEAN DEFAULT FALSE,  
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                c.execute("""
                    CREATE TABLE IF NOT EXISTS screen_diffs (
                        id SERIAL PRIMARY KEY,
                        tester_id TEXT,
                        build_id TEXT,
                        screen_name TEXT,
                        header_text TEXT,
                        removed TEXT,
                        added TEXT,
                        modified TEXT,
                        cluster_info TEXT,
                        anomaly_score DOUBLE PRECISION DEFAULT 0,
                        diff_hash TEXT UNIQUE NOT NULL,
                        diff_priority TEXT DEFAULT 'high',
                        text_diff TEXT,
                        text_overlap DOUBLE PRECISION DEFAULT 0,
                        overlap_ratio DOUBLE PRECISION DEFAULT 0,
                        ui_structure_similarity DOUBLE PRECISION DEFAULT 0,
                        cluster_id INTEGER DEFAULT -1,
                        screen_status TEXT DEFAULT 'unknown',
                        similarity_to_approved DOUBLE PRECISION DEFAULT 0,
                        approved_before BOOLEAN DEFAULT FALSE,
                        approval_status TEXT,
                        rejection_reason TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                c.execute("""
                    CREATE TABLE IF NOT EXISTS metrics_summary (
                        id SERIAL PRIMARY KEY,
                        metric_group TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value TEXT
                    );
                """)

                c.execute("""
                    CREATE TABLE IF NOT EXISTS diff_items (
                        id SERIAL PRIMARY KEY,
                        diff_id INTEGER NOT NULL REFERENCES screen_diffs(id),
                        action TEXT NOT NULL,
                        node_class TEXT,
                        node_key TEXT,
                        node_text TEXT,
                        changes_json TEXT,
                        raw_json TEXT,
                        label TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                c.execute("""
                CREATE TABLE IF NOT EXISTS ignored_changes_log (
                        id SERIAL PRIMARY KEY,
                        tester_id TEXT,
                        build_id INTEGER,
                        header_text TEXT,
                        signature TEXT,
                        class_name TEXT,
                        field TEXT,
                        old_value TEXT,
                        new_value TEXT,
                        reason TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                c.execute("""
                    CREATE TABLE IF NOT EXISTS boolean_history (
                        screen_id TEXT,
                        node_key TEXT,
                        property TEXT,
                        last_value BOOLEAN,
                        PRIMARY KEY(screen_id, node_key, property)
                    );
                """)

                c.execute("""
                    CREATE TABLE IF NOT EXISTS baseline_metadata (
                        app_name TEXT,
                        tester_id TEXT,
                        build_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (app_name, tester_id, build_id)
                    );
                """)

                c.execute("""
                    CREATE TABLE IF NOT EXISTS active_plans (
                        id SERIAL PRIMARY KEY,
                        plan_name TEXT UNIQUE NOT NULL,
                        description TEXT NOT NULL,
                        active BOOLEAN DEFAULT TRUE,
                        currency TEXT CHECK(currency IN ('USD', 'COP', 'EUR')) NOT NULL DEFAULT 'USD',
                        rate_usd DOUBLE PRECISION DEFAULT 1.0,
                        rate_cop DOUBLE PRECISION DEFAULT 4100.0,
                        rate_eur DOUBLE PRECISION DEFAULT 0.94,
                        rate DOUBLE PRECISION DEFAULT 1.0,
                        price DOUBLE PRECISION NOT NULL,
                        no_associate DOUBLE PRECISION DEFAULT 0.0,
                        max_tester INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # insertar planes por defecto
                c.execute("""
                    INSERT INTO active_plans (plan_name, description, currency, rate, price, max_tester)
                    VALUES ('BASIC', 'Basic Plan', 'USD', 1.0, 1, 1)
                    ON CONFLICT (plan_name) DO NOTHING;
                """)

                c.execute("""
                    INSERT INTO active_plans (plan_name, description, currency, rate, price, max_tester)
                    VALUES ('STANDARD', 'Standard Plan', 'USD', 1.0, 25, 5)
                    ON CONFLICT (plan_name) DO NOTHING;
                """)

                c.execute("""
                    CREATE TABLE IF NOT EXISTS login_codes (
                        codigo TEXT PRIMARY KEY,
                        usuario_id TEXT NOT NULL,
                        plan_id INTEGER REFERENCES active_plans(id),
                        generado_en BIGINT NOT NULL,
                        expira_en BIGINT NOT NULL,
                        usos_permitidos INTEGER NOT NULL,
                        usos_actuales INTEGER DEFAULT 0,
                        pago_confirmado BOOLEAN DEFAULT FALSE,
                        activo BOOLEAN DEFAULT TRUE,
                        is_paid BOOLEAN DEFAULT FALSE
                    );
                """)

                c.execute("""
                    CREATE TABLE IF NOT EXISTS usuarios (
                        id SERIAL PRIMARY KEY,
                        nombre TEXT,  
                        email TEXT,  
                        hash_password TEXT,  
                        udid TEXT,   
                        rol TEXT CHECK(rol IN ('owner', 'tester')) DEFAULT 'tester',
                        plan_id INTEGER REFERENCES active_plans(id),
                        phone TEXT,    
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        deleted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  
                    );
                """)

                c.execute("""
                    CREATE TABLE IF NOT EXISTS pagos (
                        pago_id TEXT PRIMARY KEY,
                        membresia_id TEXT NOT NULL,
                        usuario_id TEXT NOT NULL,
                        proveedor TEXT NOT NULL,
                        proveedor_id TEXT,
                        monto DOUBLE PRECISION NOT NULL,
                        moneda TEXT DEFAULT 'USD',
                        estado TEXT DEFAULT 'PENDIENTE',
                        transaccion_id TEXT,
                        cantidad_codigos INTEGER NOT NULL,
                        fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        fecha_confirmacion BIGINT
                    );
                """)

                c.execute("""
                    CREATE TABLE IF NOT EXISTS membresias (
                        membresia_id TEXT PRIMARY KEY,
                        usuario_id TEXT NOT NULL,
                        tipo_plan TEXT NOT NULL,
                        cantidad_codigos INTEGER NOT NULL,
                        fecha_inicio BIGINT NOT NULL,
                        fecha_fin BIGINT NOT NULL
                    );
                """)

                c.execute("""
                    CREATE TABLE IF NOT EXISTS pagos_log (
                        log_id SERIAL PRIMARY KEY,
                        pago_id TEXT NOT NULL REFERENCES pagos(pago_id),
                        evento TEXT NOT NULL,
                        descripcion TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                c.execute("""
                    CREATE TABLE IF NOT EXISTS transacciones (
                        transaccion_id TEXT PRIMARY KEY,
                        proveedor TEXT NOT NULL,
                        proveedor_id TEXT,
                        monto DOUBLE PRECISION NOT NULL,
                        moneda TEXT DEFAULT 'USD',
                        estado TEXT DEFAULT 'PENDIENTE',
                        fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                c.execute("""
                    CREATE TABLE IF NOT EXISTS diff_trace (
                        id SERIAL PRIMARY KEY,
                        tester_id TEXT,
                        build_id TEXT,
                        screen_name TEXT,
                        message TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                c.execute("""
                    CREATE TABLE IF NOT EXISTS diff_approvals (
                        id SERIAL PRIMARY KEY,
                        diff_id INTEGER,
                        approved BOOLEAN DEFAULT FALSE,
                        approved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                c.execute("""
                    CREATE TABLE IF NOT EXISTS diff_rejections (
                        id SERIAL PRIMARY KEY,
                        diff_id INTEGER,
                        rejection_reason TEXT,
                        rejected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                c.execute("""
                    CREATE TABLE IF NOT EXISTS password_reset_codes (
                        id SERIAL PRIMARY KEY,
                        email TEXT NOT NULL,
                        code TEXT NOT NULL,
                        expires_at BIGINT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                c.execute("""
                    CREATE TABLE IF NOT EXISTS metrics_changes (
                        id SERIAL PRIMARY KEY,
                        tester_id TEXT,
                        build_id TEXT,
                        total_events INTEGER DEFAULT 0,
                        total_changes INTEGER DEFAULT 0,
                        total_added INTEGER DEFAULT 0,
                        total_removed INTEGER DEFAULT 0,
                        total_modified INTEGER DEFAULT 0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(tester_id, build_id)
                    );
                """)
                c.execute("""
                    CREATE INDEX IF NOT EXISTS idx_data_tester_build_screen
                    ON accessibility_data(tester_id, build_id, screen_names, created_at DESC)
                """)
                c.execute("""
                    CREATE INDEX IF NOT EXISTS idx_diffs_tester_build_screen
                    ON screen_diffs(tester_id, build_id, screen_name, created_at DESC)
                """)
            
            with conn.cursor() as c:

                # diff_trace constraint
                c.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint WHERE conname = 'unique_diff_trace'
                    ) THEN
                        ALTER TABLE diff_trace
                        ADD CONSTRAINT unique_diff_trace
                        UNIQUE (tester_id, build_id, screen_name, message);
                    END IF;
                END $$;
                """)

                # diff_trace index
                c.execute("""
                CREATE INDEX IF NOT EXISTS idx_diff_trace_tester_build_screen
                ON diff_trace(tester_id, build_id, screen_name);
                """)

                # diff_items constraint
                c.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint WHERE conname = 'unique_diffitems_clean'
                    ) THEN
                        ALTER TABLE diff_items
                        ADD CONSTRAINT unique_diffitems_clean
                        UNIQUE (diff_id, action, node_key);
                    END IF;
                END $$;
                """)
            conn.commit()
            # conn.close()
            print("✔ DB inicializada correctamente")

    except Exception as e:
        print("❌ Error inicializando DB:", e)

    # finally:
    #     release_conn(conn)    
# Inicializar tablas al arrancar    
init_db()


class PostgresPool:
    pool: asyncpg.pool.Pool | None = None

    async def connect(self):
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                dsn=DATABASE_URL,
                min_size=2,
                max_size=10,
                command_timeout=10
            )
        return self.pool

    async def execute(self, query: str, params: tuple = ()):
        pool = await self.connect()
        async with pool.acquire() as conn:
            return await conn.execute(query, *params)

    async def fetchone(self, query: str, params: tuple = ()):
        pool = await self.connect()
        async with pool.acquire() as conn:
            return await conn.fetchrow(query, *params)

    async def fetchall(self, query: str, params: tuple = ()):
        pool = await self.connect()
        async with pool.acquire() as conn:
            return await conn.fetch(query, *params)


db = PostgresPool()
