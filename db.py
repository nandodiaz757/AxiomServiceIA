import asyncpg
import asyncio

DATABASE_URL = "postgresql://axiom:axiom123@localhost:5432/axiomdb"

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
