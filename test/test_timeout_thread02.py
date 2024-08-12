import asyncio
import aiosqlite


async def execute_query_with_timeout_async(query, timeout=10):
    try:
        # 连接到数据库
        conn = await aiosqlite.connect('example.db')
        cursor = await conn.cursor()

        # 使用wait_for来设置超时
        await asyncio.wait_for(cursor.execute(query), timeout)

        # 获取结果
        rows = await cursor.fetchall()

        # 处理结果...
        print(rows)

    except asyncio.TimeoutError:
        print("Query execution timed out")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # 关闭连接
        if conn:
            await conn.close()

        # 运行异步函数


asyncio.run(execute_query_with_timeout_async("SELECT * FROM some_table", timeout=5))