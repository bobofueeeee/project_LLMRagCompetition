import sqlite3
import threading

def execute_query_with_timeout(query, timeout=10):
    # 定义一个标志位，用于指示是否超时
    timeout_occurred = False

    # 定义一个内部函数来设置超时标志位
    def handle_timeout():
        nonlocal timeout_occurred
        timeout_occurred = True

        # 设置定时器

    timer = threading.Timer(timeout, handle_timeout)
    timer.start()

    try:
        # 执行查询
        conn = sqlite3.connect('example.db')
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()

        # 清除定时器（如果查询在超时前完成）
        timer.cancel()

        # 如果查询成功且未超时，则处理结果
        if not timeout_occurred:
            # 处理rows...
            print("Query executed successfully")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")

    finally:
        # 无论是否超时，都关闭连接
        if conn:
            conn.close()

            # 如果超时发生，则可能需要处理超时逻辑
    if timeout_occurred:
        print("Query execution timed out")

    # 使用示例

execute_query_with_timeout("SELECT * FROM some_table", timeout=5)