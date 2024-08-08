from pymilvus import connections, FieldSchema, CollectionSchema, DataType,Collection,MilvusClient
import random
import numpy as np

MILVUS_HOST = '139.9.92.102'  # Milvus 服务器的地址
MILVUS_PORT = '19530'  # Milvus 服务器的端口
VECTOR_DIMENSION = 128  # 向量维度


def connect_milvus():

    client = MilvusClient("milvus_demo.db")
    if client.has_collection(collection_name="demo_collection"):
        client.drop_collection(collection_name="demo_collection")

    client.create_collection(
    host=MILVUS_HOST,
    port=MILVUS_PORT,
    collection_name="demo_collection",
    dimension=768,  # The vectors we will use in this demo has 768 dimensions
    )


def create_collection(collection_name):
    # 设置字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # 主键字段
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128,description="Collection for storing embeddings")  # 向量字段
    ]

    # 设置表格
    schema = CollectionSchema(fields=fields, description="测试向量集合")
    # schema.auto_id(False)  # 禁止自动生成ID，我们手动指定

    collection = Collection(name="test_vector_collection", schema=schema)

    return collection


def insert_vectors(collection, vectors):

    c = collection

    entities = [{"embedding": vector.tolist()} for vector in vectors]


    ids = [str(i) for i in range(len(vectors))]  # 手动指定插入的向量ID

    c.insert(collection, entities, ids)

def search_vectors():
    pass

def query_vectors():
    pass

def delete_vectors():
    pass




# 示例用法
if __name__ == "__main__":
    connect_milvus()
    collection_name = "example_collection"
    collection = create_collection(collection_name)

    # 生成一些示例向量
    num_vectors = 10
    vectors = [np.random.rand(VECTOR_DIMENSION).astype(np.float32) for _ in range(num_vectors)]
    # 插入向量
    insert_vectors(collection, vectors)

    print(f"Inserted {num_vectors} vectors into collection '{collection_name}'.")
