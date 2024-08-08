from pymilvus import connections, db

# conn = connections.connect(host="139.9.92.102", port=19530)
# # database = db.create_database("fubo_database")
# print(db.list_database())

from pymilvus import connections, Collection, FieldSchema, IndexType, DataType,utility
from pymilvus import MilvusClient, DataType,CollectionSchema
# 1. Set up a Milvus client
client = MilvusClient(
    uri="http://139.9.92.102:19530/fubo_database"
)

collection_name = "fubo_collection"
if not client.has_collection(collection_name):
    client.create_collection(
        collection_name=collection_name,
        dimension=5
    )

collection = client.get_load_state(collection_name)

schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)

# 3.2. Add fields to schema
schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=5)


print(client.list_collections())

# 连接到 Milvus
