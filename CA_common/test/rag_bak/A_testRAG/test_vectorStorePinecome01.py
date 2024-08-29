from pinecone import Pinecone

pc = Pinecone(api_key="bad20e65-b265-4b00-b0d0-8bbb1bd8b3ba")
index = pc.Index("quickstart")

# index.upsert(
#     vectors=[
#         {
#             "id": "vec1",
#             "values": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#             "metadata": {"genre": "drama","page_content": "我叫付博"}
#         }, {
#             "id": "vec2",
#             "values": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
#             "metadata": {"genre": "action","page_content": "我叫张三"}
#         }, {
#             "id": "vec3",
#             "values": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
#             "metadata": {"genre": "drama","page_content": "我叫李四"}
#         }, {
#             "id": "vec4",
#             "values": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
#             "metadata": {"genre": "action","page_content": "我叫王五"}
#         }
#     ],
#     namespace= "ns1"
# )

query= index.query(
    namespace="ns1",
    vector=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    top_k=2,
    include_values=True,
    include_metadata=True
    # filter={"genre": {"$eq": "action"}}
)
print(query)