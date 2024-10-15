import json

from pymilvus import MilvusClient, DataType, connections, Collection, CollectionSchema
from H_common.untils.embedding_utils import XiaobuEmbeddingTool

class MilvusTool:
    def __init__(self, embedding_model_path):
        self.host = '139.9.92.102'
        # self.host = '127.0.0.1'
        self.port = 19530
        self.client = self.getMilvusClient()
        self.schema = self.getSchema()
        self.index_params = self.getPrepareIndexParams()
        self.collection = None
        self.embedding_model_xiaobu = XiaobuEmbeddingTool(model_path=embedding_model_path)
        connections.connect(alias="default", host=self.host, port=self.port)

    def close(self):
        connections.disconnect(alias="default")

    def getMilvusClient(self):
        """获取客户端连接"""
        client = MilvusClient(
            uri='http://' + self.host + ':' + str(self.port),
            token="root:Milvus",
            db_name="default"
        )
        return client

    def getPrepareIndexParams(self):
        """准备索引参数"""
        return self.client.prepare_index_params()

    def addIndex(self, index_params, field_name, index_type='IVF_SQ8', metric_type='L2', nlist=1024):
        """
        添加索引
        :param index_params:
        :param field_name:
        :param index_type:
            "FLAT"：平面索引，适用于高维向量的近邻搜索。
            "IVF_FLAT"：倒排列表（Inverted File）与平面索引相结合的索引结构。
            "IVF_SQ8"：倒排列表与乘积量化（Product Quantization）相结合的索引结构。
            "RNSG"：随机近邻（Random Projection Neighbor Graph）索引，适用于高维向量的近邻搜索。
        :param metric_type:
        :param nlist: 在向量量化过程中使用的聚类中心数量。‌具体来说，‌nlist=1024意味着在构建索引时，‌将使用1024个聚类中心来进行向量的量化，‌
                    这有助于提高搜索的效率和准确性。‌
        :return:
        """
        index_params.add_index(
            field_name=field_name,
            index_type=index_type,
            metric_type=metric_type,
            params={"nlist": nlist}
        )
        return index_params

    def addField(self, schema, field_name, datatype=DataType.INT64, is_primary=False, dim=None, description=''):
        """将字段添加到schema"""
        if is_primary:
            schema.add_field(field_name=field_name, datatype=datatype, is_primary=is_primary, description=description)
        else:
            if dim:
                schema.add_field(field_name=field_name, datatype=datatype, dim=dim, description=description)
            else:
                schema.add_field(field_name=field_name, datatype=datatype, description=description)

    def getSchema(self, description=''):
        """创建schema"""
        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
            description=description
        )
        return schema

    def create_collection_schema(self, collection_name, schema, index_params):
        """创建集合, 带index参数"""
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )

    def getListCollections(self):
        """列出所有现有集合"""
        result = self.client.list_collections()
        return result

    def describeCollection(self, collection_name):
        """列出某个集合的详细信息"""
        result = self.client.describe_collection(collection_name=collection_name)
        return result

    def list_indexes(self, collection_name):
        """此操作列出特定集合的所有索引"""
        res = self.client.list_indexes(collection_name=collection_name)
        return res

    def describe_index(self, collection_name, index_name):
        """此操作描述特定索引"""
        res = self.client.describe_index(
            collection_name=collection_name,
            index_name=index_name
        )
        return res

    def dropCollection(self, collection_name):
        """删除某个集合"""
        self.client.drop_collection(collection_name=collection_name)

    def insertData(self, collection_name, data):
        """
        将数据插入到特定集合中
        """
        res = self.client.insert(
            collection_name=collection_name,
            data=data
        )
        return res

    def insertListData(self, collection_name, listData: list):
        """
        批量将数据集合插入到特定集合中
        """
        res = self.client.insert(
            collection_name=collection_name,
            data=listData
        )
        print('数据插入成功：', res)
        return res

    def vector_search(self, collection_name, query, expr='', metric_type='IP', params={},
              anns_field='text', output_fields=['*'], limit=5):
        """向量检索"""
        embedding_query = self.embedding_model_xiaobu.text_embedding(query)
        collection = Collection(name=collection_name)
        search_params = {
            "metric_type": metric_type,
            "params": params
        }
        results = collection.search(data=[embedding_query], expr=expr, anns_field=anns_field, param=search_params,
                                    output_fields=output_fields, limit=limit)
        return results[0]

    def insert_pinxuan_data_to_milvus(self, collection_name, datas: list, data_type=1):
        """插入品宣数据到 Milvus中"""
        segment_datas = []
        for data in datas:
            embedding_text = self.embedding_model_xiaobu.text_embedding(data['text'])
            result = {
                "metadata": data,
                "data_type": data_type,  # 1表示原始品宣数据, 2表示模型总结后的品宣数据
                "text": embedding_text
            }
            segment_datas.append(result)
        res = self.insertListData(collection_name, segment_datas)
        return res

    def insert_dsmm_goods_name_data_to_milvus(self, collection_name, datas: list, data_type=1):
        """插入商品名称数据"""
        segment_datas = []
        for data in datas:
            embedding_text = self.embedding_model_xiaobu.text_embedding(data['goods_name'])
            result = {
                "metadata": data,
                "data_type": data_type,  # 1表示text是袋鼠妈妈商品名称数据、2表示text是商品文本段数据
                "text": embedding_text
            }
            segment_datas.append(result)
        res = self.insertListData(collection_name, segment_datas)
        return res

    def insert_dsmm_goods_data_to_milvus(self, collection_name, datas: list, data_type=2):
        """插入商品文本段数据"""
        segment_datas = []
        for data in datas:
            embedding_text = self.embedding_model_xiaobu.text_embedding(data['text_summary'])
            result = {
                "metadata": data,
                "data_type": data_type,
                "text": embedding_text
            }
            segment_datas.append(result)
        res = self.insertListData(collection_name, segment_datas)
        return res

    def euclidean_to_similarity(self, euclidean_distance):
        """将得到的欧式距离转换为相似度"""
        similarity = 1 / (1 + euclidean_distance)
        return similarity

    def hasCollection(self, collection_name):
        """检查某个集合是否存在"""
        result = self.client.has_collection(collection_name=collection_name)
        return result

    def loadCollection(self, collection_name):
        """将特定集合的数据加载到内存中"""
        self.client.load_collection(
            collection_name=collection_name
        )

    def releaseCollection(self, collection_name):
        """从内存中释放特定集合的数据"""
        self.client.release_collection(
            collection_name=collection_name
        )

def createCollectionIndex(milvusTool, schema, collection_name, data_type_desc):
    """带索引的创建集合的方式"""
    milvusTool.addField(schema=schema, field_name='id', datatype=DataType.INT64, is_primary=True, description='主键id')
    milvusTool.addField(schema=schema, field_name='metadata', datatype=DataType.JSON, description='元数据')
    milvusTool.addField(schema=schema, field_name='data_type', datatype=DataType.INT64, description=data_type_desc)
    milvusTool.addField(schema=schema, field_name='text', datatype=DataType.FLOAT_VECTOR, dim=1792, description='文本段，需要进行向量化的文本')
    index_params = milvusTool.getPrepareIndexParams()
    milvusTool.addIndex(index_params=index_params, field_name='id', index_type='STL_SORT')
    milvusTool.addIndex(index_params=index_params, field_name='data_type', index_type='STL_SORT')
    milvusTool.addIndex(index_params=index_params, field_name='text', index_type='IVF_SQ8', metric_type='IP', nlist=1024)
    milvusTool.create_collection_schema(collection_name=collection_name, schema=schema, index_params=index_params)

def insert_pinxuan_data(milvusTool, collection_name):
    """读取品宣数据插入Milvus"""
    file_path = r'D:\D盘桌面\软通\袋鼠妈妈项目\知识库\AI营销数据\袋鼠妈妈品牌介绍0618_原始拆分数据_20240813.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    insert_res = milvusTool.insert_pinxuan_data_to_milvus(collection_name=collection_name, datas=datas,
                                                          data_type=1)
    print('袋鼠妈妈品牌介绍0618_原始拆分数据_20240813   insert_res=', insert_res)
    file_path = r'D:\D盘桌面\软通\袋鼠妈妈项目\知识库\AI营销数据\袋鼠妈妈品牌介绍0618_模型总结数据_20240813.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    insert_res = milvusTool.insert_pinxuan_data_to_milvus(collection_name=collection_name, datas=datas,
                                                          data_type=2)
    print('袋鼠妈妈品牌介绍0618_模型总结数据_20240813   insert_res=', insert_res)

def insert_dsmm_goods_name_data(milvusTool, collection_name):
    """读取商品名称数据插入Milvus"""
    file_path = r'D:\D盘桌面\软通\袋鼠妈妈项目\知识库\AI营销数据\商品数据_json格式_20240821.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        datas = json.load(file)
    insert_res = milvusTool.insert_dsmm_goods_name_data_to_milvus(collection_name=collection_name, datas=datas,
                                                                  data_type=1)
    print('袋鼠妈妈商品名称数据   insert_res=', insert_res)

def insert_dsmm_goods_data(milvusTool, collection_name):
    """读取商品文本段数据插入Milvus"""
    file_path = r'D:\D盘桌面\软通\袋鼠妈妈项目\知识库\AI营销数据\商品数据_json格式_20240821.json'
    datas = []
    with open(file_path, 'r', encoding='utf-8') as file:
        datas = json.load(file)
    insert_res = milvusTool.insert_dsmm_goods_data_to_milvus(collection_name=collection_name, datas=datas,
                                                                  data_type=2)
    print('袋鼠妈妈商品文本段数据   insert_res=', insert_res)

def pinxuan_test(milvusTool):
    """品宣Demo演示"""
    collection_name = 'dsmm_pinxuan_segments'
    schema = milvusTool.getSchema('袋鼠妈妈品宣数据')
    milvusTool.dropCollection(collection_name)
    # 创建数据集
    createCollectionIndex(milvusTool, schema, collection_name, 'data_type_desc：1表示原始的品宣分段数据、2表示模型总结的品宣分段数据')
    # 插入数据
    insert_pinxuan_data(milvusTool, collection_name)
    # 向量检索
    result = milvusTool.vector_search(collection_name=collection_name,
                                      query='站在消费者视角，帮我生成袋鼠妈妈集团品宣文案，并进行儿童沐浴露单品推荐',
                                      expr='data_type == 2', output_fields=['id', 'metadata'])
    for s in result:
        print(s)

def goods_search(milvusTool, query, actuality):
    """商品名称检索"""
    collection_name = 'dsmm_goods_segments'
    # 向量检索
    result = milvusTool.vector_search(collection_name=collection_name, query=query,
                                      expr='data_type == 1', output_fields=['id', 'metadata'])
    predicted = result[0].entity.metadata['goods_name']
    if predicted == actuality:
        print('预测正确，用户query =', query, '，预测值=', predicted, '，真实值=', actuality, '，相似度=', result[0].distance)
        return 1
    else:
        print('!!!预测错误，用户query =', query, '，预测值 =', predicted, '，真实值 =', actuality, '，相似度 =', result[0].distance)
        return 0

def goods_test(milvusTool):
    """商品数据demo演示"""
    collection_name = 'dsmm_goods_segments'
    schema = milvusTool.getSchema('袋鼠妈妈商品数据')
    milvusTool.dropCollection(collection_name)
    # 创建数据集
    createCollectionIndex(milvusTool, schema, collection_name, 'data_type_desc：1表示商品名称数据、2表示商品文本段数据')
    # 插入商品名称数据
    insert_dsmm_goods_name_data(milvusTool, collection_name)
    # 插入商品文本段数据
    insert_dsmm_goods_data(milvusTool, collection_name)
    query_list = [
        '生成宝宝夏天睡觉太热的种草文案，仔细描述宝宝睡不着或者惊醒时妈妈的心得或感受',
        '生成给宝宝洗脸时的种草文案，仔细描述清洁的心得或感受',
        '生成带着宝宝去户外看海的种草文案，仔细描述看海的心得或感受',
        '站在消费者视角，帮我生成袋鼠妈妈集团品宣文案，并进行儿童沐浴露单品推荐',
        '站在主播视角，帮我生成袋鼠妈妈集团品宣文案并进行冰凉霜单品推荐',
        '站在博主的视角，生成袋鼠妈妈集团在驱蚊方向上的优势，并进行对应产品的推荐',
        '站在消费者的角度，生成夏季正确防晒或使用正确防晒产品的科普文案',
        '站在主播的角度，生成夏季防晒不正确步骤的科普文案',
        '站在消费者的角度，生成蚊虫叮咬科普文案',
        '站在消费者的角度，生成蚊虫叮咬症状的科普文案',
        '站在孕妈的角度，生成皮肤护理的科普文案',
        '站在孕妈的角度，生成婴儿成长的科普文案',
        '生成送外卖时如何护肤的科普文案',
        '生成送外卖时正确防晒的科普文案',
        '生成给宝宝洗澡时的种草文案，仔细描述洗澡的心得或感受'
    ]
    actuality = [
        '袋鼠比比儿童舒缓冰凉霜-10g',
        '袋鼠比比儿童云柔洁面泡泡-100ml',
        '袋鼠比比儿童轻阳倍护防晒乳SPF30PA+++-30g硅胶款',
        '袋鼠比比舒润守护沐浴露-80ml',
        '袋鼠比比儿童舒缓冰凉霜-10g',
        '袋鼠比比绿蓓健驱蚊草植物抑菌精油-30ml',
        '袋鼠妈妈水感清透倍呵防晒乳SPF30PA+++-50g',
        '袋鼠妈妈水感清透倍呵防晒乳SPF30PA+++-50g',
        '袋鼠比比绿蓓健驱蚊草植物抑菌精油-30ml',
        '袋鼠比比绿蓓健驱蚊草植物抑菌精油-30ml',
        '袋鼠妈妈小麦胚芽水润倍护保湿精华霜-50g',
        '袋鼠妈妈小麦胚芽水润倍护保湿精华霜-50g',
        '袋鼠妈妈卓薇焕亮清透防晒隔离霜-40g',
        '袋鼠比比儿童轻阳倍护防晒乳SPF30PA+++-30g硅胶款',
        '袋鼠比比舒润守护沐浴露-80ml',
    ]
    count = 0.0
    for i, query in enumerate(query_list):
        number = goods_search(milvusTool, query_list[i], actuality[i])
        count += number
    print('预测准确率 = ', count / len(query_list))

def select_collection_info():
    collection_name = 'dsmm_goods_segments'
    print('全部集合=', milvusTool.getListCollections())
    # 查看某个集合的具体信息
    print('集合', collection_name, '的信息=', milvusTool.describeCollection(collection_name))
    # 查看某个集合的全部索引
    print('集合', collection_name, '的全部索引信息=', milvusTool.list_indexes(collection_name))
    # 查看某个集合的某个索引的详细信息
    print('集合', collection_name, '的text索引信息=', milvusTool.describe_index(collection_name, 'text'))


def pdf_test(milvusTool):
    """pdf数据demo演示"""
    collection_name = 'dsmm_pdf_segments'
    schema = milvusTool.getSchema('袋鼠妈妈PDF数据')
    milvusTool.dropCollection(collection_name)
    # 创建数据集
    createCollectionIndex(milvusTool, schema, collection_name, 'data_type_desc：1表示企业信用报告-广东袋鼠妈妈集团有限公司_20240730.pdf')
    # 插入 企业信用报告-广东袋鼠妈妈集团有限公司_20240730.pdf 数据
    insert_pdf_json(milvusTool, collection_name)
    # 向量检索
    query = '袋鼠妈妈这个公司的地址是什么？'
    result = milvusTool.vector_search(collection_name=collection_name, query=query,
                                      expr='data_type == 1', output_fields=['id', 'metadata'])
    print('检索结果：', result)


def insert_pdf_json(milvusTool, collection_name):
    """读取品宣数据插入Milvus"""
    file_path = r'D:\D盘桌面\软通\袋鼠妈妈项目\知识库\测试数据\企业信用报告-广东袋鼠妈妈集团有限公司_20240828.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    data_type = 1
    # 插入pdf的json数据到 Milvus中
    segment_datas = []
    for data in datas:
        embedding_text = milvusTool.embedding_model_xiaobu.text_embedding(data['text'])
        result = {
            "metadata": data,
            "data_type": data_type,  # 1表示原始品宣数据, 2表示模型总结后的品宣数据
            "text": embedding_text
        }
        segment_datas.append(result)
    res = milvusTool.insertListData(collection_name, segment_datas)
    return res


if __name__ == '__main__':
    embedding_model_path=r'C:\workspace\root\AISHU AnyShare\dsmm_data\ShareCache\职能部门\信息技术中心\100、2024年项目管理\04-AI内容生成\002、袋鼠妈妈AI大模型\002、代码库\dsmm_model\com\dsmm\data\model\embedding\xiaobu-embedding-v2'
    milvusTool = MilvusTool(embedding_model_path)
    # 查看集合信息
    select_collection_info()
    # pdf demo
    # pdf_test(milvusTool)
    # 商品demo
    goods_test(milvusTool)
    # 品宣demo
    # pinxuan_test(milvusTool)
    milvusTool.close()
