import requests


def get_image_tags(registry_url, namespace, image_name):
    response = requests.get(f"{registry_url}/v2/{namespace}/{image_name}/tags/list")
    return response.json()["tags"]


# 网易云Docker镜像：http://hub-mirror.c.163.com
# 百度云Docker镜像：https://mirror.baidubce.com
# 腾讯云Docker镜像：https://ccr.ccs.tencentyun.com
# DockerProxy镜像：https://dockerproxy.com
# 阿里云Docker镜像（需要使用阿里账号自行创建专属镜像仓库）：https://cr.console.aliyun.com/
# 镜像（配置文档）：http://f1361db2.m.daocloud.io


ung2thfc_url = "https://ung2thfc.mirror.aliyuncs.com"  # hub-mirror.c.163.com  docker.mirrors.ustc.edu.cn https://mirror.baidubce.com
namespace = "milvusdb"
image_name = "milvus"

tags = get_image_tags(ung2thfc_url, namespace, image_name)
print(tags)