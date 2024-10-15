# 1. indexify

## 1.1 环境准备

```shell
# docker网络创建
docker network create --subnet=172.18.0.0/16 ruantong_network

# docker系统准备
docker pull ubuntu:20.04
```

## 1.2 设置代理

**注意：安装了proxychains-ng会导致pip install llama-cpp-python失败**

安装annconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-py311_23.11.0-1-Linux-x86_64.sh --no-check-certificate
bash Miniconda3-py311_23.11.0-1-Linux-x86_64.sh

不停输入yes
source ~/.bashrc
conda --version
```

启动shadowsocks代理

```shell
创建虚拟环境
conda create -n shadowsocks python=3.8
conda activate shadowsocks

pip3 install https://github.com/shadowsocks/shadowsocks/archive/master.zip -U
vim ss.json
-----------------------------
{
  "server":"c20s1.portablesubmarines.com", 
  "server_port":19361,
  "local_address": "127.0.0.1",
  "local_port":1080,   
  "password":"w4XPXpMKZM6Eoo9W", 
  "method":"aes-256-gcm"  
}
-----------------------------
cd shadowsocks/
sslocal -c ss.json
# 后台启动
sslocal -c ss.json -d start
```

安装proxychains

```shell
git clone https://gitcode.com/gh_mirrors/pr/proxychains-ng.git
cd proxychains-ng
./configure --prefix=/usr --sysconfdir=/etc
make
vim src/proxychains.conf
--------------------------
#socks4         127.0.0.1 9050
socks5          127.0.0.1 1080
--------------------------
./proxychains4 -f src/proxychains.conf wget google.com
```



## 1.2 indexify-server服务服务启动

```shell
docker run -itd --name indexify-v20240829 \
-p 50001:8900 \
-v /home/ruantong/indexify:/home/ruantong/indexify \
--network=ruantong_network  --ip 172.18.0.3 \
ubuntu:20.04 bash

# 进入容器
docker exec -it 91ae8cb1dac1 bash
cd /home/ruantong/indexify

# 由于Linux中下载大文件较慢且容易超时，我们手动下载，然后上传至/opt/soft/indexify目录
# 下载地址：https://github.com/tensorlakeai/indexify/releases/download/v0.1.22/indexify-0.1.22-linux-amd64
cp indexify-0.1.22-linux-amd64 indexify
vim dowload.sh
-------------------------------------------
#!/bin/sh -e
chmod a+x ./indexify || exit 1
if command -v uname >/dev/null 2>&1; then
    PLATFORM=$(uname -s)
    MACHINE=$(uname -m)
else
    PLATFORM="unknown"
    MACHINE="unknown"
fi
curl -X POST "https://getindexify.ai/api/analytics" \
    -H "Content-Type: application/json" \
    -d "{\"event\": \"indexify_download\", \"platform\": \"$PLATFORM\", \"machine\": \"$MACHINE\"}" \
    --max-time 1 -s > /dev/null 2>&1 || true
-------------------------------------------

chmod +x dowload.sh && sh dowload.sh
./indexify server -d

# 访问web页面
http://localhost:8900/ui
```

## 1.3 indexify-extractor服务启动

```shell
# 在主机上
conda create -n indexify python=3.11
# 切换虚拟环境
conda activate indexify
```

```shell
# 安装依赖
pip install indexify indexify-extractor-sdk wikipedia 
pip install llama-cpp-python 
pip install mistralai
pip install s3fs
---------------------------------
# 如果需要连外网，需要设置代理,
cd /home/ruantong/indexify/proxychains-ng
./proxychains4 -f src/proxychains.conf pip install mistralai
---------------------------------

# 安装提取器
# 将PDF文档转换为Markdown格式的提取器
indexify-extractor download tensorlake/marker
# 将文本分块的提取器（可设置块大小和重叠配置参数）
indexify-extractor download tensorlake/chunk-extractor
# 直接提取PDF文档内容的提取器（需要开启代理服务，并下载proxychains）
cd proxychains-ng
./proxychains4 -f src/proxychains.conf indexify-extractor download tensorlake/pdfextractor

# 启动indexify-extractor服务
indexify-extractor join-server
```

## 1.4 注意事项

1. 因为是centos系统，所以需要docker创建一个unbatu环境(indexify必须要ubantu)
2. indexify-extractor服务必须链接外网，（解决办法，找一台云服务器，安装ubantu系统，下载代理，安装相关的依赖，并将conda环境封装，copy到centons服务中，在映射到docker中ubantu环境中的conda）
3. 再安装indexify-server服务
4. 需要将测试环境的/root/.indexify-extractors复制到docker环境中！！



# 2. pypdf2 + fitz

```python
import base64
import re
import os, io
from io import BytesIO
from pptx import Presentation
from hashlib import md5
import fitz
from PIL import Image
import imagehash
from PyPDF2 import PdfReader

from H_common.untils.files_utils import JsonUtils

"""
1、提取PPTX数据，转为JSON格式
2、提取pdf文本
3、提取pdf图片
4、提取pdf表格
"""


def pdf_table(file_path=''):
    """提取表格数据"""
    doc = fitz.open(file_path)
    results = {}
    # 确保保存 CSV 文件的目录存在
    save_dir = 'tables'
    os.makedirs(save_dir, exist_ok=True)
    for page in doc:
        tables = page.find_tables()
        num = page.number + 1
        num_list = []
        for i, table in enumerate(tables):
            df = table.to_pandas()
            csv_path = os.path.join(save_dir, f'table_pg_{num}_{i}.csv')
            df.to_csv(csv_path, index=False)
            num_list.append(csv_path)
        if num_list:
            results[num] = num_list
    return results

def pdf_text(file_path=''):
    """提取pdf文本"""
    reader = PdfReader(file_path)
    results = {}
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text = page.extract_text()
        results[page_num + 1] = text
    return results

def pdf_images(file_path=''):
    """
    提取pdf中的图片
    返回包含字典的列表，字典的key是pdf的页数，value是图片文件的存储全路径
    """
    pdf_document = fitz.open(file_path)
    seen_hashes = set()
    results = {}
    os.makedirs('images', exist_ok=True)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)
        image_paths = []
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            image_hash = imagehash.average_hash(image)
            if image_hash not in seen_hashes:
                seen_hashes.add(image_hash)
                image_filename = f"images/page_{page_num + 1}_img_{img_index + 1}.png"
                image.save(image_filename)
                image_paths.append(image_filename)
                print(f"图片已保存为 {image_filename}")
        if image_paths:
            results[page_num + 1] = image_paths
    pdf_document.close()
    return results

def clean_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def image_to_base64(img):
    """图片转base64"""
    img_data = BytesIO()
    img.save(img_data, format=img.format)
    img_data.seek(0)
    base64_encoded_img = base64.b64encode(img_data.read()).decode('utf-8')
    return base64_encoded_img

def get_pdf_json_data(file_name, images, tables, texts):
    """获取pdf文件数据：JSON格式"""
    results = []
    for i, num in enumerate(texts):
        text = texts[num]
        table = tables[num] if num in tables else []
        image = images[num] if num in images else []
        res_num = {
            'file_name': file_name,
            "page": num,
            "text": text,
            "table_path": table,
            "image_path": image,
        }
        results.append(res_num)
    return results

def process_pptx(pptx_file):
    prs = Presentation(pptx_file)
    result_list = []
    result_dirt = {}

    for page_num, slide in enumerate(prs.slides, start=1):
        text_segments = ''
        image_data_list = []
        image_hashes = set()

        for shape in slide.shapes:
            if hasattr(shape, 'text'):
                text = shape.text
                cleaned_text = clean_text(text)
                if cleaned_text:
                    text_segments += '\t' + cleaned_text

            if shape.shape_type == 13:
                image = shape.image
                image_stream = BytesIO(image.blob)
                img = Image.open(image_stream)

                img_hash = md5(image.blob).hexdigest()
                if img_hash not in image_hashes:
                    image_hashes.add(img_hash)
                    base64_encoded_img = image_to_base64(img)
                    image_data_list.append(base64_encoded_img)

        if text_segments.strip() or image_data_list:
            result = {
                "source_file": "袋鼠妈妈品牌介绍0618.pptx",
                "page": page_num,
                "text": text_segments.strip(),
                # "image": image_data_list
            }
            result_dirt[page_num] = result
            result_list.append(result)

    return result_list, result_dirt

def main():
    pptx_file = r'D:\D盘桌面\软通\袋鼠妈妈项目\知识库\AI营销数据\袋鼠妈妈品牌介绍0618.pptx'
    segment_file = r'D:\D盘桌面\软通\袋鼠妈妈项目\知识库\AI营销数据\袋鼠妈妈品牌介绍0618_原始拆分数据_20240813.json'
    jsonUtils = JsonUtils()
    result_list, _ = process_pptx(pptx_file)
    for page_result in result_list:
        print(page_result)
        jsonUtils.write_dict_to_json(page_result, segment_file)

def main2():
    pptx_file = r'D:\D盘桌面\软通\袋鼠妈妈项目\知识库\AI营销数据\袋鼠妈妈品牌介绍0618.pptx'
    _, result_dirt = process_pptx(pptx_file)

    txt_file = r'D:\D盘桌面\软通\袋鼠妈妈项目\知识库\AI营销数据\品宣数据-模型总结后的文本段.txt'
    line_count = 0

    segment_file = r'D:\D盘桌面\软通\袋鼠妈妈项目\知识库\AI营销数据\袋鼠妈妈品牌介绍0618_模型总结数据_20240813.json'

    jsonUtils = JsonUtils()
    with open(txt_file, 'r', encoding='utf-8') as file:
        for line in file:
            line_count += 1
            text = line.strip()
            print(text)
            if line_count == 1:
                page_num = [1, 2, 3, 4, 5]
            elif line_count == 2:
                page_num = [5]
            elif line_count == 3:
                page_num = [6, 7, 8]
            elif line_count == 4:
                page_num = [9, 10, 11, 12]
            elif line_count == 5:
                page_num = [8, 12]
            elif line_count == 6:
                page_num = [13, 14]
            elif line_count == 7:
                page_num = [15]
            elif line_count == 8:
                page_num = [16, 20]
            elif line_count == 9:
                page_num = [21, 22, 23, 24, 25]
            elif line_count == 10:
                page_num = [26, 27]

            text_segments_list = []
            for i, e in enumerate(page_num):
                data = result_dirt.get(e)
                if data:
                    text_segments_list.append(data['text'])
                if e == 7:
                    text_segments_list.append('在意每一个细节，宠爱每一个你\t安心\t关心\t专心\t用心\t累计3700万妈妈的选择。')

            result = {
                "source_file": "袋鼠妈妈品牌介绍0618.pptx",
                "text": text,
                "pptx_page": page_num,
                "pptx_text": text_segments_list
            }

            jsonUtils.write_dict_to_json(result, segment_file)

def main3():
    file_path = r"D:\D盘桌面\软通\袋鼠妈妈项目\知识库\测试数据\企业信用报告-广东袋鼠妈妈集团有限公司_20240730.pdf"
    file_name_with_ext = os.path.basename(file_path)
    file_name, _ = os.path.splitext(file_name_with_ext)
    result_json_file_path = f"D:\D盘桌面\软通\袋鼠妈妈项目\知识库\测试数据\{file_name}.json"
    # 提取表格数据
    tables = pdf_table(file_path)
    # 提取pdf图片
    images = pdf_images(file_path)
    # 提取pdf文本
    texts = pdf_text(file_path)
    # 获取pdf文件数据：JSON格式
    results = get_pdf_json_data(file_name_with_ext, images, tables, texts)
    # 写入json文件
    JsonUtils.write_list_to_json(results, result_json_file_path)

if __name__ == '__main__':
    # main()
    # main2()
    main3()


```



# 3. pdfplumber

```python
import pdfplumber
from H_common.untils.textsplitter import ChineseTextSplitter
from sentence_transformers import SentenceTransformer

def pdf_load_chunk_emb_save(pdf_path):

    spliter = ChineseTextSplitter(pdf=True)
    model = SentenceTransformer(r"D:\wk\model\xiaobu-embedding-v2")
    with pdfplumber.open(pdf_path) as pdf:
        results = []
        total_pages = len(pdf.pages)
        first_page = pdf.pages[0]
        first_page_text = first_page.extract_text()
        results.append({"header": "首页", "footer":"首页", "content": first_page_text})
        print(results)

        for page_number in range(1,total_pages):
            # 尝试识别页眉和页脚（这里假设页眉和页脚在页面文本的开头和结尾）
            # 注意：这只是一个非常粗糙的示例
            pdf_dict = {}
            # 读取整页文本
            page = pdf.pages[page_number]
            text = page.extract_text()

            if text:
                lines = text.strip().split('\n')
                if lines:
                    # 假设第一行是页眉
                    header = lines[0]
                    # print("可能的页眉:")
                    # print(header)
                    pdf_dict['header'] = header

                    # 假设最后一行是页脚
                    footer = lines[-1]
                    # print("可能的页脚:")
                    # print(footer)
                    pdf_dict['footer'] = footer

                # 去除页眉页脚
                text = text[len(header):-len(footer)]
                # text_tmp = ''
                # for ele in lines[1:-1]:
                #     text_tmp = text_tmp + ele

                # 调用函数，传入PDF文件路
            # print(text_tmp)
            pdf_dict['content'] = spliter.split_text1(text)
            # pdf_dict['content_embedding'] = model.encode(pdf_dict['content'], normalize_embeddings=True)

            # 识别表格
            tables = page.extract_tables()
            pdf_dict['tables'] = tables

            print(pdf_dict)
            results.append(pdf_dict)
    return results


if __name__ == '__main__':
    pdf_path = r"D:\wk\deepLearning_LLMRagCompetition\data\2389de12d78fe1ca4fa24910e6b1573902098bc3.PDF"
    results = pdf_load_chunk_emb_save(pdf_path)
```

