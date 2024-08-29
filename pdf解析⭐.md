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

