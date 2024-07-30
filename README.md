# 基于LLM智能问答系统学习赛

## 1. 介绍
基于LLM智能问答系统学习赛

项目地址：https://tianchi.aliyun.com/competition/entrance/532172/information

## 2. 任务介绍

1. 基于文档的知识问答（文本理解任务）
2. 基于数据库数据的问答（数据查询任务）

## 3. 项目架构

### 3.1 文本理解任务

**解决方案：rag，检索增强生成**

#### 3.1.1 意图识别

![image-20240728115226311](https://gitee.com/fubob/note-pic/raw/master/image/image-20240728115226311.png)

#### 3.1.2 文档抽取

![image-20240728115302562](https://gitee.com/fubob/note-pic/raw/master/image/image-20240728115302562.png)

#### 3.1.3 文本理解

![image-20240728115408948](https://gitee.com/fubob/note-pic/raw/master/image/image-20240728115408948.png)

![image-20240728115427398](https://gitee.com/fubob/note-pic/raw/master/image/image-20240728115427398.png)

![image-20240728115456407](https://gitee.com/fubob/note-pic/raw/master/image/image-20240728115456407.png)

### 3.2 数据查询任务

#### 3.2.1 nl2sql模型指令微调

![image-20240728115546275](https://gitee.com/fubob/note-pic/raw/master/image/image-20240728115546275.png)

#### 3.2.2 prompt可控生成

![image-20240728115616030](https://gitee.com/fubob/note-pic/raw/master/image/image-20240728115616030.png)