# Face

### 环境安装
+ Milvus安装
  安装Milvus版本2.2.9
  单机安装：https://milvus.io/docs/install_standalone-docker.md
  集群安装：https://milvus.io/docs/install_cluster-milvusoperator.md

+ 项目环境配置
  安装annconda，后创建python版本为3.8的虚拟环境，之后在该虚拟环境内安装以下包
  - 安装torch2.0 或其他与本机环境适配的pytorch和cuda版本，https://pytorch.org/get-started/locally/
  - 安装onnx， pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnx
  - 安装onnxruntime-gpu, pip install onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
  - 安装opencv，pip install opencv-python==4.8.0.74
  - 安装av， pip install av==10.0.0
  - 安装pymilvus，版本要符合跟Milvus版本的对于关系，pip install pymilvus==2.2.9
  - 安装pyyaml，pip install pyyaml==6.0
  


### 项目结构
+ config/weights存放模型权重 config/config.yml 存放配置信息 config/warm_up.jpg是用于预热模型的输入图片 config.config.get_config 返回配置信息
+ model 包存放模型功能代码
+ init/initialize.py 用来初始化Milvus数据库中的表(collection)，安装完并启动Milvus后，执行initialize.py
+ milvus_tool包存放milvus数据库相关操作
+ utils包存放使用到的工具类
+ service/face_service 是人脸相关功能的接口


### 项目说明
因为模型的加载，Milvus的连接和加载都需要在程序运行时保持状态，所以在程序时启动时需要加载模型和Milvus，在程序结束时要释放Milvus。
以python框架fastapi为例:
+ 在fastapi启动类开始时加载模型和Milvus：
  ![image](https://github.com/jieblue/Face/assets/53696774/7e68352b-6a77-45c5-a955-2a5067e9f289)
+ 在fastapi类结束时释放Milvus:
  ![image](https://github.com/jieblue/Face/assets/53696774/75d2ccb3-0c46-43d2-bf8e-2bd617dd1fc3)

  



  