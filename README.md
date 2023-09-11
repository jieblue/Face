# Face

### 环境安装
+ Milvus安装
  安装Milvus版本2.2.9
  单机安装：https://milvus.io/docs/install_standalone-docker.md
  集群安装：https://milvus.io/docs/install_cluster-milvusoperator.md

+ 项目主要环境
  - av=10.0.0=py38h1983aaf_3
  - onnx=1.14.0=py38h641e5f8_1
  - python=3.8.0=hc9e8b01_5
  - python_abi=3.8=2_cp38
  - pyyaml=6.0=py38h91455d4_5
  - facexlib==0.3.0
  - numpy==1.24.1
  - onnxruntime-gpu==1.15.1
  - onnxsim==0.4.33
  - opencv-python==4.8.0.74
  - pymilvus==2.2.9
  - pytorch-lightning==2.0.5
  - torch==2.0.1+cu118
  - torchaudio==2.0.2+cu118
  - torchmetrics==1.0.0
  - torchvision==0.15.2+cu118


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

  



  
