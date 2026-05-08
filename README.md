# DeepFace 人脸识别系统

基于 DeepFace 深度学习库构建的完整人脸识别 Web 应用系统，提供人脸注册、识别、比对和视频处理功能。

## 功能特性

- **人脸注册** - 通过上传照片或摄像头拍照录入人脸信息
- **人脸比对** - 验证两张照片是否为同一人
- **图片识别** - 在图片中检测并识别所有人脸
- **视频识别** - 处理视频文件，输出带人脸标注的结果视频
- **实时识别** - 通过摄像头进行实时人脸检测和识别
- **REST API** - 提供完整的 HTTP API 接口供第三方调用

## 技术栈

### 后端
- **Web 框架**: Flask 3.1.3
- **深度学习**: DeepFace 0.0.79, TensorFlow 2.12.0, Keras 2.12.0
- **图像处理**: OpenCV 4.8.1.78, NumPy 1.23.5, Pillow 12.2.0
- **人脸检测器**: SSD DNN (默认), OpenCV Haar, MTCNN, RetinaFace
- **人脸识别模型**: Facenet512 (默认), VGG-Face, ArcFace, Facenet, DeepID, Dlib, SFace, GhostFaceNet, OpenFace

### 前端
- 原生 HTML5 / CSS3 / JavaScript (ES6+)
- 响应式设计，支持移动端
- Canvas 实时渲染识别结果
- MediaDevices API 摄像头调用

## 快速开始

### 环境要求

- Python 3.8+
- pip 包管理器

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
# 默认配置 (127.0.0.1:5001)
python run.py

# 自定义端口
python run.py --port 5000

# 绑定所有网络接口
python run.py --host 0.0.0.0 --port 5000

# 调试模式
python run.py --debug
```

启动后访问:
- **Web UI**: http://127.0.0.1:5001
- **API 文档**: http://127.0.0.1:5001/api

## 项目结构

```
Deep-face/
├── run.py                    # 程序入口
├── requirements.txt          # Python 依赖列表
├── face_system/
│   ├── __init__.py          # 包初始化
│   ├── app.py               # Flask 应用及 API 路由
│   ├── recognizer.py        # 人脸识别核心引擎
│   ├── video_processor.py   # 视频处理模块
│   ├── templates/
│   │   └── index.html       # 前端 Web UI
│   └── face_db/             # 人脸特征数据库 (pkl 文件)
├── uploads/                  # 上传的文件存储目录
└── outputs/                  # 处理后的视频输出目录
```

## API 文档

### 1. 注册人脸

**POST** `/api/register`

注册一张人脸照片到数据库。

**请求参数** (multipart/form-data):
- `name` (string): 人员姓名
- `image` (file): 人脸图片文件

**响应示例**:
```json
{
  "status": "ok",
  "name": "张三",
  "face_confidence": 0.99,
  "face_area": 12345,
  "threshold": 0.30
}
```

### 2. 列出已注册人脸

**GET** `/api/faces`

获取所有已注册的人脸列表。

**响应示例**:
```json
{
  "faces": [
    {"name": "张三", "image_path": "/path/to/image.jpg"},
    {"name": "李四", "image_path": "/path/to/image.jpg"}
  ]
}
```

### 3. 删除已注册人脸

**DELETE** `/api/face/<name>`

根据姓名删除已注册的人脸。

**响应示例**:
```json
{
  "status": "ok"
}
```

### 4. 人脸比对

**POST** `/api/verify`

验证两张图片是否为同一人。

**请求参数** (multipart/form-data):
- `image1` (file): 第一张图片
- `image2` (file): 第二张图片

**响应示例**:
```json
{
  "verified": true,
  "distance": 0.25,
  "threshold": 0.30,
  "similarity": 0.75,
  "model": "Facenet512",
  "similarity_metric": "cosine"
}
```

### 5. 图片识别

**POST** `/api/recognize`

识别图片中的所有人脸。

**请求参数** (multipart/form-data):
- `image` (file): 待识别的图片

**响应示例**:
```json
{
  "matches": [
    {
      "name": "张三",
      "distance": 0.22,
      "similarity": 0.78,
      "is_match": true,
      "region": {"x": 100, "y": 50, "w": 200, "h": 200},
      "confidence": 0.99
    }
  ]
}
```

### 6. 视频识别

**POST** `/api/recognize-video`

处理视频文件，返回带人脸标注的结果视频。

**请求参数** (multipart/form-data):
- `video` (file): 待处理的视频文件 (MP4/AVI/MOV)

**响应示例**:
```json
{
  "status": "ok",
  "total_frames": 1200,
  "fps": 30.0,
  "duration": 40.0,
  "matches": [
    {"frame": 0, "timestamp": 0.0, "name": "张三", "distance": 0.20, "similarity": 0.80}
  ],
  "match_summary": {"张三": 50},
  "output_video_url": "/outputs/out_abc123.mp4"
}
```

### 7. 实时帧识别

**POST** `/api/recognize-frame`

识别摄像头实时视频帧 (用于前端实时识别)。

**请求参数** (JSON):
- `image` (string): base64 编码的图片数据

**响应示例**:
```json
{
  "matches": [
    {
      "name": "张三",
      "distance": 0.22,
      "similarity": 0.78,
      "is_match": true,
      "region": {"x": 100, "y": 50, "w": 200, "h": 200}
    }
  ]
}
```

## 配置说明

### 人脸识别模型

系统默认使用 **Facenet512** 模型，各模型的识别阈值如下:

| 模型 | 阈值 | 说明 |
|------|------|------|
| Facenet512 | 0.30 | 默认推荐，精度高 |
| VGG-Face | 0.40 | 经典模型 |
| ArcFace | 0.40 | 高准确率 |
| Facenet | 0.40 | 128 维特征 |
| DeepFace | 0.25 | 轻量级 |
| DeepID | 0.015 | 低阈值 |
| Dlib | 0.07 | Dlib 实现 |
| SFace | 0.30 | OpenCV DNN |
| GhostFaceNet | 0.40 | 轻量化 |
| OpenFace | 0.10 | CMU 开源 |

### 人脸检测器

| 检测器 | 速度 | 精度 | 说明 |
|--------|------|------|------|
| SSD | 中等 | 高 | 默认推荐，基于 DNN |
| OpenCV Haar | 快 | 低 | 传统级联分类器 |
| MTCNN | 中等 | 高 | 多任务级联 |
| RetinaFace | 慢 | 最高 | 高精度检测 |

### 修改配置

在 `face_system/app.py` 中修改以下参数:

```python
# 检测器后端
DETECTOR = "ssd"  # 可选: ssd, opencv, mtcnn, retinaface

# 人脸识别模型
recognizer_ = FaceRecognizer(
    model_name="Facenet512",  # 替换为其他模型
    detector_backend=DETECTOR,
)

# 视频处理帧间隔
video_processor_ = VideoProcessor(
    recognizer_,
    process_every_n_frames=10,  # 每 N 帧处理一次
    detector_backend=DETECTOR,
)
```

## 使用说明

### Web 界面

1. **注册人脸**
   - 输入姓名
   - 上传照片或打开摄像头拍照
   - 点击"注册人脸"按钮

2. **视频识别**
   - 上传视频文件 (MP4/AVI/MOV)
   - 点击"开始识别"
   - 查看识别结果和输出视频

3. **实时识别**
   - 切换到"实时识别"页面
   - 点击"启动摄像头"
   - 系统自动检测并识别画面中的人脸

4. **人脸比对**
   - 上传两张需要比对的图片
   - 点击"比对两张照片"
   - 查看相似度结果

### 数据存储

- **人脸特征库**: `face_system/face_db/` - 以 `.pkl` 文件存储每个人的特征向量
- **上传文件**: `uploads/` - 临时存储上传的图片和视频
- **输出视频**: `outputs/` - 处理后的结果视频

## 注意事项

1. **首次运行**会下载模型权重文件，请确保网络畅通
2. **文件大小限制**: 默认最大上传 500MB
3. **GPU 加速**: 如需 GPU 加速，请安装 `tensorflow` (非 CPU 版本)
4. **OpenCV 编码**: 视频输出可能因系统缺少 H.264 编码器而回退到 mp4v
5. **浏览器兼容**: 输出视频在某些浏览器中可能无法直接播放，建议下载后查看

## 常见问题

### Q: 注册人脸时提示"未检测到人脸"
A: 确保上传的图片中清晰可见完整面部，光线充足，正对镜头。

### Q: 视频处理速度慢
A: 可以在 `VideoProcessor` 中增大 `process_every_n_frames` 参数，减少处理帧数。

### Q: 识别准确率低
A: 尝试更换识别模型 (如 ArcFace) 或检测器 (如 retinaface)。

### Q: 如何清空已注册人脸
A: 删除 `face_system/face_db/` 目录下的所有 `.pkl` 文件，或在前端逐个删除。

## 许可证

本项目基于 DeepFace 开源库构建，遵循其 MIT 许可证。

## 参考资料

- [DeepFace GitHub](https://github.com/serengil/deepface)
- [Flask 文档](https://flask.palletsprojects.com/)
- [OpenCV 文档](https://docs.opencv.org/)
