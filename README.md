# SeaCraft ASR App

实时语音和离线语音转写后端服务，已从 NVIDIA CUDA 生态迁移到华为昇腾 910B/NPU 环境。项目基于 FastAPI，对外提供离线转写、实时 WebSocket 转写、音频质量检测、普通话检测和文本五何问题分类等接口。

当前主要适配环境：

- Python 3.10
- Conda 环境名：`asr`
- 华为昇腾 910B
- CANN 8.2.rc1，历史 Dockerfile 中也保留 8.1.rc1 版本

## 项目结构

```text
app/
├── api/routes/          # FastAPI 路由：离线 ASR、实时 ASR、音频检测、文本分类、状态接口
├── core/                # 配置、模型加载、并发控制、日志和状态逻辑
├── entity/              # 请求参数和数据结构
├── utils/               # 音频处理、特征提取、角色识别、统计、说话人处理等工具
├── utils/test/          # 测试和压测脚本
├── utils/test-data/     # 测试音频占位目录，真实大文件由服务器存储
├── docker/              # Dockerfile 和容器构建/运行说明
├── config.toml          # 默认配置文件
├── main.py              # FastAPI 应用入口
├── nginx.conf           # Nginx 反向代理配置
├── requirements.txt     # Python 依赖
└── start.sh             # 容器/服务启动脚本
```

## 主要能力

- 离线语音识别：`POST /v1.1.8/seacraft_asr`
- 实时语音识别：`WS /v1.0.1/seacraft_asr_online`
- 服务状态：`GET /get_status`
- 清空任务列表：`DELETE /clear_tasks_list`
- 音频信噪比检测：`POST /audio/db_snr`
- 普通话检测：`POST /audio/detect_mandarin`
- 文本五何问题分类：`POST /text/question`

## 配置文件

项目使用 TOML 配置，默认文件为：

```text
app/config.toml
```

服务启动时通过 `CONFIG_PATH` 指定配置路径。容器内默认读取：

```text
/config.toml
```

关键配置说明：

```toml
version = "seacraft-asr-app-v1.1.9"
device = "npu:0"
instance_count = 4
concurrency = 5

[npu_plan]
"0" = 1
"1" = 1
```

- `device`：默认设备，未配置 `npu_plan` 时使用。
- `instance_count`：未配置 `npu_plan` 时的 worker 数。
- `concurrency`：单实例内部并发控制。
- `npu_plan`：多 NPU 部署计划，key 为 NPU ID，value 为该 NPU 上启动的 worker 数。
- `open_spk`：开启离线 ASR/说话人相关模型。
- `open_online`：开启实时 ASR 模型。
- `open_mul_lang`：开启 Whisper 多语种转写。
- `open_mul_spk`：开启多语种说话人分离。
- `open_emotion`：开启情绪识别。
- `bert_device`：五何分类 BERT 使用设备，例如 `npu`、`cpu` 或 `auto`。
- `ban_hotword`：是否禁用热词。

`npu_plan` 存在且非空时优先使用；启动脚本会为每个 NPU 生成临时 TOML 配置，并通过 Nginx 做 upstream 分发。

## 本地运行

准备 conda 环境：

```bash
conda activate asr
pip install -r requirements.txt
```

直接启动单进程调试：

```bash
cd app
CONFIG_PATH=./config.toml uvicorn main:app --host 0.0.0.0 --port 8083
```

使用项目启动脚本：

```bash
cd app
CONFIG_PATH=./config.toml APP_MODULE=main:app BASE_PORT=8000 ./start.sh
```

`start.sh` 会尝试激活 conda 环境 `asr`。可通过环境变量覆盖：

```bash
CONDA_ENV_NAME=asr CONFIG_PATH=/path/to/config.toml ./start.sh
```

## Docker 部署

常用构建方式：

```bash
cd app
docker build -f docker/Dockerfile -t jy-algorithm-app-asr-ascend:latest .
```

运行时需要挂载模型目录和配置文件：

```bash
docker run -d --name asr_server \
  --runtime=ascend \
  --ipc=host \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1 \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /home/xjtu/model_zoo/model_asr/:/model:ro \
  -v /root/config/asr_config_offline.toml:/config.toml:ro \
  -p 8081:9000 \
  jy-algorithm-app-asr-ascend:latest
```

更完整的历史构建和运行命令见 [docker/README.md](docker/README.md)。

## 测试数据

测试脚本位于：

```text
utils/test/
```

大音频测试文件不提交到 Git，由服务器存储。运行测试前，把音频文件放到：

```text
utils/test-data/
```

当前测试脚本期望的文件名见 [utils/test-data/README.md](utils/test-data/README.md)。

## Git 提交约定

应提交：

- 源码：`api/`、`core/`、`entity/`、`utils/`
- 配置模板：`config.toml`
- 部署文件：`start.sh`、`nginx.conf`、`docker/`
- 测试脚本：`utils/test/`
- 测试数据说明：`utils/test-data/README.md`

不提交：

- 运行日志：`*.log`
- Python 缓存：`__pycache__/`、`*.pyc`
- 运行统计：`asr_stats.json`
- 大音频测试文件：`*.aac`、`*.wav`、`*.pcm`
- 本地 IDE 配置和构建产物

## 注意事项

- `config.toml` 中的模型路径需要与部署机器上的实际挂载路径一致。
- 容器默认从 `/config.toml` 读取配置。
- `start.sh` 使用 Nginx 暴露统一端口，后端多个 uvicorn 实例监听 `127.0.0.1:8000+`。
- 若开启 `open_online`、`open_spk`、`open_emotion` 等功能，需要确保对应模型目录存在。
- 不使用 Git LFS 时，不要把大音频测试文件加入普通 Git 提交。
