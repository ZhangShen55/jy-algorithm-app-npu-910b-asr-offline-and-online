#!/bin/bash
set -euo pipefail

BASE_CONFIG_PATH="${CONFIG_PATH:-/config.toml}"
APP_MODULE="${APP_MODULE:-main:app}"
BASE_PORT="${BASE_PORT:-8000}"
NGINX_UPSTREAM_CONF="/etc/nginx/conf.d/backend_upstream.conf"
TMP_CONF_DIR="/tmp/app_configs"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-asr}"

CLEANUP_AGE_MINUTES="${CLEANUP_AGE_MINUTES:-120}"
CLEANUP_INTERVAL_SECONDS="${CLEANUP_INTERVAL_SECONDS:-7200}"

# 关闭 nounset，避免 set_env.sh 引用未定义变量导致退出
set +u
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

if command -v conda >/dev/null 2>&1; then
    set +u
    eval "$(conda shell.bash hook)"
    set -u
    if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
        if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ]; then
            conda activate "$CONDA_ENV_NAME"
            echo "[INFO] 已激活 Conda 环境: $CONDA_ENV_NAME"
        fi
    else
        echo "[WARN] 未找到 Conda 环境 $CONDA_ENV_NAME，继续使用当前 Python 环境"
    fi
fi

# 追加路径，避免覆盖 TBE 相关 PYTHONPATH
export PYTHONPATH="/:/app:${PYTHONPATH:-}"

if [ ! -f "$BASE_CONFIG_PATH" ]; then
    echo "[ERROR] 配置文件不存在: $BASE_CONFIG_PATH"
    exit 1
fi

if ! python -c "import toml" >/dev/null 2>&1; then
    echo "[ERROR] Python 依赖 toml 未安装，无法解析配置"
    exit 1
fi

mkdir -p "$TMP_CONF_DIR"
mkdir -p /etc/nginx/conf.d

cleanup_tmp_wav() {
    local removed
    removed="$(find /tmp -type f -name "*.wav" -mmin +"$CLEANUP_AGE_MINUTES" -print -delete 2>/dev/null | wc -l || true)"
    echo "[INFO] /tmp 清理完成，删除 ${removed} 个过期 wav 文件"
}

start_cleanup_daemon() {
    cleanup_tmp_wav
    while true; do
        sleep "$CLEANUP_INTERVAL_SECONDS"
        cleanup_tmp_wav
    done
}

start_cleanup_daemon &
cleanup_pid=$!
sleep 1
if ! kill -0 "$cleanup_pid" 2>/dev/null; then
    echo "[ERROR] 临时文件清理任务启动失败，服务不会拉起"
    exit 1
fi
echo "[INFO] 临时文件清理任务已启动，PID: $cleanup_pid"

declare -a INSTANCE_PORTS=()
declare -a INSTANCE_CONFIGS=()
declare -a INSTANCE_WORKERS=()
instance_index=0

has_npu_plan="$(python - "$BASE_CONFIG_PATH" <<'PY'
import sys
import toml

cfg = toml.load(sys.argv[1])
npu_plan = cfg.get("npu_plan")
print("true" if isinstance(npu_plan, dict) and len(npu_plan) > 0 else "false")
PY
)"

if [ "$has_npu_plan" = "true" ]; then
    mapfile -t npu_entries < <(python - "$BASE_CONFIG_PATH" <<'PY'
import sys
import toml

cfg = toml.load(sys.argv[1])
npu_plan = cfg.get("npu_plan", {})
for npu_id, count in sorted(npu_plan.items(), key=lambda item: int(item[0])):
    print(f"{npu_id}={count}")
PY
)
    for entry in "${npu_entries[@]}"; do
        npu_id="${entry%%=*}"
        count="${entry#*=}"
        if ! [[ "$count" =~ ^[0-9]+$ ]]; then
            echo "[WARN] npu_plan 中实例数非法: npu_id=$npu_id, count=$count，已跳过"
            continue
        fi
        if [ "$count" -le 0 ]; then
            continue
        fi
        cfg_path="${TMP_CONF_DIR}/config_${npu_id}.toml"
        python - "$BASE_CONFIG_PATH" "$cfg_path" "npu:${npu_id}" "$count" <<'PY'
import sys
import toml

src, dst, device, count = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cfg = toml.load(src)
cfg["device"] = device
cfg["instance_count"] = count
cfg.pop("npu_plan", None)
with open(dst, "w", encoding="utf-8") as f:
    toml.dump(cfg, f)
PY
        INSTANCE_CONFIGS+=("$cfg_path")
        INSTANCE_PORTS+=("$((BASE_PORT + instance_index))")
        INSTANCE_WORKERS+=("$count")
        instance_index=$((instance_index + 1))
    done
else
    instance_count="$(python - "$BASE_CONFIG_PATH" <<'PY'
import sys
import toml

cfg = toml.load(sys.argv[1])
print(cfg.get("instance_count", 1))
PY
)"
    if ! [[ "$instance_count" =~ ^([0-9]|[1-2][0-9]|30)$ ]]; then
        echo "[WARN] instance_count 非法（允许范围 0-30），使用默认值 1"
        instance_count=1
    fi
    if [ "$instance_count" -le 0 ]; then
        instance_count=1
    fi
    device="$(python - "$BASE_CONFIG_PATH" <<'PY'
import sys
import toml

cfg = toml.load(sys.argv[1])
print(cfg.get("device", "npu:0"))
PY
)"
    cfg_path="${TMP_CONF_DIR}/config_single.toml"
    python - "$BASE_CONFIG_PATH" "$cfg_path" "$device" "$instance_count" <<'PY'
import sys
import toml

src, dst, device, count = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cfg = toml.load(src)
cfg["device"] = device
cfg["instance_count"] = count
cfg.pop("npu_plan", None)
with open(dst, "w", encoding="utf-8") as f:
    toml.dump(cfg, f)
PY
    INSTANCE_CONFIGS+=("$cfg_path")
    INSTANCE_PORTS+=("$BASE_PORT")
    INSTANCE_WORKERS+=("$instance_count")
fi

if [ "${#INSTANCE_PORTS[@]}" -eq 0 ]; then
    echo "[ERROR] 未生成任何实例配置，请检查 config.toml 的 npu_plan 或 instance_count"
    exit 1
fi

echo "[INFO] 生成 Nginx upstream 配置：$NGINX_UPSTREAM_CONF"
{
    echo "upstream backend {"
    echo "    least_conn;"
    for port in "${INSTANCE_PORTS[@]}"; do
        echo "    server 127.0.0.1:${port};"
    done
    echo "}"
} > "$NGINX_UPSTREAM_CONF"

if nginx -t; then
    if pgrep -x "nginx" > /dev/null; then
        nginx -s reload
    else
        nginx
    fi
else
    echo "[ERROR] Nginx 配置有误，启动失败"
    exit 1
fi

monitor_and_restart() {
    local port=$1
    local cfg=$2
    local workers=$3
    while true; do
        echo "[INFO] 启动服务实例，端口: $port, workers: $workers, 配置: $cfg"
        export CONFIG_PATH="$cfg"
        uvicorn "$APP_MODULE" --host 127.0.0.1 --port "$port" --workers "$workers"
        echo "[WARN] 实例端口 $port 退出，10 秒后重启..."
        sleep 10
    done
}

for i in "${!INSTANCE_PORTS[@]}"; do
    monitor_and_restart "${INSTANCE_PORTS[$i]}" "${INSTANCE_CONFIGS[$i]}" "${INSTANCE_WORKERS[$i]}" &
done

monitor_nginx() {
    while true; do
        if ! pgrep -x "nginx" > /dev/null; then
            echo "[WARN] Nginx 已退出，尝试重启..."
            nginx
        fi
        sleep 5
    done
}

monitor_nginx &
monitor_cleanup() {
    while true; do
        if ! kill -0 "$cleanup_pid" 2>/dev/null; then
            echo "[WARN] 临时文件清理任务已退出，尝试重启..."
            start_cleanup_daemon &
            cleanup_pid=$!
            sleep 1
            if kill -0 "$cleanup_pid" 2>/dev/null; then
                echo "[INFO] 临时文件清理任务已重启，PID: $cleanup_pid"
            else
                echo "[ERROR] 临时文件清理任务重启失败"
            fi
        fi
        sleep 30
    done
}

monitor_cleanup &
wait
