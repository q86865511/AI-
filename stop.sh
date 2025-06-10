#!/bin/bash

echo "正在停止YOLO模型自動化轉換與測試系統..."
echo "==========================================="

# 停止並移除容器
docker-compose down

echo "[信息] 系統服務已停止"

# 可選：清理未使用的Docker資源
read -p "是否要清理未使用的Docker映像和容器？(y/N): " cleanup
if [[ $cleanup =~ ^[Yy]$ ]]; then
    echo "[信息] 清理未使用的Docker資源..."
    docker system prune -f
    echo "[信息] 清理完成"
fi

echo "[信息] 操作完成" 