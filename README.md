# YOLO模型自動化轉換與測試系統

## 系統概述

YOLO模型自動化轉換與測試系統是一個用於將YOLO模型轉換為TensorRT格式並進行性能測試的工具。本系統提供了友好的Web界面，允許用戶：

1. 上傳和管理不同格式的YOLO模型（PT、ONNX、TensorRT）
2. 自動將模型轉換為不同格式（支持PT -> ONNX -> TensorRT轉換流程）
3. 使用轉換後的模型進行圖像推理和目標檢測
4. 進行模型性能基準測試，比較不同格式之間的效能差異
5. 可視化性能對比結果

## 系統要求

- Windows 10/11 或 Linux 操作系統
- [Docker](https://www.docker.com/products/docker-desktop/) 及 Docker Compose
- NVIDIA GPU 及 [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- 至少 8GB RAM
- 至少 20GB 可用磁盤空間

## 快速開始

### Windows用戶

1. 確保已安裝並啟動Docker Desktop
2. 雙擊`startup.bat`啟動系統
3. 瀏覽器將自動打開Web界面

### Linux用戶

1. 確保已安裝Docker和NVIDIA Container Toolkit
2. 執行以下命令啟動系統：

```bash
# 構建並啟動容器
docker-compose up -d

# 開啟Web界面
xdg-open http://localhost:3000 || open http://localhost:3000 || echo "請手動打開瀏覽器訪問 http://localhost:3000"
```

## 系統架構

本系統採用前後端分離架構：

- **前端**：React + Ant Design，提供直觀的用戶界面
- **後端**：FastAPI，提供RESTful API服務
- **模型轉換**：使用ONNX和TensorRT進行模型格式轉換
- **推理引擎**：支持PyTorch、ONNX Runtime和TensorRT多種推理引擎

## 目錄結構

```
├── backend/               # 後端API服務
│   ├── app/               # 後端應用代碼
│   │   ├── models/        # 數據模型定義
│   │   ├── routers/       # API路由定義
│   │   ├── services/      # 業務邏輯服務
│   │   └── utils/         # 工具函數
│   ├── Dockerfile         # 後端Docker配置
│   └── requirements.txt   # Python依賴
├── frontend/              # 前端應用
│   ├── public/            # 靜態資源
│   ├── src/               # 前端源代碼
│   │   ├── components/    # React組件
│   │   ├── pages/         # 頁面組件
│   │   └── services/      # API調用服務
│   └── Dockerfile         # 前端Docker配置
├── model_repository/      # 模型存儲目錄
├── docker-compose.yml     # Docker Compose配置
├── startup.bat            # Windows啟動腳本
└── README.md              # 項目說明文檔
```

## 使用指南

### 模型上傳

1. 在Web界面中導航至"模型管理"頁面
2. 點擊"上傳模型"按鈕
3. 選擇YOLO模型文件（支持.pt、.onnx和.engine格式）
4. 填寫模型名稱和其他信息
5. 點擊"上傳"完成模型上傳

### 模型轉換

1. 在"模型管理"頁面選擇要轉換的模型
2. 點擊"轉換"按鈕
3. 選擇目標格式（ONNX或TensorRT）和精度（FP32/FP16/INT8）
4. 設置轉換參數（可選）
5. 點擊"開始轉換"開始轉換過程

### 模型推理

1. 在"模型推理"頁面選擇要使用的模型
2. 上傳圖像文件
3. 調整置信度閾值（可選）
4. 點擊"開始推理"進行目標檢測
5. 查看檢測結果和標註後的圖像

### 性能測試

1. 在"性能測試"頁面選擇要測試的模型
2. 設置批次大小和迭代次數
3. 點擊"開始測試"進行性能基準測試
4. 查看推理時間、吞吐量和顯存使用等指標

### 模型比較

1. 在"模型比較"頁面選擇多個要比較的模型
2. 設置批次大小和迭代次數
3. 點擊"開始比較"進行模型間的性能比較
4. 查看對比圖表和加速比數據

## 故障排除

- **模型轉換失敗**：檢查模型格式是否兼容，ONNX轉換需要支持的算子
- **TensorRT轉換失敗**：確保NVIDIA驅動和TensorRT版本匹配，參考TensorRT兼容性表
- **推理速度慢**：嘗試降低推理精度至FP16或INT8，設置更適合的批次大小
- **顯存溢出**：對於大型模型，嘗試使用較低精度或減小輸入圖像尺寸

## 技術文檔

更詳細的技術文檔可在系統運行後訪問：

- API文檔：http://localhost:8000/docs
- OpenAPI規範：http://localhost:8000/openapi.json

## 開發與擴展

要擴展系統功能或進行二次開發：

1. 在本地克隆代碼庫
2. 使用開發模式啟動系統 `docker-compose -f docker-compose.dev.yml up`
3. 前端代碼變更將自動熱重載
4. 後端API修改後需重啟容器或使用熱重載工具

## 許可協議

本項目使用MIT許可協議，詳見LICENSE文件。 