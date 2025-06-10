@echo off
chcp 65001
echo YOLO模型自動化轉換與測試系統啟動腳本
echo ===================================

REM 檢查Docker是否安裝
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [錯誤] 未檢測到Docker，請先安裝Docker Desktop
    goto :end
)

REM 檢查Docker是否正在運行
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [錯誤] Docker沒有運行，請啟動Docker Desktop
    goto :end
)

REM 檢查配置文件是否有更新
set REBUILD_BASE=0

echo [信息] 檢查後端基礎映像...
for %%f in (backend\requirements.txt backend\Dockerfile.base) do (
    if not exist .docker_cache\%%~nxf (
        echo [信息] 檢測到 %%~nxf 文件更新
        set REBUILD_BASE=1
    ) else (
        fc .docker_cache\%%~nxf %%f >nul 2>&1
        if errorlevel 1 (
            echo [信息] 檢測到 %%~nxf 文件更新
            set REBUILD_BASE=1
        )
    )
)

echo [信息] 檢查前端基礎映像...
for %%f in (frontend\package.json frontend\Dockerfile.base) do (
    if not exist .docker_cache\%%~nxf (
        echo [信息] 檢測到 %%~nxf 文件更新
        set REBUILD_BASE=1
    ) else (
        fc .docker_cache\%%~nxf %%f >nul 2>&1
        if errorlevel 1 (
            echo [信息] 檢測到 %%~nxf 文件更新
            set REBUILD_BASE=1
        )
    )
)

if "%REBUILD_BASE%"=="1" (
    echo [信息] 構建基礎映像...
    docker-compose build backend-base frontend-base
    
    REM 創建快取目錄並保存當前文件副本
    if not exist .docker_cache mkdir .docker_cache
    copy backend\requirements.txt .docker_cache\requirements.txt >nul
    copy backend\Dockerfile.base .docker_cache\Dockerfile.base >nul
    copy frontend\package.json .docker_cache\package.json >nul
    copy frontend\Dockerfile.base .docker_cache\Dockerfile.base >nul
) else (
    echo [信息] 基礎映像無需重建
)

echo [信息] 構建並啟動容器...
docker-compose build backend frontend
docker-compose up -d backend frontend

echo [信息] 系統服務已啟動，請稍候...
timeout /t 10 /nobreak

echo [信息] 正在打開網頁界面...
start http://localhost:3000

echo [信息] 系統已啟動：
echo - 前端界面：http://localhost:3000
echo - 後端API：http://localhost:8000
echo - API文檔：http://localhost:8000/docs

echo [信息] 按任意鍵停止服務並關閉...
pause >nul

echo [信息] 正在停止系統服務...
docker-compose down

:end
echo [信息] 操作完成
pause 