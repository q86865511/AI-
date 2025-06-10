from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query
from typing import List, Optional
import os
import uuid
import shutil
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError
from fastapi.responses import FileResponse
import tempfile
import json

from app.models import ModelInfo, ModelList, ModelType, ModelFormat
from app.services.model_service import ModelService

router = APIRouter()
model_service = ModelService()

@router.get("/", response_model=ModelList)
async def list_models(
    model_type: Optional[ModelType] = None,
    format: Optional[ModelFormat] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100)
):
    """
    獲取模型列表，可按類型和格式過濾
    """
    models = model_service.get_models(model_type, format, skip, limit)
    return {"models": models, "total": len(models)}

@router.get("/refresh")
async def refresh_models():
    """
    重新掃描模型目錄，刷新模型列表
    """
    print("收到模型刷新請求...")
    try:
        # 掃描模型儲存庫
        new_model_ids = model_service._scan_model_repository()
        # 保存更新的模型信息
        model_service._save_models()
        print(f"模型刷新成功，找到 {len(new_model_ids)} 個模型")
        return {"success": True, "message": "模型存儲庫已刷新", "new_model_ids": new_model_ids}
    except Exception as e:
        import traceback
        print(f"刷新模型列表時出錯: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"刷新模型列表失敗: {str(e)}")

@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """
    獲取特定模型的詳細信息
    """
    model = model_service.get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="找不到指定模型")
    return model

@router.post("/", response_model=ModelInfo)
async def upload_model(
    model_file: UploadFile = File(...),
    model_name: str = Form(...),
    model_type: str = Form("yolov8"),  # 改為字符串類型，稍後手動轉換為枚舉
    description: Optional[str] = Form(None)  # 添加description參數
):
    """
    上傳模型文件
    """
    try:
        # 處理model_type，將連字符替換為下劃線以匹配枚舉值
        normalized_type = model_type.replace('-', '_').upper()
        
        # 嘗試轉換為ModelType枚舉
        try:
            model_type_enum = ModelType(normalized_type)
        except ValueError:
            # 如果直接匹配失敗，嘗試匹配常見的別名
            type_map = {
                "YOLOV8": ModelType.YOLOV8,
                "YOLOV8_POSE": ModelType.YOLOV8_POSE,
                "YOLOV8POSE": ModelType.YOLOV8_POSE,
                "YOLOV8-POSE": ModelType.YOLOV8_POSE,
                "POSE": ModelType.YOLOV8_POSE,
                "YOLOV8_SEG": ModelType.YOLOV8_SEG,
                "YOLOV8SEG": ModelType.YOLOV8_SEG,
                "YOLOV8-SEG": ModelType.YOLOV8_SEG,
                "SEG": ModelType.YOLOV8_SEG
            }
            
            model_type_enum = type_map.get(normalized_type, ModelType.YOLOV8)
            print(f"模型類型 '{model_type}' 自動映射為 '{model_type_enum.value}'")
        
        # 獲取文件擴展名
        filename = model_file.filename
        ext = os.path.splitext(filename)[1].lower()
        print(f"收到上傳請求: 文件名={filename}, 模型名稱={model_name}, 模型類型={model_type_enum.value}")
        print(f"文件擴展名: {ext}")
        
        # 檢查是否為合法的模型文件
        if ext not in ['.pt', '.pth', '.onnx', '.engine', '.plan']:
            raise HTTPException(
                status_code=400, 
                detail="不支持的文件類型，僅接受.pt, .pth, .onnx, .engine, .plan格式"
            )
        
        # 創建模型目錄
        model_dir = os.path.join("model_repository", model_name)
        version_dir = os.path.join(model_dir, "1")  # 使用版本1
        
        # 創建目錄
        os.makedirs(version_dir, exist_ok=True)
        
        # 確定模型格式
        model_format = ModelFormat.PT
        if ext == '.onnx':
            model_format = ModelFormat.ONNX
        elif ext in ['.engine', '.plan']:
            model_format = ModelFormat.ENGINE
        
        # 保存文件
        model_filename = f"model{ext}"
        save_path = os.path.join(version_dir, model_filename)
        
        # 創建一個臨時文件寫入內容，然後移到目標位置
        # 這樣可以避免並發問題和寫入不完整的風險
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # 寫入上傳的模型數據
            content = await model_file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # 如果目標文件已存在，先刪除
            if os.path.exists(save_path):
                os.remove(save_path)
            
            # 移動臨時文件到目標位置
            shutil.move(temp_file.name, save_path)
        
        print(f"保存文件到: {save_path}")
        
        # 生成 config.pbtxt 配置文件
        config_path = os.path.join(model_dir, "config.pbtxt")
        
        # 設定平台和預設參數
        platform = ""
        batch_size = 1
        img_size = 640
        
        if model_format == ModelFormat.PT:
            platform = "pytorch_libtorch"
        elif model_format == ModelFormat.ONNX:
            platform = "onnxruntime_onnx"
        elif model_format == ModelFormat.ENGINE:
            platform = "tensorrt_plan"
        
        # 根據模型類型設置不同的輸出維度
        output_dims = "[ 84, 8400 ]"  # 默認為YOLOv8檢測模型
        if model_type_enum == ModelType.YOLOV8_SEG:
            # 分割模型有不同的輸出
            output_dims = "[ -1, -1, -1 ]"  # 使用動態維度
        elif model_type_enum == ModelType.YOLOV8_POSE:
            # 姿態估計模型有不同的輸出
            output_dims = "[ -1, -1, -1 ]"  # 使用動態維度
        
        # 生成基本配置
        config_content = f"""name: "{model_name}"
platform: "{platform}"
max_batch_size: {batch_size * 2}  # 允許的最大批次大小是設定值的兩倍
input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, {img_size}, {img_size} ]
  }}
]
output [
  {{
    name: "output0"
    data_type: TYPE_FP32
    dims: {output_dims}
  }}
]
default_model_filename: "{model_filename}"
instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
"""
        with open(config_path, "w", encoding="utf-8") as config_file:
            config_file.write(config_content)
        
        print(f"創建Triton配置文件: {config_path}")
        
        # 創建模型ID (使用UUID)
        model_id = str(uuid.uuid4())
        
        # 創建模型元數據文件
        metadata_file = os.path.join(model_dir, "metadata.json")
        with open(metadata_file, 'w', encoding="utf-8") as f:
            metadata = {
                "model_id": model_id,
                "upload_time": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                "original_filename": filename,
                "model_name": model_name,
                "model_type": model_type_enum.value,
                "model_format": model_format.value
            }
            json.dump(metadata, f, indent=2)
        
        print(f"創建模型元數據文件: {metadata_file}")
        
        # 計算文件大小
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        
        # 創建模型信息
        model = ModelInfo(
            id=model_id,
            name=model_name,
            type=model_type_enum,
            format=model_format,
            path=save_path,
            size_mb=round(file_size_mb, 2),
            created_at=datetime.now(timezone(timedelta(hours=8))),
            description=description or f"上傳的{model_format.value}模型: {filename}",  # 使用提供的描述或默認描述
            metadata={
                "original_filename": filename,
                "model_id": model_id,
                "model_type": model_type_enum.value
            }
        )
        
        # 保存模型信息
        saved_model = model_service.save_model(model)
        
        # 立即重新掃描模型目錄，確保新上傳的模型被識別
        model_service._scan_model_repository()
        
        print(f"模型上傳成功: id={model_id}, 名稱={model_name}, 類型={model_type_enum.value}")
        
        return saved_model
        
    except Exception as e:
        error_msg = f"模型上傳失敗: {str(e)}"
        print(error_msg)
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """
    刪除指定模型
    """
    model = model_service.get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="找不到指定模型")
    
    try:
        # 刪除模型文件和目錄
        if os.path.exists(model.path):
            try:
                os.remove(model.path)
                print(f"已刪除模型文件: {model.path}")
            except Exception as e:
                print(f"刪除模型文件時出錯: {str(e)}")
        
        # 如果有Triton模型目錄，嘗試刪除整個目錄
        triton_model_dir = model.metadata.get("triton_model_dir") if model.metadata else None
        if triton_model_dir and os.path.exists(triton_model_dir):
            try:
                shutil.rmtree(triton_model_dir)
                print(f"已刪除Triton模型目錄: {triton_model_dir}")
            except Exception as e:
                print(f"刪除Triton模型目錄時出錯: {str(e)}")
        else:
            # 如果沒有Triton目錄信息，嘗試檢查並刪除版本目錄的父目錄
            model_dir = os.path.dirname(os.path.dirname(model.path))
            if os.path.exists(model_dir) and "model_repository" in model_dir:
                try:
                    shutil.rmtree(model_dir)
                    print(f"已刪除模型目錄: {model_dir}")
                except Exception as e:
                    print(f"刪除模型目錄時出錯: {str(e)}")
        
        # 從服務中刪除模型信息
        model_service.delete_model(model_id)
        
        return {"message": "模型已成功刪除"}
    except Exception as e:
        error_msg = f"刪除模型時出錯: {str(e)}"
        print(error_msg)
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/{model_id}/download")
async def download_model(model_id: str):
    """
    下載指定模型文件
    """
    model = model_service.get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="找不到指定模型")
    
    if not os.path.exists(model.path):
        raise HTTPException(status_code=404, detail="模型文件不存在")
    
    file_extension = os.path.splitext(model.path)[1]
    file_name = f"{model.name}{file_extension}"
    
    return FileResponse(
        path=model.path,
        filename=file_name,
        media_type="application/octet-stream"
    ) 