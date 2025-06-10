from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import List, Optional, Dict, Any
import os
import uuid
import shutil
from datetime import datetime

from app.models import ModelInfo, ModelPerformance, PrecisionType, PerformanceResult
from app.services.model_service import ModelService
from app.services.inference_service import InferenceService

router = APIRouter()
model_service = ModelService()
inference_service = InferenceService()

@router.post("/predict")
async def predict_image(
    model_id: str = Form(...),
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
    iou_threshold: float = Form(0.45)
):
    """
    使用指定模型對上傳圖片進行預測
    """
    # 檢查模型是否存在
    model = model_service.get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="找不到指定模型")
    
    # 保存上傳的圖片
    upload_dir = os.path.join("uploads", "images")
    os.makedirs(upload_dir, exist_ok=True)
    
    image_id = str(uuid.uuid4())
    image_path = os.path.join(upload_dir, f"{image_id}.jpg")
    
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 執行預測
    try:
        results = inference_service.predict(
            model=model,
            image_path=image_path,
            confidence=confidence,
            iou_threshold=iou_threshold
        )
        
        # 生成結果圖片路徑
        result_image_path = os.path.join(upload_dir, f"{image_id}_result.jpg")
        
        # 如果結果包含圖片，將其保存為URL可訪問格式
        if "image" in results:
            with open(result_image_path, "wb") as buffer:
                buffer.write(results["image"])
            results["image_url"] = f"/uploads/images/{image_id}_result.jpg"
            del results["image"]
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"預測過程中發生錯誤: {str(e)}")

@router.post("/benchmark", response_model=ModelPerformance)
async def benchmark_model(
    model_id: str = Form(...),
    batch_size: int = Form(1),
    precision: PrecisionType = Form(PrecisionType.FP32),
    num_iterations: int = Form(100),
    img_size: int = Form(640)
):
    """
    對指定模型進行性能基準測試
    
    Args:
        model_id: 模型ID
        batch_size: 批次大小
        precision: 精度類型
        num_iterations: 測試迭代次數
        img_size: 圖像尺寸
    """
    # 檢查模型是否存在
    model = model_service.get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="找不到指定模型")
    
    # 執行基準測試
    try:
        performance = inference_service.benchmark(
            model=model,
            batch_size=batch_size,
            precision=precision,
            num_iterations=num_iterations,
            img_size=img_size
        )
        return performance
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"基準測試過程中發生錯誤: {str(e)}")

@router.post("/compare", response_model=PerformanceResult)
async def compare_models(
    background_tasks: BackgroundTasks,
    model_ids: List[str] = Form(...),
    batch_size: int = Form(1),
    precision: PrecisionType = Form(PrecisionType.FP32),
    num_iterations: int = Form(100),
    img_size: int = Form(640)
):
    """
    比較多個模型的性能
    
    Args:
        model_ids: 模型ID列表
        batch_size: 批次大小
        precision: 精度類型
        num_iterations: 測試迭代次數
        img_size: 圖像尺寸
    """
    # 檢查所有模型是否存在
    models = []
    for model_id in model_ids:
        model = model_service.get_model_by_id(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"找不到模型ID: {model_id}")
        models.append(model)
    
    # 在背景中執行比較任務
    performance_results = []
    
    for model in models:
        try:
            performance = inference_service.benchmark(
                model=model,
                batch_size=batch_size,
                precision=precision,
                num_iterations=num_iterations,
                img_size=img_size
            )
            performance_results.append(performance)
        except Exception as e:
            # 記錄錯誤但繼續比較其他模型
            print(f"比較模型 {model.id} 時發生錯誤: {str(e)}")
    
    # 生成比較結果
    comparison = inference_service.generate_comparison(performance_results)
    
    return {
        "results": performance_results,
        "comparison": comparison
    } 