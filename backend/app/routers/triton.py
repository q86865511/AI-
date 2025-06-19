from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import os

from app.services.triton_service import TritonService
from app.services.model_service import ModelService

router = APIRouter()
triton_service = TritonService()
model_service = ModelService()

@router.get("/health")
async def check_triton_health():
    """
    檢查Triton服務器健康狀態
    """
    try:
        result = await triton_service.health_check()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"檢查Triton健康狀態失敗: {str(e)}")

@router.post("/models/{model_id}/load")
async def load_model(model_id: str):
    """
    掛載指定模型到Triton服務器
    
    Args:
        model_id: 模型ID
    """
    try:
        # 獲取模型信息
        model = model_service.get_model_by_id(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"找不到模型: {model_id}")
        
        # 檢查模型是否為Triton兼容格式
        if not model.metadata or not model.metadata.get("triton_model_name"):
            raise HTTPException(status_code=400, detail="模型不是Triton兼容格式，無法掛載")
        
        triton_model_name = model.metadata.get("triton_model_name")
        
        # 掛載模型
        result = await triton_service.load_model(triton_model_name)
        
        if result["success"]:
            return {
                "success": True,
                "message": f"模型 {model.name} 掛載成功",
                "model_id": model_id,
                "triton_model_name": triton_model_name,
                "timestamp": result["timestamp"]
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"掛載模型失敗: {str(e)}")

@router.post("/models/{model_id}/unload")
async def unload_model(model_id: str):
    """
    從Triton服務器卸載指定模型
    
    Args:
        model_id: 模型ID
    """
    try:
        # 獲取模型信息
        model = model_service.get_model_by_id(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"找不到模型: {model_id}")
        
        # 檢查模型是否為Triton兼容格式
        if not model.metadata or not model.metadata.get("triton_model_name"):
            raise HTTPException(status_code=400, detail="模型不是Triton兼容格式，無法卸載")
        
        triton_model_name = model.metadata.get("triton_model_name")
        
        # 卸載模型
        result = await triton_service.unload_model(triton_model_name)
        
        if result["success"]:
            return {
                "success": True,
                "message": f"模型 {model.name} 卸載成功",
                "model_id": model_id,
                "triton_model_name": triton_model_name,
                "timestamp": result["timestamp"]
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"卸載模型失敗: {str(e)}")

@router.get("/models/{model_id}/status")
async def get_model_status(model_id: str):
    """
    獲取模型在Triton服務器上的狀態
    
    Args:
        model_id: 模型ID
    """
    try:
        # 獲取模型信息
        model = model_service.get_model_by_id(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"找不到模型: {model_id}")
        
        # 檢查模型是否為Triton兼容格式
        if not model.metadata or not model.metadata.get("triton_model_name"):
            return {
                "success": True,
                "model_id": model_id,
                "loaded": False,
                "ready": False,
                "reason": "模型不是Triton兼容格式"
            }
        
        triton_model_name = model.metadata.get("triton_model_name")
        
        # 檢查模型是否已準備就緒
        ready_result = await triton_service.check_model_ready(triton_model_name)
        
        return {
            "success": True,
            "model_id": model_id,
            "triton_model_name": triton_model_name,
            "loaded": ready_result["ready"],
            "ready": ready_result["ready"],
            "timestamp": ready_result["timestamp"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # 如果檢查失敗，通常表示模型未掛載
        return {
            "success": True,
            "model_id": model_id,
            "loaded": False,
            "ready": False,
            "error": str(e)
        }

@router.get("/models/{model_id}/stats")
async def get_model_stats(model_id: str):
    """
    獲取模型的性能統計信息
    
    Args:
        model_id: 模型ID
    """
    try:
        # 獲取模型信息
        model = model_service.get_model_by_id(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"找不到模型: {model_id}")
        
        # 檢查模型是否為Triton兼容格式
        if not model.metadata or not model.metadata.get("triton_model_name"):
            raise HTTPException(status_code=400, detail="模型不是Triton兼容格式，無法獲取統計信息")
        
        triton_model_name = model.metadata.get("triton_model_name")
        
        # 獲取統計信息
        result = await triton_service.get_model_stats(triton_model_name)
        
        if result["success"]:
            return {
                "success": True,
                "model_id": model_id,
                "triton_model_name": triton_model_name,
                "stats": {
                    "inference_count": result.get("inference_count", 0),
                    "avg_inference_time_ms": result.get("avg_total_inference_time_ms", 0),
                    "avg_compute_time_ms": result.get("avg_infer_time_ms", 0),
                    "success_count": result.get("success_count", 0),
                    "fail_count": result.get("fail_count", 0),
                    "last_inference": result.get("last_inference", None)
                },
                "timestamp": result["timestamp"]
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取模型統計信息失敗: {str(e)}")

@router.get("/models")
async def get_all_models_status():
    """
    獲取所有模型在Triton服務器上的狀態
    """
    try:
        result = await triton_service.get_all_models()
        
        if result["success"]:
            return {
                "success": True,
                "models": result["models"],
                "timestamp": result["timestamp"]
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取所有模型狀態失敗: {str(e)}")

@router.get("/deployment/monitoring")
async def get_deployment_monitoring():
    """
    獲取部署平台監控信息，包括所有已掛載模型的詳細信息
    """
    try:
        # 先檢查Triton健康狀態
        health_result = await triton_service.health_check()
        
        # 獲取所有已掛載模型的信息
        loaded_models = await triton_service.get_loaded_models_info()
        
        # 計算總推論次數
        total_inference_count = sum(model.get("inference_count", 0) for model in loaded_models)
        
        # 格式化監控數據
        monitoring_data = {
            "success": True,
            "triton_healthy": health_result.get("healthy", False),
            "triton_status": health_result.get("status", "unknown"),
            "loaded_models_count": len(loaded_models),
            "total_inference_count": total_inference_count,
            "models": loaded_models,
            "timestamp": datetime.now().isoformat(),
            "debug_info": {
                "health_check": health_result,
                "models_found": len(loaded_models)
            }
        }
        
        return monitoring_data
        
    except Exception as e:
        print(f"獲取部署監控信息時發生錯誤: {str(e)}")
        # 返回部分信息，即使出錯也要提供基本狀態
        return {
            "success": False,
            "triton_healthy": False,
            "triton_status": "error",
            "loaded_models_count": 0,
            "total_inference_count": 0,
            "models": [],
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/models/triton-compatible")
async def get_triton_compatible_models():
    """
    獲取所有Triton兼容的模型列表
    """
    try:
        # 獲取所有模型
        all_models = model_service.get_models()
        
        # 過濾出Triton兼容的模型
        triton_models = []
        for model in all_models:
            if (model.metadata and 
                model.metadata.get("triton_model_name") and 
                model.metadata.get("is_trt_model", False)):
                
                # 獲取模型在Triton中的狀態
                try:
                    triton_model_name = model.metadata.get("triton_model_name")
                    ready_result = await triton_service.check_model_ready(triton_model_name)
                    is_loaded = ready_result["ready"]
                except:
                    is_loaded = False
                
                triton_models.append({
                    "model_id": model.id,
                    "name": model.name,
                    "triton_model_name": model.metadata.get("triton_model_name"),
                    "format": model.format.value,
                    "type": model.type.value,
                    "precision": model.metadata.get("conversion_precision", "unknown"),
                    "is_loaded": is_loaded,
                    "created_at": model.created_at.isoformat() if model.created_at else None
                })
        
        return {
            "success": True,
            "models": triton_models,
            "total": len(triton_models)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取Triton兼容模型列表失敗: {str(e)}")

# 導入datetime模組以供使用
from datetime import datetime

@router.get("/server/metadata")
async def get_server_metadata():
    """
    獲取Triton服務器元數據信息
    """
    try:
        result = await triton_service.get_server_metadata()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取服務器元數據失敗: {str(e)}")

@router.get("/repository/stats")
async def get_repository_stats():
    """
    獲取模型倉庫統計信息
    """
    try:
        result = await triton_service.get_repository_stats()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取倉庫統計失敗: {str(e)}")

@router.get("/models/{model_name}/config")
async def get_model_config(model_name: str, version: str = "1"):
    """
    獲取特定模型的配置信息
    
    Args:
        model_name: 模型名稱
        version: 模型版本，默認為"1"
    """
    try:
        result = await triton_service.get_model_config(model_name, version)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取模型配置失敗: {str(e)}")

@router.get("/debug/api-test")
async def debug_api_test():
    """
    調試用端點：測試各種Triton API調用
    """
    try:
        results = {}
        
        # 測試健康檢查
        print("測試健康檢查...")
        results["health"] = await triton_service.health_check()
        
        # 測試服務器元數據
        print("測試服務器元數據...")
        results["server_metadata"] = await triton_service.get_server_metadata()
        
        # 測試倉庫統計
        print("測試倉庫統計...")
        results["repository_stats"] = await triton_service.get_repository_stats()
        
        # 測試所有模型
        print("測試所有模型...")
        results["all_models"] = await triton_service.get_all_models()
        
        # 測試已掛載模型信息
        print("測試已掛載模型信息...")
        results["loaded_models"] = await triton_service.get_loaded_models_info()
        
        return {
            "success": True,
            "test_results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        } 