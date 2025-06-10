from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ModelType(str, Enum):
    YOLOV8 = "yolov8"
    YOLOV8_POSE = "yolov8_pose"
    YOLOV8_SEG = "yolov8_seg"
    CUSTOM = "custom"

class ModelFormat(str, Enum):
    PT = "pt"
    ONNX = "onnx"
    ENGINE = "engine"

class ConversionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class PrecisionType(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"

class ModelInfo(BaseModel):
    """模型基本信息"""
    id: str = Field(..., description="模型ID")
    name: str = Field(..., description="模型名稱")
    type: ModelType = Field(..., description="模型類型")
    format: ModelFormat = Field(..., description="模型格式")
    path: str = Field(..., description="模型文件路徑")
    size_mb: float = Field(..., description="模型大小(MB)")
    created_at: datetime = Field(default_factory=datetime.now, description="創建時間")
    description: Optional[str] = Field(None, description="模型描述")
    metadata: Optional[Dict[str, Any]] = Field(None, description="額外元數據")
    
    model_config = {
        "protected_namespaces": ()
    }

class ConversionJob(BaseModel):
    """模型轉換任務"""
    id: str = Field(..., description="轉換任務ID")
    source_model_id: str = Field(..., description="來源模型ID")
    target_format: ModelFormat = Field(..., description="目標格式")
    precision: PrecisionType = Field(PrecisionType.FP32, description="精度類型")
    status: ConversionStatus = Field(ConversionStatus.PENDING, description="轉換狀態")
    created_at: datetime = Field(default_factory=datetime.now, description="創建時間")
    completed_at: Optional[datetime] = Field(None, description="完成時間")
    error_message: Optional[str] = Field(None, description="錯誤訊息")
    target_model_id: Optional[str] = Field(None, description="轉換後的模型ID")
    parameters: Optional[Dict[str, Any]] = Field(None, description="轉換參數")
    
    model_config = {
        "protected_namespaces": ()
    }

class ModelPerformance(BaseModel):
    """模型性能指標"""
    model_id: str = Field(..., description="模型ID")
    inference_time_ms: float = Field(..., description="推理時間(毫秒)")
    throughput: float = Field(..., description="吞吐量(FPS)")
    memory_usage_mb: float = Field(..., description="顯存使用量(MB)")
    precision: PrecisionType = Field(..., description="精度類型")
    batch_size: int = Field(1, description="批次大小")
    timestamp: datetime = Field(default_factory=datetime.now, description="測試時間")
    device_info: Dict[str, Any] = Field(..., description="設備信息")
    
    model_config = {
        "protected_namespaces": ()
    }

class ModelList(BaseModel):
    """模型列表響應"""
    models: List[ModelInfo] = Field(..., description="模型列表")
    total: int = Field(..., description="總模型數")
    
    model_config = {
        "protected_namespaces": ()
    }

class ConversionJobList(BaseModel):
    """轉換任務列表響應"""
    jobs: List[ConversionJob] = Field(..., description="轉換任務列表")
    total: int = Field(..., description="總任務數")
    
    model_config = {
        "protected_namespaces": ()
    }

class PerformanceResult(BaseModel):
    """性能測試結果響應"""
    results: List[ModelPerformance] = Field(..., description="性能測試結果")
    comparison: Optional[Dict[str, Any]] = Field(None, description="模型性能比較")
    
    model_config = {
        "protected_namespaces": ()
    } 