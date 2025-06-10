from app.models.model import (
    ModelType, 
    ModelFormat, 
    ConversionStatus, 
    PrecisionType, 
    ModelInfo, 
    ConversionJob, 
    ModelPerformance,
    ModelList,
    ConversionJobList,
    PerformanceResult
)

# 設置Pydantic全局配置以禁用protected_namespaces警告
import pydantic
pydantic.config.ConfigDict = {"protected_namespaces": ()} 