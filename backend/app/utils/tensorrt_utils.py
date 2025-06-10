import os
import subprocess
from typing import Dict, Any, List, Optional

def check_tensorrt_installation() -> Dict[str, Any]:
    """
    檢查TensorRT安裝狀態和版本信息
    """
    result = {
        "installed": False,
        "version": None,
        "cuda_version": None,
        "details": {}
    }
    
    try:
        import tensorrt as trt
        result["installed"] = True
        result["version"] = trt.__version__
        
        # 獲取更多TensorRT信息
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        
        result["details"]["platform_has_fp16"] = builder.platform_has_fast_fp16
        result["details"]["platform_has_int8"] = builder.platform_has_fast_int8
        result["details"]["max_workspace_size"] = builder.max_workspace_size
        result["details"]["num_DLA_cores"] = builder.num_DLA_cores
        
        # 獲取CUDA版本
        try:
            import torch
            result["cuda_version"] = torch.version.cuda
        except ImportError:
            pass
            
    except ImportError:
        result["error"] = "TensorRT not installed"
    except Exception as e:
        result["error"] = str(e)
    
    return result

def run_trtexec_benchmark(model_path: str, batch_size: int = 1, 
                         iterations: int = 100, precision: str = "fp32") -> Dict[str, Any]:
    """
    使用trtexec運行模型基準測試
    
    Args:
        model_path: TensorRT引擎文件路徑
        batch_size: 批次大小
        iterations: 迭代次數
        precision: 精度類型 (fp32, fp16, int8)
    
    Returns:
        包含基準測試結果的字典
    """
    results = {
        "success": False,
        "throughput": 0,
        "latency_mean_ms": 0,
        "latency_median_ms": 0,
        "latency_percentile_99_ms": 0,
        "error": None
    }
    
    try:
        # 檢查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        
        # 構建trtexec命令
        cmd = [
            "trtexec",
            f"--loadEngine={model_path}",
            f"--batch={batch_size}",
            f"--iterations={iterations}",
            "--avgRuns=10",
            "--warmUp=5"
        ]
        
        # 執行命令
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        # 解析輸出結果
        output = stdout.decode('utf-8')
        
        if process.returncode != 0:
            results["error"] = stderr.decode('utf-8')
            return results
        
        # 從輸出中提取性能數據
        for line in output.split('\n'):
            if "Throughput" in line:
                parts = line.split()
                if len(parts) >= 2:
                    results["throughput"] = float(parts[-2])
            
            elif "Latency" in line:
                if "mean" in line.lower():
                    parts = line.split()
                    if len(parts) >= 2:
                        results["latency_mean_ms"] = float(parts[-2])
                
                elif "median" in line.lower():
                    parts = line.split()
                    if len(parts) >= 2:
                        results["latency_median_ms"] = float(parts[-2])
                
                elif "percentile" in line.lower() and "99%" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        results["latency_percentile_99_ms"] = float(parts[-2])
        
        results["success"] = True
    
    except Exception as e:
        results["error"] = str(e)
    
    return results

def get_trt_layer_info(engine_path: str) -> List[Dict[str, Any]]:
    """
    獲取TensorRT引擎的層信息
    
    Args:
        engine_path: TensorRT引擎文件路徑
    
    Returns:
        包含每層信息的列表
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # 載入引擎
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        # 獲取引擎信息
        layers_info = []
        
        # 獲取綁定信息（輸入和輸出）
        for i in range(engine.num_bindings):
            binding_info = {
                "name": engine.get_binding_name(i),
                "is_input": engine.binding_is_input(i),
                "data_type": str(engine.get_binding_dtype(i)),
                "shape": engine.get_binding_shape(i)
            }
            layers_info.append(binding_info)
        
        # 清理
        del engine
        
        return layers_info
    
    except Exception as e:
        print(f"獲取TensorRT層信息時出錯: {str(e)}")
        return []

def generate_trt_config(input_model: str, output_path: str, precision: str = "fp32",
                      batch_size: int = 1, workspace: int = 4) -> Optional[str]:
    """
    生成TensorRT配置文件
    
    Args:
        input_model: 源模型文件路徑 (ONNX)
        output_path: 輸出配置文件路徑
        precision: 精度類型 (fp32, fp16)
        batch_size: 批次大小
        workspace: 工作空間大小(GB)
    
    Returns:
        配置文件路徑，如果失敗則返回None
    """
    try:
        # 創建基本配置
        config_str = f"""
# TensorRT Engine配置文件
model_type: "yolov8"
input_model: "{input_model}"
output_path: "{output_path}"
precision: "{precision}"
batch_size: {batch_size}
workspace_size: {workspace}
dynamic_shapes:
  min: [1, 3, 640, 640]
  opt: [{batch_size}, 3, 640, 640]
  max: [{batch_size*2}, 3, 1280, 1280]
optimizations:
  tf32: true
  refittable: false
  restricted: false
  builder_optimization_level: 3
        """
        
        # 寫入配置文件
        config_file_path = os.path.join(os.path.dirname(output_path), "trt_config.yaml")
        with open(config_file_path, 'w', encoding='utf-8') as f:
            f.write(config_str.strip())
        
        return config_file_path
        
    except Exception as e:
        print(f"生成TensorRT配置時出錯: {str(e)}")
        return None 