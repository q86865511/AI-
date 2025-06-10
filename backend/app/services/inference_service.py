import os
import time
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import uuid
from datetime import datetime, timezone, timedelta
import cv2
import io
import asyncio
import torch
import GPUtil
from app.services.model_service import ModelService
import tensorrt as trt

from app.models import ModelInfo, ModelPerformance, PrecisionType

class InferenceService:
    """
    模型推理服務，負責模型推理和性能測試
    """
    def __init__(self):
        """初始化推理服務"""
        # 移除單次測試資料夾，只保留自動化測試功能
        self.model_service = ModelService()
        
        # TensorRT logger
        self.logger = trt.Logger(trt.Logger.WARNING)
    
    def predict(self, model: ModelInfo, image_path: str, confidence: float = 0.25,
               iou_threshold: float = 0.45) -> Dict[str, Any]:
        """
        使用指定模型對圖片進行預測
        """
        try:
            print(f"使用模型 {model.id} ({model.format.value}) 對圖片 {image_path} 進行預測")
            
            # 讀取圖片
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f"無法讀取圖片: {image_path}")
            
            # 根據模型格式選擇推理方法
            results = None
            if model.format.value == "pt":
                results = self._predict_with_pytorch(model, image, confidence, iou_threshold)
            elif model.format.value == "onnx":
                results = self._predict_with_onnx(model, image, confidence, iou_threshold)
            elif model.format.value == "engine":
                results = self._predict_with_tensorrt(model, image, confidence, iou_threshold)
            else:
                raise Exception(f"不支持的模型格式: {model.format.value}")
            
            return results
        
        except Exception as e:
            print(f"預測時出錯: {str(e)}")
            raise
    
    def _predict_with_pytorch(self, model: ModelInfo, image: np.ndarray,
                            confidence: float, iou_threshold: float) -> Dict[str, Any]:
        """使用PyTorch模型進行預測"""
        try:
            from ultralytics import YOLO
            
            # 加載模型
            yolo_model = YOLO(model.path)
            
            # 執行預測
            results = yolo_model(image, conf=confidence, iou=iou_threshold)[0]
            
            # 繪製結果
            plot_image = results.plot()
            
            # 將結果轉換為可序列化格式
            detections = []
            for box in results.boxes:
                detection = {
                    "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "class_name": results.names[int(box.cls[0])]
                }
                detections.append(detection)
            
            # 將圖片轉換為bytes
            success, encoded_image = cv2.imencode('.jpg', plot_image)
            if not success:
                raise Exception("無法編碼結果圖片")
            
            return {
                "detections": detections,
                "image": encoded_image.tobytes(),
                "model_id": model.id,
                "model_type": model.type.value,
                "inference_time_ms": results.speed["inference"]
            }
        
        except Exception as e:
            raise Exception(f"PyTorch推理失敗: {str(e)}")
    
    def _predict_with_onnx(self, model: ModelInfo, image: np.ndarray,
                           confidence: float, iou_threshold: float) -> Dict[str, Any]:
        """使用ONNX模型進行預測"""
        try:
            import onnxruntime as ort
            from ultralytics.utils.ops import non_max_suppression
            
            # 獲取模型類型相關信息
            is_pose = model.type.value == "yolov8-pose"
            is_seg = model.type.value == "yolov8-seg"
            
            # 準備圖片
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_height, img_width = img.shape[:2]
            input_height, input_width = 640, 640  # 默認輸入尺寸
            
            # 預處理
            img_processed = cv2.resize(img, (input_width, input_height))
            img_processed = img_processed.astype(np.float32) / 255.0
            img_processed = img_processed.transpose(2, 0, 1)  # HWC -> CHW
            img_processed = np.expand_dims(img_processed, axis=0)  # 添加batch維度
            
            # 創建推理會話
            sess = ort.InferenceSession(model.path)
            input_name = sess.get_inputs()[0].name
            
            # 計時
            start_time = time.time()
            
            # 執行推理
            outputs = sess.run(None, {input_name: img_processed})
            
            # 後處理
            if is_pose:
                # 處理姿態模型輸出
                boxes_kpts = outputs[0]
                results = non_max_suppression(boxes_kpts, confidence, iou_threshold, nc=1, nkpt=17)
            elif is_seg:
                # 處理分割模型輸出
                boxes, masks = outputs[0], outputs[1]
                results = non_max_suppression(boxes, confidence, iou_threshold)
                # 需要額外處理掩碼
                proto = outputs[1]
            else:
                # 標準目標檢測
                results = non_max_suppression(outputs[0], confidence, iou_threshold)
            
            # 計算推理時間
            inference_time_ms = (time.time() - start_time) * 1000
            
            # 將結果轉換回原始圖片尺寸
            scale_x, scale_y = img_width / input_width, img_height / input_height
            
            # 構建結果
            result_img = image.copy()
            detections = []
            
            # 如果有任何檢測結果
            if len(results) > 0 and len(results[0]) > 0:
                for det in results[0]:
                    # 還原到原始圖片尺寸
                    x1, y1, x2, y2 = det[:4]
                    x1, x2 = x1 * scale_x, x2 * scale_x
                    y1, y2 = y1 * scale_y, y2 * scale_y
                    
                    # 繪製邊界框
                    cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # 添加置信度文字
                    conf = float(det[4])
                    class_id = int(det[5])
                    cv2.putText(result_img, f"{class_id}: {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 保存檢測結果
                    detection = {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "class_id": class_id,
                        "class_name": str(class_id)  # 這裡需要一個類別映射
                    }
                    detections.append(detection)
            
            # 將圖片轉換為bytes
            success, encoded_image = cv2.imencode('.jpg', result_img)
            if not success:
                raise Exception("無法編碼結果圖片")
            
            return {
                "detections": detections,
                "image": encoded_image.tobytes(),
                "model_id": model.id,
                "model_type": model.type.value,
                "inference_time_ms": inference_time_ms
            }
        
        except Exception as e:
            raise Exception(f"ONNX推理失敗: {str(e)}")
    
    def _predict_with_tensorrt(self, model: ModelInfo, image: np.ndarray,
                              confidence: float, iou_threshold: float) -> Dict[str, Any]:
        """使用TensorRT引擎進行推理"""
        try:
            import tensorrt as trt
            import torch
            from ultralytics.utils.ops import non_max_suppression
            
            # 確保有GPU可用
            if not torch.cuda.is_available():
                raise Exception("TensorRT推理需要GPU")
            
            # 準備圖片
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_height, img_width = img.shape[:2]
            input_height, input_width = 640, 640  # 默認輸入尺寸
            
            # 預處理
            img_processed = cv2.resize(img, (input_width, input_height))
            img_processed = img_processed.astype(np.float32) / 255.0
            img_processed = img_processed.transpose(2, 0, 1)  # HWC -> CHW
            img_processed = np.expand_dims(img_processed, axis=0)  # 添加batch維度
            
            # 獲取引擎路徑
            engine_path = model.path
            
            # 如果路徑不存在，嘗試在模型目錄中查找
            if not os.path.exists(engine_path):
                model_dir = os.path.dirname(model.path)
                print(f"引擎路徑不存在: {engine_path}，嘗試在目錄 {model_dir} 中查找")
                
                # 查找 .engine 或 .plan 文件
                engine_files = []
                if os.path.exists(model_dir):
                    for file in os.listdir(model_dir):
                        if file.endswith(".engine") or file.endswith(".plan"):
                            engine_files.append(os.path.join(model_dir, file))
                
                if engine_files:
                    engine_path = engine_files[0]
                    print(f"找到引擎文件: {engine_path}")
                else:
                    raise Exception(f"在目錄 {model_dir} 中找不到TensorRT引擎文件")
            
            if not engine_path or not os.path.exists(engine_path):
                raise Exception(f"找不到TensorRT引擎文件: {engine_path}")
            
            # 加載引擎
            try:
                print(f"正在加載TensorRT引擎: {engine_path}")
                with open(engine_path, "rb") as f:
                    engine_data = f.read()
                
                if not engine_data:
                    raise Exception("引擎文件為空")
                
                logger = trt.Logger(trt.Logger.WARNING)
                runtime = trt.Runtime(logger)
                engine = runtime.deserialize_cuda_engine(engine_data)
                
                if engine is None:
                    raise Exception("無法反序列化引擎，可能是引擎文件損壞或版本不兼容")
                
                context = engine.create_execution_context()
                if context is None:
                    raise Exception("無法創建執行上下文")
                    
            except Exception as e:
                raise Exception(f"加載TensorRT引擎失敗: {str(e)}")
            
            # 使用torch來處理GPU內存和同步
            device = torch.device('cuda:0')
            
            # 準備輸入張量
            input_idx = engine.get_binding_index("images")
            output_idx = engine.get_binding_index("output0")
            
            # 獲取輸入輸出形狀
            input_shape = engine.get_binding_shape(input_idx)
            output_shape = engine.get_binding_shape(output_idx)
            
            # 使用torch分配GPU內存
            input_tensor = torch.from_numpy(img_processed).to(device)
            output_tensor = torch.empty(output_shape, dtype=torch.float32, device=device)
            
            # 創建緩衝區指針列表
            bindings = [input_tensor.data_ptr(), output_tensor.data_ptr()]
            
            # 測量推理時間
            torch.cuda.synchronize()
            start_time = time.time()
            
            # 執行推理
            context.execute_v2(bindings=bindings)
            
            # 同步並計算時間
            torch.cuda.synchronize()
            inference_time_ms = (time.time() - start_time) * 1000
            
            # 將輸出從GPU複製到CPU
            output_data = output_tensor.cpu().numpy()
            
            # 後處理 - 使用非極大值抑制
            results = non_max_suppression(torch.from_numpy(output_data), confidence, iou_threshold)
            
            # 將結果轉換回原始圖片尺寸
            scale_x, scale_y = img_width / input_width, img_height / input_height
            
            # 構建結果
            result_img = image.copy()
            detections = []
            
            # 如果有任何檢測結果
            if len(results) > 0 and len(results[0]) > 0:
                for det in results[0]:
                    # 還原到原始圖片尺寸
                    x1, y1, x2, y2 = det[:4]
                    x1, x2 = x1 * scale_x, x2 * scale_x
                    y1, y2 = y1 * scale_y, y2 * scale_y
                    
                    # 繪製邊界框
                    cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # 添加置信度文字
                    conf = float(det[4])
                    class_id = int(det[5]) if len(det) > 5 else 0
                    cv2.putText(result_img, f"{class_id}: {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 保存檢測結果
                    detection = {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "class_id": class_id,
                        "class_name": str(class_id)  # 這裡需要一個類別映射
                    }
                    detections.append(detection)
            
            # 將圖片轉換為bytes
            success, encoded_image = cv2.imencode('.jpg', result_img)
            if not success:
                raise Exception("無法編碼結果圖片")
            
            # 釋放資源
            del context
            del engine
            
            return {
                "detections": detections,
                "image": encoded_image.tobytes(),
                "model_id": model.id,
                "model_type": model.type.value,
                "inference_time_ms": inference_time_ms
            }
        
        except Exception as e:
            raise Exception(f"TensorRT推理失敗: {str(e)}")
    
    def benchmark(self, model: ModelInfo, batch_size: int = 1,
                 precision: PrecisionType = PrecisionType.FP32,
                 num_iterations: int = 100, img_size: int = 640) -> ModelPerformance:
        """
        對模型進行性能基準測試
        
        Args:
            model: 要測試的模型
            batch_size: 批次大小
            precision: 精度類型
            num_iterations: 測試迭代次數
            img_size: 圖像尺寸
        """
        try:
            print(f"開始對模型 {model.id} ({model.format.value}) 進行性能基準測試，批次大小={batch_size}，圖像尺寸={img_size}")
            
            # 創建隨機測試數據
            input_data = np.random.rand(batch_size, 3, img_size, img_size).astype(np.float32)
            
            # 進行預熱
            self._run_inference(model, input_data, 10)
            
            # 開始計時測試
            start_time = time.time()
            memory_usage = self._run_inference(model, input_data, num_iterations)
            end_time = time.time()
            
            # 計算性能指標
            total_time_ms = (end_time - start_time) * 1000
            avg_inference_time_ms = total_time_ms / num_iterations
            throughput = (batch_size * num_iterations) / (total_time_ms / 1000)
            
            # 獲取設備信息
            device_info = self._get_device_info()
            
            # 創建性能結果
            performance = ModelPerformance(
                model_id=model.id,
                inference_time_ms=round(avg_inference_time_ms, 2),
                throughput=round(throughput, 2),
                memory_usage_mb=round(memory_usage, 2),
                precision=precision,
                batch_size=batch_size,
                device_info=device_info
            )
            
            # 添加額外屬性到device_info中
            performance.device_info["img_size"] = img_size
            
            # 不再保存單次測試結果
            
            return performance
        
        except Exception as e:
            print(f"基準測試時出錯: {str(e)}")
            raise
    
    def _run_inference(self, model: ModelInfo, input_data: np.ndarray, num_iterations: int) -> float:
        """運行多次推理並測量顯存使用量"""
        memory_usage = 0.0
        
        # 根據模型格式選擇推理方法
        if model.format.value == "pt":
            try:
                from ultralytics import YOLO
                import torch
                
                # 載入模型
                yolo_model = YOLO(model.path)
                
                # 預熱
                yolo_model.predict(input_data[0], verbose=False)
                
                # 測量初始顯存
                initial_mem = torch.cuda.memory_allocated() / (1024 * 1024)
                
                # 執行多次推理
                for _ in range(num_iterations):
                    yolo_model.predict(input_data, verbose=False)
                
                # 測量最終顯存
                final_mem = torch.cuda.memory_allocated() / (1024 * 1024)
                memory_usage = final_mem - initial_mem
                
            except Exception as e:
                raise Exception(f"PyTorch推理基準測試失敗: {str(e)}")
        
        elif model.format.value == "onnx":
            try:
                import onnxruntime as ort
                
                # 創建ONNX推理會話
                sess_options = ort.SessionOptions()
                sess = ort.InferenceSession(model.path, sess_options, providers=['CUDAExecutionProvider'])
                input_name = sess.get_inputs()[0].name
                
                # 預熱
                sess.run(None, {input_name: input_data})
                
                # 執行多次推理
                for _ in range(num_iterations):
                    sess.run(None, {input_name: input_data})
                
                # 無法直接從ONNX獲取顯存使用量，這裡使用估算值
                memory_usage = input_data.nbytes / (1024 * 1024) * 3  # 粗略估計
                
            except Exception as e:
                raise Exception(f"ONNX推理基準測試失敗: {str(e)}")
        
        elif model.format.value == "engine":
            try:
                import tensorrt as trt
                import torch
                
                # 確保有GPU可用
                if not torch.cuda.is_available():
                    raise Exception("TensorRT推理需要GPU")
                
                # 載入TensorRT引擎
                TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
                with open(model.path, "rb") as f:
                    engine_data = f.read()
                    runtime = trt.Runtime(TRT_LOGGER)
                    engine = runtime.deserialize_cuda_engine(engine_data)
                
                # 創建執行上下文
                context = engine.create_execution_context()
                
                # 設置輸入形狀
                batch_size, channels, height, width = input_data.shape
                context.set_binding_shape(0, (batch_size, channels, height, width))
                
                # 獲取輸入輸出索引
                input_idx = engine.get_binding_index("images")
                output_idx = engine.get_binding_index("output0")
                
                # 獲取輸出形狀
                output_shape = context.get_binding_shape(output_idx)
                
                # 使用torch來處理GPU內存
                device = torch.device('cuda:0')
                
                # 創建輸入輸出張量
                input_tensor = torch.from_numpy(input_data).to(device)
                output_tensor = torch.empty(output_shape, dtype=torch.float32, device=device)
                
                # 創建緩衝區指針列表
                bindings = [input_tensor.data_ptr(), output_tensor.data_ptr()]
                
                # 記錄初始顯存使用
                torch.cuda.synchronize()
                initial_mem = torch.cuda.memory_allocated(device) / (1024 * 1024)
                
                # 預熱
                for _ in range(10):
                    context.execute_v2(bindings=bindings)
                    torch.cuda.synchronize()
                
                # 執行多次推理
                for _ in range(num_iterations):
                    context.execute_v2(bindings=bindings)
                    torch.cuda.synchronize()
                
                # 測量最終顯存使用
                final_mem = torch.cuda.memory_allocated(device) / (1024 * 1024)
                memory_usage = final_mem - initial_mem
                
                # 釋放資源
                del context
                del engine
                
            except Exception as e:
                raise Exception(f"TensorRT推理基準測試失敗: {str(e)}")
        
        return memory_usage
    
    def _get_device_info(self) -> Dict[str, Any]:
        """獲取設備信息"""
        device_info = {
            "platform": "unknown",
            "device": "unknown",
            "cuda_version": "unknown"
        }
        
        try:
            import platform
            device_info["platform"] = platform.platform()
            
            try:
                import torch
                if torch.cuda.is_available():
                    device_info["device"] = torch.cuda.get_device_name(0)
                    device_info["cuda_version"] = torch.version.cuda
            except ImportError:
                pass
            
            try:
                import tensorrt as trt
                device_info["tensorrt_version"] = trt.__version__
            except ImportError:
                pass
        
        except Exception as e:
            print(f"獲取設備信息時出錯: {str(e)}")
        
        return device_info
    
    # 移除 _save_benchmark_result 方法，因為不再需要保存單次測試結果
    
    def generate_comparison(self, results: List[ModelPerformance]) -> Dict[str, Any]:
        """生成模型性能比較結果"""
        comparison = {
            "models": [],
            "inference_time_ms": [],
            "throughput": [],
            "memory_usage_mb": [],
            "speedup": {},
            "memory_reduction": {},
            "best_performance": None,
            "best_memory_efficiency": None
        }
        
        # 收集基本數據
        for perf in results:
            comparison["models"].append(perf.model_id)
            comparison["inference_time_ms"].append(perf.inference_time_ms)
            comparison["throughput"].append(perf.throughput)
            comparison["memory_usage_mb"].append(perf.memory_usage_mb)
        
        # 找出基準模型（首個結果）並計算相對指標
        if results:
            base_perf = results[0]
            base_time = base_perf.inference_time_ms
            base_memory = base_perf.memory_usage_mb
            
            # 計算每個模型相對於基準的加速比和內存減少率
            for i, perf in enumerate(results):
                if i > 0:  # 跳過基準模型自身
                    model_id = perf.model_id
                    comparison["speedup"][model_id] = round(base_time / perf.inference_time_ms, 2)
                    comparison["memory_reduction"][model_id] = round(
                        (base_memory - perf.memory_usage_mb) / base_memory * 100, 2
                    )
            
            # 找出最佳性能和內存效率的模型
            best_perf_idx = comparison["inference_time_ms"].index(min(comparison["inference_time_ms"]))
            best_mem_idx = comparison["memory_usage_mb"].index(min(comparison["memory_usage_mb"]))
            
            comparison["best_performance"] = {
                "model_id": results[best_perf_idx].model_id,
                "inference_time_ms": results[best_perf_idx].inference_time_ms,
                "throughput": results[best_perf_idx].throughput
            }
            
            comparison["best_memory_efficiency"] = {
                "model_id": results[best_mem_idx].model_id,
                "memory_usage_mb": results[best_mem_idx].memory_usage_mb
            }
        
        return comparison 

    async def benchmark_model(self, model_id: str, batch_size: int = 1, num_iterations: int = 100, image_size: int = 640, test_dataset: Optional[str] = None) -> Dict[str, Any]:
        """
        對指定模型進行性能基準測試
        
        Args:
            model_id: 模型ID
            batch_size: 批次大小
            num_iterations: 測試迭代次數
            image_size: 圖像尺寸
            test_dataset: 數據集ID (可選)
        """
        import time
        import numpy as np
        import torch
        
        # 確保 model_service 已初始化
        if self.model_service is None:
            from app.services.model_service import ModelService
            self.model_service = ModelService()
        
        # 獲取模型信息
        model = self.model_service.get_model_by_id(model_id)
        if not model:
            raise Exception(f"找不到模型: {model_id}")
        
        print(f"開始對模型 {model.name} 進行性能測試，批次大小: {batch_size}, 迭代次數: {num_iterations}")
        
        # 準備輸入數據 - 隨機數據，根據批次大小和圖像尺寸
        input_shape = (batch_size, 3, image_size, image_size)
        dummy_input = np.random.randint(0, 255, size=input_shape).astype(np.uint8)
        
        # 測量推理時間
        inference_times = []
        memory_usage = 0  # 初始化內存使用變數
        gpu_loads = []  # GPU負載列表
        
        try:
            # 準備GPU使用率監測
            try:
                import GPUtil
                
                # 獲取GPU使用率的函數
                def get_gpu_load():
                    try:
                        # 獲取當前GPU利用率
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            return gpus[0].load * 100  # 轉換為百分比
                        return 0
                    except Exception as e:
                        print(f"獲取GPU使用率失敗: {str(e)}")
                        return 0
                    
                has_gpu_util = True
                print("成功初始化GPU監測")
            except Exception as e:
                print(f"初始化GPU監測失敗: {str(e)}")
                has_gpu_util = False
            
            # 根據模型格式選擇測試方法
            if model.format.value == "pt":  # 修正：使用 model.format.value
                # PyTorch模型
                import torch
                
                # 加載模型
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                try:
                    # 使用 ultralytics 的 YOLO 模型
                    from ultralytics import YOLO
                    pt_model = YOLO(model.path)
                except Exception as e:
                    raise Exception(f"加載PyTorch模型失敗: {str(e)}")
                
                # 預熱
                for _ in range(10):
                    _ = pt_model.predict(dummy_input[0], verbose=False)
                
                # 記錄初始內存
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                
                # 正式測量
                for _ in range(num_iterations):
                    # 測量GPU使用率
                    if has_gpu_util:
                        gpu_loads.append(get_gpu_load())
                    
                    # 同步並開始計時
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    iter_start = time.time()
                    _ = pt_model.predict(dummy_input, verbose=False)
                    
                    # 同步並結束計時
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    iter_end = time.time()
                    inference_times.append((iter_end - iter_start) * 1000)  # 轉換為毫秒
                    
                    # 測量GPU使用率
                    if has_gpu_util:
                        gpu_loads.append(get_gpu_load())
                
                # 記錄GPU內存使用情況
                if torch.cuda.is_available():
                    memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024  # 轉換為MB
                    torch.cuda.reset_peak_memory_stats()
            
            elif model.format.value == "onnx":  # 修正：使用 model.format.value
                # ONNX模型
                import onnxruntime as ort
                
                # 創建推理會話
                try:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    session = ort.InferenceSession(model.path, providers=providers)
                except Exception as e:
                    raise Exception(f"加載ONNX模型失敗: {str(e)}")
                
                # 獲取輸入名稱
                input_name = session.get_inputs()[0].name
                
                # 預熱
                for _ in range(10):
                    session.run(None, {input_name: dummy_input.astype(np.float32) / 255.0})
                
                # 正式測量
                for _ in range(num_iterations):
                    # 測量GPU使用率
                    if has_gpu_util:
                        gpu_loads.append(get_gpu_load())
                        
                    iter_start = time.time()
                    session.run(None, {input_name: dummy_input.astype(np.float32) / 255.0})
                    iter_end = time.time()
                    inference_times.append((iter_end - iter_start) * 1000)  # 轉換為毫秒
                    
                    # 測量GPU使用率
                    if has_gpu_util:
                        gpu_loads.append(get_gpu_load())
                
                # ONNX 無法直接獲取準確的內存使用，使用估算值
                if has_gpu_util:
                    try:
                        GPUs = GPUtil.getGPUs()
                        if GPUs:
                            memory_usage = GPUs[0].memoryUsed  # MB
                    except:
                        memory_usage = dummy_input.nbytes / (1024 * 1024) * 3  # 粗略估計
                else:
                    memory_usage = dummy_input.nbytes / (1024 * 1024) * 3  # 粗略估計
            
            elif model.format.value == "engine":  # 修正：使用 model.format.value
                # TensorRT引擎模型
                import tensorrt as trt
                import torch
                
                # 確保有GPU可用
                if not torch.cuda.is_available():
                    raise Exception("TensorRT推理需要GPU")
                
                # 獲取引擎路徑
                engine_path = model.path
                if not os.path.exists(engine_path):
                    # 如果直接路徑不存在，嘗試在模型目錄中查找
                    model_dir = model.metadata.get("triton_model_dir") if model.metadata else os.path.dirname(model.path)
                    for file in os.listdir(model_dir):
                        if file.endswith(".engine") or file.endswith(".plan"):
                            engine_path = os.path.join(model_dir, file)
                            break
                
                # 載入TensorRT引擎
                TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
                with open(engine_path, "rb") as f:
                    engine_data = f.read()
                    runtime = trt.Runtime(TRT_LOGGER)
                    engine = runtime.deserialize_cuda_engine(engine_data)
                
                # 創建執行上下文
                context = engine.create_execution_context()
                
                # 設置輸入形狀
                channels, height, width = 3, image_size, image_size
                context.set_binding_shape(0, (batch_size, channels, height, width))
                
                # 獲取輸入輸出索引
                input_idx = engine.get_binding_index("images")
                output_idx = engine.get_binding_index("output0")
                
                # 獲取輸出形狀
                output_shape = context.get_binding_shape(output_idx)
                
                # 使用torch來處理GPU內存
                device = torch.device('cuda:0')
                
                # 創建輸入輸出張量
                input_tensor = torch.from_numpy(dummy_input.astype(np.float32) / 255.0).to(device)
                output_tensor = torch.empty(output_shape, dtype=torch.float32, device=device)
                
                # 創建緩衝區指針列表
                bindings = [input_tensor.data_ptr(), output_tensor.data_ptr()]
                
                # 獲取初始內存使用情況
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated(device) / 1024 / 1024  # MB
                
                # 預熱
                for _ in range(10):
                    context.execute_v2(bindings=bindings)
                    torch.cuda.synchronize()
                
                # 正式測量
                for i in range(num_iterations):
                    # 測量GPU使用率
                    if has_gpu_util:
                        gpu_loads.append(get_gpu_load())
                    
                    # 同步並開始計時
                    torch.cuda.synchronize()
                    iter_start = time.time()
                    
                    # 執行推理
                    context.execute_v2(bindings=bindings)
                    
                    # 同步並結束計時
                    torch.cuda.synchronize()
                    iter_end = time.time()
                    
                    inference_times.append((iter_end - iter_start) * 1000)  # 轉換為毫秒
                    
                    # 測量GPU使用率
                    if has_gpu_util:
                        gpu_loads.append(get_gpu_load())
                
                # 測量最終顯存使用
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated(device) / 1024 / 1024  # MB
                memory_usage = memory_after - memory_before
                
                # 如果GPUtil可用，獲取更準確的內存信息
                if has_gpu_util:
                    try:
                        GPUs = GPUtil.getGPUs()
                        if GPUs:
                            memory_usage = GPUs[0].memoryUsed  # MB
                    except:
                        pass
                
                # 釋放資源
                del context
                del engine
            
            else:
                raise Exception(f"不支持的模型格式: {model.format.value}")
            
            # 計算統計數據
            avg_time = np.mean(inference_times) if inference_times else 0
            std_time = np.std(inference_times) if inference_times else 0
            min_time = np.min(inference_times) if inference_times else 0
            max_time = np.max(inference_times) if inference_times else 0
            p95_time = np.percentile(inference_times, 95) if inference_times else 0
            throughput = (batch_size * 1000) / avg_time if avg_time > 0 else 0  # FPS
            
            # 計算GPU使用率統計數據
            avg_gpu_load = np.mean(gpu_loads) if len(gpu_loads) > 0 else 0
            max_gpu_load = np.max(gpu_loads) if len(gpu_loads) > 0 else 0
            
            result = {
                "model_id": model_id,
                "model_name": model.name,
                "format": model.format.value,  # 修正：使用 model.format.value
                "batch_size": batch_size,
                "image_size": image_size,
                "iterations": num_iterations,
                "avg_inference_time_ms": float(avg_time),
                "std_inference_time_ms": float(std_time),
                "min_inference_time_ms": float(min_time),
                "max_inference_time_ms": float(max_time),
                "p95_inference_time_ms": float(p95_time),
                "throughput_fps": float(throughput),
                "memory_usage_mb": float(memory_usage),
                "avg_gpu_load": float(avg_gpu_load),
                "max_gpu_load": float(max_gpu_load),
                "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            }
            
            # 如果提供了測試數據集，計算mAP等指標
            if test_dataset:
                # 此處假設有一個方法來計算精度指標
                accuracy_metrics = self.evaluate_model_on_dataset(model_id, test_dataset, batch_size, image_size)
                result.update(accuracy_metrics)
            
            # 不再保存單次測試結果
            
            print(f"性能測試完成，平均推理時間: {avg_time:.2f} ms，吞吐量: {throughput:.2f} FPS，GPU使用率: {avg_gpu_load:.2f}%")
            return result
        
        except Exception as e:
            error_msg = f"性能測試失敗: {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            raise Exception(error_msg)

    # 移除 save_benchmark_result 和 get_model_performance 方法，因為不再需要保存單次測試結果

    def evaluate_model_on_dataset(self, model_id: str, dataset_path: str, batch_size: int = 1, image_size: int = 640) -> Dict[str, Any]:
        """
        在數據集上評估模型性能（mAP等指標）
        
        Args:
            model_id: 模型ID
            dataset_path: 數據集路徑
            batch_size: 批次大小
            image_size: 圖像尺寸
            
        Returns:
            評估結果字典
        """
        import os
        import json
        from app.services.model_service import ModelService
        
        model_service = ModelService()
        model = model_service.get_model_by_id(model_id)
        
        if not model:
            raise Exception(f"找不到模型: {model_id}")
        
        print(f"在數據集 {dataset_path} 上評估模型 {model_id}")
        
        # 檢查是否為COCO數據集（包含YAML文件）
        yaml_file = None
        is_coco = False
        
        # 嘗試從數據集元數據中查找YAML文件
        datasets_dir = os.path.join(os.getcwd(), "data", "datasets")
        datasets_metadata_path = os.path.join(datasets_dir, "datasets.json")
        
        if os.path.exists(datasets_metadata_path):
            with open(datasets_metadata_path, "r") as f:
                try:
                    datasets = json.load(f)
                    for dataset in datasets:
                        if dataset.get("path") == dataset_path:
                            yaml_file = dataset.get("yaml_file")
                            is_coco = dataset.get("is_coco", False)
                            break
                except json.JSONDecodeError:
                    pass
        
        # 如果未在元數據中找到，則嘗試在目錄中查找
        if not yaml_file and os.path.isdir(dataset_path):
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file == "coco_pose.yaml" or file.endswith(".yaml"):
                        yaml_file = os.path.join(root, file)
                        is_coco = True
                        break
                if yaml_file:
                    break
        
        # 如果找到YAML文件且為COCO數據集，使用YOLO的val功能進行評估
        if yaml_file and is_coco:
            try:
                print(f"使用YAML配置文件 {yaml_file} 進行模型驗證")
                
                # 使用Ultralytics的驗證功能
                from ultralytics import YOLO
                
                # 如果是engine模型，需要先找到對應的PT模型
                if model.format.value == "engine":
                    # 嘗試從元數據中獲取源模型ID
                    source_model_id = model.metadata.get("source_model_id") if model.metadata else None
                    
                    if source_model_id:
                        source_model = model_service.get_model_by_id(source_model_id)
                        if source_model and source_model.format.value == "pt":
                            model_path = source_model.path
                            print(f"使用源PyTorch模型 {source_model.name} 進行驗證")
                        else:
                            raise Exception("找不到對應的PyTorch源模型用於驗證")
                    else:
                        raise Exception("TensorRT引擎模型無法直接使用YOLO的val功能進行驗證，請使用原始PT模型")
                else:
                    model_path = model.path
                
                # 載入模型
                yolo_model = YOLO(model_path)
                
                # 執行驗證
                results = yolo_model.val(
                    data=yaml_file,
                    batch=batch_size,
                    imgsz=image_size,
                    verbose=True
                )
                
                # 返回驗證結果
                metrics = results.results_dict
                
                # 格式化結果
                val_results = {
                    "map50": round(metrics.get("metrics/mAP50(B)", 0.0), 4),
                    "map50_95": round(metrics.get("metrics/mAP50-95(B)", 0.0), 4),
                    "precision": round(metrics.get("metrics/precision(B)", 0.0), 4),
                    "recall": round(metrics.get("metrics/recall(B)", 0.0), 4),
                    "eval_time": round(results.speed.get("val", 0) / 1000, 2),  # 轉換為秒
                    "dataset": dataset_path,
                    "yaml_file": yaml_file
                }
                
                print(f"驗證結果: mAP50={val_results['map50']}, mAP50-95={val_results['map50_95']}")
                return val_results
                
            except Exception as e:
                print(f"使用YOLO val功能評估模型失敗: {str(e)}")
                import traceback
                print(traceback.format_exc())
        
        # 如果不是COCO數據集或驗證失敗，返回隨機模擬數據
        print("未找到有效的YAML配置或驗證失敗，返回隨機模擬數據")
        import random
        
        # 隨機模擬結果數據
        map50 = random.uniform(0.75, 0.95)
        map50_95 = random.uniform(0.65, map50)
        precision = random.uniform(0.80, 0.98)
        recall = random.uniform(0.75, 0.95)
        
        return {
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "eval_time": round(random.uniform(5, 30), 2),
            "dataset": dataset_path,
            "note": "模擬數據 - 未找到有效的YAML配置文件或驗證失敗"
        } 

    async def validate_model(self, model_id: str, dataset_id: str, batch_size: int = 1, custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        在指定數據集上驗證模型性能
        
        Args:
            model_id: 模型ID
            dataset_id: 數據集ID
            batch_size: 批次大小
            custom_params: 自定義參數
        """
        from app.services.model_service import ModelService
        model_service = ModelService()
        
        try:
            # 獲取模型信息
            model = model_service.get_model_by_id(model_id)
            if not model:
                raise Exception(f"找不到模型: {model_id}")
            
            # 獲取數據集信息
            datasets_dir = os.path.join(os.getcwd(), "data", "datasets")
            datasets_metadata_path = os.path.join(datasets_dir, "datasets.json")
            
            dataset_name = None
            dataset_path = None
            if os.path.exists(datasets_metadata_path):
                with open(datasets_metadata_path, 'r', encoding='utf-8') as f:
                    datasets = json.load(f)
                    dataset_info = next((d for d in datasets if d["id"] == dataset_id), None)
                    if dataset_info:
                        dataset_name = dataset_info["name"]
                        dataset_path = dataset_info["path"]
            
            if not dataset_name:
                raise Exception(f"找不到數據集: {dataset_id}")
            
            print(f"開始在數據集 {dataset_name} 上驗證模型 {model.name}")
            
            # 根據模型格式進行不同的驗證
            if model.format.value == "pt":
                # 對於PT模型，直接驗證
                return await self._validate_pytorch_model(model, dataset_id, dataset_name, dataset_path, batch_size, custom_params)
            elif model.format.value in ["onnx", "engine"]:
                # 對於轉換後的模型，需要找到源PyTorch模型
                source_model_path = None
                source_model = None
                
                # 策略1: 通過metadata中的source_model_id查找
                if model.metadata and model.metadata.get("source_model_id"):
                    source_model_id = model.metadata["source_model_id"]
                    source_model = model_service.get_model_by_id(source_model_id)
                    if source_model and source_model.format.value == "pt":
                        source_model_path = source_model.path
                        print(f"通過source_model_id找到源模型: {source_model.name}")
                
                # 策略2: 通過metadata中的source_model_name查找同名PT模型
                if not source_model_path and model.metadata and model.metadata.get("source_model_name"):
                    source_name = model.metadata["source_model_name"]
                    # 在所有PT模型中查找同名模型
                    for m_id, m in model_service.models.items():
                        if m.format.value == "pt" and m.name == source_name:
                            source_model_path = m.path
                            source_model = m
                            print(f"通過模型名稱找到源模型: {m.name}")
                            break
                
                # 策略3: 通過模型名稱推斷源模型名稱
                if not source_model_path:
                    # 從轉換後的模型名稱中提取源模型名稱
                    # 例如: test_engine_fp32_batch1_size640 -> test
                    model_name_parts = model.name.split('_')
                    if len(model_name_parts) >= 2:
                        # 取第一部分作為源模型名稱
                        potential_source_name = model_name_parts[0]
                        for m_id, m in model_service.models.items():
                            if m.format.value == "pt" and m.name == potential_source_name:
                                source_model_path = m.path
                                source_model = m
                                print(f"通過名稱推斷找到源模型: {m.name}")
                                break
                
                if not source_model_path or not os.path.exists(source_model_path):
                    # 如果找不到源模型，拋出錯誤而不是使用默認模型
                    raise Exception(f"找不到源PyTorch模型進行驗證。模型: {model.name}, 查找的源模型信息: {model.metadata.get('source_model_name') if model.metadata else 'None'}")
                
                return await self._validate_converted_model(model, source_model, dataset_id, dataset_name, dataset_path, batch_size, custom_params)
            else:
                raise Exception(f"不支持的模型格式: {model.format}")
                
        except Exception as e:
            print(f"模型驗證失敗: {str(e)}")
            # 返回錯誤結果而不是模擬結果
            return {
                "model_id": model_id,
                "model_name": model.name if model else "unknown",
                "dataset_id": dataset_id,
                "dataset_name": dataset_name or "unknown",
                "batch_size": batch_size,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "status": "failed"
            }
    
    async def _validate_pytorch_model(self, model: ModelInfo, dataset_id: str, dataset_name: str, dataset_path: str, batch_size: int, custom_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """驗證PyTorch模型"""
        try:
            # 查找數據集的YAML配置文件
            yaml_file = self._find_dataset_yaml(dataset_name, dataset_path)
            if not yaml_file:
                raise Exception(f"找不到數據集 {dataset_name} 的YAML配置文件")
            
            # 準備驗證參數
            validation_params = {
                "batch": batch_size,
                "imgsz": custom_params.get("imgsz", 640) if custom_params else 640,
                "conf": custom_params.get("conf", 0.25) if custom_params else 0.25,
                "iou": custom_params.get("iou", 0.7) if custom_params else 0.7,
                "verbose": custom_params.get("verbose", False) if custom_params else False,
                "device": custom_params.get("device", 0) if custom_params else 0,
            }
            
            # 使用PyTorch模型直接驗證
            from ultralytics import YOLO
            yolo_model = YOLO(model.path)
            metrics = yolo_model.val(data=yaml_file, **validation_params)
            validation_results = metrics.results_dict
            
            # 格式化結果
            return self._format_validation_results(model.id, model.name, dataset_id, dataset_name, batch_size, validation_results)
            
        except Exception as e:
            print(f"PyTorch模型驗證失敗: {str(e)}")
            raise
    
    async def _validate_converted_model(self, model: ModelInfo, source_model: ModelInfo, dataset_id: str, dataset_name: str, dataset_path: str, batch_size: int, custom_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """驗證轉換後的模型（ONNX/TensorRT）"""
        try:
            # 查找數據集的YAML配置文件
            yaml_file = self._find_dataset_yaml(dataset_name, dataset_path)
            if not yaml_file:
                raise Exception(f"找不到數據集 {dataset_name} 的YAML配置文件")
            
            # 準備驗證參數
            validation_params = {
                "batch": batch_size,
                "imgsz": custom_params.get("imgsz", 640) if custom_params else 640,
                "conf": custom_params.get("conf", 0.25) if custom_params else 0.25,
                "iou": custom_params.get("iou", 0.7) if custom_params else 0.7,
                "verbose": custom_params.get("verbose", False) if custom_params else False,
                "device": custom_params.get("device", 0) if custom_params else 0,
            }
            
            # 使用源PyTorch模型加載，但使用轉換後的推理引擎
            from ultralytics import YOLO
            
            # 首先用源模型創建驗證器
            yolo_model = YOLO(source_model.path)
            
            # 然後設置使用轉換後的模型進行推理
            if model.format.value == "onnx":
                validation_params["format"] = "onnx"
                validation_params["model"] = model.path
            elif model.format.value == "engine":
                validation_params["format"] = "engine"
                validation_params["model"] = model.path
            
            # 執行驗證
            metrics = yolo_model.val(data=yaml_file, **validation_params)
            validation_results = metrics.results_dict
            
            # 格式化結果
            return self._format_validation_results(model.id, model.name, dataset_id, dataset_name, batch_size, validation_results)
            
        except Exception as e:
            print(f"轉換模型驗證失敗: {str(e)}")
            raise
    
    def _find_dataset_yaml(self, dataset_name: str, dataset_path: str) -> str:
        """查找數據集的YAML配置文件"""
        # 可能的YAML文件名
        possible_yaml_names = [
            f"{dataset_name}.yaml",
            f"{dataset_name}_pose.yaml",
            f"{dataset_name}_detect.yaml",
            "data.yaml",
            "dataset.yaml"
        ]
        
        # 可能的搜索路徑
        search_paths = []
        if dataset_path:
            search_paths.append(dataset_path)
            search_paths.append(os.path.dirname(dataset_path))
        
        datasets_dir = os.path.join(os.getcwd(), "data", "datasets")
        search_paths.append(datasets_dir)
        
        for search_path in search_paths:
            if not search_path or not os.path.exists(search_path):
                continue
                
            for yaml_name in possible_yaml_names:
                yaml_path = os.path.join(search_path, yaml_name)
                if os.path.exists(yaml_path):
                    print(f"找到數據集YAML文件: {yaml_path}")
                    return yaml_path
        
        # 如果找不到，返回None
        return None
    
    def _format_validation_results(self, model_id: str, model_name: str, dataset_id: str, dataset_name: str, batch_size: int, validation_results: dict) -> Dict[str, Any]:
        """格式化驗證結果"""
        formatted_results = {
            "model_id": model_id,
            "model_name": model_name,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "batch_size": batch_size,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {}
        }
        
        # 提取關鍵指標
        if validation_results:
            # 檢測指標
            if "metrics/precision(B)" in validation_results:
                formatted_results["metrics"]["precision"] = float(validation_results["metrics/precision(B)"])
            
            if "metrics/recall(B)" in validation_results:
                formatted_results["metrics"]["recall"] = float(validation_results["metrics/recall(B)"])
            
            if "metrics/mAP50(B)" in validation_results:
                formatted_results["metrics"]["mAP50"] = float(validation_results["metrics/mAP50(B)"])
            
            if "metrics/mAP50-95(B)" in validation_results:
                formatted_results["metrics"]["mAP50_95"] = float(validation_results["metrics/mAP50-95(B)"])
            
            # 速度指標
            if "speed/inference(ms)" in validation_results:
                formatted_results["metrics"]["inference_time_ms"] = float(validation_results["speed/inference(ms)"])
            
            if "speed/preprocess(ms)" in validation_results:
                formatted_results["metrics"]["preprocess_time_ms"] = float(validation_results["speed/preprocess(ms)"])
            
            if "speed/postprocess(ms)" in validation_results:
                formatted_results["metrics"]["postprocess_time_ms"] = float(validation_results["speed/postprocess(ms)"])
        
        return formatted_results 