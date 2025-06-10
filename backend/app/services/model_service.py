import os
import json
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone, timedelta
from app.models import ModelInfo, ModelType, ModelFormat

class ModelService:
    """
    模型服務，負責模型管理和存儲庫掃描
    完全基於model_repository目錄中的metadata，廢除models.json
    """

    def __init__(self):
        """初始化模型服務"""
        self.models: Dict[str, ModelInfo] = {}
        self.model_repository_path = os.environ.get("MODEL_REPOSITORY_PATH", "model_repository")
        
        # 確保模型存儲庫目錄存在
        os.makedirs(self.model_repository_path, exist_ok=True)
        
        # 初始化時掃描模型存儲庫
        self._scan_model_repository()

    def get_models(self, model_type: Optional[ModelType] = None, 
                   format: Optional[ModelFormat] = None,
                   skip: int = 0, limit: int = 10) -> List[ModelInfo]:
        """獲取模型列表，支持過濾和分頁"""
        # 每次查詢前重新掃描，確保獲取最新狀態
        self._scan_model_repository()
        
        models = list(self.models.values())
        
        # 過濾
        if model_type:
            models = [m for m in models if m.type == model_type]
        if format:
            models = [m for m in models if m.format == format]
        
        # 按創建時間排序
        models.sort(key=lambda m: m.created_at, reverse=True)
        
        # 分頁
        return models[skip:skip + limit]

    def get_model_by_id(self, model_id: str) -> Optional[ModelInfo]:
        """根據ID獲取模型"""
        # 每次查詢前重新掃描，確保獲取最新狀態
        self._scan_model_repository()
        return self.models.get(model_id)

    def get_model_by_name(self, name: str) -> Optional[ModelInfo]:
        """根據名稱獲取模型"""
        # 每次查詢前重新掃描，確保獲取最新狀態
        self._scan_model_repository()
        for model in self.models.values():
            if model.name == name:
                return model
        return None

    def save_model(self, model: ModelInfo) -> ModelInfo:
        """保存模型信息到model_repository中的metadata"""
        self.models[model.id] = model
        
        # 根據模型路徑確定metadata文件位置
        model_dir = self._get_model_directory(model.path)
        if model_dir:
            self._write_metadata_file(model_dir, model)
        
        return model

    def delete_model(self, model_id: str) -> bool:
        """刪除模型"""
        model = self.models.get(model_id)
        if not model:
            return False
        
        try:
            # 獲取模型目錄
            model_dir = self._get_model_directory(model.path)
            
            if model_dir and os.path.exists(model_dir):
                # 刪除整個模型目錄
                import shutil
                shutil.rmtree(model_dir)
                print(f"已刪除模型目錄: {model_dir}")
            
            # 從內存中刪除
            del self.models[model_id]
            
            return True
        except Exception as e:
            print(f"刪除模型失敗: {str(e)}")
            return False

    def _scan_model_repository(self):
        """掃描模型存儲庫，重新載入所有模型"""
        print(f"掃描模型存儲庫: {self.model_repository_path}")
        
        # 清空現有模型列表
        self.models.clear()
        
        if not os.path.exists(self.model_repository_path):
            print(f"模型存儲庫目錄不存在: {self.model_repository_path}")
            return
        
        # 遍歷所有子目錄
        for item in os.listdir(self.model_repository_path):
            item_path = os.path.join(self.model_repository_path, item)
            
            if os.path.isdir(item_path):
                try:
                    self._scan_model_directory(item_path)
                except Exception as e:
                    print(f"掃描模型目錄 {item_path} 時出錯: {str(e)}")
        
        print(f"掃描完成，找到 {len(self.models)} 個模型")

    def _scan_model_directory(self, model_dir: str):
        """掃描單個模型目錄"""
        metadata_file = os.path.join(model_dir, "metadata.json")
        
        # 檢查是否有metadata文件
        if os.path.exists(metadata_file):
            self._load_model_from_metadata(model_dir, metadata_file)
        else:
            # 如果沒有metadata，嘗試自動檢測並創建
            self._detect_and_create_model(model_dir)

    def _load_model_from_metadata(self, model_dir: str, metadata_file: str):
        """從metadata文件載入模型"""
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 獲取模型ID，如果沒有則生成新的
            model_id = metadata.get("model_id")
            if not model_id:
                model_id = str(uuid.uuid4())
                metadata["model_id"] = model_id
                # 更新metadata文件
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # 查找模型文件
            model_file_path = self._find_model_file(model_dir)
            if not model_file_path:
                print(f"在目錄 {model_dir} 中找不到模型文件")
                return
            
            # 確定模型格式
            model_format = self._determine_model_format(model_file_path)
            
            # 創建ModelInfo對象
            model = ModelInfo(
                id=model_id,
                name=metadata.get("display_name", os.path.basename(model_dir)),
                type=ModelType(metadata.get("type", "yolov8")),
                format=model_format,
                path=model_file_path,
                size_mb=os.path.getsize(model_file_path) / (1024 * 1024),
                created_at=self._parse_datetime(metadata.get("created_at")),
                description=metadata.get("description", f"模型: {metadata.get('display_name', os.path.basename(model_dir))}"),
                metadata=metadata
            )
            
            self.models[model_id] = model
            print(f"載入模型: {model.name} (ID: {model_id})")
            
        except Exception as e:
            print(f"載入metadata文件 {metadata_file} 時出錯: {str(e)}")

    def _detect_and_create_model(self, model_dir: str):
        """自動檢測並創建模型metadata"""
        model_file_path = self._find_model_file(model_dir)
        if not model_file_path:
            return
        
        # 生成新的模型ID
        model_id = str(uuid.uuid4())
        
        # 確定模型格式和類型
        model_format = self._determine_model_format(model_file_path)
        model_name = os.path.basename(model_dir)
        
        # 創建metadata
        metadata = {
            "model_id": model_id,
            "display_name": model_name,
            "type": "yolov8",  # 默認類型
            "format": model_format.value,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "is_trt_model": True,
            "triton_model_name": model_name,
            "triton_model_dir": model_dir,
            "version": "1",
            "platform": self._get_platform_from_format(model_format)
        }
        
        # 如果是轉換後的模型，嘗試從名稱推斷源模型
        if "_engine_" in model_name or "_onnx_" in model_name:
            source_name = model_name.split("_")[0]  # 取第一部分作為源模型名稱
            metadata["source_model_name"] = source_name
        
        # 保存metadata文件
        metadata_file = os.path.join(model_dir, "metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 創建ModelInfo對象
        model = ModelInfo(
            id=model_id,
            name=model_name,
            type=ModelType.YOLOV8,
            format=model_format,
            path=model_file_path,
            size_mb=os.path.getsize(model_file_path) / (1024 * 1024),
            created_at=datetime.now(timezone.utc),
            description=f"自動檢測的模型: {model_name}",
            metadata=metadata
        )
        
        self.models[model_id] = model
        print(f"自動創建模型: {model.name} (ID: {model_id})")

    def _find_model_file(self, model_dir: str) -> Optional[str]:
        """在模型目錄中查找模型文件"""
        # 查找版本目錄（通常是1）
        version_dirs = ["1", "2", "3"]
        model_extensions = [".pt", ".onnx", ".engine", ".plan"]
        
        for version in version_dirs:
            version_dir = os.path.join(model_dir, version)
            if os.path.exists(version_dir):
                for file in os.listdir(version_dir):
                    file_path = os.path.join(version_dir, file)
                    if os.path.isfile(file_path):
                        for ext in model_extensions:
                            if file.endswith(ext):
                                return file_path
        
        # 如果版本目錄中沒找到，直接在模型目錄中查找
        for file in os.listdir(model_dir):
            file_path = os.path.join(model_dir, file)
            if os.path.isfile(file_path):
                for ext in model_extensions:
                    if file.endswith(ext):
                        return file_path
        
        return None

    def _determine_model_format(self, file_path: str) -> ModelFormat:
        """根據文件擴展名確定模型格式"""
        if file_path.endswith(".pt"):
            return ModelFormat.PT
        elif file_path.endswith(".onnx"):
            return ModelFormat.ONNX
        elif file_path.endswith(".engine") or file_path.endswith(".plan"):
            return ModelFormat.ENGINE
        else:
            return ModelFormat.PT  # 默認

    def _get_platform_from_format(self, model_format: ModelFormat) -> str:
        """根據模型格式獲取平台名稱"""
        if model_format == ModelFormat.PT:
            return "pytorch_libtorch"
        elif model_format == ModelFormat.ONNX:
            return "onnxruntime_onnx"
        elif model_format == ModelFormat.ENGINE:
            return "tensorrt_plan"
        else:
            return "unknown"

    def _get_model_directory(self, model_path: str) -> Optional[str]:
        """根據模型路徑獲取模型目錄"""
        # 模型路徑通常是 /path/to/model_repository/model_name/1/model.ext
        # 我們需要返回 /path/to/model_repository/model_name
        path_parts = model_path.replace("\\", "/").split("/")
        
        # 找到model_repository的位置
        try:
            repo_index = -1
            for i, part in enumerate(path_parts):
                if "model_repository" in part:
                    repo_index = i
                    break
            
            if repo_index >= 0 and repo_index + 1 < len(path_parts):
                # 模型目錄是model_repository後的第一個目錄
                model_dir_parts = path_parts[:repo_index + 2]
                return "/".join(model_dir_parts)
        except Exception as e:
            print(f"解析模型目錄路徑時出錯: {str(e)}")
        
        return None

    def _parse_datetime(self, datetime_str: Optional[str]) -> datetime:
        """解析日期時間字符串"""
        if not datetime_str:
            return datetime.now(timezone.utc)
        
        try:
            # 嘗試解析ISO格式
            return datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
        except:
            return datetime.now(timezone.utc)

    def _write_metadata_file(self, model_dir: str, model: ModelInfo):
        """寫入metadata文件"""
        metadata_file = os.path.join(model_dir, "metadata.json")
        
        # 構建metadata
        metadata = {
            "model_id": model.id,
            "display_name": model.name,
            "type": model.type.value,
            "format": model.format.value,
            "created_at": model.created_at.isoformat(),
            "description": model.description,
            "is_trt_model": True,
            "triton_model_name": model.name,
            "triton_model_dir": model_dir,
            "version": "1",
            "platform": self._get_platform_from_format(model.format)
        }
        
        # 合併現有metadata
        if model.metadata:
            metadata.update(model.metadata)
            # 確保關鍵字段不被覆蓋
            metadata["model_id"] = model.id
            metadata["display_name"] = model.name
            metadata["type"] = model.type.value
            metadata["format"] = model.format.value
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    async def refresh_models(self):
        """刷新模型列表"""
        self._scan_model_repository()
        return len(self.models)

    def create_model_from_upload(self, name: str, model_type: str, file_path: str, description: str = None) -> ModelInfo:
        """從上傳文件創建模型"""
        # 生成唯一ID
        model_id = str(uuid.uuid4())
        
        # 確定模型格式
        model_format = self._determine_model_format(file_path)
        
        # 創建模型目錄
        model_dir = os.path.join(self.model_repository_path, name)
        version_dir = os.path.join(model_dir, "1")
        os.makedirs(version_dir, exist_ok=True)
        
        # 移動文件到版本目錄
        import shutil
        target_file = os.path.join(version_dir, f"model{os.path.splitext(file_path)[1]}")
        shutil.move(file_path, target_file)
        
        # 創建模型對象
        model = ModelInfo(
            id=model_id,
            name=name,
            type=ModelType(model_type),
            format=model_format,
            path=target_file,
            size_mb=os.path.getsize(target_file) / (1024 * 1024),
            created_at=datetime.now(timezone.utc),
            description=description or f"上傳的模型: {name}",
            metadata={
                "model_id": model_id,
                "display_name": name,
                "type": model_type,
                "format": model_format.value,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "is_trt_model": True,
                "triton_model_name": name,
                "triton_model_dir": model_dir,
                "version": "1",
                "platform": self._get_platform_from_format(model_format),
                "uploaded": True
            }
        )
        
        # 保存metadata
        self._write_metadata_file(model_dir, model)
        
        # 創建config.pbtxt
        self._create_triton_config(model_dir, model)
        
        # 添加到模型列表
        self.models[model_id] = model
        
        return model

    def _create_triton_config(self, model_dir: str, model: ModelInfo):
        """創建Triton配置文件"""
        config_path = os.path.join(model_dir, "config.pbtxt")
        
        # 獲取模型文件名
        model_filename = os.path.basename(model.path)
        
        # 根據格式設置平台
        platform = self._get_platform_from_format(model.format)
        
        config_content = f"""name: "{model.name}"
platform: "{platform}"
max_batch_size: 2
input [
  {{
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }}
]
output [
  {{
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 84, 8400 ]
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
        
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content) 