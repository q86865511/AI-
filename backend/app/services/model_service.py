import os
import json
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime, timezone, timedelta
import re

from app.models import ModelInfo, ModelType, ModelFormat

class ModelService:
    """
    模型服務，負責模型的增刪改查
    """
    def __init__(self):
        """初始化模型服務"""
        self.models: Dict[str, ModelInfo] = {}
        self.data_file = "data/models.json"
        
        # 確保數據目錄存在
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        # 首先清空現有模型數據，確保重新掃描
        self.models = {}
        
        # 掃描模型目錄載入現有模型
        self._scan_model_repository()
        
        # 保存所有模型數據，確保新掃描的模型被持久化
        self._save_models()
        
        print(f"模型服務初始化完成，載入了 {len(self.models)} 個模型")
        print(f"模型ID列表: {list(self.models.keys())}")
    
    def get_models(self, model_type: Optional[ModelType] = None, 
                   format: Optional[ModelFormat] = None,
                   skip: int = 0, limit: int = 10) -> List[ModelInfo]:
        """獲取模型列表，支持過濾和分頁"""
        models = list(self.models.values())
        
        # 按類型和格式過濾
        if model_type:
            models = [m for m in models if m.type == model_type]
        if format:
            models = [m for m in models if m.format == format]
        
        # 按創建時間排序（最新的先顯示）
        models.sort(key=lambda m: m.created_at, reverse=True)
        
        # 分頁
        return models[skip:skip + limit]
    
    def get_model_by_id(self, model_id: str) -> Optional[ModelInfo]:
        """根據ID獲取模型"""
        return self.models.get(model_id)
    
    def save_model(self, model: ModelInfo) -> ModelInfo:
        """保存模型信息"""
        self.models[model.id] = model
        self._save_models()
        return model
    
    def delete_model(self, model_id: str) -> bool:
        """刪除模型"""
        if model_id in self.models:
            model = self.models[model_id]
            
            # 檢查是否是Triton格式的模型，如果是，刪除Triton模型目錄
            if model.metadata and model.metadata.get("is_trt_model"):
                triton_model_dir = model.metadata.get("triton_model_dir")
                if triton_model_dir and os.path.exists(triton_model_dir):
                    try:
                        print(f"刪除Triton模型目錄: {triton_model_dir}")
                        import shutil
                        shutil.rmtree(triton_model_dir)
                    except Exception as e:
                        print(f"刪除Triton模型目錄時出錯: {str(e)}")
            
            # 刪除模型記錄
            del self.models[model_id]
            self._save_models()
            return True
        return False
    
    def _load_models(self):
        """從文件載入模型數據"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    models_data = json.load(f)
                    for model_data in models_data:
                        model = ModelInfo.parse_obj(model_data)
                        self.models[model.id] = model
            except Exception as e:
                print(f"載入模型數據時出錯: {str(e)}")
    
    def _save_models(self):
        """保存模型數據到文件"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                models_data = [model.dict() for model in self.models.values()]
                # 處理datetime序列化
                for model_data in models_data:
                    model_data['created_at'] = model_data['created_at'].isoformat()
                json.dump(models_data, f, indent=2)
        except Exception as e:
            print(f"保存模型數據時出錯: {str(e)}")
    
    def _scan_model_repository(self):
        """掃描模型目錄，添加現有模型"""
        model_repo = os.environ.get("MODEL_REPOSITORY_PATH", "model_repository")
        if not os.path.exists(model_repo):
            os.makedirs(model_repo, exist_ok=True)
            return
        
        print(f"掃描模型目錄: {model_repo}")
        
        # 記錄掃描前的模型ID
        previous_ids = set(self.models.keys())
        new_ids = set()
        
        # 掃描模型目錄下的所有子目錄
        for model_dir in os.listdir(model_repo):
            model_path = os.path.join(model_repo, model_dir)
            if os.path.isdir(model_path):
                print(f"檢查目錄: {model_path}")
                
                # 首先檢查是否有metadata.json文件，優先使用其中的模型ID
                metadata_path = os.path.join(model_path, 'metadata.json')
                specified_id = None
                
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            if 'model_id' in metadata:
                                specified_id = metadata['model_id']
                                print(f"從metadata.json獲取模型ID: {specified_id}")
                    except Exception as e:
                        print(f"讀取metadata.json時出錯: {str(e)}")
                
                # 檢查是否是標準的Triton格式模型目錄
                if os.path.exists(os.path.join(model_path, 'config.pbtxt')):
                    # 添加模型，並獲取返回的ID
                    result_id = self._add_trt_model(model_path, model_dir, specified_id)
                    if result_id:
                        new_ids.add(result_id)
                        print(f"添加Triton模型: {model_dir}, ID: {result_id}")
                        
                # 如果不是Triton格式，檢查是否是PyTorch格式
                elif os.path.exists(os.path.join(model_path, '1', 'model.pt')):
                    # 添加PyTorch模型
                    result_id = self._add_pt_model(model_path, model_dir, specified_id)
                    if result_id:
                        new_ids.add(result_id)
                        print(f"添加PyTorch模型: {model_dir}, ID: {result_id}")
                else:
                    print(f"跳過非標準模型目錄: {model_path}")
        
        # 刪除不再存在的模型記錄
        ids_to_remove = previous_ids - new_ids
        if ids_to_remove:
            for old_id in ids_to_remove:
                if old_id in self.models:
                    print(f"刪除不再存在的模型記錄: {old_id}")
                    del self.models[old_id]
        
        # 輸出當前模型列表
        if self.models:
            print(f"掃描後的模型列表:")
            for m_id, model in self.models.items():
                print(f"  - ID: {m_id}, 名稱: {model.name}, 格式: {model.format.value}, 路徑: {model.path}")
        else:
            print(f"掃描後沒有找到任何模型")
            
        # 返回所有找到的模型ID
        return list(new_ids)
    
    def _add_trt_model(self, model_path: str, model_dir: str, specified_id: str = None):
        """添加Triton Inference Server模型，可以指定模型ID"""
        try:
            # 檢查是否已存在相同Triton模型目錄的記錄
            existing_model_id = None
            for m_id, model in self.models.items():
                if (model.metadata and 
                    model.metadata.get("triton_model_dir") == model_path):
                    existing_model_id = m_id
                    print(f"已存在相同Triton模型目錄的記錄，ID: {m_id}")
                    break
            
            if existing_model_id:
                # 如果已存在記錄，直接返回現有ID，不做任何修改
                print(f"保持現有模型ID不變: {existing_model_id}")
                return existing_model_id
            
            # 確定模型類型
            model_type = ModelType.YOLOV8
            if 'pose' in model_dir.lower():
                model_type = ModelType.YOLOV8_POSE
            elif 'cls' in model_dir.lower() or 'classification' in model_dir.lower():
                model_type = ModelType.YOLOV8_CLS
            elif 'seg' in model_dir.lower() or 'segment' in model_dir.lower():
                model_type = ModelType.YOLOV8_SEG
            
            # 檢查模型格式
            model_format = None
            config_path = os.path.join(model_path, "config.pbtxt")
            model_file_path = None
            
            # 查找版本目錄下的模型文件
            version_dirs = [d for d in os.listdir(model_path) if d.isdigit()]
            if version_dirs:
                version_dir = sorted(version_dirs)[-1]  # 使用最新版本
                version_path = os.path.join(model_path, version_dir)
                
                for file in os.listdir(version_path):
                    if file.endswith('.pt'):
                        model_format = ModelFormat.PT
                        model_file_path = os.path.join(version_path, file)
                        break
                    elif file.endswith('.onnx'):
                        model_format = ModelFormat.ONNX
                        model_file_path = os.path.join(version_path, file)
                        break
                    elif file.endswith('.engine') or file.endswith('.plan'):
                        model_format = ModelFormat.ENGINE
                        model_file_path = os.path.join(version_path, file)
                        break
            
            if not model_format or not model_file_path:
                print(f"跳過無法識別格式的模型目錄: {model_path}")
                return None
            
            # 計算文件大小
            file_size_mb = os.path.getsize(model_file_path) / (1024 * 1024)
            
            # 讀取現有的metadata.json（如果存在）
            metadata_file = os.path.join(model_path, "metadata.json")
            existing_metadata = {}
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        existing_metadata = json.load(f)
                        print(f"讀取現有metadata: {metadata_file}")
                except Exception as e:
                    print(f"讀取metadata文件失敗: {str(e)}")
            
            # 使用指定的ID或現有metadata中的ID，如果都沒有則生成新的
            if specified_id:
                model_id = specified_id
                print(f"使用指定的模型ID: {model_id}")
            elif existing_metadata.get("model_id"):
                model_id = existing_metadata["model_id"]
                print(f"從metadata.json獲取模型ID: {model_id}")
            else:
                model_id = str(uuid.uuid4())
                print(f"生成新的模型ID: {model_id}")
            
            print(f"添加Triton模型: {model_id}, 名稱: {model_dir}, 類型: {model_type}, 格式: {model_format}")
            
            # 構建完整的metadata，保留現有的源模型信息
            metadata = {
                "is_trt_model": True,
                "triton_model_name": model_dir,
                "triton_model_dir": model_path,
                "version": version_dir if 'version_dir' in locals() else "1",
                "platform": "pytorch_libtorch" if model_format == ModelFormat.PT else 
                           "tensorrt_plan" if model_format == ModelFormat.ENGINE else "onnxruntime_onnx",
                "original_id": model_id
            }
            
            # 保留現有metadata中的源模型信息
            source_fields = [
                "source_model_id", "source_model_name", "source_model_path", 
                "source_model_type", "source_model_format", "conversion_precision", 
                "conversion_target_format"
            ]
            for field in source_fields:
                if field in existing_metadata:
                    metadata[field] = existing_metadata[field]
            
            # 創建模型信息
            model_info = ModelInfo(
                id=model_id,
                name=model_dir,
                type=model_type,
                format=model_format,
                path=model_file_path,
                size_mb=file_size_mb,
                created_at=datetime.now(timezone.utc),
                description=f"Triton Inference Server模型: {model_dir}",
                metadata=metadata
            )
            
            # 更新metadata文件，保留所有字段
            updated_metadata = {**existing_metadata, **metadata}
            try:
                with open(metadata_file, 'w', encoding="utf-8") as f:
                    json.dump(updated_metadata, f, indent=2, ensure_ascii=False)
                print(f"更新模型元數據文件: {metadata_file}")
            except Exception as e:
                print(f"更新元數據文件時出錯: {str(e)}")
            
            # 保存模型
            self.models[model_id] = model_info
            print(f"添加Triton模型: {model_dir}, ID: {model_id}")
            
            return model_id
            
        except Exception as e:
            print(f"添加Triton模型時出錯: {str(e)}")
            return None
    
    def _add_pt_model(self, model_path: str, model_dir: str, specified_id: str = None):
        """添加PyTorch模型，可以指定模型ID"""
        try:
            # 檢查是否已存在相同路徑的記錄
            existing_model_id = None
            for m_id, model in self.models.items():
                if model.path and os.path.dirname(os.path.dirname(model.path)) == model_path:
                    existing_model_id = m_id
                    print(f"已存在相同路徑的模型記錄，ID: {m_id}")
                    break
            
            if existing_model_id:
                # 如果指定了ID且與現有ID不同，使用指定的ID更新記錄
                if specified_id and specified_id != existing_model_id:
                    print(f"更新模型ID: 從 {existing_model_id} 到 {specified_id}")
                    model = self.models[existing_model_id]
                    del self.models[existing_model_id]
                    model.id = specified_id
                    self.models[specified_id] = model
                    
                    # 更新metadata.json
                    self._update_metadata_file(model_path, specified_id)
                    
                    return specified_id
            else:
                    return existing_model_id
            
            # 確定模型類型
            model_type = ModelType.YOLOV8
            if 'pose' in model_dir.lower():
                model_type = ModelType.YOLOV8_POSE
            elif 'seg' in model_dir.lower():
                model_type = ModelType.YOLOV8_SEG
            
            # 查找模型文件
            model_file_path = os.path.join(model_path, '1', 'model.pt')
            if not os.path.exists(model_file_path):
                print(f"模型文件不存在: {model_file_path}")
                return None
            
            # 計算文件大小
            file_size_mb = os.path.getsize(model_file_path) / (1024 * 1024)
            
            # 創建或使用模型ID
            model_id = specified_id or str(uuid.uuid4())
            
            # 創建模型信息
            model = ModelInfo(
                id=model_id,
                name=model_dir,
                type=model_type,
                format=ModelFormat.PT,
                path=model_file_path,
                size_mb=round(file_size_mb, 2),
                created_at=datetime.now(timezone(timedelta(hours=8))),
                description=f"PyTorch模型: {model_dir}",
                metadata={
                    "model_id": model_id,
                    "model_dir": model_path,
                    "model_name": model_dir
                }
            )
            
            # 保存模型信息
            self.models[model_id] = model
            
            # 創建或更新metadata.json
            self._update_metadata_file(model_path, model_id)
            
            print(f"成功添加PyTorch模型: id={model_id}, 名稱={model_dir}, 路徑={model_file_path}")
            return model_id
            
        except Exception as e:
            print(f"添加PyTorch模型時出錯: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def _update_metadata_file(self, model_dir: str, model_id: str):
        """創建或更新模型目錄中的metadata.json文件"""
        try:
            metadata_file = os.path.join(model_dir, "metadata.json")
            metadata = {}
            
            # 如果文件已存在，讀取現有內容
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except Exception:
                    metadata = {}
            
            # 更新模型ID
            metadata["model_id"] = model_id
            
            # 寫入文件
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"更新模型元數據文件，模型ID: {model_id}")
            
        except Exception as e:
            print(f"更新模型元數據文件時出錯: {str(e)}")

    def _write_metadata_file(self, model_dir, model):
        """寫入模型元數據文件"""
        try:
            metadata_file = os.path.join(model_dir, "metadata.json")
            model_data = {
                "model_id": model.id,
                "name": model.name,
                "type": model.type.value,
                "format": model.format.value,
                "created_at": model.created_at.isoformat(),
                "description": model.description
            }
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, indent=2)
            return True
        except Exception as e:
            print(f"寫入元數據文件時出錯: {str(e)}")
            return False

    async def refresh_models(self):
        """
        強制重新掃描和加載模型存儲庫中的模型
        """
        print("正在重新掃描模型目錄...")
        self._scan_model_repository()
        return self.models 