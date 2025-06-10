import React, { useState, useEffect } from 'react';
import { Table, Button, Upload, Form, Input, Select, Modal, Card, message, Tooltip } from 'antd';
import { UploadOutlined, PlusOutlined, InfoCircleOutlined, ReloadOutlined } from '@ant-design/icons';
import axios from 'axios';

const { Option } = Select;

const ModelsPage = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [uploadForm] = Form.useForm();

  // 當組件掛載時和每次訪問頁面時刷新模型列表
  useEffect(() => {
    // 讓模型服務掃描目錄後再獲取模型列表
    const refreshAndFetch = async () => {
      try {
        // 先嘗試刷新模型庫
        await axios.get('http://localhost:8000/api/models/refresh');
        console.log('模型庫刷新成功');
        // 然後獲取最新的模型列表
        await fetchModels();
      } catch (error) {
        console.error('刷新模型庫失敗，直接獲取模型列表', error);
        fetchModels();
      }
    };

    // 執行刷新和獲取
    refreshAndFetch();

    // 添加頁面可見性變化事件監聽器
    document.addEventListener('visibilitychange', handleVisibilityChange);

    // 清理函數，當組件卸載時移除事件監聽器
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  // 處理頁面可見性變化，當用戶從其他頁面返回時刷新數據
  const handleVisibilityChange = () => {
    if (document.visibilityState === 'visible') {
      // 當頁面再次變為可見時，刷新模型庫然後獲取模型列表
      axios.get('http://localhost:8000/api/models/refresh')
        .then(() => fetchModels())
        .catch(() => fetchModels());
    }
  };

  const fetchModels = async () => {
    setLoading(true);
    try {
      const response = await axios.get('http://localhost:8000/api/models/');
      const rawModels = response.data.models;
      
      // 過濾掉重複模型（相同triton_model_dir路徑的模型）
      const modelsByPath = {};
      const filteredModels = [];
      
      rawModels.forEach(model => {
        // 如果模型有metadata和triton_model_dir
        if (model.metadata && model.metadata.triton_model_dir) {
          const path = model.metadata.triton_model_dir;
          
          // 如果這個路徑還沒有記錄過，或這個模型更新
          if (!modelsByPath[path] || new Date(model.created_at) > new Date(modelsByPath[path].created_at)) {
            modelsByPath[path] = model;
          }
        } else {
          // 沒有triton_model_dir的模型直接保留
          filteredModels.push(model);
        }
      });
      
      // 將所有唯一路徑的最新模型添加到結果
      Object.values(modelsByPath).forEach(model => {
        filteredModels.push(model);
      });
      
      // 按創建時間排序
      filteredModels.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      
      console.log(`從 ${rawModels.length} 個模型中過濾出 ${filteredModels.length} 個不重複模型`);
      setModels(filteredModels);
    } catch (error) {
      console.error('獲取模型失敗:', error);
      message.error('獲取模型列表失敗');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (modelId) => {
    try {
      await axios.delete(`http://localhost:8000/api/models/${modelId}`);
      message.success('刪除模型成功');
      fetchModels();
    } catch (error) {
      console.error('刪除模型失敗:', error);
      message.error('刪除模型失敗');
    }
  };

  const handleUpload = async (values) => {
    try {
      const { fileList, name, model_type, description } = values;
      
      console.log('提交的模型類型:', model_type);
      console.log('提交的文件:', fileList);
      
      if (!fileList || !fileList[0] || !fileList[0].originFileObj) {
        message.error('請選擇有效的模型文件');
        return;
      }
      
      const formData = new FormData();
      formData.append('model_file', fileList[0].originFileObj);
      formData.append('model_name', name);
      formData.append('model_type', model_type);
      if (description) {
        formData.append('description', description);
      }
      
      // 檢查FormData內容
      for (let pair of formData.entries()) {
        console.log(pair[0] + ': ' + (pair[0] === 'model_file' ? '文件對象' : pair[1]));
      }
      
      const response = await axios.post('http://localhost:8000/api/models/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      message.success('上傳模型成功');
      setModalVisible(false);
      uploadForm.resetFields();
      
      // 上傳完成後立即刷新模型列表，並設置短暫延遲確保後端處理完成
      setTimeout(() => {
        fetchModels();
      }, 1000);
    } catch (error) {
      console.error('上傳模型失敗:', error.response?.data || error);
      message.error(`上傳模型失敗: ${error.response?.data?.detail || error.message}`);
    }
  };

  const columns = [
    {
      title: '名稱',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <Tooltip title={`ID: ${record.id}`}>
          <span>{text} <InfoCircleOutlined style={{ fontSize: '12px', color: '#1890ff' }} /></span>
        </Tooltip>
      ),
    },
    {
      title: '類型',
      dataIndex: 'type',
      key: 'type',
    },
    {
      title: '格式',
      dataIndex: 'format',
      key: 'format',
    },
    {
      title: '大小',
      dataIndex: 'size_mb',
      key: 'size_mb',
      render: (size) => `${size} MB`,
    },
    {
      title: '上傳時間',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date) => new Date(date).toLocaleString(),
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <>
          <Button type="link" href={`/models/${record.id}`}>
            詳情
          </Button>
          <Button 
            type="link" 
            onClick={() => window.open(`http://localhost:8000/api/models/${record.id}/download`)}
          >
            下載
          </Button>
          <Button type="link" danger onClick={() => handleDelete(record.id)}>
            刪除
          </Button>
        </>
      ),
    },
  ];

  return (
    <div>
      <Card
        title="模型管理"
        extra={
          <>
            <Button 
              onClick={fetchModels} 
              icon={<ReloadOutlined />} 
              style={{ marginRight: 8 }}
            >
              刷新
            </Button>
            <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalVisible(true)}>
              上傳模型
            </Button>
          </>
        }
      >
        <Table
          columns={columns}
          dataSource={models}
          rowKey="id"
          loading={loading}
        />
      </Card>

      <Modal
        title="上傳模型"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={null}
      >
        <Form
          form={uploadForm}
          layout="vertical"
          onFinish={handleUpload}
        >
          <Form.Item
            name="fileList"
            label="模型文件"
            rules={[{ required: true, message: '請上傳模型文件' }]}
            valuePropName="fileList"
            getValueFromEvent={(e) => {
              if (Array.isArray(e)) {
                return e;
              }
              return e?.fileList;
            }}
          >
            <Upload 
              accept=".pt,.onnx,.engine" 
              beforeUpload={() => false}
              maxCount={1}
              listType="text"
            >
              <Button icon={<UploadOutlined />}>選擇文件</Button>
            </Upload>
          </Form.Item>
          <Form.Item
            name="name"
            label="模型名稱"
            rules={[{ required: true, message: '請輸入模型名稱' }]}
          >
            <Input />
          </Form.Item>
          <Form.Item
            name="model_type"
            label="模型類型"
            rules={[{ required: true, message: '請選擇模型類型' }]}
          >
            <Select>
              <Option value="yolov8">YOLOv8</Option>
              <Option value="yolov8_pose">YOLOv8-Pose</Option>
              <Option value="yolov8_seg">YOLOv8-Seg</Option>
              <Option value="custom">自定義</Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="description"
            label="描述"
          >
            <Input.TextArea rows={4} />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit">
              上傳
            </Button>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default ModelsPage; 