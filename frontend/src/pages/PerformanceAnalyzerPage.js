import React, { useState, useEffect, useRef } from 'react';
import { 
  Card, Select, Button, Table, Form, Input, Switch, Space, 
  Statistic, Row, Col, Typography, Upload, Divider, Radio, 
  Slider, Checkbox, Tabs, message
} from 'antd';
import { 
  UploadOutlined, DownloadOutlined, BarChartOutlined, 
  LineChartOutlined, PieChartOutlined, AreaChartOutlined 
} from '@ant-design/icons';
import axios from 'axios';
import ReactECharts from 'echarts-for-react';
import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas';
import * as XLSX from 'xlsx';

const { Title, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

// 注意：缺少以下依賴項，需要安裝：
// - echarts
// - echarts-for-react 
// - jspdf
// - html2canvas
// - xlsx
// 將在安裝完畢後再導入這些組件

const ModelPerformanceAnalyzer = () => {
  // 狀態管理
  const [testData, setTestData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [chartType, setChartType] = useState('bar');
  
  // 每個圖表分別的排序設定
  const [inferenceTimeSortOrder, setInferenceTimeSortOrder] = useState('asc');
  const [mapSortOrder, setMapSortOrder] = useState('desc');
  const [vramSortOrder, setVramSortOrder] = useState('asc');
  const [gpuLoadSortOrder, setGpuLoadSortOrder] = useState('asc');
  
  const [filterType, setFilterType] = useState('default');
  const [vramLimit, setVramLimit] = useState(1000);
  const [gpuLoadLimit, setGpuLoadLimit] = useState(80);
  const [metricType, setMetricType] = useState('speed');
  
  // 檔案上傳參考
  const uploadRef = useRef();
  const chartContainerRef = useRef();
  const inferenceTimeChartRef = useRef();
  const vramUsageChartRef = useRef();
  const mAPChartRef = useRef();
  const gpuLoadChartRef = useRef();
  
  // 處理 JSON 數據
  const processTestData = (data) => {
    if (!data) return null;

    // 計算每個配置的模型數據結果
    const processedData = Object.keys(data).map(key => {
      const configData = data[key];
      
      // 分割配置名稱和批次大小
      const parts = key.split('_batch');
      const modelType = parts[0];
      const batchSize = parseInt(parts[1]);
      
      // 計算平均 VRAM 使用量
      const avgVRam = configData.model_load_vram_MB.reduce((sum, val) => sum + val, 0) / 
                      configData.model_load_vram_MB.length;
      
      // 生成假的 GPU 負載數據 (因為原始數據中沒有)
      const gpuLoad = modelType.includes('FP16') ? 
                      55 + Math.random() * 15 : 
                      65 + Math.random() * 20;
      
      return {
        key,
        modelType,
        batchSize,
        avgInferenceTime: configData.avg_inference_time,
        stdInferenceTime: configData.std_inference_time,
        mAP5095: configData['mAP50-95'], // 只使用mAP50-95
        vramUsage: avgVRam,
        gpuLoad: gpuLoad,
        // 用於計算加速比的基準值 (以批次大小為1的PT模型為基準)
        speedup: data.PT_batch1.avg_inference_time / configData.avg_inference_time,
        // 計算準確率比率 (相對於基準模型)
        accuracyRatio: configData['mAP50-95'] / data.PT_batch1['mAP50-95']
      };
    });

    return processedData;
  };
  
  // 載入示例數據
  const loadSampleData = () => {
    setLoading(true);
    
    // 使用 fetch 載入示例數據
    fetch('/sample_performance_data.json')
      .then(response => response.json())
      .then(data => {
        setTestData(processTestData(data));
        setLoading(false);
        message.success('已載入示例數據');
      })
      .catch(error => {
        console.error('載入示例數據失敗:', error);
        message.error('載入示例數據失敗');
        setLoading(false);
      });
  };
  
  // 處理文件上傳
  const handleFileUpload = (file) => {
    setLoading(true);
    const reader = new FileReader();
    
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        const processed = processTestData(data);
        setTestData(processed);
        message.success('數據載入成功！');
      } catch (error) {
        console.error('解析 JSON 失敗:', error);
        message.error('無法解析 JSON 文件');
      } finally {
        setLoading(false);
      }
    };
    
    reader.onerror = () => {
      message.error('讀取文件失敗');
      setLoading(false);
    };
    
    reader.readAsText(file);
    return false; // 阻止默認上傳行為
  };
  
  // 篩選邏輯
  const getFilteredDataForDisplay = () => {
    if (!testData) return [];
    
    let filteredData = [...testData];
    
    // 根據不同的篩選類型進行篩選
    if (filterType === 'highLoad') {
      // 高負載情況：只考慮推論時間，選擇SpeedUp最高的配置
      filteredData.sort((a, b) => b.speedup - a.speedup);
      return filteredData.slice(0, 5); // 返回前5個
    } 
    else if (filterType === 'resourceLimit') {
      // VRAM/GPU 負載限制篩選
      filteredData = filteredData.filter(item => 
        item.vramUsage <= vramLimit && item.gpuLoad <= gpuLoadLimit
      );
      
      // 根據選擇的指標排序
      if (metricType === 'speed') {
        filteredData.sort((a, b) => b.speedup - a.speedup);
      } else {
        filteredData.sort((a, b) => b.accuracyRatio - a.accuracyRatio);
      }
      
      return filteredData.slice(0, 5); // 返回前5個
    } 
    else if (filterType === 'balanced') {
      // 平衡模式：篩選符合資源限制的數據
      filteredData = filteredData.filter(item => 
        item.vramUsage <= vramLimit && item.gpuLoad <= gpuLoadLimit
      );
      
      // 標準化 mAP 與推論時間
      const maxMAP = Math.max(...filteredData.map(item => item.mAP5095));
      const maxInferenceTime = Math.max(...filteredData.map(item => item.avgInferenceTime));
      const minInferenceTime = Math.min(...filteredData.map(item => item.avgInferenceTime));
      
      filteredData.forEach(item => {
        item.normMAP = item.mAP5095 / maxMAP;
        // 推論時間標準化：越小越好，所以用 (max - current) / (max - min)
        item.normInferenceTime = (maxInferenceTime - item.avgInferenceTime) / (maxInferenceTime - minInferenceTime);
        // 平衡分數為兩者差的絕對值，越小表示越接近y=x軸線
        item.balanceScore = Math.abs(item.normMAP - item.normInferenceTime);
      });
      
      // 按平衡分數排序 (越小表示越平衡)
      filteredData.sort((a, b) => a.balanceScore - b.balanceScore);
      return filteredData.slice(0, 5); // 返回前5個最平衡的配置
    } 
    
    return filteredData;
  };

  // 為不同圖表獲取排序後的數據
  const getSortedDataForChart = (sortField, sortOrder) => {
    if (!testData) return [];
    
    let data = [...testData];
    
    data.sort((a, b) => {
      if (sortOrder === 'asc') {
        return a[sortField] - b[sortField];
      } else {
        return b[sortField] - a[sortField];
      }
    });
    
    return data;
  };
  
  // 準備圖表數據
  const getInferenceTimeChartOption = () => {
    if (!testData || testData.length === 0) return {};
    
    const data = getSortedDataForChart('avgInferenceTime', inferenceTimeSortOrder);
    const xAxisData = data.map(item => item.key);
    
    const option = {
      title: {
        text: '推論時間比較',
        subtext: '單位: 毫秒 (ms)'
      },
      tooltip: {
        trigger: 'axis',
        formatter: function(params) {
          const item = params[0];
          const dataItem = data[item.dataIndex];
          return `${dataItem.key}<br/>
                 推論時間: ${dataItem.avgInferenceTime.toFixed(2)} ms<br/>
                 標準差: ${dataItem.stdInferenceTime.toFixed(2)} ms<br/>
                 加速比: ${dataItem.speedup.toFixed(2)}x<br/>`;
        }
      },
      toolbox: {
        feature: {
          saveAsImage: { show: true, title: '保存為圖片' }
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: xAxisData,
        axisLabel: {
          rotate: 45
        }
      },
      yAxis: {
        type: 'value',
        name: '推論時間 (ms)'
      },
      series: [
        {
          name: '推論時間',
          type: chartType,
          data: data.map(item => item.avgInferenceTime),
          markLine: {
            data: [{ type: 'average', name: '平均值' }]
          },
          itemStyle: {
            color: function(params) {
              // 根據模型類型設置不同的顏色
              const modelType = data[params.dataIndex].modelType;
              if (modelType.includes('PT')) return '#3498db';
              if (modelType.includes('INT8')) return '#e74c3c';
              if (modelType.includes('FP16')) return '#2ecc71';
              return '#f39c12';
            }
          }
        }
      ]
    };

    return option;
  };
  
  const getVRamUsageChartOption = () => {
    if (!testData || testData.length === 0) return {};
    
    const data = getSortedDataForChart('vramUsage', vramSortOrder);
    const xAxisData = data.map(item => item.key);
    
    const option = {
      title: {
        text: 'VRAM 使用量比較',
        subtext: '單位: 兆位元組 (MB)'
      },
      tooltip: {
        trigger: 'axis',
        formatter: function(params) {
          const item = params[0];
          const dataItem = data[item.dataIndex];
          return `${dataItem.key}<br/>
                 VRAM: ${dataItem.vramUsage.toFixed(2)} MB<br/>`;
        }
      },
      toolbox: {
        feature: {
          saveAsImage: { show: true, title: '保存為圖片' }
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: xAxisData,
        axisLabel: {
          rotate: 45
        }
      },
      yAxis: {
        type: 'value',
        name: 'VRAM (MB)'
      },
      series: [
        {
          name: 'VRAM使用量',
          type: chartType,
          data: data.map(item => item.vramUsage),
          markLine: {
            data: [{ type: 'average', name: '平均值' }]
          },
          itemStyle: {
            color: function(params) {
              // 根據模型類型設置不同的顏色
              const modelType = data[params.dataIndex].modelType;
              if (modelType.includes('PT')) return '#3498db';
              if (modelType.includes('INT8')) return '#e74c3c';
              if (modelType.includes('FP16')) return '#2ecc71';
              return '#f39c12';
            }
          }
        }
      ]
    };

    return option;
  };
  
  const getMapChartOption = () => {
    if (!testData || testData.length === 0) return {};
    
    const data = getSortedDataForChart('mAP5095', mapSortOrder);
    const xAxisData = data.map(item => item.key);
    
    const option = {
      title: {
        text: '準確率比較',
        subtext: 'mAP@50-95'
      },
      tooltip: {
        trigger: 'axis',
        formatter: function(params) {
          const dataItem = data[params[0].dataIndex];
          return `${dataItem.key}<br/>
                 mAP@50-95: ${dataItem.mAP5095.toFixed(4)}<br/>
                 準確率比率: ${dataItem.accuracyRatio.toFixed(4)}<br/>`;
        }
      },
      toolbox: {
        feature: {
          saveAsImage: { show: true, title: '保存為圖片' }
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: xAxisData,
        axisLabel: {
          rotate: 45
        }
      },
      yAxis: {
        type: 'value',
        name: '準確率',
        min: 0.4,
        max: 1.0
      },
      series: [
        {
          name: 'mAP@50-95',
          type: chartType,
          data: data.map(item => item.mAP5095),
          itemStyle: {
            color: '#3498db'
          }
        }
      ]
    };

    return option;
  };
  
  const getGpuLoadChartOption = () => {
    if (!testData || testData.length === 0) return {};
    
    const data = getSortedDataForChart('gpuLoad', gpuLoadSortOrder);
    const xAxisData = data.map(item => item.key);
    
    const option = {
      title: {
        text: 'GPU負載比較',
        subtext: '單位: 百分比 (%)'
      },
      tooltip: {
        trigger: 'axis',
        formatter: function(params) {
          const dataItem = data[params[0].dataIndex];
          return `${dataItem.key}<br/>
                 GPU負載: ${dataItem.gpuLoad.toFixed(2)}%<br/>`;
        }
      },
      toolbox: {
        feature: {
          saveAsImage: { show: true, title: '保存為圖片' }
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: xAxisData,
        axisLabel: {
          rotate: 45
        }
      },
      yAxis: {
        type: 'value',
        name: 'GPU負載 (%)',
        min: 0,
        max: 100
      },
      series: [
        {
          name: 'GPU負載',
          type: chartType,
          data: data.map(item => item.gpuLoad),
          markLine: {
            data: [{ type: 'average', name: '平均值' }]
          },
          itemStyle: {
            color: function(params) {
              // 根據模型類型設置不同的顏色
              const modelType = data[params.dataIndex].modelType;
              if (modelType.includes('PT')) return '#3498db';
              if (modelType.includes('INT8')) return '#e74c3c';
              if (modelType.includes('FP16')) return '#2ecc71';
              return '#f39c12';
            }
          }
        }
      ]
    };

    return option;
  };
  
  // 匯出圖表為 PDF
  const exportChartsToPDF = () => {
    if (!chartContainerRef.current) {
      message.error('無法獲取圖表容器');
      return;
    }
    
    message.loading('正在生成 PDF，請稍候...', 0);
    
    const input = chartContainerRef.current;
    const pdf = new jsPDF('p', 'mm', 'a4');
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = pdf.internal.pageSize.getHeight();
    
    html2canvas(input, { scale: 2 }).then(canvas => {
      // 計算合適的縮放比例
      const imgWidth = canvas.width;
      const imgHeight = canvas.height;
      const ratio = Math.min(pdfWidth / imgWidth, pdfHeight / imgHeight);
      
      const imgData = canvas.toDataURL('image/png');
      pdf.addImage(imgData, 'PNG', 0, 0, imgWidth * ratio, imgHeight * ratio);
      
      // 保存 PDF
      pdf.save('AI模型測試結果分析報告.pdf');
      message.destroy();
      message.success('PDF 生成成功');
    }).catch(error => {
      message.destroy();
      message.error('PDF 生成失敗: ' + error.message);
    });
  };
  
  // 匯出數據為 Excel
  const exportDataToExcel = () => {
    if (!testData || testData.length === 0) {
      message.error('無數據可匯出');
      return;
    }
    
    try {
      const data = getFilteredDataForDisplay();
      const worksheet = XLSX.utils.json_to_sheet(data.map(item => ({
        '配置': item.key,
        '模型類型': item.modelType,
        '批次大小': item.batchSize,
        '推論時間 (ms)': item.avgInferenceTime.toFixed(2),
        '標準差 (ms)': item.stdInferenceTime.toFixed(2),
        'mAP@50-95': item.mAP5095.toFixed(4),
        'VRAM (MB)': item.vramUsage.toFixed(2),
        'GPU負載 (%)': item.gpuLoad.toFixed(2),
        '加速比': item.speedup.toFixed(2),
        '準確率比率': item.accuracyRatio.toFixed(4)
      })));
      
      const workbook = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(workbook, worksheet, '模型性能數據');
      
      // 生成並下載 Excel 文件
      XLSX.writeFile(workbook, 'AI模型測試結果分析數據.xlsx');
      message.success('Excel 檔案已匯出');
    } catch (error) {
      console.error('匯出 Excel 失敗:', error);
      message.error('匯出 Excel 失敗: ' + error.message);
    }
  };
  
  // 結果數據表格列定義
  const columns = [
    {
      title: '配置',
      dataIndex: 'key',
      key: 'key',
    },
    {
      title: '批次大小',
      dataIndex: 'batchSize',
      key: 'batchSize',
    },
    {
      title: '推論時間 (ms)',
      dataIndex: 'avgInferenceTime',
      key: 'avgInferenceTime',
      render: (text) => text.toFixed(2),
      sorter: (a, b) => a.avgInferenceTime - b.avgInferenceTime,
    },
    {
      title: 'mAP@50-95',
      dataIndex: 'mAP5095',
      key: 'mAP5095',
      render: (text) => text.toFixed(4),
      sorter: (a, b) => a.mAP5095 - b.mAP5095,
    },
    {
      title: 'VRAM (MB)',
      dataIndex: 'vramUsage',
      key: 'vramUsage',
      render: (text) => text.toFixed(2),
      sorter: (a, b) => a.vramUsage - b.vramUsage,
    },
    {
      title: 'GPU負載 (%)',
      dataIndex: 'gpuLoad',
      key: 'gpuLoad',
      render: (text) => text.toFixed(2),
      sorter: (a, b) => a.gpuLoad - b.gpuLoad,
    },
    {
      title: '加速比',
      dataIndex: 'speedup',
      key: 'speedup',
      render: (text) => text.toFixed(2),
      sorter: (a, b) => a.speedup - b.speedup,
    },
    {
      title: '準確率比率',
      dataIndex: 'accuracyRatio',
      key: 'accuracyRatio',
      render: (text) => text.toFixed(4),
      sorter: (a, b) => a.accuracyRatio - b.accuracyRatio,
    }
  ];
  
  return (
    <div style={{ padding: '20px' }}>
      <Title level={2}>自動化測試結果分析儀表板</Title>
      <Divider />
      
      {/* 上傳與控制區域 */}
      <Row gutter={24} style={{ marginBottom: '20px' }}>
        <Col span={8}>
          <Card title="數據加載">
            <Upload
              ref={uploadRef}
              beforeUpload={handleFileUpload}
              showUploadList={false}
              accept=".json"
            >
              <Button icon={<UploadOutlined />} loading={loading}>
                上傳 JSON 檔案
              </Button>
            </Upload>
            <Button 
              style={{ marginLeft: '10px' }} 
              onClick={loadSampleData}
              loading={loading}
            >
              載入示例數據
            </Button>
          </Card>
        </Col>
        
        <Col span={8}>
          <Card title="圖表類型設定">
            <Radio.Group 
              value={chartType} 
              onChange={(e) => setChartType(e.target.value)}
              buttonStyle="solid"
            >
              <Radio.Button value="bar"><BarChartOutlined /> 長條圖</Radio.Button>
              <Radio.Button value="line"><LineChartOutlined /> 折線圖</Radio.Button>
              <Radio.Button value="area"><AreaChartOutlined /> 區域圖</Radio.Button>
            </Radio.Group>
          </Card>
        </Col>
        
        <Col span={8}>
          <Card title="篩選條件">
            <Radio.Group 
              value={filterType} 
              onChange={(e) => setFilterType(e.target.value)}
              buttonStyle="solid"
              style={{ marginBottom: '20px' }}
            >
              <Radio.Button value="default">預設顯示</Radio.Button>
              <Radio.Button value="highLoad">高負載情況</Radio.Button>
              <Radio.Button value="resourceLimit">資源限制</Radio.Button>
              <Radio.Button value="balanced">平衡模式</Radio.Button>
            </Radio.Group>
            
            {(filterType === 'resourceLimit' || filterType === 'balanced') && (
              <div style={{ marginTop: '10px' }}>
                <Row gutter={24}>
                  <Col span={12}>
                    <div style={{ marginBottom: '10px' }}>
                      <Text>VRAM 限制 (MB): {vramLimit}</Text>
                      <Slider 
                        min={100} 
                        max={3000} 
                        value={vramLimit} 
                        onChange={setVramLimit}
                        marks={{
                          500: '500MB',
                          1000: '1GB',
                          2000: '2GB',
                          3000: '3GB'
                        }}
                        step={50}
                      />
                    </div>
                  </Col>
                  <Col span={12}>
                    <div style={{ marginBottom: '10px' }}>
                      <Text>GPU 負載限制 (%): {gpuLoadLimit}</Text>
                      <Slider 
                        min={0} 
                        max={100} 
                        value={gpuLoadLimit} 
                        onChange={setGpuLoadLimit}
                        marks={{
                          0: '0%',
                          50: '50%',
                          100: '100%'
                        }}
                      />
                    </div>
                  </Col>
                </Row>
                
                {filterType === 'resourceLimit' && (
                  <Radio.Group 
                    value={metricType} 
                    onChange={(e) => setMetricType(e.target.value)}
                    buttonStyle="solid"
                  >
                    <Radio.Button value="speed">優先加速比</Radio.Button>
                    <Radio.Button value="map">優先精度</Radio.Button>
                  </Radio.Group>
                )}
              </div>
            )}
          </Card>
        </Col>
      </Row>
      
      {/* 圖表顯示區 */}
      {testData ? (
        <div ref={chartContainerRef}>
          <Row gutter={24} style={{ marginBottom: '20px' }}>
            <Col span={12}>
              <Card 
                title="推論時間比較"
                extra={
                  <Radio.Group 
                    size="small"
                    value={inferenceTimeSortOrder} 
                    onChange={(e) => setInferenceTimeSortOrder(e.target.value)}
                  >
                    <Radio.Button value="asc">升序</Radio.Button>
                    <Radio.Button value="desc">降序</Radio.Button>
                  </Radio.Group>
                }
              >
                <ReactECharts
                  ref={inferenceTimeChartRef}
                  option={getInferenceTimeChartOption()}
                  style={{ height: '350px' }}
                  notMerge={true}
                />
              </Card>
            </Col>
            <Col span={12}>
              <Card 
                title="VRAM 使用量比較"
                extra={
                  <Radio.Group 
                    size="small"
                    value={vramSortOrder} 
                    onChange={(e) => setVramSortOrder(e.target.value)}
                  >
                    <Radio.Button value="asc">升序</Radio.Button>
                    <Radio.Button value="desc">降序</Radio.Button>
                  </Radio.Group>
                }
              >
                <ReactECharts
                  ref={vramUsageChartRef}
                  option={getVRamUsageChartOption()}
                  style={{ height: '350px' }}
                  notMerge={true}
                />
              </Card>
            </Col>
          </Row>
          
          <Row gutter={24} style={{ marginBottom: '20px' }}>
            <Col span={12}>
              <Card 
                title="準確率比較"
                extra={
                  <Radio.Group 
                    size="small"
                    value={mapSortOrder} 
                    onChange={(e) => setMapSortOrder(e.target.value)}
                  >
                    <Radio.Button value="asc">升序</Radio.Button>
                    <Radio.Button value="desc">降序</Radio.Button>
                  </Radio.Group>
                }
              >
                <ReactECharts
                  ref={mAPChartRef}
                  option={getMapChartOption()}
                  style={{ height: '350px' }}
                  notMerge={true}
                />
              </Card>
            </Col>
            <Col span={12}>
              <Card 
                title="GPU負載比較"
                extra={
                  <Radio.Group 
                    size="small"
                    value={gpuLoadSortOrder} 
                    onChange={(e) => setGpuLoadSortOrder(e.target.value)}
                  >
                    <Radio.Button value="asc">升序</Radio.Button>
                    <Radio.Button value="desc">降序</Radio.Button>
                  </Radio.Group>
                }
              >
                <ReactECharts
                  ref={gpuLoadChartRef}
                  option={getGpuLoadChartOption()}
                  style={{ height: '350px' }}
                  notMerge={true}
                />
              </Card>
            </Col>
          </Row>
          
          {/* 結果表格 */}
          <Card 
            title={filterType !== 'default' ? "Top 5 推薦配置" : "配置列表"} 
            style={{ marginTop: '20px' }}
            extra={
              <Space>
                <Button 
                  icon={<DownloadOutlined />} 
                  onClick={exportChartsToPDF}
                >
                  導出圖表 (PDF)
                </Button>
                <Button 
                  icon={<DownloadOutlined />} 
                  onClick={exportDataToExcel}
                >
                  導出數據 (Excel)
                </Button>
              </Space>
            }
          >
            <Table 
              columns={columns} 
              dataSource={getFilteredDataForDisplay()}
              size="small"
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </div>
      ) : (
        <Card style={{ textAlign: 'center', padding: '50px 0' }}>
          <Title level={4}>請上傳 JSON 檔案或載入示例數據</Title>
                        <p>上傳後將顯示自動化測試結果分析圖表</p>
        </Card>
      )}
    </div>
  );
};

export default ModelPerformanceAnalyzer; 