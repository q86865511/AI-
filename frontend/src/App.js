import React, { useState } from 'react';
import { Route, Routes, Link } from 'react-router-dom';
import { Layout, Menu, Typography, ConfigProvider } from 'antd';
import zhTW from 'antd/locale/zh_TW';
import {
  HomeOutlined,
  CloudUploadOutlined,
  DashboardOutlined,
  FundViewOutlined,
  BarChartOutlined
} from '@ant-design/icons';
import './App.css';

// 導入頁面組件
import HomePage from './pages/HomePage';
import ModelsPage from './pages/ModelsPage';
import ConversionPage from './pages/ConversionPage';
import BenchmarkPage from './pages/BenchmarkPage';
import PerformanceAnalyzerPage from './pages/PerformanceAnalyzerPage';

const { Header, Sider, Content, Footer } = Layout;
const { Title } = Typography;

function App() {
  const [collapsed, setCollapsed] = useState(false);

  const toggleCollapsed = () => {
    setCollapsed(!collapsed);
  };

  return (
    <ConfigProvider locale={zhTW}>
      <Layout style={{ minHeight: '100vh' }}>
        <Sider 
          collapsible 
          collapsed={collapsed} 
          onCollapse={toggleCollapsed}
          theme="dark"
        >
            <div style={{ height: 64, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Title level={4} style={{ color: 'white', margin: 0 }}>
                {!collapsed ? 'AI部署平台' : 'AI'}
              </Title>
            </div>
            <Menu
              theme="dark"
              defaultSelectedKeys={['1']}
              mode="inline"
              items={[
                {
                  key: '1',
                  icon: <HomeOutlined />,
                  label: <Link to="/">首頁</Link>,
                },
                {
                  key: '2',
                  icon: <CloudUploadOutlined />,
                  label: <Link to="/upload">模型優化</Link>,
                },
                {
                  key: '3',
                  icon: <DashboardOutlined />,
                  label: <Link to="/models">模型管理</Link>,
                },
                {
                  key: '4',
                  icon: <FundViewOutlined />,
                  label: <Link to="/benchmark">自動化轉換與測試</Link>,
                },
                {
                  key: '6',
                  icon: <BarChartOutlined />,
                  label: <Link to="/performance-analyzer">自動化結果分析</Link>,
                }
              ]}
            />
          </Sider>
          <Layout>
            <Header style={{ 
              background: '#fff', 
              padding: '0 24px',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
              display: 'flex',
              alignItems: 'center'
            }}>
              <Title level={3} style={{ margin: 0 }}>
                具自動化測試模型優化和即時推論資源監控之AI部署平台
              </Title>
            </Header>
            <Content style={{ margin: '24px 16px' }}>
              <div style={{ padding: 24, background: '#fff', minHeight: 360 }}>
                <Routes>
                  <Route path="/" element={<HomePage />} />
                  <Route path="/upload" element={<ConversionPage />} />
                  <Route path="/models" element={<ModelsPage />} />
                  <Route path="/benchmark" element={<BenchmarkPage />} />
                  <Route path="/performance-analyzer" element={<PerformanceAnalyzerPage />} />
                </Routes>
              </div>
            </Content>
            <Footer style={{ textAlign: 'center' }}>
              AI部署平台 ©2024
            </Footer>
          </Layout>
        </Layout>
      </ConfigProvider>
    );
}

export default App; 