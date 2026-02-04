# 扬州空气质量预测系统

一个端到端的空气质量预测系统，展示完整的数据科学工作流程——从数据采集、清洗、特征工程、模型训练到预测和验证。

## 项目特点

- **多源数据整合**：WAQI 实时数据、quotsoft.net 历史数据、Open-Meteo 天气数据
- **完整特征工程**：150+ 候选特征，涵盖时间、滞后、滚动、气象交互等类别
- **专业模型评估**：基线对比、交叉验证、消融实验、SHAP 解释
- **可视化 Dashboard**：8 个 Tab 页面展示完整分析过程

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行完整管道

```bash
python run_pipeline.py
```

这将依次执行：
1. 下载历史空气质量数据
2. 下载历史天气数据
3. 获取节假日数据
4. 数据清洗
5. 数据合并
6. 特征工程
7. 特征筛选
8. 模型训练
9. SHAP 模型解释
10. 生成预测数据

### 3. 查看 Dashboard

直接用浏览器打开 `dashboard/index.html`

## 项目结构

```
yangzhou-aqi-prediction/
├── README.md                    # 项目说明
├── requirements.txt             # Python 依赖
├── run_pipeline.py              # 一键运行脚本
├── data/
│   ├── raw/                     # 原始数据
│   │   ├── historical_aqi/      # quotsoft.net 历史空气质量
│   │   └── historical_weather/  # Open-Meteo 历史天气
│   ├── processed/               # 清洗后的数据
│   └── features/                # 特征工程后的数据
├── src/
│   ├── config.py                # 配置文件
│   ├── data_collection/         # 数据采集脚本
│   ├── data_processing/         # 数据处理
│   ├── feature_engineering/     # 特征工程
│   └── modeling/                # 模型训练
├── models/                      # 保存的模型文件
├── dashboard/                   # 前端展示
│   ├── index.html               # 主页面
│   └── data/                    # Dashboard 数据
└── docs/                        # 文档
```

## API Keys

项目已配置以下 API：

- **WAQI**: 空气质量实时数据
- **高德地图**: 天气和交通数据
- **Open-Meteo**: 历史天气数据（免费，无需 Key）
- **quotsoft.net**: 历史空气质量（免费，无需 Key）

## 模型性能

| 模型 | MAE | RMSE | R² |
|------|-----|------|-----|
| Naive (24h) | 22.4 | 31.2 | 0.78 |
| Moving Avg | 19.8 | 27.5 | 0.82 |
| Linear | 15.3 | 21.8 | 0.88 |
| **XGBoost** | **12.5** | **18.3** | **0.92** |

## 特征类别

- **时间特征**: 小时、星期、月份、季节、节假日
- **滞后特征**: 1h, 3h, 6h, 12h, 24h, 48h 滞后值
- **滚动统计**: 6h, 12h, 24h, 48h 窗口的均值、标准差
- **气象特征**: 温度、湿度、风速、气压、降水
- **交互特征**: 温湿度乘积、风向分解、气压变化
- **代理变量**: 逆温层、秸秆焚烧、供暖季

## 局限性

- 无法预测突发事件（工厂排放、交通事故）
- 跨区域污染传输存在不确定性
- 超过 24 小时的预测准确率下降
- 极端天气情况预测能力有限

## License

MIT License
