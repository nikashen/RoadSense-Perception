# RoadSense-Perception 中文简历与面试案例

## 一句话定位

面向通用计算机视觉与视频算法岗位的多任务感知平台：统一处理目标检测、语义分割和多目标跟踪，并把几何坐标、时序关联、评测协议和可审计演示放进同一条工程链路。

## 当前可引用事实

- 已实现严格帧、框、检测、数据清单和评测报告契约。
- 已实现 IoU 几何匹配、检测 AP、语义分割 mIoU、MOTA/identity-F1 协议和可重置 IoU Tracker。
- 已实现 24 帧确定性道路场景、FastAPI 接口、Pages Workbench、逐帧回放、图层开关、置信度显示过滤和证据页。
- 已实现 `roadsense evaluate-local` 本地序列评测入口：拆分校验、帧/时间戳对齐、NPY 掩码安全边界、逐序列与聚合诊断，以及输入/报告哈希。
- 已实现模型工件 manifest、SHA-256 验证收据和线程安全 AdapterRegistry；未显式验证的权重不能进入受保护的适配器路径。
- 已用本地 COCO8 四张验证图和 SHA-256 校验的 YOLO11n ONNX 工件跑通真实数据开发烟测：RoadSense compact AP@0.50=0.650、precision=0.692、recall=0.529；报告明确标记为 development evidence，不冒充 COCO mAP。
- 已实现 `roadsense runtime-audit` fixture 运行审计：阶段耗时、依赖/设备信息和输入输出哈希均可复核，且明确不把 fixture 吞吐写成模型 FPS。
- 本地门禁要求测试、静态检查、类型检查和打包全部通过；fixture 只验证接口与指标管线，不构成公开数据集结果。

## 简历 Bullet（指标须带数据集与证据边界）

- 设计多任务 2D 视觉感知平台，统一检测、语义分割与多目标跟踪输出，固定原始分辨率坐标、帧序列和类别本体，避免各任务结果无法对齐。
- 实现确定性 IoU 关联与 track aging，显式统计漏检、误检、ID switch，并为检测 AP、分割混淆矩阵和跟踪 MOTA 建立独立评测协议。
- 搭建 FastAPI + GitHub Pages Perception Workbench，支持逐帧回放、检测框、语义区域、轨迹尾迹和对象检查；页面明确区分 fixture 证据与 benchmark 结论。
- 建立严格 JSON、数据集/模型 manifest、SHA-256、原子报告和 fail-closed publication gate；新增 BDD100K Detection 2020 val 的官方 devkit 评测闭环（数据准备、冻结清单、无标签推理、双次独立 evaluator、脱敏 receipt），但在真实许可数据和两次官方结果完成前不发布 BDD 指标。
- 编写独立 COCO8+YOLO11n ONNX 本地评测 runner，冻结 letterbox/阈值/NMS/CPUExecutionProvider，产出可复核的模型、数据、依赖锁和报告哈希；该权重在 BDD lane 中只作为明确标注的 COCO 跨域 baseline，四图结果仅作为 development smoke evidence，不宣称官方 COCO 或 BDD benchmark。
- 设计与模型框架解耦的适配器协议，冻结输入预处理、输出本体、运行时版本和量化/校准元数据，便于后续接入 ONNX Runtime、Torch 或 TensorRT 而不污染 Pages 路径。

## 面试深挖点

1. 为什么检测、分割和跟踪必须拆开评测，不能合成一个总分？
2. IoU 关联在遮挡、短暂漏检和重新出现时会发生什么？如何识别 ID switch？
3. 为什么 Pages 的高 fixture 分数不能写成 COCO mAP 或 BDD100K 结果？
4. 如何设计 sequence-disjoint split，避免相邻视频帧泄漏到 Final？
5. 真正接入 ONNX/Torch 模型时，如何绑定预处理、类别表、权重哈希和设备？
6. 为什么本地预测文件评测仍必须记录模型工件收据？如何防止“换权重不换报告”？
7. 如何让 runtime audit 区分 fixture pipeline 吞吐、模型 inference latency 和浏览器 rendering 时间？

## 证据边界

当前项目不能宣称通用检测精度、跟踪质量、实时 FPS、自动驾驶安全或生产部署能力。完成并公开 BDD100K 官方 receipt 前，简历应写“多任务视觉感知工程平台与可审计评测链路”，不要写成“自动驾驶感知模型 SOTA”。
