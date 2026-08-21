# RoadSense-Perception 中文简历与面试案例

## 一句话定位

面向通用计算机视觉与视频算法岗位的多任务感知平台：统一处理目标检测、语义分割和多目标跟踪，并把几何坐标、时序关联、评测协议和可审计演示放进同一条工程链路。

## 当前可引用事实

- 已实现严格帧、框、检测、数据清单和评测报告契约。
- 已实现 IoU 几何匹配、检测 AP、语义分割 mIoU、MOTA/identity-F1 协议和可重置 IoU Tracker。
- 已实现 24 帧确定性道路场景、FastAPI 接口、Pages Workbench、逐帧回放、图层开关、置信度显示过滤和证据页。
- 本地首轮门禁为 `53 passed`；fixture 只验证接口与指标管线，不构成公开数据集结果。

## 简历 Bullet（真实数据授权后再加入指标）

- 设计多任务 2D 视觉感知平台，统一检测、语义分割与多目标跟踪输出，固定原始分辨率坐标、帧序列和类别本体，避免各任务结果无法对齐。
- 实现确定性 IoU 关联与 track aging，显式统计漏检、误检、ID switch，并为检测 AP、分割混淆矩阵和跟踪 MOTA 建立独立评测协议。
- 搭建 FastAPI + GitHub Pages Perception Workbench，支持逐帧回放、检测框、语义区域、轨迹尾迹和对象检查；页面明确区分 fixture 证据与 benchmark 结论。
- 建立严格 JSON、数据集 manifest、SHA-256、原子报告和 fail-closed publication gate，为后续 BDD100K 真实评测保留可审计证据链。

## 面试深挖点

1. 为什么检测、分割和跟踪必须拆开评测，不能合成一个总分？
2. IoU 关联在遮挡、短暂漏检和重新出现时会发生什么？如何识别 ID switch？
3. 为什么 Pages 的高 fixture 分数不能写成 COCO mAP 或 BDD100K 结果？
4. 如何设计 sequence-disjoint split，避免相邻视频帧泄漏到 Final？
5. 真正接入 ONNX/Torch 模型时，如何绑定预处理、类别表、权重哈希和设备？

## 证据边界

当前项目不能宣称通用检测精度、跟踪质量、实时 FPS、自动驾驶安全或生产部署能力。完成 BDD100K 真实评测前，简历应写“多任务视觉感知工程平台”，不要写成“自动驾驶感知模型 SOTA”。

