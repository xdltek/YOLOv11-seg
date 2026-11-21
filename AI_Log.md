# AI Development Log

| Date | Stage | Prompt Summary | AI Output Summary | Manual Changes | Final Status | Reviewer | Artifact/Commit |
|---|---|---|---|---|---|---|---|
| 2026-11-21 | Release scope consolidation | 收敛为 640×640 n/s/m/l/x 与 960×960 n/s，清理实验修复并更新双语支持与性能文档 | 删除实验上采样替换、sigmoid 工具与专用测试；保留支持路径的精度修正；新增按图输入/首层卷积校验发布边界；同步七规格支持表与带来源的性能表 | ONNX/SDK/服务未修改；原始历史工程记录留存在本地清理证据目录；640 性能沿用 1.6.11，960 使用最近十帧实测，均未标为本次新跑分 | Build and host checks PASS; fresh target regression/performance pending while device busy | local workspace | `build/release_scope_20261121/`; `README.md`; `ReleaseNote.md` |
| 2026-11-21 | Reboot regression and fresh performance | 补齐支持规格性能，复验精度修正及 busy 关联 | 七规格 RGB/I420 十帧测量和严格精度复验；修复 x640 图事件池不足的启动配置；检查范围外诊断及被撤回的排序 kernel 证据 | 原始 ONNX 与安装 SDK 不变；仅生成本地 RPP_CFG_FILE；生产支持边界不扩展 | PASS: 14 performance cases, 28 accuracy frames, three retained-module regressions; no busy reproduced | local workspace | `build/reboot_release_20261121/`; `README.md`; `ReleaseNote.md` |

## Current delivery

- Release scope: YOLO11 n/s/m/l/x at 640×640; YOLO11 n/s at 960×960.
- Each application process builds a fresh engine from original ONNX. No local
  engine cache, serialization, deserialization or prepared-model requirement.
- Retained RPP convolution precision corrections, DFL/prototype adaptation,
  FP32 box decoding and stable RT NMS compaction serve this release scope.
- Model files are supplied locally and remain excluded from Git and release packages.
- `scripts/` contains only build, run and benchmark entry points.
- All seven profiles have fresh RGB/I420 performance and sample accuracy evidence
  after reboot. Performance remains slower than the original 1.6.11 target.
- The startup script configures 512 common / 3456 graph events in a local server
  configuration, respecting explicit overrides and preserving installed SDK files.
- Device box/DFL, stable compaction and precision-module regressions pass. This
  audit did not reproduce busy; it does not prove its absence under every workload.
