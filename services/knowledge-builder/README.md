# knowledge-builder（TODO 第 3 部分实现）

本目录用于实现 `TODOlist.md` 第 3 部分：制作教学提示与纠错提示。

## 已覆盖能力

### 3.1 提示模板体系

- 提示层级：`pre_notice` / `step_hint` / `instant_correction` / `safety_alert`
- 每个 step 自动补齐：
  - `next_step_hint`
  - `common_mistake_warning`
  - `if_error_then_intervention`
- 输出模式：`text_only`（后续可扩展 TTS）

### 3.2 提示触发策略（MVP）

- 按步骤状态触发：
  - `not_started` -> 预告提示
  - `in_progress` -> 步骤提示
  - `stuck` -> 步骤提示（卡住分支）
  - `deviation` -> 即时纠偏
- 按时间触发：
  - `elapsed_in_step_sec > timeout_hint_sec` -> 温和步骤提示
- 按错误模式触发：
  - 命中 `error_type` -> 即时纠偏
- 防提示轰炸：
  - `cooldown_sec` 冷却
  - `max_repeat` 最大重复次数

### 3.3 文案策略

- 语气规则：简短、可执行、先肯定后纠偏
- 禁用文案过滤：模糊 / 不可执行 / 过度责备（基础词表）

## 文件说明

- `build_knowledge.py`
  - 输入：`data/processed/analysis/structured_teaching_knowledge.json`
  - 输出：`data/processed/knowledge/prompt_knowledge.json`
  - 作用：构建提示知识、补齐 step 文案字段、生成错误目录等

- `prompt_engine.py`
  - 运行时提示引擎（按事件产出提示）
  - 支持 4 层提示优先级、状态/超时/错误触发、冷却抑制

- `demo_run.py`
  - 演示脚本，模拟事件流并打印提示输出

## 运行方式

在项目根目录执行：

```powershell
python services/knowledge-builder/build_knowledge.py
python services/knowledge-builder/demo_run.py
```

## 下一步建议

- 将 `prompt_engine.py` 接入 `realtime-coach` 事件流
- 把 `prompt` 输出直接传给 `apps/student-web` 展示
- 增加文案分档（新手/熟练者）
