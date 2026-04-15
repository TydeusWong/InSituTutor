# Input Context

- 任务名称: {{task_name}}
- 场景: {{environment}}
- 视频标识: {{video_id}}
- 视频时长(秒): {{duration_sec}}
- FPS: {{fps}}
- 分析模式: {{analysis_mode}}  # full_video 或 segmented
- 分段信息: {{segment_hint}}

# Task

请基于输入视频，输出结构化教学知识 JSON，字段必须满足约定 schema。

重点提取：

1. 教学主步骤（按顺序）
2. 每步关键动作（动作主体、对象、姿态或空间关系）
3. 每步观测点（系统在线可检测）
4. 每步常见错误与触发条件
5. 每个错误对应的纠偏策略与提示文案
6. 可用于 MVP 在线判断的简化规则（顺序/存在性/姿态粗判/空间关系/超时）

# Extra Requirements

- 如果有老师口头讲解，请综合语音信息增强步骤命名和纠偏建议。
- 若步骤存在并行可能，MVP 仍需给出一个主线顺序流程。
- 输出中的字段命名严格遵循 schema，不要新增未定义字段。
- 仅返回 JSON。
- 严格按照视频信息，减少幻觉。
