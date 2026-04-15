const DATA_PATH = "../../data/processed/analysis/structured_teaching_knowledge.json";

const state = {
  steps: [],
  errorsAndInterventions: [],
  currentIndex: 0,
  meta: null,
};

const loadingStateEl = document.getElementById("loadingState");
const errorStateEl = document.getElementById("errorState");
const stepStateEl = document.getElementById("stepState");
const doneStateEl = document.getElementById("doneState");
const errorTextEl = document.getElementById("errorText");
const metaTextEl = document.getElementById("metaText");
const progressTextEl = document.getElementById("progressText");
const stepTitleEl = document.getElementById("stepTitle");
const timeRangeEl = document.getElementById("timeRange");
const actionsListEl = document.getElementById("actionsList");
const attentionListEl = document.getElementById("attentionList");
const mistakeListEl = document.getElementById("mistakeList");
const completionTextEl = document.getElementById("completionText");
const nextBtn = document.getElementById("nextBtn");
const restartBtn = document.getElementById("restartBtn");

function showOnly(section) {
  loadingStateEl.classList.add("hidden");
  errorStateEl.classList.add("hidden");
  stepStateEl.classList.add("hidden");
  doneStateEl.classList.add("hidden");
  section.classList.remove("hidden");
}

function setList(listEl, items) {
  listEl.innerHTML = "";
  if (!items || items.length === 0) {
    const li = document.createElement("li");
    li.textContent = "暂无";
    listEl.appendChild(li);
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    listEl.appendChild(li);
  });
}

function getInterventionTextsForStep(step) {
  const matched = state.errorsAndInterventions
    .filter((entry) => {
      const warning = step.common_mistake_warning || "";
      return (
        warning.includes(entry.error_type) ||
        warning.includes(entry.trigger_condition) ||
        warning.includes(entry.prompt_text)
      );
    })
    .map((entry) => `【${entry.error_type}】${entry.prompt_text}`);

  const fallback = step.common_mistake_warning
    ? [`易错点：${step.common_mistake_warning}`]
    : ["暂无明确易错点"];

  return matched.length > 0 ? matched : fallback;
}

function renderStep() {
  const total = state.steps.length;
  const index = state.currentIndex;
  const step = state.steps[index];

  progressTextEl.textContent = `步骤 ${index + 1} / ${total}`;
  stepTitleEl.textContent = `${step.step_order}. ${step.step_name}`;
  timeRangeEl.textContent = `时间范围：${step.time_range || "未知"}`;

  const actionTexts = (step.actions || []).map(
    (a) => `${a.description || a.action_type || "动作信息缺失"}`
  );
  setList(actionsListEl, actionTexts);
  setList(attentionListEl, step.attention_cues || []);
  setList(mistakeListEl, getInterventionTextsForStep(step));

  completionTextEl.textContent =
    step.completion_criterion ||
    (step.completion_criteria && step.completion_criteria.join("；")) ||
    "暂无";

  nextBtn.textContent = index === total - 1 ? "完成" : "下一步";
}

function onNext() {
  if (state.currentIndex < state.steps.length - 1) {
    state.currentIndex += 1;
    renderStep();
    return;
  }
  showOnly(doneStateEl);
}

function onRestart() {
  state.currentIndex = 0;
  renderStep();
  showOnly(stepStateEl);
}

async function init() {
  showOnly(loadingStateEl);
  try {
    const response = await fetch(DATA_PATH);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}：无法读取数据文件`);
    }
    const data = await response.json();
    state.steps = Array.isArray(data.steps) ? data.steps : [];
    state.errorsAndInterventions = Array.isArray(data.errors_and_interventions)
      ? data.errors_and_interventions
      : [];
    state.meta = data.meta || {};

    if (state.steps.length === 0) {
      throw new Error("JSON 中未找到有效 steps");
    }

    const taskName = state.meta.task_name || "未命名任务";
    const sceneId = state.meta.scene_id || "未知场景";
    metaTextEl.textContent = `任务：${taskName} | 场景：${sceneId} | 共 ${state.steps.length} 步`;

    renderStep();
    showOnly(stepStateEl);
  } catch (error) {
    errorTextEl.textContent = String(error.message || error);
    showOnly(errorStateEl);
    metaTextEl.textContent = "数据加载失败";
  }
}

nextBtn.addEventListener("click", onNext);
restartBtn.addEventListener("click", onRestart);

init();
