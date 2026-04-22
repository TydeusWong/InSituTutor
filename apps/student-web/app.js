const state = {
  sessionId: null,
  running: false,
  timer: null,
  pending: false,
};

const el = {
  statusPill: document.getElementById("status-pill"),
  sectionName: document.getElementById("section-name"),
  sectionGoal: document.getElementById("section-goal"),
  stepPrompt: document.getElementById("step-prompt"),
  focusList: document.getElementById("focus-list"),
  mistakeList: document.getElementById("mistake-list"),
  sectionProgress: document.getElementById("section-progress"),
  stepProgress: document.getElementById("step-progress"),
  detectorSource: document.getElementById("detector-source"),
  omniResult: document.getElementById("omni-result"),
  videoFeed: document.getElementById("video-feed"),
  serverMessage: document.getElementById("server-message"),
  btnStart: document.getElementById("btn-start"),
  btnNext: document.getElementById("btn-next"),
  btnRetry: document.getElementById("btn-retry"),
  btnReset: document.getElementById("btn-reset"),
};

function setList(target, items) {
  target.innerHTML = "";
  (Array.isArray(items) ? items : []).forEach((text) => {
    const li = document.createElement("li");
    li.textContent = String(text);
    target.appendChild(li);
  });
}

function render(snapshot) {
  if (!snapshot) return;
  const section = snapshot.section || {};
  const step = snapshot.step || {};
  const lastEval = snapshot.last_eval || {};
  const lastOmni = snapshot.last_omni || {};

  el.statusPill.textContent = snapshot.state || "-";
  el.sectionName.textContent = section.section_name || "未进入章节";
  el.sectionGoal.textContent = section.section_goal || "-";
  const zh = (step.prompt || {}).zh;
  const en = (step.prompt || {}).en;
  el.stepPrompt.textContent = zh || en || snapshot.message || "-";
  setList(el.focusList, step.focus_points || []);
  setList(el.mistakeList, step.common_mistakes || []);
  el.sectionProgress.textContent = `${(section.index ?? 0) + 1}/${section.count ?? 0}`;
  el.stepProgress.textContent = `${(step.index ?? 0) + 1}/${step.count ?? 0}`;
  el.detectorSource.textContent = lastEval.detector_source || "-";
  if (typeof lastOmni.confidence === "number") {
    el.omniResult.textContent = `${lastOmni.pass ? "PASS" : "RETRY"} (${lastOmni.confidence.toFixed(3)})`;
  } else {
    el.omniResult.textContent = "-";
  }
  el.serverMessage.textContent = snapshot.message || "";

  if (snapshot.video_feed_url && el.videoFeed.src !== snapshot.video_feed_url) {
    el.videoFeed.src = snapshot.video_feed_url;
  }

  if (snapshot.is_done) {
    state.running = false;
    stopLoop();
  }
}

async function api(path, method = "GET", body = null) {
  const opts = { method, headers: {} };
  if (body) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(path, opts);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${text}`);
  }
  return await resp.json();
}

async function initSession() {
  const data = await api("/api/realtime/session/init");
  state.sessionId = data.session_id;
  render(data);
}

async function startSession() {
  const data = await api("/api/realtime/session/start", "POST", { session_id: state.sessionId });
  render(data);
  state.running = true;
  startLoop();
}

async function nextSection() {
  const data = await api("/api/realtime/section/next", "POST", { session_id: state.sessionId });
  render(data);
}

async function retrySection() {
  const data = await api("/api/realtime/section/retry", "POST", { session_id: state.sessionId });
  render(data);
  state.running = true;
  startLoop();
}

async function resetSession() {
  const data = await api("/api/realtime/session/reset", "POST", { session_id: state.sessionId });
  render(data);
  state.running = true;
  startLoop();
}

async function evaluateTick() {
  if (!state.running || state.pending) return;
  state.pending = true;
  try {
    const data = await api("/api/realtime/step/evaluate", "POST", { session_id: state.sessionId });
    render(data);
  } catch (err) {
    el.serverMessage.textContent = `请求失败: ${err.message}`;
  } finally {
    state.pending = false;
  }
}

function startLoop() {
  if (state.timer) return;
  state.timer = setInterval(evaluateTick, 300);
}

function stopLoop() {
  if (!state.timer) return;
  clearInterval(state.timer);
  state.timer = null;
}

el.btnStart.addEventListener("click", startSession);
el.btnNext.addEventListener("click", nextSection);
el.btnRetry.addEventListener("click", retrySection);
el.btnReset.addEventListener("click", resetSession);

initSession().catch((err) => {
  el.serverMessage.textContent = `初始化失败: ${err.message}`;
});
