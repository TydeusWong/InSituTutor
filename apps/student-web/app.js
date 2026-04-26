const state = {
  sessionId: null,
  running: false,
  timer: null,
  overlayTimer: null,
  pending: false,
  overlayPending: false,
  overlayUrl: "",
  overlayData: null,
  drawQueued: false,
  resizeObserver: null,
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
  videoStage: document.getElementById("video-stage"),
  videoFeed: document.getElementById("video-feed"),
  videoOverlay: document.getElementById("video-overlay"),
  serverMessage: document.getElementById("server-message"),
  btnStart: document.getElementById("btn-start"),
  btnNext: document.getElementById("btn-next"),
  btnRetry: document.getElementById("btn-retry"),
  btnReset: document.getElementById("btn-reset"),
};

const overlayCtx = el.videoOverlay.getContext("2d");

function setList(target, items) {
  target.innerHTML = "";
  (Array.isArray(items) ? items : []).forEach((text) => {
    const li = document.createElement("li");
    li.textContent = String(text);
    target.appendChild(li);
  });
}

function requestOverlayDraw() {
  if (state.drawQueued) return;
  state.drawQueued = true;
  window.requestAnimationFrame(drawOverlay);
}

function syncOverlayCanvasSize() {
  const rect = el.videoStage.getBoundingClientRect();
  const cssWidth = Math.max(1, Math.round(rect.width));
  const cssHeight = Math.max(1, Math.round(rect.height));
  const dpr = window.devicePixelRatio || 1;
  const pixelWidth = Math.max(1, Math.round(cssWidth * dpr));
  const pixelHeight = Math.max(1, Math.round(cssHeight * dpr));

  if (el.videoOverlay.width !== pixelWidth || el.videoOverlay.height !== pixelHeight) {
    el.videoOverlay.width = pixelWidth;
    el.videoOverlay.height = pixelHeight;
  }
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width: cssWidth, height: cssHeight };
}

function getFrameRect(frameWidth, frameHeight, boxWidth, boxHeight) {
  if (!frameWidth || !frameHeight || !boxWidth || !boxHeight) {
    return null;
  }
  const scale = Math.min(boxWidth / frameWidth, boxHeight / frameHeight);
  const width = frameWidth * scale;
  const height = frameHeight * scale;
  const x = (boxWidth - width) / 2;
  const y = (boxHeight - height) / 2;
  return { x, y, width, height };
}

function drawOverlay() {
  state.drawQueued = false;
  const canvasSize = syncOverlayCanvasSize();
  overlayCtx.clearRect(0, 0, canvasSize.width, canvasSize.height);

  const data = state.overlayData;
  if (!data || !data.ok || !data.fresh) {
    return;
  }

  const frameWidth = Number(data.frame_width || el.videoFeed.naturalWidth || 0);
  const frameHeight = Number(data.frame_height || el.videoFeed.naturalHeight || 0);
  const frameRect = getFrameRect(frameWidth, frameHeight, canvasSize.width, canvasSize.height);
  if (!frameRect) {
    return;
  }

  const detections = Array.isArray(data.detections) ? data.detections : [];
  overlayCtx.lineWidth = 2;
  overlayCtx.font = "600 14px 'Space Grotesk', 'Noto Sans SC', sans-serif";
  overlayCtx.textBaseline = "top";

  detections.forEach((det) => {
    const bbox = (det && det.bbox_xyxy) || {};
    const x1 = frameRect.x + Number(bbox.x1 || 0) * frameRect.width;
    const y1 = frameRect.y + Number(bbox.y1 || 0) * frameRect.height;
    const x2 = frameRect.x + Number(bbox.x2 || 0) * frameRect.width;
    const y2 = frameRect.y + Number(bbox.y2 || 0) * frameRect.height;
    const w = Math.max(0, x2 - x1);
    const h = Math.max(0, y2 - y1);
    if (!w || !h) return;

    const label = String(det.label || "");
    const score = Number(det.score || 0);
    const text = label ? `${label} ${score.toFixed(2)}` : score.toFixed(2);

    overlayCtx.strokeStyle = "#27d36f";
    overlayCtx.fillStyle = "#27d36f";
    overlayCtx.strokeRect(x1, y1, w, h);
    overlayCtx.fillRect(x1, Math.max(frameRect.y, y1 - 22), Math.max(72, text.length * 8 + 10), 20);
    overlayCtx.fillStyle = "#08111c";
    overlayCtx.fillText(text, x1 + 5, Math.max(frameRect.y + 2, y1 - 20));
  });

  const status = data.matched ? "MATCHED" : "RUNNING";
  const stepId = data.step_id || "-";
  overlayCtx.fillStyle = data.matched ? "#1cd35c" : "#ffb020";
  overlayCtx.fillRect(frameRect.x + 10, frameRect.y + 10, Math.max(136, (stepId.length + status.length) * 8 + 22), 24);
  overlayCtx.fillStyle = "#08111c";
  overlayCtx.fillText(`${stepId} | ${status}`, frameRect.x + 16, frameRect.y + 14);
}

function attachOverlayObservers() {
  window.addEventListener("resize", requestOverlayDraw);
  el.videoFeed.addEventListener("load", requestOverlayDraw);
  if ("ResizeObserver" in window) {
    state.resizeObserver = new ResizeObserver(() => requestOverlayDraw());
    state.resizeObserver.observe(el.videoStage);
  }
}

function render(snapshot) {
  if (!snapshot) return;
  const section = snapshot.section || {};
  const step = snapshot.step || {};
  const lastEval = snapshot.last_eval || {};
  const lastOmni = snapshot.last_omni || {};

  el.statusPill.textContent = snapshot.state || "-";
  el.sectionName.textContent = section.section_name || "Current section";
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

  const videoUrl = snapshot.video_primary_url || snapshot.video_feed_url || snapshot.video_feed_stream_url || snapshot.video_feed_overlay_url;
  if (videoUrl && el.videoFeed.dataset.streamUrl !== videoUrl) {
    el.videoFeed.dataset.streamUrl = videoUrl;
    el.videoFeed.src = videoUrl;
  }

  state.overlayUrl = snapshot.overlay_meta_url || "";
  requestOverlayDraw();

  if (snapshot.is_done) {
    state.running = false;
    state.overlayData = null;
    stopLoop();
    requestOverlayDraw();
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

async function overlayTick() {
  if (!state.running || state.overlayPending || !state.overlayUrl) return;
  state.overlayPending = true;
  try {
    const data = await api(state.overlayUrl);
    state.overlayData = data;
    requestOverlayDraw();
  } catch (err) {
    state.overlayData = null;
    requestOverlayDraw();
  } finally {
    state.overlayPending = false;
  }
}

function startLoop() {
  if (!state.timer) {
    state.timer = setInterval(evaluateTick, 300);
  }
  if (!state.overlayTimer) {
    state.overlayTimer = setInterval(overlayTick, 100);
  }
  overlayTick();
}

function stopLoop() {
  if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
  if (state.overlayTimer) {
    clearInterval(state.overlayTimer);
    state.overlayTimer = null;
  }
}

el.btnStart.addEventListener("click", startSession);
el.btnNext.addEventListener("click", nextSection);
el.btnRetry.addEventListener("click", retrySection);
el.btnReset.addEventListener("click", resetSession);

attachOverlayObservers();

initSession().catch((err) => {
  el.serverMessage.textContent = `初始化失败: ${err.message}`;
});
