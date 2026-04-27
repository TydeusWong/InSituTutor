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
  selfEvolutionRecordingEnabled: false,
  mediaRecorder: null,
  mediaStream: null,
  recordedChunks: [],
  recordingStarted: false,
  mediaUploaded: false,
  recordingDiscarded: false,
  captureTarget: null,
  captureActual: null,
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
  errorCorrection: document.getElementById("error-correction"),
  errorMessage: document.getElementById("error-message"),
  errorEvidence: document.getElementById("error-evidence"),
  selfEvolutionRecording: document.getElementById("self-evolution-recording"),
  recordingStatus: document.getElementById("recording-status"),
  recordingMeta: document.getElementById("recording-meta"),
  btnStart: document.getElementById("btn-start"),
  btnNextStep: document.getElementById("btn-next-step"),
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
    const skipped = String(lastOmni.reason || "").includes("omni_validation_skipped");
    const status = skipped ? "SKIPPED/PASS" : (lastOmni.pass ? "PASS" : "RETRY");
    el.omniResult.textContent = `${status} (${lastOmni.confidence.toFixed(3)})`;
  } else {
    el.omniResult.textContent = "-";
  }
  el.serverMessage.textContent = snapshot.message || "";
  renderErrorCorrection(snapshot.last_error || null);
  renderRecordingState(snapshot);

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
    finalizeSelfEvolutionRecording().catch((err) => {
      el.serverMessage.textContent = `Recording upload failed: ${err.message}`;
    });
  }
}

function renderErrorCorrection(lastError) {
  if (!el.errorCorrection || !el.errorMessage || !el.errorEvidence) return;
  if (!lastError || !lastError.error_id) {
    el.errorCorrection.classList.add("hidden");
    el.errorMessage.textContent = "";
    el.errorEvidence.textContent = "";
    el.stepPrompt.classList.remove("step-prompt-error");
    return;
  }
  const evidence = lastError.evidence || {};
  el.errorCorrection.classList.remove("hidden");
  el.errorMessage.textContent = lastError.message || "Please correct the current action before continuing.";
  el.errorEvidence.textContent = `${lastError.error_id} | ${evidence.detector_source || "detector"} | ${evidence.matched_condition || ""}`;
  el.stepPrompt.classList.add("step-prompt-error");
}

function renderRecordingState(snapshot) {
  const se = (snapshot && snapshot.self_evolution) || {};
  const capture = se.capture || {};
  if (capture.target_width && capture.target_height) {
    state.captureTarget = {
      width: Number(capture.target_width || 0),
      height: Number(capture.target_height || 0),
      fps: Number(capture.target_fps || 0),
    };
  }
  const target = state.captureTarget;
  const actual = state.captureActual || {
    width: Number(capture.actual_width || 0),
    height: Number(capture.actual_height || 0),
    fps: Number(capture.actual_fps || 0),
  };
  const targetText = target && target.width
    ? `Target: ${target.width}x${target.height}@${target.fps || "-"}fps`
    : "Target: -";
  const actualText = actual && actual.width
    ? `Actual: ${actual.width}x${actual.height}@${actual.fps || "-"}fps`
    : "Actual: -";
  el.recordingMeta.textContent = `Video: IP camera | Audio: microphone | ${targetText} | ${actualText}`;
  if (state.recordingStarted) {
    el.recordingStatus.textContent = `Recording session ${state.sessionId || "-"}`;
    el.recordingStatus.classList.add("active");
  } else if (state.mediaUploaded) {
    el.recordingStatus.textContent = "Recording uploaded";
    el.recordingStatus.classList.remove("active");
  } else if (state.recordingDiscarded) {
    el.recordingStatus.textContent = "Recording discarded";
    el.recordingStatus.classList.remove("active");
  } else {
    el.recordingStatus.textContent = state.selfEvolutionRecordingEnabled ? "Recording armed" : "Recording off";
    el.recordingStatus.classList.toggle("active", state.selfEvolutionRecordingEnabled);
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

function getPreferredMimeType() {
  const candidates = [
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm",
  ];
  return candidates.find((x) => window.MediaRecorder && MediaRecorder.isTypeSupported(x)) || "";
}

async function startSelfEvolutionRecording() {
  state.recordedChunks = [];
  state.mediaUploaded = false;
  state.recordingDiscarded = false;
  const target = state.captureTarget || { width: 1280, height: 720, fps: 10 };
  const constraints = { audio: true, video: false };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  state.mediaStream = stream;
  const audioTrack = stream.getAudioTracks()[0];
  const settings = audioTrack ? audioTrack.getSettings() : {};
  state.captureActual = {
    actual_width: target.width || 0,
    actual_height: target.height || 0,
    actual_fps: target.fps || 0,
    audio_sample_rate: Number(settings.sampleRate || 0),
    audio_channel_count: Number(settings.channelCount || 0),
    video_source: "ipcam",
    audio_source: "browser_microphone",
  };
  const mimeType = getPreferredMimeType();
  const options = mimeType ? { mimeType } : {};
  const recorder = new MediaRecorder(stream, options);
  recorder.addEventListener("dataavailable", (event) => {
    if (event.data && event.data.size > 0) {
      state.recordedChunks.push(event.data);
    }
  });
  recorder.start(1000);
  state.mediaRecorder = recorder;
  state.recordingStarted = true;
  renderRecordingState({});
}

async function uploadSelfEvolutionRecording() {
  if (!state.selfEvolutionRecordingEnabled || state.mediaUploaded || !state.recordedChunks.length) return;
  const mimeType = (state.mediaRecorder && state.mediaRecorder.mimeType) || "video/webm";
  const blob = new Blob(state.recordedChunks, { type: mimeType });
  const resp = await fetch(`/api/self-evolution/session/media?session_id=${encodeURIComponent(state.sessionId)}`, {
    method: "POST",
    headers: { "Content-Type": mimeType, "X-Session-Id": state.sessionId || "" },
    body: blob,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${text}`);
  }
  await resp.json();
  const finishData = await api("/api/self-evolution/session/finish", "POST", { session_id: state.sessionId });
  state.mediaUploaded = true;
  state.recordedChunks = [];
  if (finishData.review_media_path) {
    el.serverMessage.textContent = `Review media saved: ${finishData.review_media_path}`;
  } else if (finishData.review_media_error) {
    el.serverMessage.textContent = `Review media mux failed: ${finishData.review_media_error}`;
  }
  renderRecordingState({});
}

async function finalizeSelfEvolutionRecording() {
  if (!state.selfEvolutionRecordingEnabled || !state.recordingStarted) return;
  const recorder = state.mediaRecorder;
  if (recorder && recorder.state !== "inactive") {
    await new Promise((resolve) => {
      recorder.addEventListener("stop", resolve, { once: true });
      recorder.stop();
    });
  }
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach((track) => track.stop());
  }
  state.recordingStarted = false;
  state.mediaRecorder = null;
  state.mediaStream = null;
  const keep = window.confirm("是否保留本次 Self-evolution recording 数据？选择“取消”会舍弃本次录音录像和 v5 session 数据。");
  if (keep) {
    await uploadSelfEvolutionRecording();
    return;
  }
  state.recordedChunks = [];
  state.recordingDiscarded = true;
  await api("/api/self-evolution/session/discard", "POST", { session_id: state.sessionId }).catch(() => {});
  renderRecordingState({});
}

async function startSession() {
  state.selfEvolutionRecordingEnabled = Boolean(el.selfEvolutionRecording && el.selfEvolutionRecording.checked);
  try {
    if (state.selfEvolutionRecordingEnabled) {
      await startSelfEvolutionRecording();
    }
    const data = await api("/api/realtime/session/start", "POST", {
      session_id: state.sessionId,
      self_evolution_recording_enabled: state.selfEvolutionRecordingEnabled,
      capture_meta: state.captureActual || {},
    });
    render(data);
    state.running = true;
    startLoop();
  } catch (err) {
    if (state.recordingStarted) {
      await finalizeSelfEvolutionRecording().catch(() => {});
    }
    el.serverMessage.textContent = `Start failed: ${err.message}`;
  }
}

async function nextSection() {
  const data = await api("/api/realtime/section/next", "POST", { session_id: state.sessionId });
  render(data);
}

async function nextStep() {
  const data = await api("/api/realtime/step/next", "POST", { session_id: state.sessionId });
  render(data);
  state.running = true;
  startLoop();
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
el.btnNextStep.addEventListener("click", nextStep);
el.btnNext.addEventListener("click", nextSection);
el.btnRetry.addEventListener("click", retrySection);
el.btnReset.addEventListener("click", resetSession);
if (el.selfEvolutionRecording) {
  el.selfEvolutionRecording.addEventListener("change", () => {
    state.selfEvolutionRecordingEnabled = Boolean(el.selfEvolutionRecording.checked);
    renderRecordingState({});
  });
}

attachOverlayObservers();

initSession().catch((err) => {
  el.serverMessage.textContent = `初始化失败: ${err.message}`;
});
