const state = {
  defaults: null,
  referenceFrame: null,
  streamFrame: null,
  referenceImage: null,
  streamImage: null,
  objects: [],
  activeObjectId: null,
  detections: [],
  objectScores: [],
  warnings: [],
  socket: null,
  drawing: null,
};

const elements = {
  failedDataset: document.querySelector("#failedDataset"),
  replayDataset: document.querySelector("#replayDataset"),
  urdfPath: document.querySelector("#urdfPath"),
  artifactRoot: document.querySelector("#artifactRoot"),
  cameraKey: document.querySelector("#cameraKey"),
  referenceEpisode: document.querySelector("#referenceEpisode"),
  referenceFrame: document.querySelector("#referenceFrame"),
  replayEpisode: document.querySelector("#replayEpisode"),
  reviewStatus: document.querySelector("#reviewStatus"),
  failureCategory: document.querySelector("#failureCategory"),
  reviewNote: document.querySelector("#reviewNote"),
  saveReview: document.querySelector("#saveReview"),
  objectLabel: document.querySelector("#objectLabel"),
  addObject: document.querySelector("#addObject"),
  refineObject: document.querySelector("#refineObject"),
  objectsList: document.querySelector("#objectsList"),
  loadFrames: document.querySelector("#loadFrames"),
  startStream: document.querySelector("#startStream"),
  saveMatch: document.querySelector("#saveMatch"),
  overlayCanvas: document.querySelector("#overlayCanvas"),
  overlayOpacity: document.querySelector("#overlayOpacity"),
  guidanceList: document.querySelector("#guidanceList"),
  warningsList: document.querySelector("#warningsList"),
  jointList: document.querySelector("#jointList"),
  readyBadge: document.querySelector("#readyBadge"),
  scoreBadge: document.querySelector("#scoreBadge"),
  lineage: document.querySelector("#lineage"),
  referenceMeta: document.querySelector("#referenceMeta"),
  streamMeta: document.querySelector("#streamMeta"),
};

async function boot() {
  state.defaults = await requestJson("/api/defaults");
  elements.failedDataset.value = state.defaults.failed_dataset_path;
  elements.replayDataset.value = state.defaults.mock_robot_dataset_path;
  elements.artifactRoot.value = state.defaults.artifact_root;
  elements.referenceEpisode.value = String(state.defaults.reference_episode_index);
  elements.referenceFrame.value = String(state.defaults.reference_frame_index);
  elements.replayEpisode.value = String(state.defaults.replay_episode_index);
  elements.failureCategory.innerHTML = state.defaults.failure_categories
    .map((category) => `<option value="${escapeHtml(category)}">${escapeHtml(category)}</option>`)
    .join("");
  await refreshDataset();
  ensureDefaultObjects();
  await loadFrames();
  bindEvents();
  renderAll();
}

function bindEvents() {
  elements.failedDataset.addEventListener("change", refreshDataset);
  elements.cameraKey.addEventListener("change", loadFrames);
  elements.referenceEpisode.addEventListener("change", loadFrames);
  elements.referenceFrame.addEventListener("change", loadFrames);
  elements.replayEpisode.addEventListener("change", updateLineage);
  elements.urdfPath.addEventListener("change", () => loadJointComparison(0));
  elements.loadFrames.addEventListener("click", loadFrames);
  elements.saveReview.addEventListener("click", saveReview);
  elements.addObject.addEventListener("click", addObject);
  elements.refineObject.addEventListener("click", refineActiveObject);
  elements.startStream.addEventListener("click", startStream);
  elements.saveMatch.addEventListener("click", saveMatch);
  elements.overlayCanvas.addEventListener("pointerdown", beginRoiDraw);
  elements.overlayCanvas.addEventListener("pointermove", updateRoiDraw);
  elements.overlayCanvas.addEventListener("pointerup", finishRoiDraw);
  elements.overlayOpacity.addEventListener("input", renderAll);
}

async function refreshDataset() {
  const dataset = await requestJson(`/api/dataset?path=${encodeURIComponent(elements.failedDataset.value)}`);
  const selected = elements.cameraKey.value || state.defaults?.camera_key || dataset.camera_keys[0];
  elements.cameraKey.innerHTML = dataset.camera_keys
    .map((key) => `<option value="${escapeHtml(key)}">${escapeHtml(key)}</option>`)
    .join("");
  elements.cameraKey.value = dataset.camera_keys.includes(selected) ? selected : dataset.camera_keys[0];
  updateLineage();
}

async function loadFrames() {
  state.referenceFrame = await fetchFrame(
    elements.failedDataset.value,
    elements.cameraKey.value,
    numberValue(elements.referenceEpisode),
    numberValue(elements.referenceFrame),
  );
  state.streamFrame = await fetchFrame(
    elements.replayDataset.value,
    elements.cameraKey.value,
    numberValue(elements.replayEpisode),
    0,
  );
  state.referenceImage = await loadImage(state.referenceFrame.image);
  state.streamImage = await loadImage(state.streamFrame.image);
  elements.referenceMeta.textContent = `Reference episode ${numberValue(elements.referenceEpisode)}, frame ${numberValue(elements.referenceFrame)}`;
  elements.streamMeta.textContent = `Replay episode ${numberValue(elements.replayEpisode)}, frame 0`;
  updateLineage();
  await loadJointComparison(0);
  renderAll();
}

async function fetchFrame(datasetPath, cameraKey, episodeIndex, frameIndex) {
  const params = new URLSearchParams({
    dataset_path: datasetPath,
    camera_key: cameraKey,
    episode_index: String(episodeIndex),
    frame_index: String(frameIndex),
  });
  return requestJson(`/api/frame?${params.toString()}`);
}

async function saveReview() {
  const result = await postJson("/api/reviews", {
    dataset_path: elements.failedDataset.value,
    episode_index: numberValue(elements.referenceEpisode),
    status: elements.reviewStatus.value,
    failure_category: elements.failureCategory.value,
    note: elements.reviewNote.value,
  });
  elements.saveReview.textContent = result.ok ? "Review Saved" : "Save Review";
  setTimeout(() => (elements.saveReview.textContent = "Save Review"), 1200);
}

function ensureDefaultObjects() {
  if (state.objects.length > 0) return;
  state.objects.push(
    {
      id: crypto.randomUUID(),
      label: "green_block",
      bbox: { x_min: 0.50, y_min: 0.47, x_max: 0.54, y_max: 0.56 },
    },
    {
      id: crypto.randomUUID(),
      label: "gray_blocks",
      bbox: { x_min: 0.55, y_min: 0.34, x_max: 0.61, y_max: 0.50 },
    },
  );
  state.activeObjectId = state.objects[0].id;
}

function addObject() {
  const label = elements.objectLabel.value.trim() || `object_${state.objects.length + 1}`;
  const offset = Math.min(0.18, state.objects.length * 0.04);
  const object = {
    id: crypto.randomUUID(),
    label,
    bbox: { x_min: 0.35 + offset, y_min: 0.35, x_max: 0.60 + offset, y_max: 0.60 },
  };
  state.objects.push(object);
  state.activeObjectId = object.id;
  renderAll();
}

async function refineActiveObject() {
  const active = activeObject();
  if (!active) return;
  const result = await postJson("/api/refine-roi", {
    dataset_path: elements.failedDataset.value,
    camera_key: elements.cameraKey.value,
    episode_index: numberValue(elements.referenceEpisode),
    frame_index: numberValue(elements.referenceFrame),
    bbox: active.bbox,
  });
  active.bbox = result.bbox;
  renderAll();
}

function beginRoiDraw(event) {
  const active = activeObject();
  if (!active || !state.referenceImage) return;
  const point = canvasPoint(elements.overlayCanvas, event);
  state.drawing = { objectId: active.id, start: point, current: point };
}

function updateRoiDraw(event) {
  if (!state.drawing) return;
  state.drawing.current = canvasPoint(elements.overlayCanvas, event);
  const active = state.objects.find((object) => object.id === state.drawing.objectId);
  if (active) active.bbox = bboxFromPoints(state.drawing.start, state.drawing.current);
  renderAll();
}

function finishRoiDraw() {
  state.drawing = null;
  renderAll();
}

function startStream() {
  closeSocket();
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws/match`);
  state.socket = socket;
  setStatus("Streaming", "neutral");
  socket.addEventListener("open", () => socket.send(JSON.stringify(matchPayload(false, 24))));
  socket.addEventListener("message", async (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "frame") {
      if (message.reference_frame) {
        state.referenceFrame = message.reference_frame;
        state.referenceImage = await loadImage(message.reference_frame.image);
      }
      state.streamFrame = message.frame;
      state.streamImage = await loadImage(message.frame.image);
      state.detections = message.detections || [];
      state.objectScores = message.object_scores || [];
      state.warnings = message.warnings || [];
      elements.streamMeta.textContent = `Replay episode ${numberValue(elements.replayEpisode)}, frame ${message.frame_index}`;
      await loadJointComparison(message.frame_index);
      setResult(message.ready, message.overall_score);
      renderAll();
    }
    if (message.type === "result") {
      elements.saveMatch.textContent = "Result Streamed";
      setTimeout(() => (elements.saveMatch.textContent = "Save Match"), 1200);
    }
    if (message.type === "error") {
      state.warnings = [message.message];
      setStatus("Error", "not-ready");
      renderAll();
    }
  });
}

async function saveMatch() {
  const result = await postJson("/api/match", matchPayload(true, 8));
  state.objectScores = result.object_scores || [];
  state.warnings = result.warnings || [];
  setResult(Boolean(result.ready), Number(result.overall_score));
  elements.saveMatch.textContent = "Match Saved";
  setTimeout(() => (elements.saveMatch.textContent = "Save Match"), 1200);
  renderAll();
}

async function loadJointComparison(replayFrameIndex) {
  const params = new URLSearchParams({
    reference_dataset_path: elements.failedDataset.value,
    replay_dataset_path: elements.replayDataset.value,
    reference_episode_index: String(numberValue(elements.referenceEpisode)),
    reference_frame_index: String(numberValue(elements.referenceFrame)),
    replay_episode_index: String(numberValue(elements.replayEpisode)),
    replay_frame_index: String(replayFrameIndex),
    urdf_path: elements.urdfPath.value.trim(),
  });
  try {
    const comparison = await requestJson(`/api/joints?${params.toString()}`);
    renderJoints(comparison);
  } catch (error) {
    elements.jointList.innerHTML = `<div class="warning-item">${escapeHtml(error.message)}</div>`;
  }
}

function matchPayload(save, streamFrames) {
  return {
    reference_dataset_path: elements.failedDataset.value,
    replay_dataset_path: elements.replayDataset.value,
    camera_key: elements.cameraKey.value,
    reference_episode_index: numberValue(elements.referenceEpisode),
    reference_frame_index: numberValue(elements.referenceFrame),
    replay_episode_index: numberValue(elements.replayEpisode),
    replay_start_frame_index: 0,
    stream_frames: streamFrames,
    save,
    objects: state.objects.map((object) => ({ label: object.label, bbox: object.bbox })),
  };
}

function renderAll() {
  drawOverlay();
  renderObjects();
  renderGuidance();
  renderWarnings();
  updateLineage();
}

function drawOverlay() {
  const canvas = elements.overlayCanvas;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.referenceImage) {
    drawImageCover(ctx, state.referenceImage, canvas);
  }
  if (state.streamImage) {
    ctx.save();
    ctx.globalAlpha = Number(elements.overlayOpacity.value || "0.55");
    drawImageCover(ctx, state.streamImage, canvas);
    ctx.restore();
  }
  for (const object of state.objects) {
    drawBox(ctx, object.bbox, object.id === state.activeObjectId ? "#ffbf00" : "#28a745", object.label);
  }
  for (const detection of state.detections) drawBox(ctx, detection.bbox, "#00b4d8", detection.label);
  for (const score of state.objectScores) {
    const object = state.objects.find((item) => item.label === score.label);
    const detection = state.detections.find((item) => item.label === score.label);
    if (object && detection) drawArrow(ctx, detection.centroid, centroid(object.bbox));
  }
}

function renderObjects() {
  elements.objectsList.innerHTML = "";
  for (const object of state.objects) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `object-item ${object.id === state.activeObjectId ? "active" : ""}`;
    item.innerHTML = `<strong>${escapeHtml(object.label)}</strong><span>${formatBbox(object.bbox)}</span>`;
    item.addEventListener("click", () => {
      state.activeObjectId = object.id;
      renderAll();
    });
    elements.objectsList.appendChild(item);
  }
}

function renderGuidance() {
  elements.guidanceList.innerHTML = "";
  const scores = state.objectScores.length
    ? state.objectScores
    : state.objects.map((object) => ({ label: object.label, score: 0, guidance: "Start stream to score setup" }));
  for (const score of scores) {
    const item = document.createElement("div");
    item.className = "guidance-item";
    const scoreText = Number.isFinite(score.score) ? Number(score.score).toFixed(3) : "--";
    item.innerHTML = `<strong>${escapeHtml(score.label)}</strong><span>${escapeHtml(score.guidance || "No guidance")}</span><span>score ${scoreText}</span>`;
    elements.guidanceList.appendChild(item);
  }
}

function renderWarnings() {
  elements.warningsList.innerHTML = "";
  const warnings = state.warnings.length ? state.warnings : ["No warnings yet"];
  for (const warning of warnings) {
    const item = document.createElement("div");
    item.className = "warning-item";
    item.textContent = warning;
    elements.warningsList.appendChild(item);
  }
}

function renderJoints(comparison) {
  const joints = comparison.joints || [];
  if (!joints.length) {
    elements.jointList.innerHTML = '<div class="joint-item">No joint positions available</div>';
    return;
  }
  const maxAbsValue = Math.max(
    1,
    ...joints.flatMap((joint) => [Math.abs(joint.reference), Math.abs(joint.replay)]),
  );
  elements.jointList.innerHTML = "";
  for (const joint of joints) {
    const range = joint.lower !== null && joint.upper !== null && joint.lower !== undefined && joint.upper !== undefined
      ? { lower: Number(joint.lower), upper: Number(joint.upper) }
      : { lower: -maxAbsValue, upper: maxAbsValue };
    const referencePercent = valueToPercent(joint.reference, range.lower, range.upper);
    const replayPercent = valueToPercent(joint.replay, range.lower, range.upper);
    const item = document.createElement("div");
    item.className = "joint-item";
    item.innerHTML = `
      <div class="joint-row">
        <div class="joint-name">${escapeHtml(joint.name)}</div>
        <div>
          <div class="joint-track">
            <span class="joint-reference" style="left: ${referencePercent}%"></span>
            <span class="joint-replay" style="left: ${replayPercent}%"></span>
          </div>
          <div class="joint-delta">delta ${formatJointValue(joint.delta)}${joint.in_urdf ? " · URDF" : ""}</div>
        </div>
        <div class="joint-values">ref ${formatJointValue(joint.reference)} · replay ${formatJointValue(joint.replay)}</div>
      </div>`;
    elements.jointList.appendChild(item);
  }
}

function drawImageCover(ctx, image, canvas) {
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
}

function drawBox(ctx, bbox, color, label) {
  const x = bbox.x_min * ctx.canvas.width;
  const y = bbox.y_min * ctx.canvas.height;
  const width = (bbox.x_max - bbox.x_min) * ctx.canvas.width;
  const height = (bbox.y_max - bbox.y_min) * ctx.canvas.height;
  ctx.save();
  ctx.lineWidth = 3;
  ctx.strokeStyle = color;
  ctx.fillStyle = "rgba(0, 0, 0, 0.62)";
  ctx.strokeRect(x, y, width, height);
  ctx.fillRect(x, Math.max(0, y - 22), Math.max(72, ctx.measureText(label).width + 18), 22);
  ctx.fillStyle = "#ffffff";
  ctx.font = "13px sans-serif";
  ctx.fillText(label, x + 8, Math.max(14, y - 7));
  ctx.restore();
}

function drawArrow(ctx, from, to) {
  const fromX = from[0] * ctx.canvas.width;
  const fromY = from[1] * ctx.canvas.height;
  const toX = to[0] * ctx.canvas.width;
  const toY = to[1] * ctx.canvas.height;
  const angle = Math.atan2(toY - fromY, toX - fromX);
  ctx.save();
  ctx.strokeStyle = "#ffbf00";
  ctx.fillStyle = "#ffbf00";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(toX - 12 * Math.cos(angle - Math.PI / 6), toY - 12 * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(toX - 12 * Math.cos(angle + Math.PI / 6), toY - 12 * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function canvasPoint(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: clamp((event.clientX - rect.left) / rect.width, 0, 1),
    y: clamp((event.clientY - rect.top) / rect.height, 0, 1),
  };
}

function bboxFromPoints(start, current) {
  const xMin = Math.min(start.x, current.x);
  const yMin = Math.min(start.y, current.y);
  const xMax = Math.max(start.x, current.x);
  const yMax = Math.max(start.y, current.y);
  return {
    x_min: clamp(xMin, 0, 0.99),
    y_min: clamp(yMin, 0, 0.99),
    x_max: clamp(Math.max(xMax, xMin + 0.01), 0.01, 1),
    y_max: clamp(Math.max(yMax, yMin + 0.01), 0.01, 1),
  };
}

function centroid(bbox) {
  return [(bbox.x_min + bbox.x_max) / 2, (bbox.y_min + bbox.y_max) / 2];
}

function activeObject() {
  return state.objects.find((object) => object.id === state.activeObjectId) || null;
}

function updateLineage() {
  elements.lineage.textContent = `Reference episode ${numberValue(elements.referenceEpisode)}, replay episode ${numberValue(elements.replayEpisode)}, ${elements.cameraKey.value || "camera"}`;
}

function setResult(ready, score) {
  setStatus(ready ? "Ready" : "Not Ready", ready ? "ready" : "not-ready");
  elements.scoreBadge.textContent = `Score ${Number(score).toFixed(3)}`;
}

function setStatus(text, className) {
  elements.readyBadge.textContent = text;
  elements.readyBadge.className = `badge ${className}`;
}

async function requestJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || response.statusText);
  return result;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = src;
  });
}

function numberValue(input) {
  return Number.parseInt(input.value || "0", 10);
}

function formatBbox(bbox) {
  return [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max].map((value) => value.toFixed(3)).join(", ");
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function valueToPercent(value, lower, upper) {
  if (!Number.isFinite(lower) || !Number.isFinite(upper) || lower === upper) return 50;
  return clamp(((value - lower) / (upper - lower)) * 100, 0, 100);
}

function formatJointValue(value) {
  return Number(value).toFixed(2);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "'": "&#39;",
    '"': "&quot;",
  })[char]);
}

function closeSocket() {
  if (state.socket && state.socket.readyState < WebSocket.CLOSING) state.socket.close();
}

boot().catch((error) => {
  state.warnings = [error.message];
  setStatus("Error", "not-ready");
  renderWarnings();
});
