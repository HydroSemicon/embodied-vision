"use strict";

const elements = {
  connection: document.querySelector("#connectionStatus"),
  cameraFeed: document.querySelector("#cameraFeed"),
  cameraPlaceholder: document.querySelector("#cameraPlaceholder"),
  cameraState: document.querySelector("#cameraState"),
  recognitionState: document.querySelector("#recognitionState"),
  modelState: document.querySelector("#modelState"),
  peopleCount: document.querySelector("#peopleCount"),
  lastUpdated: document.querySelector("#lastUpdated"),
  eventSwitch: document.querySelector("#eventSwitch"),
  trackCount: document.querySelector("#trackCount"),
  trackList: document.querySelector("#trackList"),
  trackSelect: document.querySelector("#trackSelect"),
  personName: document.querySelector("#personName"),
  enrollForm: document.querySelector("#enrollForm"),
  enrollButton: document.querySelector("#enrollButton"),
  enrollHint: document.querySelector("#enrollHint"),
  registryList: document.querySelector("#registryList"),
  refreshRegistry: document.querySelector("#refreshRegistry"),
  activityLog: document.querySelector("#activityLog"),
  clearLog: document.querySelector("#clearLog"),
  toast: document.querySelector("#toast"),
};

let currentTracks = [];
let previousTrackSignatures = new Map();
let tracksInitialized = false;
let eventDeliveryEnabled = false;
let enrollmentRequestActive = false;
let toastTimer = null;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function getJson(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `${response.status} ${response.statusText}`);
  }
  return data;
}

function setConnection(online) {
  elements.connection.classList.toggle("online", online);
  elements.connection.classList.toggle("offline", !online);
  elements.connection.querySelector("span:last-child").textContent = online
    ? "Pythonに接続中"
    : "Pythonに接続できません";
}

function showToast(message, error = false) {
  window.clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.toggle("error", error);
  elements.toast.classList.add("show");
  toastTimer = window.setTimeout(() => elements.toast.classList.remove("show"), 3200);
}

function setEventSwitch(enabled) {
  eventDeliveryEnabled = Boolean(enabled);
  elements.eventSwitch.setAttribute("aria-checked", String(eventDeliveryEnabled));
  elements.eventSwitch.querySelector(".switch-label").textContent = eventDeliveryEnabled ? "ON" : "OFF";
}

function statusLabel(status) {
  const labels = {
    recognized: "認識済み",
    pending: "照合中",
    unknown: "未登録",
    unavailable: "利用不可",
    ready: "稼働中",
    starting: "起動中",
    disabled: "無効",
    error: "エラー",
  };
  return labels[status] || status || "不明";
}

function displayName(track) {
  const identity = track.identity || {};
  if (identity.status === "recognized" && identity.name) return identity.name;
  if (identity.status === "unknown") return "Unknown";
  if (identity.status === "unavailable") return "顔認識なし";
  return "照合しています";
}

function candidateMarkup(track) {
  const candidate = track.latest_candidate || track.best_candidate;
  if (!candidate) {
    return `<div class="match-line"><span>最近傍候補</span><strong>まだありません</strong></div>
      <div class="distance-bar"><span style="width:0%"></span></div>`;
  }
  const distance = Number(candidate.distance);
  const threshold = Number(candidate.threshold);
  const quality = Math.max(0, Math.min(100, (1 - distance) * 100));
  const matchClass = candidate.within_threshold ? "match" : "";
  return `<div class="match-line"><span>候補 ${escapeHtml(candidate.name)}</span>
      <strong>${distance.toFixed(4)} / ${threshold.toFixed(2)}</strong></div>
    <div class="distance-bar" title="距離は小さいほど一致"><span class="${matchClass}" style="width:${quality}%"></span></div>`;
}

function renderTracks(tracks) {
  currentTracks = tracks;
  elements.trackCount.textContent = `${tracks.length} track${tracks.length === 1 ? "" : "s"}`;

  if (!tracks.length) {
    elements.trackList.innerHTML = '<div class="empty-state">人物は検出されていません</div>';
  } else {
    elements.trackList.innerHTML = tracks.map((track) => {
      const status = track.identity?.status || "pending";
      const enrollment = track.enrollment;
      const enrollmentResult = track.enrollment_result;
      const quality = track.face_quality?.quality;
      const rejection = track.face_rejection?.reason;
      const sampleStatus = enrollment
        ? `登録中 ${Number(enrollment.sample_count)}/${Number(enrollment.target_samples)}<br>残り ${Number(enrollment.seconds_remaining).toFixed(1)}秒`
        : enrollmentResult?.status === "failed"
          ? `登録失敗 ${Number(enrollmentResult.selected_sample_count)}/${Number(enrollmentResult.captured_sample_count)}枚採用<br>${escapeHtml(enrollmentResult.message)}`
          : enrollmentResult?.status === "completed"
            ? `登録完了 ${Number(enrollmentResult.selected_sample_count)}枚採用<br>計${Number(enrollmentResult.embedding_count)}サンプル`
        : track.face_sample_ready
          ? `顔サンプル取得済み<br>${quality == null ? `証拠 ${Number(track.evidence_samples || 0)}件` : `品質 ${Math.round(Number(quality) * 100)}%`}`
          : `顔サンプル待ち<br>${rejection ? escapeHtml(rejection) : `証拠 ${Number(track.evidence_samples || 0)}件`}`;
      return `<article class="track-card">
        <div class="track-id">#${escapeHtml(track.track_id)}</div>
        <div class="track-person">
          <strong>${escapeHtml(displayName(track))}</strong>
          <span class="badge ${escapeHtml(status)}">${escapeHtml(statusLabel(status))}</span>
        </div>
        <div class="match-data">${candidateMarkup(track)}</div>
        <div class="face-ready ${track.face_sample_ready ? "yes" : ""}">
          ${sampleStatus}
        </div>
      </article>`;
    }).join("");
  }

  const selected = elements.trackSelect.value;
  elements.trackSelect.innerHTML = tracks.length
    ? tracks.map((track) => `<option value="${escapeHtml(track.track_id)}">#${escapeHtml(track.track_id)} · ${escapeHtml(displayName(track))}${track.face_sample_ready ? "" : "（顔待ち）"}</option>`).join("")
    : '<option value="">人物を待っています</option>';
  if (tracks.some((track) => String(track.track_id) === selected)) {
    elements.trackSelect.value = selected;
  }
  updateEnrollAvailability();
  recordTrackChanges(tracks);
}

function updateEnrollAvailability() {
  const track = currentTracks.find((item) => String(item.track_id) === elements.trackSelect.value);
  const ready = Boolean(track?.face_sample_ready);
  elements.enrollButton.disabled = !ready || enrollmentRequestActive || Boolean(track?.enrollment);
  elements.enrollHint.textContent = !track
    ? "人物が検出されると選択できます。"
    : track.enrollment
      ? `${track.enrollment.name} の登録中です。正面からゆっくり左右へ顔を向けてください。`
    : ready
      ? "開始後、最大30秒間収集します。正面からゆっくり左右へ顔を向けてください。"
      : `track #${track.track_id} は顔サンプルの取得待ちです。正面を向いてください。`;
}

function trackSignature(track) {
  const candidate = track.latest_candidate;
  return JSON.stringify([
    track.identity?.status,
    track.identity?.name,
    candidate?.name,
    candidate?.distance,
    track.face_sample_ready,
    track.enrollment?.name,
    track.enrollment_result?.status,
    track.enrollment_result?.completed_at,
  ]);
}

function addActivity(message) {
  if (elements.activityLog.querySelector(".muted")) elements.activityLog.innerHTML = "";
  const item = document.createElement("li");
  const time = document.createElement("time");
  time.textContent = new Date().toLocaleTimeString("ja-JP");
  item.append(time, document.createTextNode(message));
  elements.activityLog.prepend(item);
  while (elements.activityLog.children.length > 60) elements.activityLog.lastElementChild.remove();
}

function recordTrackChanges(tracks) {
  const next = new Map(tracks.map((track) => [String(track.track_id), trackSignature(track)]));
  if (tracksInitialized) {
    for (const track of tracks) {
      const id = String(track.track_id);
      if (!previousTrackSignatures.has(id)) {
        addActivity(`track #${id} を検出しました`);
      } else if (previousTrackSignatures.get(id) !== next.get(id)) {
        const candidate = track.latest_candidate;
        const enrollmentResult = track.enrollment_result;
        const detail = enrollmentResult?.status === "completed"
          ? `${enrollmentResult.name} の登録が完了しました（${Number(enrollmentResult.selected_sample_count)}枚採用）`
          : enrollmentResult?.status === "failed"
            ? `登録に失敗しました: ${enrollmentResult.message}`
          : track.identity?.status === "recognized"
          ? `${track.identity.name} と認識しました`
          : candidate
            ? `候補 ${candidate.name}, 距離 ${Number(candidate.distance).toFixed(4)}`
            : statusLabel(track.identity?.status);
        addActivity(`track #${id}: ${detail}`);
      }
    }
    for (const id of previousTrackSignatures.keys()) {
      if (!next.has(id)) addActivity(`track #${id} が画面から離れました`);
    }
  }
  previousTrackSignatures = next;
  tracksInitialized = true;
}

function renderRegistry(people) {
  elements.peopleCount.textContent = String(people.length);
  if (!people.length) {
    elements.registryList.innerHTML = '<div class="empty-state">登録人物はいません</div>';
    return;
  }
  elements.registryList.innerHTML = people.map((person) => `
    <article class="registry-item">
      <div class="avatar">${escapeHtml(person.name.slice(0, 1).toUpperCase())}</div>
      <div class="registry-copy">
        <strong>${escapeHtml(person.name)}</strong>
        <span>${escapeHtml(person.person_id.slice(0, 12))}…</span>
      </div>
      <div class="sample-count"><strong>${Number(person.embedding_count)}</strong>samples</div>
    </article>`).join("");
}

async function refreshStatus() {
  try {
    const response = await fetch("/status", { cache: "no-store" });
    const status = await response.json();
    setConnection(true);
    elements.cameraState.textContent = status.camera === "ready" ? "稼働中" : "映像待ち";
    if (status.camera === "ready") {
      elements.cameraFeed.classList.add("ready");
      elements.cameraPlaceholder.hidden = true;
    }
    const face = status.face_recognition || {};
    elements.recognitionState.textContent = face.state === "ready" ? "稼働中" : statusLabel(face.state);
    elements.modelState.textContent = face.model ? `${face.model} / ${face.detector}` : "--";
    elements.peopleCount.textContent = String(face.registered_people ?? 0);
    setEventSwitch(status.face_events?.effective_enabled ?? status.event_delivery);
    elements.lastUpdated.textContent = new Date().toLocaleTimeString("ja-JP");
  } catch (error) {
    setConnection(false);
    elements.cameraState.textContent = "切断";
    elements.recognitionState.textContent = "切断";
  }
}

async function refreshTracks() {
  try {
    const data = await getJson("/tracks");
    renderTracks(Array.isArray(data.tracks) ? data.tracks : []);
  } catch (error) {
    // The connection indicator is managed by refreshStatus.
  }
}

async function refreshRegistry() {
  try {
    const data = await getJson("/faces");
    renderRegistry(Array.isArray(data.people) ? data.people : []);
  } catch (error) {
    elements.registryList.innerHTML = '<div class="empty-state">登録データを取得できません</div>';
  }
}

function startCameraStream() {
  elements.cameraFeed.src = `/stream/annotated?t=${Date.now()}`;
}

elements.cameraFeed.addEventListener("load", () => {
  elements.cameraFeed.classList.add("ready");
  elements.cameraPlaceholder.hidden = true;
});

elements.cameraFeed.addEventListener("error", () => {
  elements.cameraFeed.classList.remove("ready");
  elements.cameraPlaceholder.hidden = false;
  window.setTimeout(startCameraStream, 1000);
});

elements.eventSwitch.addEventListener("click", async () => {
  elements.eventSwitch.disabled = true;
  try {
    const result = await getJson("/face-events", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: !eventDeliveryEnabled }),
    });
    setEventSwitch(result.effective_enabled);
    addActivity(`Aliceへの顔・人物イベントを${result.effective_enabled ? "有効" : "無効"}にしました`);
    showToast(`顔・人物イベント: ${result.effective_enabled ? "ON" : "OFF"}`);
  } catch (error) {
    showToast(`切り替えに失敗しました: ${error.message}`, true);
  } finally {
    elements.eventSwitch.disabled = false;
  }
});

elements.trackSelect.addEventListener("change", updateEnrollAvailability);

elements.enrollForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const trackId = elements.trackSelect.value;
  const name = elements.personName.value.trim();
  if (!trackId || !name) return;

  enrollmentRequestActive = true;
  elements.enrollButton.disabled = true;
  elements.enrollButton.textContent = "登録しています…";
  try {
    const result = await getJson("/faces/enroll", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ track_id: trackId, name }),
    });
    const person = result.person;
    showToast(`${person.name} の顔収集を開始しました`);
    addActivity(`track #${trackId}: ${person.name} の登録を開始しました`);
    elements.personName.value = "";
    await refreshTracks();
  } catch (error) {
    showToast(`登録できませんでした: ${error.message}`, true);
  } finally {
    enrollmentRequestActive = false;
    elements.enrollButton.textContent = "この顔を登録";
    updateEnrollAvailability();
  }
});

elements.refreshRegistry.addEventListener("click", refreshRegistry);
elements.clearLog.addEventListener("click", () => {
  elements.activityLog.innerHTML = '<li class="muted">状態の変化がここに表示されます</li>';
});

startCameraStream();
refreshStatus();
refreshTracks();
refreshRegistry();
window.setInterval(refreshStatus, 500);
window.setInterval(refreshTracks, 200);
window.setInterval(refreshRegistry, 3000);
