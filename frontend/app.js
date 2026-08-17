const $ = (id) => document.getElementById(id);

async function health() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    $("hReady").textContent = data.ready ? data.shards.join(" · ") : "NO INDEX";
    $("hStt").textContent = data.stt_configured ? "Sarvam" : "missing key";
    $("hLlm").textContent = data.llm_configured ? "Groq" : "extractive";
    $("hFaiss").textContent = data.faiss ? "FAISS HNSW" : "numpy";
  } catch {
    $("hReady").textContent = "offline";
  }
}

function renderStages(timings) {
  const ol = $("stages");
  ol.innerHTML = "";
  if (!timings || !timings.length) {
    ol.innerHTML = '<li class="idle">No stages</li>';
    return;
  }
  for (const t of timings) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${t.name}</span><span class="ms">${t.ms.toFixed(1)} ms</span>`;
    ol.appendChild(li);
  }
}

function renderResult(data) {
  $("answerSheet").hidden = false;
  $("fileNo").textContent = data.refused ? "FILE · REFUSED" : "FILE · GROUNDED";
  $("asked").textContent = data.query
    ? `Q · ${data.query}${data.detected_language ? "  [" + data.detected_language + "]" : ""}`
    : "";
  $("answer").textContent = data.answer || "—";
  const badge = $("badge");
  badge.className = "badge " + (data.refused ? "no" : "ok");
  badge.textContent = data.refused
    ? `REFUSE · ${data.refuse_reason || "guardrail"}`
    : `GROUNDED · conf ${Number(data.confidence).toFixed(2)} · ${data.generation_mode}`;
  renderStages(data.timings_ms);
  const budget = $("budget");
  budget.textContent = `retrieve ${data.retrieval_ms.toFixed(1)} ms  ·  e2e ${data.end_to_end_ms.toFixed(1)} ms  ·  retrieve<200ms ${data.retrieval_under_200ms ? "yes" : "no"}`;
  const cites = $("cites");
  cites.innerHTML = "";
  for (const c of data.citations || []) {
    const div = document.createElement("div");
    div.className = "cite";
    div.innerHTML = `<div class="meta">${c.chunk_id} · ${c.language} · ${c.strategy} · score ${c.score.toFixed(4)}</div><div>${c.parent_text}</div>`;
    cites.appendChild(div);
  }
}

async function askText(query, mode) {
  $("hint").textContent = "Retrieving…";
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, mode }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || res.statusText);
  renderResult(data);
  $("hint").textContent = "Hindi or English. 30 seconds max. Sarvam saaras:v3.";
}

async function askVoice(blob, mode) {
  $("hint").textContent = "Transcribing with Sarvam…";
  const fd = new FormData();
  fd.append("file", blob, "query.webm");
  fd.append("mode", mode);
  const res = await fetch("/api/ask/voice", { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || res.statusText);
  renderResult(data);
  $("hint").textContent = "Hindi or English. 30 seconds max. Sarvam saaras:v3.";
}

function selectedMode() {
  const picked = document.querySelector('input[name="mode"]:checked');
  return picked ? picked.value : "auto";
}

$("textForm").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const q = $("query").value.trim();
  if (!q) return;
  try {
    await askText(q, selectedMode());
  } catch (err) {
    $("hint").textContent = String(err.message || err);
  }
});

let media = null;
let recorder = null;
let chunks = [];

async function startRec() {
  media = await navigator.mediaDevices.getUserMedia({ audio: true });
  chunks = [];
  const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
    ? "audio/webm;codecs=opus"
    : "audio/webm";
  recorder = new MediaRecorder(media, { mimeType: mime });
  recorder.ondataavailable = (e) => {
    if (e.data.size) chunks.push(e.data);
  };
  recorder.onstop = async () => {
    media.getTracks().forEach((t) => t.stop());
    const blob = new Blob(chunks, { type: recorder.mimeType });
    try {
      await askVoice(blob, selectedMode());
    } catch (err) {
      $("hint").textContent = String(err.message || err);
    }
  };
  recorder.start();
  $("mic").setAttribute("aria-pressed", "true");
  $("micLabel").textContent = "Listening… tap to send";
}

function stopRec() {
  if (recorder && recorder.state !== "inactive") recorder.stop();
  $("mic").setAttribute("aria-pressed", "false");
  $("micLabel").textContent = "Hold to speak";
}

$("mic").addEventListener("click", async () => {
  try {
    if (!recorder || recorder.state === "inactive") await startRec();
    else stopRec();
  } catch (err) {
    $("hint").textContent = "Microphone permission is required.";
  }
});

health();
