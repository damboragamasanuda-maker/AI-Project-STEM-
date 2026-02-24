const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const dropzone = document.getElementById("dropzone");
const uploadResult = document.getElementById("uploadResult");
const progressBar = document.getElementById("progressBar");

const questionInput = document.getElementById("questionInput");
const askBtn = document.getElementById("askBtn");
const answerBox = document.getElementById("answerBox");

const statusText = document.getElementById("statusText");
const statusDot = document.getElementById("statusDot");

let isIndexed = false;

function setStatus(text, mode = "ready") {
  statusText.textContent = text;

  // mode: ready | busy | error
  if (!statusDot) return;
  if (mode === "ready") {
    statusDot.style.background = "var(--b)";
    statusDot.style.boxShadow = "0 0 0 6px rgba(34,197,94,.18)";
  } else if (mode === "busy") {
    statusDot.style.background = "var(--e)";
    statusDot.style.boxShadow = "0 0 0 6px rgba(234,179,8,.20)";
  } else {
    statusDot.style.background = "var(--d)";
    statusDot.style.boxShadow = "0 0 0 6px rgba(249,115,22,.20)";
  }
}

function setProgress(pct) {
  if (!progressBar) return;
  progressBar.style.width = `${pct}%`;
}

function setAskEnabled(enabled) {
  askBtn.disabled = !enabled;
  askBtn.style.opacity = enabled ? "1" : ".6";
  askBtn.style.cursor = enabled ? "pointer" : "not-allowed";
}

setAskEnabled(false);
setStatus("Ready", "ready");

browseBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async () => {
  if (!fileInput.files?.length) return;
  await uploadPdf(fileInput.files[0]);
});

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.style.borderColor = "rgba(34,197,94,.40)";
});

dropzone.addEventListener("dragleave", () => {
  dropzone.style.borderColor = "rgba(255,255,255,.16)";
});

dropzone.addEventListener("drop", async (e) => {
  e.preventDefault();
  dropzone.style.borderColor = "rgba(255,255,255,.16)";
  const file = e.dataTransfer.files?.[0];
  if (file) await uploadPdf(file);
});

async function uploadPdf(file) {
  isIndexed = false;
  setAskEnabled(false);

  if (file.type !== "application/pdf") {
    uploadResult.textContent = "Only PDF files are supported.";
    setStatus("PDF only", "error");
    setProgress(0);
    return;
  }

  uploadResult.textContent = "Uploading & indexing...";
  answerBox.textContent = "—";
  setStatus("Indexing…", "busy");
  setProgress(15);

  const formData = new FormData();
  formData.append("file", file);

  try {
    // fake progress animation while waiting
    const tick = setInterval(() => {
      const current = parseFloat(progressBar.style.width || "15");
      if (current < 85) setProgress(current + 5);
    }, 250);

    const res = await fetch("/index-pdf", { method: "POST", body: formData });
    const data = await res.json();

    clearInterval(tick);

    if (!res.ok) throw new Error(data.detail || "Upload failed");

    setProgress(100);
    isIndexed = true;
    setAskEnabled(true);
    setStatus("Ready", "ready");

    uploadResult.textContent = `✅ Uploaded: ${data.filename} • chunks: ${data.chunks_indexed}`;
  } catch (err) {
    setProgress(0);
    setStatus("Error", "error");
    uploadResult.textContent = `❌ Error: ${err.message}`;
  }
}

askBtn.addEventListener("click", askQuestion);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") askQuestion();
});

async function askQuestion() {
  const q = questionInput.value.trim();
  if (!q) return;

  if (!isIndexed) {
    answerBox.textContent = "Upload and index a PDF first.";
    return;
  }

  answerBox.textContent = "Thinking...";
  setStatus("Answering…", "busy");

  try {
    const res = await fetch("/qa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");

    answerBox.textContent = data.answer ?? JSON.stringify(data, null, 2);
    setStatus("Ready", "ready");
  } catch (err) {
    answerBox.textContent = `❌ Error: ${err.message}`;
    setStatus("Error", "error");
  }
}