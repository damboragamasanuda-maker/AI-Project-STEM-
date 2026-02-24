const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const dropzone = document.getElementById("dropzone");
const uploadResult = document.getElementById("uploadResult");

const questionInput = document.getElementById("questionInput");
const askBtn = document.getElementById("askBtn");
const answerBox = document.getElementById("answerBox");

browseBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async () => {
  if (!fileInput.files?.length) return;
  await uploadPdf(fileInput.files[0]);
});

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.style.borderColor = "rgba(72,209,168,.55)";
});

dropzone.addEventListener("dragleave", () => {
  dropzone.style.borderColor = "rgba(255,255,255,.14)";
});

dropzone.addEventListener("drop", async (e) => {
  e.preventDefault();
  dropzone.style.borderColor = "rgba(255,255,255,.14)";
  const file = e.dataTransfer.files?.[0];
  if (file) await uploadPdf(file);
});

async function uploadPdf(file) {
  if (file.type !== "application/pdf") {
    uploadResult.textContent = "Only PDF files are supported.";
    return;
  }
  uploadResult.textContent = "Uploading & indexing...";
  answerBox.textContent = "—";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/index-pdf", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    uploadResult.textContent = `Uploaded: ${data.filename} • chunks: ${data.chunks_indexed}`;
  } catch (err) {
    uploadResult.textContent = `Error: ${err.message}`;
  }
}

askBtn.addEventListener("click", askQuestion);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") askQuestion();
});

async function askQuestion() {
  const q = questionInput.value.trim();
  if (!q) return;

  answerBox.textContent = "Thinking...";
  try {
    const res = await fetch("/qa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");

    answerBox.textContent = data.answer ?? JSON.stringify(data, null, 2);
  } catch (err) {
    answerBox.textContent = `Error: ${err.message}`;
  }
}