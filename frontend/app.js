const chatBox = document.getElementById("chatBox");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");

let conversationId = null;

function addMessage(role, text) {
  const empty = chatBox.querySelector(".empty");
  if (empty) empty.remove();

  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
  return div;
}

async function sendQuestion() {
  const q = questionInput.value.trim();
  if (!q) return;

  addMessage("user", q);
  questionInput.value = "";
  sendBtn.disabled = true;

  const placeholder = addMessage("assistant", "思考中...");
  let started = false;
  let fullText = "";

  try {
    const resp = await fetch("/api/v1/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: q,
        conversation_id: conversationId,
      }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || "请求失败");
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // 按 \n\n 切出完整 SSE 事件
      let sepIdx;
      while ((sepIdx = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, sepIdx);
        buffer = buffer.slice(sepIdx + 2);

        // 提取 data: 开头的行
        const dataLines = rawEvent
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trim());
        if (!dataLines.length) continue;

        let event;
        try {
          event = JSON.parse(dataLines.join("\n"));
        } catch {
          continue;
        }

        if (event.type === "delta") {
          if (!started) {
            started = true;
            placeholder.textContent = "";
          }
          fullText += event.text;
          placeholder.textContent = fullText;
          chatBox.scrollTop = chatBox.scrollHeight;
        } else if (event.type === "done") {
          if (event.citations && event.citations.length) {
            renderCitations(placeholder, event.citations);
          }
          if (event.conversation_id) {
            conversationId = event.conversation_id; // 多轮 ID 持久化
          }
        }
      }
    }
  } catch (err) {
    placeholder.textContent = "出错了: " + err.message;
  } finally {
    sendBtn.disabled = false;
    questionInput.focus();
  }
}

function renderCitations(parentDiv, citations) {
  const wrap = document.createElement("div");
  wrap.className = "citations";

  citations.forEach((c, i) => {
    const card = document.createElement("div");
    card.className = "citation";
    const preview = c.chunk_text.length > 200
      ? c.chunk_text.slice(0, 200) + "…"
      : c.chunk_text;
    card.innerHTML = `
      <div class="citation-header">
        <span class="citation-num">来源 ${i + 1}</span>
        <span class="citation-doc">${escapeHtml(c.doc_name)}</span>
        <span class="citation-score">相似度 ${c.score.toFixed(3)}</span>
      </div>
      <div class="citation-text">${escapeHtml(preview)}</div>
    `;
    wrap.appendChild(card);
  });

  parentDiv.appendChild(wrap);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (m) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[m]);
}



async function uploadDocument() {
  const file = fileInput.files[0];
  if (!file) {
    uploadStatus.textContent = "请先选择文件";
    return;
  }

  uploadStatus.textContent = "上传中...";
  uploadBtn.disabled = true;

  const form = new FormData();
  form.append("file", file);

  try {
    const resp = await fetch("/api/v1/documents/upload", {
      method: "POST",
      body: form,
    });
    const data = await resp.json();
    uploadStatus.textContent = `已入库: ${data.filename}`;
    fileInput.value = "";
  } catch (err) {
    uploadStatus.textContent = "上传失败: " + err.message;
  } finally {
    uploadBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", sendQuestion);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) sendQuestion();
});
uploadBtn.addEventListener("click", uploadDocument);