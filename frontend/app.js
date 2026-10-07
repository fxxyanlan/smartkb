const chatBox = document.getElementById("chatBox");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");

let conversationId = null;
let sending = false;

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
  if (sending) return;
  const q = questionInput.value.trim();
  if (!q) return;

  sending = true;
  addMessage("user", q);
  questionInput.value = "";
  sendBtn.disabled = true;

  const placeholder = addMessage("assistant", "思考中...");
  placeholder.classList.add("streaming");

  let started = false;
  let doneReceived = false;

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
          placeholder.textContent += event.text;
          chatBox.scrollTop = chatBox.scrollHeight;
        } else if (event.type === "done") {
          doneReceived = true;
          if (event.citations && event.citations.length) {
            renderCitations(placeholder, event.citations);
          }
          if (event.conversation_id) {
            conversationId = event.conversation_id; // 多轮 ID 持久化
          }
        } else if (event.type === "error") {
          doneReceived = true;
          if (!started) placeholder.textContent = "";
          placeholder.classList.add("error");
          placeholder.textContent +=
            (placeholder.textContent ? "\n\n" : "") + "⚠ " + (event.message || "生成失败");
        }
      }
    }

    // 连接意外中断（未收到 done/error）：明确提示回答不完整，避免误以为答完
    if (!doneReceived) {
      if (!started) placeholder.textContent = "";
      placeholder.classList.add("error");
      placeholder.textContent +=
        (placeholder.textContent ? "\n\n" : "") + "⚠ 连接中断，回答可能不完整";
    }
  } catch (err) {
    placeholder.classList.add("error");
    placeholder.textContent = "出错了: " + err.message;
  } finally {
    placeholder.classList.remove("streaming");
    sending = false;
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
    const raw = c.chunk_text || "";
    const preview = raw.length > 200 ? raw.slice(0, 200) + "…" : raw;
    const score =
      typeof c.score === "number" && isFinite(c.score) ? c.score.toFixed(3) : "-";
    card.innerHTML = `
      <div class="citation-header">
        <span class="citation-num">来源 ${i + 1}</span>
        <span class="citation-doc">${escapeHtml(c.doc_name || "")}</span>
        <span class="citation-score">相似度 ${score}</span>
      </div>
      <div class="citation-text">${escapeHtml(preview)}</div>
    `;
    wrap.appendChild(card);
  });

  parentDiv.appendChild(wrap);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (m) => ({
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
    showUploadStatus("请先选择文件", true);
    return;
  }

  showUploadStatus("上传中...", false);
  uploadBtn.disabled = true;

  const form = new FormData();
  form.append("file", file);

  try {
    const resp = await fetch("/api/v1/documents/upload", {
      method: "POST",
      body: form,
    });
    const data = await resp.json().catch(() => ({}));

    if (!resp.ok) {
      // 服务端返回统一错误体 {detail: "..."}
      throw new Error(data.detail || resp.statusText || "上传失败");
    }

    if (data.status === "exists") {
      showUploadStatus(`文档已存在，未重复入库: ${data.filename}`, false);
    } else {
      showUploadStatus(`已入库: ${data.filename}`, false);
    }
    fileInput.value = "";
  } catch (err) {
    showUploadStatus("上传失败: " + err.message, true);
  } finally {
    uploadBtn.disabled = false;
  }
}

function showUploadStatus(text, isError) {
  uploadStatus.textContent = text;
  uploadStatus.classList.toggle("error", !!isError);
}

sendBtn.addEventListener("click", sendQuestion);
questionInput.addEventListener("keydown", (e) => {
  // isComposing:中文输入法选词时的回车不应触发发送
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) sendQuestion();
});
uploadBtn.addEventListener("click", uploadDocument);
