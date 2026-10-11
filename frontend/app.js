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
            renderCitations(placeholder, event.citations, q);
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

const PREVIEW_LEN = 200;

/**
 * 把问题切成高亮词：中文取相邻双字，英文/数字取整词（长度 >= 2）。
 * 单字中文会把几乎整段都点亮，所以刻意不用。
 */
function extractTerms(question) {
  const terms = new Set();
  const normalized = (question || "").toLowerCase();

  // 英文/数字整词
  (normalized.match(/[a-z0-9_]+/g) || []).forEach((w) => {
    if (w.length >= 2) terms.add(w);
  });

  // 中文/日文等连续片段取相邻双字
  (normalized.match(/[\u4e00-\u9fff]+/g) || []).forEach((seg) => {
    for (let i = 0; i < seg.length - 1; i++) terms.add(seg.slice(i, i + 2));
  });

  return [...terms];
}

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * 在原始文本上匹配、对每个片段分别转义，避免 `&` 实体被误当成普通文本匹配到。
 * 长词优先，防止短词把长词切断。
 */
function highlight(text, terms) {
  const source = String(text);
  if (!terms.length) return escapeHtml(source);

  const pattern = [...terms]
    .sort((a, b) => b.length - a.length)
    .map(escapeRegExp)
    .join("|");
  if (!pattern) return escapeHtml(source);

  const parts = source.split(new RegExp(`(${pattern})`, "gi"));
  return parts
    .map((part, i) => (i % 2 === 1 ? `<mark>${escapeHtml(part)}</mark>` : escapeHtml(part)))
    .join("");
}

function renderCitations(parentDiv, citations, question) {
  const wrap = document.createElement("div");
  wrap.className = "citations";

  const terms = extractTerms(question);

  citations.forEach((c, i) => {
    const card = document.createElement("div");
    card.className = "citation";

    const raw = c.chunk_text || "";
    const preview = raw.length > PREVIEW_LEN ? raw.slice(0, PREVIEW_LEN) + "…" : raw;
    const expandable = raw.length > PREVIEW_LEN;
    const score =
      typeof c.score === "number" && isFinite(c.score) ? c.score.toFixed(3) : "-";

    card.innerHTML = `
      <div class="citation-header"${expandable ? ' role="button" tabindex="0"' : ""}>
        <span class="citation-num">来源 ${i + 1}</span>
        <span class="citation-doc">${escapeHtml(c.doc_name || "")}</span>
        <span class="citation-score">相似度 ${score}</span>
        ${expandable ? '<span class="citation-toggle">展开原文</span>' : ""}
      </div>
      <div class="citation-text">${highlight(preview, terms)}</div>
    `;

    if (expandable) {
      const toggle = () => {
        const expanded = card.classList.toggle("expanded");
        card.querySelector(".citation-text").innerHTML = highlight(
          expanded ? raw : preview,
          terms
        );
        card.querySelector(".citation-toggle").textContent = expanded
          ? "收起"
          : "展开原文";
      };
      const header = card.querySelector(".citation-header");
      header.addEventListener("click", toggle);
      header.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          toggle();
        }
      });
    }

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
