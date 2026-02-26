(function () {
  const cfg = window.REPLY_WEBCHAT_CONFIG || {};
  const baseUrl = String(cfg.baseUrl || "").replace(/\/+$/, "");
  const token = String(cfg.token || "").trim();
  const userId = String(cfg.userId || "employee-webchat").trim();
  const title = String(cfg.title || "{reply} Assistant");

  if (!baseUrl) {
    console.error("[reply-webchat] Missing baseUrl in window.REPLY_WEBCHAT_CONFIG");
    return;
  }

  const style = document.createElement("style");
  style.textContent = `
    .reply-chatbox{position:fixed;right:18px;bottom:18px;width:340px;max-width:calc(100vw - 24px);height:460px;background:#0f1724;border:1px solid #28405d;border-radius:14px;box-shadow:0 18px 52px rgba(0,0,0,.45);z-index:999999;display:flex;flex-direction:column;font-family:"Avenir Next","Segoe UI",sans-serif;color:#e8f1ff}
    .reply-chatbox *{box-sizing:border-box}
    .reply-chat-head{padding:12px 14px;border-bottom:1px solid #21324b;font-weight:700;background:#13223a;border-radius:14px 14px 0 0}
    .reply-chat-log{flex:1;overflow:auto;padding:12px;display:flex;flex-direction:column;gap:8px}
    .reply-chat-msg{padding:9px 10px;border-radius:10px;line-height:1.4;font-size:14px;white-space:pre-wrap}
    .reply-chat-msg.user{background:#1a3350;align-self:flex-end;max-width:88%}
    .reply-chat-msg.bot{background:#152338;align-self:flex-start;max-width:95%}
    .reply-chat-msg.warn{background:#58352f}
    .reply-chat-controls{display:flex;gap:8px;padding:10px;border-top:1px solid #21324b}
    .reply-chat-input{flex:1;background:#0d1828;color:#e8f1ff;border:1px solid #2b4668;border-radius:10px;min-height:42px;padding:9px 10px;font-size:14px}
    .reply-chat-send{background:#1ab6ff;color:#001322;border:1px solid #45c5ff;border-radius:10px;min-width:72px;font-weight:700;cursor:pointer}
    .reply-chat-send:disabled{opacity:.6;cursor:default}
    .reply-chat-status{padding:0 10px 8px;color:#99b4d8;font-size:12px}
    @media (max-width:700px){.reply-chatbox{right:8px;left:8px;bottom:8px;width:auto;height:70vh}}
  `;
  document.head.appendChild(style);

  const root = document.createElement("div");
  root.className = "reply-chatbox";
  root.innerHTML = `
    <div class="reply-chat-head">${title}</div>
    <div class="reply-chat-log" id="replyChatLog"></div>
    <div class="reply-chat-controls">
      <textarea class="reply-chat-input" id="replyChatInput" placeholder="Kerdezz a vallalati dokumentaciobol..."></textarea>
      <button class="reply-chat-send" id="replyChatSend" type="button">Send</button>
    </div>
    <div class="reply-chat-status" id="replyChatStatus">Connecting...</div>
  `;
  document.body.appendChild(root);

  const log = root.querySelector("#replyChatLog");
  const input = root.querySelector("#replyChatInput");
  const send = root.querySelector("#replyChatSend");
  const status = root.querySelector("#replyChatStatus");
  let sessionId = "";

  function headers() {
    const h = { "Content-Type": "application/json" };
    if (token) {
      h["Authorization"] = "Bearer " + token;
      h["X-API-Token"] = token;
    }
    return h;
  }

  function addMessage(kind, text) {
    const div = document.createElement("div");
    div.className = "reply-chat-msg " + kind;
    div.textContent = text;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  async function initSession() {
    const payload = {
      user_id: userId || "employee-webchat",
      source: "embedded-webchat",
      metadata: { origin: window.location.origin || "" },
    };
    const res = await fetch(baseUrl + "/chat/session", {
      method: "POST",
      headers: headers(),
      body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || "session init failed");
    }
    sessionId = String(data.session_id || "");
  }

  async function sendMessage() {
    const text = (input.value || "").trim();
    if (!text || !sessionId) {
      return;
    }
    input.value = "";
    addMessage("user", text);
    send.disabled = true;
    status.textContent = "Thinking...";
    try {
      const res = await fetch(baseUrl + "/chat/message", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({
          session_id: sessionId,
          message: text,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        addMessage("warn", "Request failed: " + String(data.detail || "unknown error"));
        status.textContent = "Error";
        return;
      }
      addMessage("bot", String(data.assistant_message || ""));
      status.textContent = "Ready";
    } catch (err) {
      addMessage("warn", "Request failed: " + String(err));
      status.textContent = "Network error";
    } finally {
      send.disabled = false;
      input.focus();
    }
  }

  send.addEventListener("click", sendMessage);
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  initSession()
    .then(function () {
      status.textContent = "Ready";
      addMessage("bot", "Szia! Segitek a vallalati dokumentaciobol.");
      input.focus();
    })
    .catch(function (err) {
      status.textContent = "Connection failed";
      addMessage("warn", "Chat init failed: " + String(err));
    });
})();
