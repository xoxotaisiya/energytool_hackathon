const messages = document.querySelector('#messages');
const form = document.querySelector('#chatForm');
const input = document.querySelector('#messageInput');
const sendButton = document.querySelector('#sendButton');
const newChatButton = document.querySelector('#newChat');
const debugToggle = document.querySelector('#debugToggle');
const debugPanel = document.querySelector('#debugPanel');
const debugTime = document.querySelector('#debugTime');
const debugContent = document.querySelector('#debugContent');
const historyDialog = document.querySelector('#historyDialog');
const historyList = document.querySelector('#historyList');
let sessionId = localStorage.getItem('energytool_session_id');
let busy = false;

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function addMessage(role, text, meta = '') {
  const article = document.createElement('article');
  article.className = `message ${role === 'user' ? 'user-message' : 'assistant-message'}`;
  if (role !== 'user') {
    const label = document.createElement('div');
    label.className = 'message-label';
    label.textContent = 'ENERGYTOOL AI';
    article.append(label);
  }
  const paragraph = document.createElement('p');
  paragraph.textContent = text;
  article.append(paragraph);
  if (meta) {
    const detail = document.createElement('div');
    detail.className = 'message-meta';
    detail.textContent = meta;
    article.append(detail);
  }
  messages.append(article);
  scrollToBottom();
  return article;
}

function addLeadCard(lead) {
  if (!lead || lead.status !== 'complete') return;
  const existing = messages.querySelector('.lead-card');
  if (existing) existing.remove();
  const card = document.createElement('section');
  card.className = 'lead-card';
  const title = document.createElement('strong');
  title.textContent = 'ЗАЯВКА СФОРМИРОВАНА';
  card.append(title);
  [['ФИО', lead.full_name], ['Телефон', lead.phone], ['Компания', lead.company], ['Резюме', lead.summary]].forEach(([label, value]) => {
    const row = document.createElement('p');
    row.textContent = `${label}: ${value || '—'}`;
    card.append(row);
  });
  messages.append(card);
  scrollToBottom();
}

function showTyping() {
  const node = document.createElement('article');
  node.className = 'message assistant-message typing';
  node.id = 'typing';
  node.innerHTML = '<i></i><i></i><i></i>';
  messages.append(node);
  scrollToBottom();
}

function updateDebug(data) {
  const debug = data.debug || {};
  debugTime.textContent = Number.isFinite(debug.elapsed_ms) ? `${debug.elapsed_ms} мс` : '—';
  debugContent.replaceChildren();
  const sources = data.sources || [];
  if (!sources.length) {
    debugContent.textContent = `Этап: ${debug.stage || data.status || '—'}. Источники не использованы.`;
    return;
  }
  sources.forEach((source, index) => {
    const line = document.createElement('div');
    line.className = 'debug-source';
    line.textContent = `${index + 1}. ${source.path || source.file || source.id}`;
    debugContent.append(line);
  });
}

async function sendMessage(text) {
  if (busy || !text.trim()) return;
  busy = true;
  sendButton.disabled = true;
  document.querySelector('#suggestions')?.remove();
  addMessage('user', text.trim(), 'Вы');
  showTyping();
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text.trim(), session_id: sessionId, debug: debugToggle.checked}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Ошибка сервера');
    sessionId = data.session_id;
    localStorage.setItem('energytool_session_id', sessionId);
    document.querySelector('#typing')?.remove();
    addMessage('assistant', data.answer, data.status === 'handoff' ? 'Сценарий заявки' : 'Ответ по базе знаний');
    addLeadCard(data.lead);
    updateDebug(data);
  } catch (error) {
    document.querySelector('#typing')?.remove();
    addMessage('assistant', `Не удалось отправить сообщение: ${error.message}`, 'Ошибка');
  } finally {
    busy = false;
    sendButton.disabled = false;
    input.focus();
  }
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  const text = input.value;
  input.value = '';
  input.style.height = 'auto';
  sendMessage(text);
});

input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
});
document.querySelectorAll('#suggestions button').forEach((button) => {
  button.addEventListener('click', () => sendMessage(button.textContent));
});

debugToggle.addEventListener('change', () => {
  debugPanel.hidden = !debugToggle.checked;
});

newChatButton.addEventListener('click', async () => {
  if (busy) return;
  try {
    const response = await fetch('/api/sessions', {method: 'POST'});
    const data = await response.json();
    sessionId = data.session_id;
    localStorage.setItem('energytool_session_id', sessionId);
  } catch (_) {
    sessionId = null;
    localStorage.removeItem('energytool_session_id');
  }
  messages.replaceChildren();
  addMessage('assistant', 'Начат новый диалог. Расскажите, какую задачу вы хотите решить.', 'Контекст очищен');
  debugContent.textContent = 'Источники появятся после ответа.';
  debugTime.textContent = '—';
  input.focus();
});

async function loadHistory() {
  historyList.innerHTML = '<div class="empty-history">Загрузка…</div>';
  historyDialog.showModal();
  try {
    const response = await fetch('/api/sessions?limit=100');
    const data = await response.json();
    historyList.replaceChildren();
    if (!data.sessions.length) {
      historyList.innerHTML = '<div class="empty-history">Сохранённых диалогов пока нет.</div>';
      return;
    }
    data.sessions.forEach((session) => {
      const button = document.createElement('button');
      button.className = 'history-item';
      const title = document.createElement('strong');
      title.textContent = session.title;
      const date = document.createElement('span');
      date.textContent = `${new Date(session.updated_at).toLocaleString('ru-RU')} · ${session.message_count} сообщ.`;
      button.append(title, date);
      if (session.lead_status === 'complete') {
        const badge = document.createElement('em');
        badge.textContent = 'ЗАЯВКА';
        button.append(badge);
      }
      button.addEventListener('click', () => openSession(session.session_id));
      historyList.append(button);
    });
  } catch (error) {
    historyList.innerHTML = `<div class="empty-history">Не удалось загрузить историю.</div>`;
  }
}

async function openSession(id) {
  try {
    const response = await fetch(`/api/sessions/${encodeURIComponent(id)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail);
    sessionId = data.session_id;
    localStorage.setItem('energytool_session_id', sessionId);
    messages.replaceChildren();
    data.messages.forEach((message) => addMessage(message.role, message.content));
    addLeadCard(data.lead);
    historyDialog.close();
  } catch (_) {
    historyList.innerHTML = '<div class="empty-history">Не удалось открыть диалог.</div>';
  }
}

document.querySelector('#historyButton').addEventListener('click', loadHistory);
document.querySelector('#closeHistory').addEventListener('click', () => historyDialog.close());
historyDialog.addEventListener('click', (event) => {
  if (event.target === historyDialog) historyDialog.close();
});

async function restoreCurrentSession() {
  if (!sessionId) return;
  try {
    const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
    if (!response.ok) throw new Error('session not found');
    const data = await response.json();
    if (!data.messages.length) return;
    messages.replaceChildren();
    data.messages.forEach((message) => addMessage(message.role, message.content));
    addLeadCard(data.lead);
  } catch (_) {
    sessionId = null;
    localStorage.removeItem('energytool_session_id');
  }
}

restoreCurrentSession().finally(() => input.focus());
