const chatWindow = document.getElementById('chatWindow');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const quickPills = document.querySelectorAll('.pill');

function addMessage(text, role) {
  const message = document.createElement('div');
  message.className = `message ${role}`;
  message.innerHTML = role === 'user' ? `<p>${text}</p>` : `<strong>NovaBot</strong><p>${text}</p>`;
  chatWindow.appendChild(message);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function sendMessage(message) {
  if (!message.trim()) return;

  addMessage(message, 'user');
  messageInput.value = '';

  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    });

    const data = await response.json();
    addMessage(data.reply, 'bot');
  } catch (error) {
    addMessage('Sorry, the assistant is unavailable right now.', 'bot');
  }
}

chatForm.addEventListener('submit', (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  sendMessage(message);
});

quickPills.forEach((pill) => {
  pill.addEventListener('click', () => {
    const question = pill.dataset.question;
    sendMessage(question);
  });
});
