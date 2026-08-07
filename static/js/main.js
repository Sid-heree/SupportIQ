// --- NEW: Load Real Tickets from Database on Page Load ---
document.addEventListener("DOMContentLoaded", async () => {
    try {
        const response = await fetch('/api/tickets');
        const tickets = await response.json();
        const tableBody = document.getElementById('ticket-table-body');
        
        if (tableBody && tickets.length > 0) {
            // Clear any loading state
            tableBody.innerHTML = '';
            
            tickets.forEach((ticket, index) => {
                // FIXED: Now mapping to 'subject' and 'queue' from your actual dataset
                tableBody.innerHTML += `
                    <tr>
                        <td>#TK-${1000 + index}</td>
                        <td>${escapeHtml(ticket.subject || "No description").substring(0, 80)}...</td>
                        <td><span class="status-badge" style="background: #e0f2fe; color: #0284c7;">${escapeHtml(ticket.queue || "General")}</span></td>
                    </tr>
                `;
            });
        }
    } catch (err) {
        console.error("Failed to load tickets:", err);
    }
});

// Tab Switcher
function switchTab(tabId, btnElement) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));

    document.getElementById(tabId).classList.add('active');
    btnElement.classList.add('active');
    document.getElementById('header-title').innerText = btnElement.innerText.trim();
}

// Send Message Handler
async function handleSend() {
    const input = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const contextContainer = document.getElementById('context-container');
    const query = input.value.trim();

    if (!query) return;

    // 1. Render User Message
    const userRow = document.createElement('div');
    userRow.className = 'message-row user-row';
    userRow.innerHTML = `<div class="bubble">${escapeHtml(query)}</div>`;
    chatMessages.appendChild(userRow);

    input.value = '';
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // 2. Render Loading State
    const loadingId = 'loading-' + Date.now();
    const botRow = document.createElement('div');
    botRow.className = 'message-row bot-row';
    botRow.id = loadingId;
    botRow.innerHTML = `
        <div class="bubble">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    chatMessages.appendChild(botRow);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Update Context Panel State
    contextContainer.innerHTML = `
        <h3 style="font-size:16px; margin-bottom:12px;">Vector DB Retrieval</h3>
        <p style="color:var(--text-muted); font-size:13px;">Searching FAISS index for semantic matches...</p>
    `;

    try {
        // 3. Request API Response
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });

        const data = await response.json();
        const loadingElement = document.getElementById(loadingId);

        if (response.ok) {
            // --- NEW: Reveal Predictive ML Badges ---
            const metaBar = document.getElementById('ticket-meta');
            if (metaBar) {
                metaBar.style.display = 'flex';
                document.getElementById('meta-dept').innerText = data.department || 'N/A';
                document.getElementById('meta-priority').innerText = `Priority: ${data.priority || 'N/A'}`;
                document.getElementById('meta-eta').innerText = `Est. Time: ${data.eta || 'N/A'}`;
            }

            // Render Parsed Markdown Answer
            const parsedAnswer = marked.parse(data.answer);
            loadingElement.innerHTML = `
                <div class="bubble">${parsedAnswer}</div>
                <button class="copy-btn" onclick="copyToClipboard(this)">
                    <i class="fa-regular fa-copy"></i> Copy Resolution
                </button>
            `;

            // Render Retrieved Vector Context Cards
            let contextHTML = `<h3 style="font-size:16px; margin-bottom:12px;">Vector DB Retrieval</h3>`;
            if (data.context && data.context.length > 0) {
                data.context.forEach((item, index) => {
                    let cleanContext = escapeHtml(item).replace(/\\n/g, '<br>').replace(/\n/g, '<br>');
                    
                    contextHTML += `
                        <div class="context-card">
                            <strong>Matched Ticket #${index + 1}</strong>
                            <p style="margin-top:6px;">${cleanContext}</p>
                        </div>
                    `;
                });
            } else {
                contextHTML += `<p style="color:var(--text-muted); font-size:13px;">No relevant context retrieved.</p>`;
            }
            contextContainer.innerHTML = contextHTML;

        } else {
            loadingElement.innerHTML = `<div class="bubble" style="color:red;">Error: ${data.error}</div>`;
        }
    } catch (err) {
        document.getElementById(loadingId).innerHTML = `<div class="bubble" style="color:red;">Server connection failed.</div>`;
    }

    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Copy to Clipboard Utility
function copyToClipboard(button) {
    const bubbleText = button.previousElementSibling.innerText;
    navigator.clipboard.writeText(bubbleText).then(() => {
        const originalHTML = button.innerHTML;
        button.innerHTML = `<i class="fa-solid fa-check"></i> Copied!`;
        setTimeout(() => {
            button.innerHTML = originalHTML;
        }, 2000);
    });
}

// Security Helper to Prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.innerText = text;
    return div.innerHTML;
}