// js/ui_modules/chat_ui.js

let chatMessagesContainerElement;
let agentThinkingStatusElement;
let chatTextareaElement;
let chatSendButtonElement;
let onSendMessageCallback = (messageText) => console.warn("[ChatUI] onSendMessageCallback not set.");

function initChatUI(elements, callbacks) {
    chatMessagesContainerElement = elements.chatMessagesContainer;
    agentThinkingStatusElement = elements.agentThinkingStatusEl;
    chatTextareaElement = elements.chatTextareaEl;
    chatSendButtonElement = elements.chatSendButtonEl;
    onSendMessageCallback = callbacks.onSendMessage || onSendMessageCallback;

    chatSendButtonElement.addEventListener('click', handleSendButtonClick);
    chatTextareaElement.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleSendButtonClick();
        }
    });
}

function handleSendButtonClick() {
    const messageText = chatTextareaElement.value.trim();
    if (messageText) {
        onSendMessageCallback(messageText);
        chatTextareaElement.value = '';
    }
}

function addChatMessageToUI(messageData, type, options = {}) {
    if (!chatMessagesContainerElement) return;

    const messageElement = document.createElement('div');
    messageElement.classList.add('message', `message-${type}`);
    
    let content = typeof messageData === 'object' ? messageData.content : messageData;
    
    // Simple HTML escaping
    const escapeHTML = (str) => str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
    
    // Basic markdown for bold and italics
    content = escapeHTML(content)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>');
        
    messageElement.innerHTML = content.replace(/\n/g, '<br>');

    if (options.isError) {
        messageElement.style.color = 'var(--error-color)';
    }

    chatMessagesContainerElement.appendChild(messageElement);
    scrollToBottomChat();
}

function showAgentThinkingStatusInUI(show, statusText = "Thinking...") {
    if (!agentThinkingStatusElement) return;
    agentThinkingStatusElement.textContent = statusText;
    agentThinkingStatusElement.style.display = show ? 'block' : 'none';
    if (show) {
        chatMessagesContainerElement.appendChild(agentThinkingStatusElement);
        scrollToBottomChat();
    }
}

function clearChatMessagesUI() {
    if (chatMessagesContainerElement) {
        chatMessagesContainerElement.innerHTML = '';
        // Re-add the thinking status element so it can be used again
        if (agentThinkingStatusElement) {
            agentThinkingStatusElement.style.display = 'none';
            chatMessagesContainerElement.appendChild(agentThinkingStatusElement);
        }
    }
}

function scrollToBottomChat() {
    if (chatMessagesContainerElement) {
        chatMessagesContainerElement.scrollTop = chatMessagesContainerElement.scrollHeight;
    }
}
