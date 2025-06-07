/**
 * This script acts as the main orchestrator for the AI Agent UI.
 * - Initializes StateManager and UI modules.
 * - Manages the core application lifecycle.
 * - Routes events/messages between UI modules, StateManager, and WebSocket communication.
 * - Handles WebSocket message dispatching.
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log("AI Agent UI Script Loaded and DOM ready! Initializing StateManager...");
    
    // Ensure StateManager is loaded
    if (typeof StateManager === 'undefined' || typeof StateManager.initStateManager !== 'function') {
        console.error("FATAL: StateManager is not loaded. Ensure state_manager.js is loaded before script.js.");
        alert("Application critical error: State manager failed to load.");
        return;
    }
    StateManager.initStateManager(); 

    // DOM Element References
    const taskListUl = document.getElementById('task-list');
    const newTaskButton = document.getElementById('new-task-button');
    const chatMessagesContainer = document.getElementById('chat-messages');
    const monitorLogAreaElement = document.getElementById('monitor-log-area');
    const monitorArtifactArea = document.getElementById('monitor-artifact-area');
    const artifactNav = document.querySelector('.artifact-nav');
    const artifactPrevBtn = document.getElementById('artifact-prev-btn');
    const artifactNextBtn = document.getElementById('artifact-next-btn');
    const artifactCounterElement = document.getElementById('artifact-counter');
    const chatTextarea = document.querySelector('.chat-input-area textarea');
    const chatSendButton = document.querySelector('.chat-input-area button');
    const currentTaskTitleElement = document.getElementById('current-task-title');
    const statusDotElement = document.getElementById('status-dot');
    const monitorStatusTextElement = document.getElementById('monitor-status-text');
    const stopButtonElement = document.getElementById('stop-button');
    const fileUploadInputElement = document.getElementById('file-upload-input');
    const uploadFileButtonElement = document.getElementById('upload-file-button');
    
    // Simplified for the new backend
    const executorLlmSelectElement = document.getElementById('llm-select');

    const httpBackendBaseUrl = 'http://localhost:8766'; 

    // --- Global Message Dispatcher ---
    window.dispatchWsMessage = (message) => {
        try {
            switch (message.type) {
                case 'history_start': 
                    if (typeof addChatMessageToUI === 'function') addChatMessageToUI("Loading history...", "status_message");
                    updateGlobalMonitorStatus('running', 'Loading History...'); 
                    break;
                case 'history_end': 
                    const loadingMsg = Array.from(chatMessagesContainer.children).find(el => el.textContent.includes("Loading history..."));
                    if(loadingMsg) loadingMsg.remove();
                    updateGlobalMonitorStatus('idle', 'Idle');
                    if (typeof scrollToBottomChat === 'function') scrollToBottomChat();
                    break;
                case 'agent_thinking_update': 
                    if (message.content && typeof message.content.status === 'string') {
                        const status = message.content.status.toLowerCase();
                        if (status === 'idle.') {
                            updateGlobalMonitorStatus('idle', 'Idle');
                        } else {
                            updateGlobalMonitorStatus('running', message.content.status);
                        }
                    }
                    break;
                case 'agent_message': 
                    if (typeof addChatMessageToUI === 'function') addChatMessageToUI(message.content, 'agent_message');
                    updateGlobalMonitorStatus('idle', 'Idle');
                    break;
                case 'status_message':
                    const isError = message.content?.isError || String(message.content?.text || message.content).toLowerCase().includes("error");
                    const isCancelled = String(message.content?.text || message.content).toLowerCase().includes("cancelled");
                    if (typeof addChatMessageToUI === 'function') addChatMessageToUI(message.content, 'status_message', { isError });
                    if (isError) {
                        updateGlobalMonitorStatus('error', message.content?.text || 'Error');
                    } else if (isCancelled) {
                        updateGlobalMonitorStatus('idle', 'Cancelled');
                    }
                    break;
                case 'monitor_log': 
                    if (typeof addLogEntryToMonitor === 'function') addLogEntryToMonitor(message.content);
                    break;
                case 'update_artifacts': 
                    if (Array.isArray(message.content)) { 
                        StateManager.setCurrentTaskArtifacts(message.content);
                        if(typeof updateArtifactDisplayUI === 'function') updateArtifactDisplayUI(StateManager.getCurrentTaskArtifacts(), 0);
                    }
                    break;
                case 'available_models': 
                    if (message.content && typeof message.content === 'object') { 
                        StateManager.setAvailableModels({gemini: message.content.gemini || [], ollama: message.content.ollama || []});
                        if (typeof populateAllLlmSelectorsUI === 'function') populateAllLlmSelectorsUI(StateManager.getAvailableModels(), message.content.default_executor_llm_id);
                    } 
                    break;
                default: 
                    if (typeof addChatMessageToUI === 'function' && ['user'].includes(message.type)) {
                        addChatMessageToUI(message.content, message.type);
                    } else {
                        console.warn("[Script.js] Received unhandled message type:", message.type);
                    }
            }
        } catch (error) {
            console.error("[Script.js] Failed to process dispatched WS message:", error);
        }
    };

    function updateGlobalMonitorStatus(status, text) { 
        StateManager.setIsAgentRunning(status === 'running');
        if (typeof updateMonitorStatusUI === 'function') { 
            updateMonitorStatusUI(status, text, StateManager.getIsAgentRunning()); 
        } 
    }

    const handleSendMessageFromUI = (messageText) => { 
        if (!StateManager.getCurrentTaskId()) { alert("Please select or create a task first."); return; } 
        if (StateManager.getIsAgentRunning()) { 
            if (typeof addChatMessageToUI === 'function') addChatMessageToUI("Agent is currently busy. Please wait.", "status_message", { isError: true });
            return;
        } 
        if (typeof addChatMessageToUI === 'function') addChatMessageToUI(messageText, 'user'); 
        if (typeof sendWsMessage === 'function') sendWsMessage("user_message", { content: messageText }); 
        updateGlobalMonitorStatus('running', 'Initializing...');
    };
    
    const handleStopAgentRequest = () => { 
        if (StateManager.getIsAgentRunning()) {
            if (typeof sendWsMessage === 'function') sendWsMessage("cancel_agent", {});
            updateGlobalMonitorStatus('running', 'Cancelling...');
        }
    };
    
    const handleTaskSelection = (taskId) => {
        StateManager.selectTask(taskId);
        const newActiveTaskId = StateManager.getCurrentTaskId();
        if (typeof renderTaskList === 'function') renderTaskList(StateManager.getTasks(), newActiveTaskId);
        const selectedTaskObject = StateManager.getTasks().find(t => t.id === newActiveTaskId);
        if (newActiveTaskId && selectedTaskObject) {
            if (typeof clearChatMessagesUI === 'function') clearChatMessagesUI();
            if (typeof clearMonitorLogUI === 'function') clearMonitorLogUI();
            if (typeof sendWsMessage === 'function') sendWsMessage("context_switch", { taskId: selectedTaskObject.id, taskTitle: selectedTaskObject.title });
        }
    };
    
    const handleWsOpen = () => { 
        if (typeof addChatMessageToUI === 'function') addChatMessageToUI("Connected to backend.", "status_message");
        StateManager.setIsAgentRunning(false);
        updateGlobalMonitorStatus('idle', 'Idle');
        
        if (typeof sendWsMessage === 'function') { 
            sendWsMessage("get_available_models", {}); 
            const activeTask = StateManager.getTasks().find(task => task.id === StateManager.getCurrentTaskId()); 
            if (activeTask) {
                if (typeof sendWsMessage === 'function') sendWsMessage("context_switch", { taskId: activeTask.id, taskTitle: activeTask.title }); 
            } else {
                updateGlobalMonitorStatus('idle', 'No Task');
            }
        } 
    };

    const handleWsClose = () => { 
        if (typeof addChatMessageToUI === 'function') addChatMessageToUI("Connection closed. Please refresh.", "status_message", { isError: true });
        updateGlobalMonitorStatus('disconnected', 'Disconnected'); 
    };
    const handleWsError = () => { 
        if (typeof addChatMessageToUI === 'function') addChatMessageToUI("Connection error. Is the backend running?", "status_message", { isError: true });
        updateGlobalMonitorStatus('error', 'Connection Error'); 
    };
    
    // Simplified init calls
    if (typeof initTaskUI === 'function') initTaskUI({ taskListUl: taskListUl, currentTaskTitleEl: currentTaskTitleElement, uploadFileBtn: uploadFileButtonElement }, { onTaskSelect: handleTaskSelection, onNewTask: () => handleTaskSelection(StateManager.addTask().id), onDeleteTask: (id, title) => { const wasActive = id === StateManager.getCurrentTaskId(); StateManager.deleteTask(id); if (wasActive) handleTaskSelection(StateManager.getCurrentTaskId()); else renderTaskList(StateManager.getTasks(), StateManager.getCurrentTaskId()); }, onRenameTask: (id, oldTitle, newTitle) => { StateManager.renameTask(id, newTitle); renderTaskList(StateManager.getTasks(), StateManager.getCurrentTaskId()); } });
    if (typeof initChatUI === 'function') initChatUI({ chatMessagesContainer: chatMessagesContainer, agentThinkingStatusEl: document.getElementById('agent-thinking-status'), chatTextareaEl: chatTextarea, chatSendButtonEl: chatSendButton }, { onSendMessage: handleSendMessageFromUI });
    if (typeof initMonitorUI === 'function') initMonitorUI({ monitorLogArea: monitorLogAreaElement, statusDot: statusDotElement, monitorStatusText: monitorStatusTextElement, stopButton: stopButtonElement }, { onStopAgent: handleStopAgentRequest });
    if (typeof initArtifactUI === 'function') initArtifactUI({ monitorArtifactArea, artifactNav, prevBtn: artifactPrevBtn, nextBtn: artifactNextBtn, counterEl: artifactCounterElement }, { onNavigate: (dir) => { const newIndex = StateManager.navigateArtifacts(dir); if (newIndex !== -1) updateArtifactDisplayUI(StateManager.getCurrentTaskArtifacts(), newIndex); } });
    if (typeof initLlmSelectorsUI === 'function') initLlmSelectorsUI({ executorLlmSelect: executorLlmSelectElement, roleSelectors: [] }, { onExecutorLlmChange: (id) => { StateManager.setCurrentExecutorLlmId(id); if (typeof sendWsMessage === 'function') sendWsMessage("set_llm", { llm_id: id }); }, onRoleLlmChange: ()=>{} });
    
    if (typeof renderTaskList === 'function') renderTaskList(StateManager.getTasks(), StateManager.getCurrentTaskId());

    // Establish WebSocket Connection
    if (typeof connectWebSocket === 'function') {
        updateGlobalMonitorStatus('disconnected', 'Connecting...');
        connectWebSocket(handleWsOpen, handleWsClose, handleWsError);
    }
});
