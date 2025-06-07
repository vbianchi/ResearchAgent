/**
 * This script acts as the main orchestrator for the AI Agent UI.
 * - Initializes StateManager and UI modules.
 * - Manages the core application lifecycle.
 * - Routes events/messages between UI modules, StateManager, and WebSocket communication.
 * - Handles WebSocket message dispatching.
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log("AI Agent UI Script Loaded and DOM ready! Initializing StateManager...");
    
    if (typeof StateManager === 'undefined' || typeof StateManager.initStateManager !== 'function') {
        console.error("FATAL: StateManager is not loaded or initStateManager is not a function. Ensure state_manager.js is loaded before script.js.");
        alert("Application critical error: State manager failed to load. Please check console and refresh.");
        return;
    }
    StateManager.initStateManager(); 
    console.log("[Script.js] StateManager.initStateManager() called.");

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
    const agentThinkingStatusElement = document.getElementById('agent-thinking-status'); 

    const roleSelectorsMetaForInit = [
        { element: document.getElementById('intent-llm-select'), role: 'intent_classifier', storageKey: 'sessionIntentClassifierLlmId', label: 'Intent Classifier' },
        { element: document.getElementById('planner-llm-select'), role: 'planner', storageKey: 'sessionPlannerLlmId', label: 'Planner' },
        { element: document.getElementById('controller-llm-select'), role: 'controller', storageKey: 'sessionControllerLlmId', label: 'Controller' },
        { element: document.getElementById('evaluator-llm-select'), role: 'evaluator', storageKey: 'sessionEvaluatorLlmId', label: 'Evaluator' }
    ];
    const executorLlmSelectElement = document.getElementById('llm-select');

    const httpBackendBaseUrl = 'http://localhost:8766'; 
    let isLoadingHistory = false; 

    window.dispatchWsMessage = (message) => {
        try {
            switch (message.type) {
                case 'session_established':
                    if (message.content && message.content.session_id && typeof StateManager !== 'undefined' && typeof StateManager.setCurrentSessionId === 'function') {
                        StateManager.setCurrentSessionId(message.content.session_id);
                    }
                    break;
                case 'history_start': 
                    isLoadingHistory = true;
                    if (typeof addChatMessageToUI === 'function') addChatMessageToUI("Loading history...", "status_message", { component_hint: "SYSTEM"});
                    updateGlobalMonitorStatus('running', 'Loading History...'); 
                    break;
                case 'history_end': 
                    isLoadingHistory = false;
                    const loadingMsgWrapper = Array.from(chatMessagesContainer.querySelectorAll('.message-outer-blue-line'))
                                               .find(wrapper => wrapper.querySelector('.message-system-status-content')?.textContent.startsWith("Loading history..."));
                    if (loadingMsgWrapper) loadingMsgWrapper.remove();
                    if (typeof scrollToBottomChat === 'function') scrollToBottomChat();
                    if (typeof scrollToBottomMonitorLog === 'function') scrollToBottomMonitorLog(); 
                    
                    // <<< FIX: Always set status to Idle when history loading ends to prevent deadlock >>>
                    updateGlobalMonitorStatus('idle', 'Idle');
                    break;
                case 'agent_thinking_update': 
                    if (message.content && typeof message.content.status === 'string') {
                        updateGlobalMonitorStatus('running', message.content.status);
                    }
                    break;
                case 'agent_message': 
                    if (typeof addChatMessageToUI === 'function') addChatMessageToUI(message.content, message.type);
                    updateGlobalMonitorStatus('idle', 'Idle');
                    break;
                case 'llm_token_usage': 
                    if (message.content && typeof message.content === 'object') { 
                        handleTokenUsageUpdate(message.content);
                    } 
                    break;
                case 'user': 
                    if (typeof addChatMessageToUI === 'function') addChatMessageToUI(message.content, message.type);
                    break;
                case 'status_message': 
                    if (typeof addChatMessageToUI === 'function') addChatMessageToUI(message.content, message.type, { isError: String(message.content?.text || message.content).toLowerCase().includes("error") });
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
                        if (typeof populateAllLlmSelectorsUI === 'function') populateAllLlmSelectorsUI(StateManager.getAvailableModels(), message.content.default_executor_llm_id, message.content.role_llm_defaults);
                    } 
                    break;
                default: 
                    console.warn("[Script.js] Received unknown message type:", message.type);
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
    function handleTokenUsageUpdate(lastCallUsage = null) { 
        StateManager.updateCurrentTaskTotalTokens(lastCallUsage); 
        if (typeof updateTokenDisplayUI === 'function') { 
            updateTokenDisplayUI(lastCallUsage, StateManager.getCurrentTaskTotalTokens()); 
        } 
    }
    
    function clearChatAndMonitor(addLog = true) { 
        if (typeof clearChatMessagesUI === 'function') clearChatMessagesUI(); 
        if (typeof clearMonitorLogUI === 'function') clearMonitorLogUI(); 
        StateManager.setCurrentTaskArtifacts([]); 
        if (typeof clearArtifactDisplayUI === 'function') clearArtifactDisplayUI(); 
        if (addLog && typeof addLogEntryToMonitor === 'function') { 
            addLogEntryToMonitor({text: "[SYSTEM_EVENT] Cleared context.", log_source: "SYSTEM_EVENT"}); 
        } 
    };
    
    const handleTaskSelection = (taskId) => { 
        StateManager.selectTask(taskId); 
        const newActiveTaskId = StateManager.getCurrentTaskId(); 
        if (typeof renderTaskList === 'function') renderTaskList(StateManager.getTasks(), newActiveTaskId);
        if (typeof updateTokenDisplayUI === 'function') updateTokenDisplayUI(null, StateManager.getCurrentTaskTotalTokens());
        const selectedTaskObject = StateManager.getTasks().find(t => t.id === newActiveTaskId);
        if (newActiveTaskId && selectedTaskObject) {
            clearChatAndMonitor(false); 
            if (typeof sendWsMessage === 'function') sendWsMessage("context_switch", { taskId: selectedTaskObject.id, taskTitle: selectedTaskObject.title });
        } else {
            clearChatAndMonitor();
        }
    };

    const handleNewTaskCreation = () => { const newTask = StateManager.addTask(); handleTaskSelection(newTask.id); };
    const handleTaskDeletion = (taskId, taskTitle) => { const wasActiveTask = StateManager.getCurrentTaskId() === taskId; StateManager.deleteTask(taskId); if (typeof sendWsMessage === 'function') sendWsMessage("delete_task", { taskId: taskId }); if (wasActiveTask) { handleTaskSelection(StateManager.getCurrentTaskId()); } else { if (typeof renderTaskList === 'function') renderTaskList(StateManager.getTasks(), StateManager.getCurrentTaskId()); } };
    const handleTaskRename = (taskId, oldTitle, newTitle) => { if (StateManager.renameTask(taskId, newTitle)) { if (typeof renderTaskList === 'function') renderTaskList(StateManager.getTasks(), StateManager.getCurrentTaskId()); if (taskId === StateManager.getCurrentTaskId() && typeof updateCurrentTaskTitleUI === 'function') updateCurrentTaskTitleUI(StateManager.getTasks(), StateManager.getCurrentTaskId()); if (typeof sendWsMessage === 'function') sendWsMessage("rename_task", { taskId: taskId, newName: newTitle }); } };
    
    const handleSendMessageFromUI = (messageText) => { 
        if (!StateManager.getCurrentTaskId()) { alert("Please select or create a task first."); return; } 
        if (StateManager.getIsAgentRunning()) { addChatMessageToUI("Agent is currently busy. Please wait.", "status_message", { isError: true }); return; } 
        if (typeof addChatMessageToUI === 'function') addChatMessageToUI(messageText, 'user'); 
        if (typeof sendWsMessage === 'function') sendWsMessage("user_message", { content: messageText }); 
        updateGlobalMonitorStatus('running', 'Initializing...');
    };
    
    const handleStopAgentRequest = () => { if (StateManager.getIsAgentRunning()) { if (typeof sendWsMessage === 'function') sendWsMessage("cancel_agent", {}); updateGlobalMonitorStatus('running', 'Cancelling...'); } };
    const handleArtifactNavigation = (direction) => { const newIndex = StateManager.navigateArtifacts(direction); if (newIndex !== -1 && typeof updateArtifactDisplayUI === 'function') updateArtifactDisplayUI(StateManager.getCurrentTaskArtifacts(), newIndex); };
    const handleExecutorLlmChange = (selectedId) => { StateManager.setCurrentExecutorLlmId(selectedId); if (typeof sendWsMessage === 'function') sendWsMessage("set_llm", { llm_id: selectedId }); };
    
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
    
    // Initialize UI Modules with all their required elements and callbacks
    if (newTaskButton) newTaskButton.addEventListener('click', handleNewTaskCreation);
    if (typeof initTaskUI === 'function') initTaskUI({ taskListUl: taskListUl, currentTaskTitleEl: currentTaskTitleElement, uploadFileBtn: uploadFileButtonElement }, { onTaskSelect: handleTaskSelection, onNewTask: handleNewTaskCreation, onDeleteTask: handleTaskDeletion, onRenameTask: handleTaskRename });
    if (typeof initChatUI === 'function') initChatUI({ chatMessagesContainer: chatMessagesContainer, agentThinkingStatusEl: agentThinkingStatusElement, chatTextareaEl: chatTextarea, chatSendButtonEl: chatSendButton }, { onSendMessage: handleSendMessageFromUI });
    if (typeof initMonitorUI === 'function') initMonitorUI({ monitorLogArea: monitorLogAreaElement, statusDot: statusDotElement, monitorStatusText: monitorStatusTextElement, stopButton: stopButtonElement }, { onStopAgent: handleStopAgentRequest });
    if (typeof initArtifactUI === 'function') initArtifactUI({ monitorArtifactArea: monitorArtifactArea, artifactNav: artifactNav, prevBtn: artifactPrevBtn, nextBtn: artifactNextBtn, counterEl: artifactCounterElement }, { onNavigate: handleArtifactNavigation });
    if (typeof initLlmSelectorsUI === 'function') initLlmSelectorsUI({ executorLlmSelect: executorLlmSelectElement, roleSelectors: roleSelectorsMetaForInit }, { onExecutorLlmChange: handleExecutorLlmChange, onRoleLlmChange: () => {} /* No-op for now */ });
    if (typeof initTokenUsageUI === 'function') initTokenUsageUI({});
    if (typeof initFileUploadUI === 'function') initFileUploadUI({ fileUploadInputEl: fileUploadInputElement, uploadFileButtonEl: uploadFileButtonElement }, { httpBackendBaseUrl: httpBackendBaseUrl }, { getCurrentTaskId: StateManager.getCurrentTaskId, addLog: (logData) => addLogEntryToMonitor(logData), addChatMsg: (msgText, msgType, options) => addChatMessageToUI(msgText, msgType, options) });

    // Initial Render
    if (typeof renderTaskList === 'function') renderTaskList(StateManager.getTasks(), StateManager.getCurrentTaskId());
    if (typeof updateTokenDisplayUI === 'function') updateTokenDisplayUI(null, StateManager.getCurrentTaskTotalTokens());

    // Establish WebSocket Connection
    if (typeof connectWebSocket === 'function') {
        updateGlobalMonitorStatus('disconnected', 'Connecting...');
        connectWebSocket(handleWsOpen, handleWsClose, handleWsError);
    } else {
        console.error("connectWebSocket function not found.");
        updateGlobalMonitorStatus('error', 'Initialization Error');
    }
});
