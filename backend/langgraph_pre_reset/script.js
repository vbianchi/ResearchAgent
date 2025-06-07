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
            // console.log("[Script.js] Dispatching WS Message:", message.type, message.content); 
            switch (message.type) {
                case 'session_established':
                    if (message.content && message.content.session_id && typeof StateManager !== 'undefined' && typeof StateManager.setCurrentSessionId === 'function') {
                        StateManager.setCurrentSessionId(message.content.session_id);
                        console.log(`[Script.js] Full session ID established via 'session_established': ${message.content.session_id}`);
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
                    if (loadingMsgWrapper) { 
                        loadingMsgWrapper.remove();
                    } 
                    if (typeof scrollToBottomChat === 'function') scrollToBottomChat();
                    if (typeof scrollToBottomMonitorLog === 'function') scrollToBottomMonitorLog(); 
                    
                    if (!StateManager.getIsAgentRunning()) { 
                        updateGlobalMonitorStatus('idle', 'Idle'); 
                        if (typeof showAgentThinkingStatusInUI === 'function') {
                            showAgentThinkingStatusInUI(true, { message: "Idle.", status_key: "IDLE", component_hint: "SYSTEM" });
                        }
                    } else {
                        console.log("[Script.js] History loaded, but agent is already running. Not setting to Idle.");
                    }
                    break;
                case 'agent_major_step_announcement': 
                    if (message.content && typeof message.content.description === 'string') {
                        if (typeof displayMajorStepAnnouncementUI === 'function') {
                            displayMajorStepAnnouncementUI(message.content);
                        }
                    } else {
                        console.warn("[Script.js] Invalid agent_major_step_announcement data:", message.content);
                    }
                    break;
                case 'agent_thinking_update': 
                    if (message.content && typeof message.content === 'object' && (message.content.message || message.content.sub_type)) { 
                        if (typeof showAgentThinkingStatusInUI === 'function') {
                            showAgentThinkingStatusInUI(true, message.content); 
                        }
                    } else if (message.content && typeof message.content.status === 'string') { 
                         if (typeof showAgentThinkingStatusInUI === 'function') {
                            showAgentThinkingStatusInUI(true, { message: message.content.status, status_key: "LEGACY_STATUS", component_hint: "SYSTEM" });
                        }
                    }
                    break;
                case 'agent_message': 
                    if (typeof showAgentThinkingStatusInUI === 'function') showAgentThinkingStatusInUI(false); 
                    if (typeof addChatMessageToUI === 'function') addChatMessageToUI(message.content, message.type, { component_hint: message.content.component_hint || 'DEFAULT' });
                    updateGlobalMonitorStatus('idle', 'Idle');
                    if (typeof showAgentThinkingStatusInUI === 'function') showAgentThinkingStatusInUI(true, { message: "Idle.", status_key: "IDLE", component_hint: "SYSTEM" });
                    break;
                case 'confirmed_plan_log': 
                    if (typeof addChatMessageToUI === 'function') {
                        addChatMessageToUI(message.content, message.type); 
                    }
                    break;
                case 'llm_token_usage': 
                    console.log("[Script.js] Received 'llm_token_usage':", JSON.stringify(message.content)); 
                    if (message.content && typeof message.content === 'object') { 
                        handleTokenUsageUpdate(message.content);
                    } 
                    break;
                case 'propose_plan_for_confirmation':
                    if (message.content && message.content.human_summary && message.content.structured_plan && message.content.plan_id) {
                        StateManager.setCurrentDisplayedPlan(message.content.structured_plan);
                        StateManager.setCurrentPlanProposalId(message.content.plan_id);         
                
                        if (typeof addChatMessageToUI === 'function') { 
                            const planDataForUI = {
                                ...message.content,
                                onConfirm: handlePlanConfirmRequest,
                                onCancel: handlePlanCancelRequest,
                                onViewDetails: handlePlanViewDetailsRequest
                            };
                            addChatMessageToUI(planDataForUI, message.type);
                        }
                        updateGlobalMonitorStatus('idle', 'Awaiting Plan Confirmation'); 
                        if (typeof showAgentThinkingStatusInUI === 'function') showAgentThinkingStatusInUI(true, { message: "Awaiting plan confirmation...", status_key: "AWAITING_PLAN_CONFIRMATION", component_hint: "SYSTEM" });
                    } else {
                        console.error("[Script.js] Invalid propose_plan_for_confirmation data received:", message.content);
                        if (typeof addChatMessageToUI === 'function') addChatMessageToUI("Error: Received invalid plan proposal from backend.", "status_message", {component_hint: "ERROR", isError: true });
                    }
                    break;
                // --- START: New case for tool_result_for_chat (Phase 3) ---
                case 'tool_result_for_chat':
                    console.debug("[Script.js] Received 'tool_result_for_chat':", message.content);
                    if (message.content && typeof message.content === 'object') {
                        if (typeof displayToolOutputMessageUI === 'function') {
                            displayToolOutputMessageUI(message.content);
                        } else {
                            console.error("[Script.js] displayToolOutputMessageUI function not found in chat_ui.js for 'tool_result_for_chat'.");
                        }
                    } else {
                        console.warn("[Script.js] Invalid 'tool_result_for_chat' data received:", message.content);
                    }
                    break;
                // --- END: New case for tool_result_for_chat (Phase 3) ---
                case 'user': 
                    if (typeof addChatMessageToUI === 'function') addChatMessageToUI(message.content, message.type);
                    break;
                case 'status_message': 
                    let statusText = "";
                    let statusComponentHint = "SYSTEM";
                    let isErrorStatus = false;

                    if (typeof message.content === 'string') {
                        statusText = message.content;
                    } else if (message.content && typeof message.content.text === 'string') {
                        statusText = message.content.text;
                        statusComponentHint = message.content.component_hint || statusComponentHint;
                    } else if (message.content) {
                        statusText = String(message.content);
                    }

                    const lowerStatusText = statusText.toLowerCase();
                    isErrorStatus = lowerStatusText.includes("error");

                    if (typeof addChatMessageToUI === 'function') addChatMessageToUI(statusText, message.type, { component_hint: statusComponentHint, isError: isErrorStatus });
                    
                    if (isErrorStatus) { 
                        updateGlobalMonitorStatus('error', statusText);
                        if (typeof showAgentThinkingStatusInUI === 'function') showAgentThinkingStatusInUI(true, { message: "Error occurred.", status_key: "ERROR", component_hint: "ERROR" });
                    } else if (lowerStatusText.includes("complete") || lowerStatusText.includes("cancelled") || lowerStatusText.includes("plan proposal cancelled") || lowerStatusText.includes("task context ready")) { 
                        updateGlobalMonitorStatus('idle', 'Idle');
                        if (typeof showAgentThinkingStatusInUI === 'function') showAgentThinkingStatusInUI(true, { message: "Idle.", status_key: "IDLE", component_hint: "SYSTEM" });
                    }
                    break;
                case 'monitor_log': 
                    if (message.content && typeof message.content.text === 'string') {
                        if (typeof addLogEntryToMonitor === 'function') addLogEntryToMonitor(message.content);
                    } else {
                        if (typeof message.content === 'string' && typeof addLogEntryToMonitor === 'function') {
                             addLogEntryToMonitor({ text: message.content, log_source: "UNKNOWN_STRING_LOG" });
                        }
                    }
                    break;
                case 'update_artifacts': 
                    // console.log(`[Script.js CRITICAL DEBUG] Received 'update_artifacts'. Agent running: ${StateManager.getIsAgentRunning()}`, message.content); // Reduced verbosity
                    if (Array.isArray(message.content)) { 
                        StateManager.setCurrentTaskArtifacts(message.content);
                        const newArtifacts = StateManager.getCurrentTaskArtifacts(); 
                        const newIndexToSet = newArtifacts.length > 0 ? 0 : -1;
                        StateManager.setCurrentArtifactIndex(newIndexToSet);
                        
                        // console.log(`[Script.js] Artifacts updated. New count: ${newArtifacts.length}. Displaying index: ${newIndexToSet}`); // Reduced verbosity
                        // console.log(`[Script.js CRITICAL DEBUG] About to call updateArtifactDisplayUI. Agent running: ${StateManager.getIsAgentRunning()}`); // Reduced verbosity
                        if(typeof updateArtifactDisplayUI === 'function') { 
                            updateArtifactDisplayUI(newArtifacts, StateManager.getCurrentArtifactIndex()); 
                        } 
                    } 
                    break;
                case 'trigger_artifact_refresh': 
                    // console.log(`[Script.js CRITICAL DEBUG] Received 'trigger_artifact_refresh' for taskId: ${message.content?.taskId}. Current task: ${StateManager.getCurrentTaskId()}. Agent running: ${StateManager.getIsAgentRunning()}`); // Reduced verbosity
                    const taskIdToRefresh = message.content?.taskId;
                    if (taskIdToRefresh && taskIdToRefresh === StateManager.getCurrentTaskId()) { 
                        if (typeof addLogEntryToMonitor === 'function') addLogEntryToMonitor({text: `[SYSTEM_EVENT] File event detected for task ${taskIdToRefresh}, requesting artifact list update...`, log_source: "SYSTEM_EVENT"});
                        if (typeof sendWsMessage === 'function') {
                            // console.log(`[Script.js CRITICAL DEBUG] Conditions met. Sending 'get_artifacts_for_task' for task ${StateManager.getCurrentTaskId()}`); // Reduced verbosity
                            sendWsMessage('get_artifacts_for_task', { taskId: StateManager.getCurrentTaskId() });
                        }
                    } else {
                        // console.log(`[Script.js CRITICAL DEBUG] Conditions NOT met for sending 'get_artifacts_for_task'. taskIdToRefresh: ${taskIdToRefresh}, currentTaskId: ${StateManager.getCurrentTaskId()}`); // Reduced verbosity
                    }
                    break;
                case 'available_models': 
                    if (message.content && typeof message.content === 'object') { 
                        StateManager.setAvailableModels({gemini: message.content.gemini || [], ollama: message.content.ollama || []});
                        const backendDefaultExecutorLlmId = message.content.default_executor_llm_id || null; 
                        const backendRoleDefaults = message.content.role_llm_defaults || {};
                        if (typeof populateAllLlmSelectorsUI === 'function') { populateAllLlmSelectorsUI(StateManager.getAvailableModels(), backendDefaultExecutorLlmId, backendRoleDefaults); } 
                    } 
                    break;
                case 'error_parsing_message': 
                    console.error("Error parsing message from WebSocket:", message.content);
                    if (typeof addLogEntryToMonitor === 'function') addLogEntryToMonitor({text: `[SYSTEM_ERROR] Error parsing WebSocket message: ${message.content}`, log_source: "SYSTEM_ERROR"});
                    if (typeof addChatMessageToUI === 'function') addChatMessageToUI("Error: Received an unreadable message from the backend.", "status_message", {component_hint: "ERROR", isError: true}); 
                    break;
                default: 
                    console.warn("[Script.js] Received unknown message type:", message.type, "Content:", message.content);
                    if (typeof addLogEntryToMonitor === 'function') addLogEntryToMonitor({text: `[SYSTEM_WARNING] Unknown message type received: ${message.type}`, log_source: "SYSTEM_WARNING"});
            }
        } catch (error) {
            console.error("[Script.js] Failed to process dispatched WS message:", error, "Original Message:", message);
            if (typeof addLogEntryToMonitor === 'function') addLogEntryToMonitor({text: `[SYSTEM_ERROR] Error processing dispatched message: ${error.message}.`, log_source: "SYSTEM_ERROR"});
            updateGlobalMonitorStatus('error', 'Processing Error');
            if (typeof showAgentThinkingStatusInUI === 'function') showAgentThinkingStatusInUI(false); 
        }
    };

    function updateGlobalMonitorStatus(status, text) { 
        StateManager.setIsAgentRunning(status === 'running' || status === 'cancelling'); 
        if (typeof updateMonitorStatusUI === 'function') { 
            updateMonitorStatusUI(status, text, StateManager.getIsAgentRunning()); 
        } 
    }
    function handleTokenUsageUpdate(lastCallUsage = null) { 
        // console.log("[Script.js] handleTokenUsageUpdate called with:", JSON.stringify(lastCallUsage)); // Reduced verbosity
        StateManager.updateCurrentTaskTotalTokens(lastCallUsage); 
        if (typeof updateTokenDisplayUI === 'function') { 
            updateTokenDisplayUI(lastCallUsage, StateManager.getCurrentTaskTotalTokens()); 
        } 
    }
    
    function clearChatAndMonitor(addLog = true) { 
        if (typeof clearChatMessagesUI === 'function') clearChatMessagesUI(); 
        if (typeof clearMonitorLogUI === 'function') clearMonitorLogUI(); 
        StateManager.setCurrentTaskArtifacts([]); 
        StateManager.setCurrentArtifactIndex(-1); 
        if (typeof clearArtifactDisplayUI === 'function') clearArtifactDisplayUI(); 
        if (addLog && typeof addLogEntryToMonitor === 'function') { 
            addLogEntryToMonitor({text: "[SYSTEM_EVENT] Cleared context.", log_source: "SYSTEM_EVENT"}); 
        } 
        StateManager.setCurrentDisplayedPlan(null); 
        StateManager.setCurrentPlanProposalId(null); 
    };
    
    const handleTaskSelection = (taskId) => { 
        console.log(`[MainScript] Task selection requested for: ${taskId}`);
        const previousActiveTaskId = StateManager.getCurrentTaskId();
        
        StateManager.setIsAgentRunning(false); 
        const selectedTaskObjectForTitle = StateManager.getTasks().find(t => t.id === taskId);
        const taskTitleForStatus = selectedTaskObjectForTitle ? selectedTaskObjectForTitle.title : (taskId ? 'Selected Task' : 'New Task');
        
        updateGlobalMonitorStatus('idle', `Initializing ${taskTitleForStatus}...`);
        if (typeof showAgentThinkingStatusInUI === 'function') {
            showAgentThinkingStatusInUI(true, { 
                message: `Initializing ${taskTitleForStatus}...`, 
                status_key: "TASK_INIT", 
                component_hint: "SYSTEM" 
            });
        }
        if (typeof clearCurrentMajorStepUI === 'function') {
             clearCurrentMajorStepUI();
        }

        StateManager.selectTask(taskId); 
        const newActiveTaskId = StateManager.getCurrentTaskId(); 
        console.log(`[MainScript] StateManager updated. Previous active: ${previousActiveTaskId}, New active: ${newActiveTaskId}`);
        
        if (typeof renderTaskList === 'function') { renderTaskList(StateManager.getTasks(), newActiveTaskId); } 
        
        if (typeof updateTokenDisplayUI === 'function') {
            updateTokenDisplayUI(null, StateManager.getCurrentTaskTotalTokens());
        }
        
        const selectedTaskObject = StateManager.getTasks().find(t => t.id === newActiveTaskId);
        
        if (newActiveTaskId && selectedTaskObject) {
            clearChatAndMonitor(false); 
            if (typeof sendWsMessage === 'function') sendWsMessage("context_switch", { task: selectedTaskObject.title, taskId: selectedTaskObject.id });
        } else if (!newActiveTaskId) { 
            clearChatAndMonitor(); 
            if (typeof addChatMessageToUI === 'function') addChatMessageToUI("No task selected.", "status_message", {component_hint: "SYSTEM"});
            if (typeof addLogEntryToMonitor === 'function') addLogEntryToMonitor({text: "[SYSTEM_EVENT] No task selected.", log_source: "SYSTEM_EVENT"});
            updateGlobalMonitorStatus('idle', 'No Task');
            if (typeof showAgentThinkingStatusInUI === 'function') {
                showAgentThinkingStatusInUI(true, { message: "No task selected.", status_key: "IDLE", component_hint: "SYSTEM" });
            }
            if (typeof updateTokenDisplayUI === 'function') {
                 updateTokenDisplayUI(null, { overall: { input: 0, output: 0, total: 0 }, roles: {} });
            }
        }
    };

    const handleNewTaskCreation = () => { const newTask = StateManager.addTask(); handleTaskSelection(newTask.id); };
    const handleTaskDeletion = (taskId, taskTitle) => { const wasActiveTask = StateManager.getCurrentTaskId() === taskId; StateManager.deleteTask(taskId); 
        if (typeof sendWsMessage === 'function') sendWsMessage("delete_task", { taskId: taskId }); if (wasActiveTask) { handleTaskSelection(StateManager.getCurrentTaskId()); 
        } else { if (typeof renderTaskList === 'function') { renderTaskList(StateManager.getTasks(), StateManager.getCurrentTaskId()); } } };
    const handleTaskRename = (taskId, oldTitle, newTitle) => { if (StateManager.renameTask(taskId, newTitle)) { if (typeof renderTaskList === 'function') renderTaskList(StateManager.getTasks(), StateManager.getCurrentTaskId()); 
        if (taskId === StateManager.getCurrentTaskId()) { if (typeof updateCurrentTaskTitleUI === 'function') updateCurrentTaskTitleUI(StateManager.getTasks(), StateManager.getCurrentTaskId()); 
        } if (typeof sendWsMessage === 'function') sendWsMessage("rename_task", { taskId: taskId, newName: newTitle }); } };
    
    const handleSendMessageFromUI = (messageText) => { 
        if (!StateManager.getCurrentTaskId()) { alert("Please select or create a task first."); return; } 
        if (StateManager.getIsAgentRunning()) { if (typeof addChatMessageToUI === 'function') addChatMessageToUI("Agent is currently busy. Please wait or stop the current process.", "status_message", {component_hint: "SYSTEM"}); return; } 
        if (typeof addChatMessageToUI === 'function') addChatMessageToUI(messageText, 'user'); 
        if (typeof addMessageToInputHistory === 'function') addMessageToInputHistory(messageText); 
        if (typeof showAgentThinkingStatusInUI === 'function') showAgentThinkingStatusInUI(false); 
        if (typeof sendWsMessage === 'function') sendWsMessage("user_message", { content: messageText }); 
        else { if (typeof addChatMessageToUI === 'function') addChatMessageToUI("Error: Cannot send message. Connection issue.", "status_message", {component_hint: "ERROR", isError: true});} 
        updateGlobalMonitorStatus('running', 'Classifying intent...'); 
        if (typeof showAgentThinkingStatusInUI === 'function') showAgentThinkingStatusInUI(true, { message: "Classifying intent...", status_key: "INTENT_CLASSIFICATION_START", component_hint: "INTENT_CLASSIFIER" });
    };
    
    const handlePlanConfirmRequest = (planId) => {
        console.log(`[Script.js] Plan confirmed by user for plan ID: ${planId}`);
        const currentPlan = StateManager.getCurrentDisplayedPlan(); 
        if (!currentPlan) {
            console.error(`[Script.js] Cannot confirm plan ${planId}: No plan found in state.`);
            if (typeof addChatMessageToUI === 'function') addChatMessageToUI("Error: Could not find plan to confirm.", "status_message", {component_hint: "ERROR", isError: true});
            updateGlobalMonitorStatus('error', 'Plan Confirmation Error');
            return;
        }
        if (typeof sendWsMessage === 'function') {
            sendWsMessage('execute_confirmed_plan', { plan_id: planId, confirmed_plan: currentPlan });
        }
        if (typeof transformToConfirmedPlanUI === 'function') {
            transformToConfirmedPlanUI(planId);
        }
        updateGlobalMonitorStatus('running', 'Executing Plan...');
    };
    const handlePlanCancelRequest = (planId) => {
        console.log(`[Script.js] Plan cancelled by user for plan ID: ${planId}`);
        if (typeof sendWsMessage === 'function') {
            sendWsMessage('cancel_plan_proposal', { plan_id: planId });
        }
        const planConfirmContainer = chatMessagesContainer.querySelector(`.plan-confirmation-wrapper[data-plan-id="${planId}"]`);
        if (planConfirmContainer) {
            planConfirmContainer.remove();
        }
        if (typeof addChatMessageToUI === 'function') addChatMessageToUI("Plan proposal cancelled by user.", "status_message", {component_hint: "SYSTEM"});
        updateGlobalMonitorStatus('idle', 'Idle'); 
        if (typeof showAgentThinkingStatusInUI === 'function') showAgentThinkingStatusInUI(true, { message: "Idle.", status_key: "IDLE", component_hint: "SYSTEM" });
        StateManager.setCurrentDisplayedPlan(null);
        StateManager.setCurrentPlanProposalId(null);
    };
    const handlePlanViewDetailsRequest = (planId, isNowVisible) => { if (typeof addLogEntryToMonitor === 'function') { addLogEntryToMonitor({text: `[UI_ACTION] User toggled plan details for proposal ${planId}. Details are now ${isNowVisible ? 'visible' : 'hidden'}.`, log_source: "UI_EVENT"}); } };
    const handleStopAgentRequest = () => { if (StateManager.getIsAgentRunning()) { if (typeof addLogEntryToMonitor === 'function') addLogEntryToMonitor({text: "[SYSTEM_EVENT] Stop request sent by user.", log_source: "SYSTEM_EVENT"}); if (typeof sendWsMessage === 'function') sendWsMessage("cancel_agent", {}); updateGlobalMonitorStatus('cancelling', 'Cancelling...'); if (typeof showAgentThinkingStatusInUI === 'function') showAgentThinkingStatusInUI(true, { message: "Cancelling...", status_key: "CANCELLING", component_hint: "SYSTEM" }); } };
    const handleArtifactNavigation = (direction) => { let currentIndex = StateManager.getCurrentArtifactIndex(); const currentArtifacts = StateManager.getCurrentTaskArtifacts(); let newIndex = currentIndex; if (direction === "prev") { if (currentIndex > 0) newIndex = currentIndex - 1; } else if (direction === "next") { if (currentIndex < currentArtifacts.length - 1) newIndex = currentIndex + 1; } if (newIndex !== currentIndex) { StateManager.setCurrentArtifactIndex(newIndex); if(typeof updateArtifactDisplayUI === 'function') { updateArtifactDisplayUI(currentArtifacts, newIndex); } } };
    const handleExecutorLlmChange = (selectedId) => { StateManager.setCurrentExecutorLlmId(selectedId); if (typeof sendWsMessage === 'function') { sendWsMessage("set_llm", { llm_id: selectedId }); }};
    const handleRoleLlmChange = (role, selectedId) => { StateManager.setRoleLlmOverride(role, selectedId); if (typeof sendWsMessage === 'function') { sendWsMessage("set_session_role_llm", { role: role, llm_id: selectedId }); }};
    const handleThinkingStatusClick = () => { if (typeof scrollToBottomMonitorLog === 'function') { scrollToBottomMonitorLog(); } };
    
    const handleWsOpen = (event) => { 
        if (typeof addLogEntryToMonitor === 'function') addLogEntryToMonitor({text: `[SYSTEM_CONNECTION] WebSocket connection established.`, log_source: "SYSTEM_CONNECTION"}); 
        if (typeof addChatMessageToUI === 'function') addChatMessageToUI("Connected to backend.", "status_message", {component_hint: "SYSTEM"}); 
        updateGlobalMonitorStatus('idle', 'Idle'); 
        if (typeof sendWsMessage === 'function') { 
            sendWsMessage("get_available_models", {}); 
            const activeTaskFromStorage = StateManager.getTasks().find(task => task.id === StateManager.getCurrentTaskId()); 
            if (StateManager.getCurrentTaskId() && activeTaskFromStorage) { 
                console.log(`[Script.js] WS Open: Active task ${activeTaskFromStorage.id} found. Sending context_switch.`);
                StateManager.setIsAgentRunning(false); 
                 if (typeof showAgentThinkingStatusInUI === 'function') {
                    showAgentThinkingStatusInUI(true, { 
                        message: `Initializing Task: ${activeTaskFromStorage.title}...`, 
                        status_key: "TASK_INIT", 
                        component_hint: "SYSTEM" 
                    });
                }
                sendWsMessage("context_switch", { task: activeTaskFromStorage.title, taskId: activeTaskFromStorage.id }); 
            } else { 
                console.log("[Script.js] WS Open: No active task found in StateManager on initial load.");
                updateGlobalMonitorStatus('idle', 'No Task'); 
                if(typeof clearArtifactDisplayUI === 'function') clearArtifactDisplayUI(); 
                if (typeof showAgentThinkingStatusInUI === 'function') showAgentThinkingStatusInUI(true, { message: "No task selected.", status_key: "IDLE", component_hint: "SYSTEM" });
            } 
        } 
    };
    const handleWsClose = (event) => { let reason = event.reason || 'No reason given'; let advice = ""; if (event.code === 1000 || event.wasClean) { reason = "Normal"; } else { reason = `Abnormal (Code: ${event.code})`; advice = " Backend down or network issue?"; } if (typeof addChatMessageToUI === 'function') addChatMessageToUI(`Connection closed.${advice}`, "status_message", {component_hint: "ERROR", isError: true}); if (typeof addLogEntryToMonitor === 'function') addLogEntryToMonitor({text: `[SYSTEM_CONNECTION] WebSocket disconnected. ${reason}`, log_source: "SYSTEM_CONNECTION"}); updateGlobalMonitorStatus('disconnected', 'Disconnected');  if (typeof disableAllLlmSelectorsUI === 'function') disableAllLlmSelectorsUI(); if (typeof showAgentThinkingStatusInUI === 'function') showAgentThinkingStatusInUI(true, { message: "Disconnected.", status_key: "DISCONNECTED", component_hint: "ERROR" }); };
    const handleWsError = (event, isCreationError = false) => { const errorMsg = isCreationError ? "FATAL: Failed to initialize WebSocket connection." : "ERROR: Cannot connect to backend."; if (typeof addChatMessageToUI === 'function') addChatMessageToUI(errorMsg, "status_message", {component_hint: "ERROR", isError: true}); if (typeof addLogEntryToMonitor === 'function') addLogEntryToMonitor({text: `[SYSTEM_ERROR] WebSocket error occurred.`, log_source: "SYSTEM_ERROR"}); updateGlobalMonitorStatus('error', isCreationError ? 'Connection Init Failed' : 'Connection Error'); if (typeof disableAllLlmSelectorsUI === 'function') disableAllLlmSelectorsUI(); if (typeof showAgentThinkingStatusInUI === 'function') showAgentThinkingStatusInUI(true, { message: "Connection Error.", status_key: "ERROR", component_hint: "ERROR" }); };

    // Initialize UI Modules
    if (newTaskButton) { newTaskButton.addEventListener('click', handleNewTaskCreation); }
    document.body.addEventListener('click', event => { if (event.target.classList.contains('action-btn')) { const commandText = event.target.textContent.trim(); if (typeof addLogEntryToMonitor === 'function') addLogEntryToMonitor({text: `[UI_ACTION] Clicked: ${commandText}`, log_source: "UI_EVENT"}); if (typeof sendWsMessage === 'function') sendWsMessage("action_command", { command: commandText }); } });
    if (typeof initTaskUI === 'function') { initTaskUI( { taskListUl: taskListUl, currentTaskTitleEl: currentTaskTitleElement, uploadFileBtn: uploadFileButtonElement }, { onTaskSelect: handleTaskSelection, onNewTask: handleNewTaskCreation, onDeleteTask: handleTaskDeletion, onRenameTask: handleTaskRename }); if (typeof renderTaskList === 'function') renderTaskList(StateManager.getTasks(), StateManager.getCurrentTaskId()); }
    if (typeof initChatUI === 'function') { initChatUI( { chatMessagesContainer: chatMessagesContainer, agentThinkingStatusEl: agentThinkingStatusElement, chatTextareaEl: chatTextarea, chatSendButtonEl: chatSendButton }, { onSendMessage: handleSendMessageFromUI, onThinkingStatusClick: handleThinkingStatusClick }); }
    if (typeof initMonitorUI === 'function') { initMonitorUI( { monitorLogArea: monitorLogAreaElement, statusDot: statusDotElement, monitorStatusText: monitorStatusTextElement, stopButton: stopButtonElement }, { onStopAgent: handleStopAgentRequest }); }
    
    if (typeof initArtifactUI === 'function') { 
        console.log("[Script.js DEBUG] Before initArtifactUI: StateManager type:", typeof StateManager);
        if (StateManager) {
            console.log("[Script.js DEBUG] StateManager object keys:", Object.keys(StateManager));
            console.log("[Script.js DEBUG] StateManager.getCurrentTaskArtifacts type:", typeof StateManager.getCurrentTaskArtifacts);
            console.log("[Script.js DEBUG] StateManager.getCurrentArtifactIndex type:", typeof StateManager.getCurrentArtifactIndex);
        } else {
            console.error("[Script.js DEBUG] StateManager is undefined or null before initArtifactUI!");
        }

        initArtifactUI( 
            { monitorArtifactArea: monitorArtifactArea, artifactNav: artifactNav, prevBtn: artifactPrevBtn, nextBtn: artifactNextBtn, counterEl: artifactCounterElement }, 
            { onNavigate: handleArtifactNavigation }
        ); 
        
        if (typeof updateArtifactDisplayUI === 'function') { 
            if (StateManager && typeof StateManager.getCurrentTaskArtifacts === 'function' && typeof StateManager.getCurrentArtifactIndex === 'function') {
                updateArtifactDisplayUI(StateManager.getCurrentTaskArtifacts(), StateManager.getCurrentArtifactIndex()); 
            } else {
                console.error("[Script.js CRITICAL ERROR] StateManager or its methods are still not available right before calling updateArtifactDisplayUI.", StateManager);
                if (typeof clearArtifactDisplayUI === 'function') clearArtifactDisplayUI();
                if (monitorArtifactArea && artifactNav) {
                    const placeholder = document.createElement('div');
                    placeholder.className = 'artifact-placeholder';
                    placeholder.textContent = 'Error initializing artifacts (StateManager issue).';
                    monitorArtifactArea.insertBefore(placeholder, artifactNav);
                }
            }
        } 
    }

    if (typeof initLlmSelectorsUI === 'function') { initLlmSelectorsUI( { executorLlmSelect: executorLlmSelectElement, roleSelectors: roleSelectorsMetaForInit }, { onExecutorLlmChange: handleExecutorLlmChange, onRoleLlmChange: handleRoleLlmChange }); }
    if (typeof initTokenUsageUI === 'function') { 
        initTokenUsageUI({}); 
        updateTokenDisplayUI(null, StateManager.getCurrentTaskTotalTokens());
    }
    if (typeof initFileUploadUI === 'function') { initFileUploadUI( { fileUploadInputEl: fileUploadInputElement, uploadFileButtonEl: uploadFileButtonElement }, { httpBackendBaseUrl: httpBackendBaseUrl }, { getCurrentTaskId: StateManager.getCurrentTaskId, addLog: (logData) => { if (typeof addLogEntryToMonitor === 'function') addLogEntryToMonitor(logData); }, addChatMsg: (msgText, msgType, options) => { if (typeof addChatMessageToUI === 'function') addChatMessageToUI(msgText, msgType, options); } }); }

    // Establish WebSocket Connection
    if (typeof connectWebSocket === 'function') { if (typeof addLogEntryToMonitor === 'function') addLogEntryToMonitor({text: "[SYSTEM_CONNECTION] Attempting to connect to backend...", log_source: "SYSTEM_CONNECTION"}); updateGlobalMonitorStatus('disconnected', 'Connecting...'); connectWebSocket(handleWsOpen, handleWsClose, handleWsError);
    } else { console.error("connectWebSocket function not found."); if (typeof addChatMessageToUI === 'function') addChatMessageToUI("ERROR: WebSocket manager not loaded.", "status_message", {component_hint: "ERROR", isError: true}); updateGlobalMonitorStatus('error', 'Initialization Error'); }
});

