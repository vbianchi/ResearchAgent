// js/ui_modules/chat_ui.js

/**
 * Manages the Chat UI.
 * - Renders chat messages (user, agent, status, step announcements, sub-statuses, thoughts, tool outputs).
 * - Handles Markdown formatting.
 * - Displays plan proposals and confirmed plans.
 * - Manages chat input and thinking status.
 * - Handles collapsibility for long tool outputs and copy-to-clipboard.
 * - Implements collapsible major steps.
 */

let chatMessagesContainerElement;
let agentThinkingStatusElement; // Global status line at the bottom
let chatTextareaElement;
let chatSendButtonElement;
let onSendMessageCallback = (messageText) => console.warn("[ChatUI] onSendMessageCallback not set.");
let onThinkingStatusClickCallback = () => console.warn("[ChatUI] onThinkingStatusClickCallback not set.");

let chatInputHistory = [];
const MAX_CHAT_HISTORY = 50;
let chatHistoryIndex = -1;
let currentInputBuffer = "";

let currentMajorStepDiv = null; // To hold the div of the current major step for appending sub-statuses/thoughts

// Component hint to CSS class mapping for side-lines
const componentBorderColorMap = {
    DEFAULT: 'agent-line-default', // Fallback if no specific hint
    USER: 'user-message-line-color', // Not a border, but used for consistency
    SYSTEM: 'agent-line-system',
    INTENT_CLASSIFIER: 'agent-line-intent-classifier',
    PLANNER: 'agent-line-planner',
    CONTROLLER: 'agent-line-controller',
    EXECUTOR: 'agent-line-executor',
    EVALUATOR_STEP: 'agent-line-evaluator-step',
    EVALUATOR_OVERALL: 'agent-line-evaluator-overall',
    TOOL: 'agent-line-tool',
    LLM_CORE: 'agent-line-llm-core',
    WARNING: 'agent-line-warning',
    ERROR: 'agent-line-error'
};

const MAX_CHARS_TOOL_OUTPUT_PREVIEW = 500;
const MAX_LINES_TOOL_OUTPUT_PREVIEW = 10;


async function handleCopyToClipboard(textToCopy, buttonElement) {
    if (!navigator.clipboard) {
        console.warn('[ChatUI] Clipboard API not available. Falling back to execCommand if possible.');
        try {
            const textArea = document.createElement("textarea");
            textArea.value = textToCopy;
            textArea.style.position = "fixed"; 
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            const successful = document.execCommand('copy');
            document.body.removeChild(textArea);

            if (successful) {
                console.log('[ChatUI] Text copied to clipboard using execCommand.');
                if (buttonElement) {
                    const originalText = buttonElement.innerHTML; 
                    buttonElement.innerHTML = 'Copied ✓';
                    buttonElement.disabled = true;
                    setTimeout(() => {
                        buttonElement.innerHTML = originalText;
                        buttonElement.disabled = false;
                    }, 1500);
                }
            } else {
                throw new Error('execCommand failed');
            }
        } catch (err) {
            console.error('[ChatUI] Failed to copy text using execCommand: ', err);
            if (buttonElement) {
                const originalText = buttonElement.innerHTML;
                buttonElement.innerHTML = 'Error';
                setTimeout(() => { buttonElement.innerHTML = originalText; }, 2000);
            }
        }
        return;
    }
    try {
        await navigator.clipboard.writeText(textToCopy);
        console.log('[ChatUI] Text copied to clipboard:', textToCopy.substring(0, 50) + "...");
        if (buttonElement) {
            const originalText = buttonElement.innerHTML; 
            buttonElement.innerHTML = 'Copied ✓';
            buttonElement.disabled = true;
            setTimeout(() => {
                buttonElement.innerHTML = originalText;
                buttonElement.disabled = false;
            }, 1500);
        }
    } catch (err) {
        console.error('[ChatUI] Failed to copy text using navigator.clipboard: ', err);
        if (buttonElement) {
            const originalText = buttonElement.innerHTML;
            buttonElement.innerHTML = 'Failed!';
            setTimeout(() => { buttonElement.innerHTML = originalText; }, 2000);
        }
    }
}

function _createCopyButton(getTextToCopyFn, buttonText = '📋&nbsp;Copy') {
    const copyButton = document.createElement('button');
    copyButton.className = 'chat-copy-btn';
    copyButton.innerHTML = buttonText; 
    copyButton.title = 'Copy to clipboard';
    copyButton.onclick = (e) => {
        e.stopPropagation(); 
        const textToCopy = getTextToCopyFn();
        handleCopyToClipboard(textToCopy, copyButton);
    };
    return copyButton;
}

function _addCopyButtonsToPreBlocks(parentElement) {
    if (!parentElement) return;
    const preBlocks = parentElement.querySelectorAll('pre:not(.copy-btn-added)');
    preBlocks.forEach(preElement => {
        if (preElement.parentElement.classList.contains('pre-wrapper-with-copy')) {
            preElement.classList.add('copy-btn-added'); 
            return;
        }
        const textToCopyFn = () => preElement.textContent || "";
        const copyButton = _createCopyButton(textToCopyFn, 'Copy Code');
        const wrapper = document.createElement('div');
        wrapper.className = 'pre-wrapper-with-copy';
        preElement.parentNode.insertBefore(wrapper, preElement);
        wrapper.appendChild(preElement); 
        wrapper.appendChild(copyButton);
        preElement.classList.add('copy-btn-added'); 
    });
}

function initChatUI(elements, callbacks) {
    console.log("[ChatUI] Initializing...");
    chatMessagesContainerElement = elements.chatMessagesContainer;
    agentThinkingStatusElement = elements.agentThinkingStatusEl; 
    chatTextareaElement = elements.chatTextareaEl;
    chatSendButtonElement = elements.chatSendButtonEl;

    if (!chatMessagesContainerElement || !agentThinkingStatusElement || !chatTextareaElement || !chatSendButtonElement) {
        console.error("[ChatUI] One or more essential UI elements not provided!");
        return;
    }

    onSendMessageCallback = callbacks.onSendMessage || onSendMessageCallback;
    onThinkingStatusClickCallback = callbacks.onThinkingStatusClick || onThinkingStatusClickCallback;

    chatSendButtonElement.addEventListener('click', handleSendButtonClick);
    chatTextareaElement.addEventListener('keydown', handleChatTextareaKeydown);
    chatTextareaElement.addEventListener('input', handleChatTextareaInput);
    
    agentThinkingStatusElement.addEventListener('click', () => {
        if (typeof onThinkingStatusClickCallback === 'function') {
            onThinkingStatusClickCallback();
        }
    });
    console.log("[ChatUI] Initialized.");
}

function handleSendButtonClick() {
    const messageText = chatTextareaElement.value.trim();
    if (messageText) {
        if (typeof onSendMessageCallback === 'function') {
            onSendMessageCallback(messageText);
        }
        addMessageToInputHistory(messageText);
        chatTextareaElement.value = '';
        adjustTextareaHeight();
        currentInputBuffer = ""; 
        chatHistoryIndex = -1;    
    }
    chatTextareaElement.focus();
}

function handleChatTextareaKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        handleSendButtonClick();
    } else if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
        if (chatInputHistory.length === 0 && chatTextareaElement.value.trim() === "") return;
        if (chatHistoryIndex === -1 && (chatTextareaElement.value.trim() !== "" || event.key === 'ArrowUp')) {
            if (chatInputHistory.length > 0 || event.key === 'ArrowUp') {
                 currentInputBuffer = chatTextareaElement.value;
            }
        }
        let newHistoryIndex = chatHistoryIndex;
        if (event.key === 'ArrowUp') {
            if (chatInputHistory.length > 0) { 
                newHistoryIndex = (chatHistoryIndex === -1) ? chatInputHistory.length - 1 : Math.max(0, chatHistoryIndex - 1);
            } else { return; } 
        } else if (event.key === 'ArrowDown') {
            if (chatHistoryIndex !== -1 && chatHistoryIndex < chatInputHistory.length - 1) {
                newHistoryIndex++;
            } else { 
                newHistoryIndex = -1; 
            }
        }
        if (newHistoryIndex !== chatHistoryIndex || (event.key === 'ArrowDown' && chatHistoryIndex === chatInputHistory.length - 1) ) {
            event.preventDefault(); 
            chatHistoryIndex = newHistoryIndex;
            chatTextareaElement.value = (chatHistoryIndex === -1) ? currentInputBuffer : chatInputHistory[chatHistoryIndex];
            chatTextareaElement.selectionStart = chatTextareaElement.selectionEnd = chatTextareaElement.value.length;
            adjustTextareaHeight();
        }
    } else {
        chatHistoryIndex = -1; 
    }
}

function handleChatTextareaInput() {
    adjustTextareaHeight();
    if (chatHistoryIndex !== -1) { 
        currentInputBuffer = chatTextareaElement.value;
        chatHistoryIndex = -1;
    }
}

function adjustTextareaHeight() {
    if (!chatTextareaElement) return;
    chatTextareaElement.style.height = 'auto'; 
    chatTextareaElement.style.height = (chatTextareaElement.scrollHeight) + 'px';
}

function addMessageToInputHistory(messageText) {
    if (chatInputHistory[chatInputHistory.length - 1] !== messageText) {
        chatInputHistory.push(messageText);
        if (chatInputHistory.length > MAX_CHAT_HISTORY) {
            chatInputHistory.shift(); 
        }
    }
}

function escapeHTML(str) {
    if (typeof str !== 'string') return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}

function formatMessageContentInternal(text, isThoughtOrToolContentBox = false) {
    if (typeof text !== 'string') {
        text = String(text); 
    }
    const codeBlockPlaceholders = [];
    let tempText = text.replace(/```(\w*)\n([\s\S]*?)\n?```/g, (match, lang, code) => {
        const escapedCode = escapeHTML(code); 
        const langClass = lang ? ` class="language-${lang}"` : '';
        const placeholder = `%%CODEBLOCK_${codeBlockPlaceholders.length}%%`;
        codeBlockPlaceholders.push(`<pre data-is-code-block="true"><code${langClass}>${escapedCode}</code></pre>`);
        return placeholder;
    });
    const inlineCodePlaceholders = [];
    tempText = tempText.replace(/`([^`]+?)`/g, (match, code) => {
        const escapedCode = escapeHTML(code); 
        const placeholder = `%%INLINECODE_${inlineCodePlaceholders.length}%%`;
        inlineCodePlaceholders.push(`<code>${escapedCode}</code>`);
        return placeholder;
    });
    tempText = tempText.replace(/\[([^<>[\]]+?)\]\((https?:\/\/[^\s)]+)\)/g, (match, linkText, linkUrl) => {
        const safeLinkText = escapeHTML(linkText); 
        const safeLinkUrl = escapeHTML(linkUrl); 
        return `<a href="${safeLinkUrl}" target="_blank" rel="noopener noreferrer">${safeLinkText}</a>`;
    });
    tempText = tempText.replace(/(\*\*\*|___)(?=\S)([\s\S]*?\S)\1/g, (match, wrapper, content) => `<strong><em>${escapeHTML(content)}</em></strong>`);
    tempText = tempText.replace(/(\*\*|__)(?=\S)([\s\S]*?\S)\1/g, (match, wrapper, content) => `<strong>${escapeHTML(content)}</strong>`);
    tempText = tempText.replace(/(?<![`*\w\\])(?:(\*|_))(?=\S)([\s\S]*?\S)\1(?![`*\w])/g, (match, wrapper, content) => `<em>${escapeHTML(content)}</em>`);
    if (!isThoughtOrToolContentBox) {
        const partsForNewline = tempText.split(/(%%CODEBLOCK_\d+%%|%%INLINECODE_\d+%%)/g);
        for (let i = 0; i < partsForNewline.length; i++) {
            if (!partsForNewline[i].startsWith('%%CODEBLOCK_') && !partsForNewline[i].startsWith('%%INLINECODE_')) {
                partsForNewline[i] = partsForNewline[i].replace(/\n/g, '<br>');
            }
        }
        tempText = partsForNewline.join('');
    }
    tempText = tempText.replace(/%%INLINECODE_(\d+)%%/g, (match, index) => inlineCodePlaceholders[parseInt(index)]);
    tempText = tempText.replace(/%%CODEBLOCK_(\d+)%%/g, (match, index) => codeBlockPlaceholders[parseInt(index)]);
    return tempText;
}

function getComponentClass(componentHint) {
    const hint = String(componentHint).toUpperCase(); 
    if (componentBorderColorMap[hint]) {
        return componentBorderColorMap[hint];
    }
    if (hint.startsWith("TOOL_")) { 
        return componentBorderColorMap.TOOL; 
    }
    return componentBorderColorMap.SYSTEM; 
}

function displayMajorStepAnnouncementUI(data) {
    if (!chatMessagesContainerElement) {
        console.error("[ChatUI] Chat container missing! Cannot display major step.");
        return;
    }
    
    const { step_number, total_steps, description } = data; 
    
    const stepWrapperDiv = document.createElement('div');
    stepWrapperDiv.className = 'message message-agent-step'; 
    
    const titleDiv = document.createElement('div');
    titleDiv.className = 'step-title'; 
    titleDiv.innerHTML = formatMessageContentInternal(`<strong>Step ${step_number}/${total_steps}: ${description}</strong>`);
    
    titleDiv.addEventListener('click', () => {
        stepWrapperDiv.classList.toggle('step-collapsed');
    });
    stepWrapperDiv.appendChild(titleDiv);
    
    const subContentContainer = document.createElement('div');
    subContentContainer.className = 'sub-content-container';
    stepWrapperDiv.appendChild(subContentContainer);
    
    currentMajorStepDiv = stepWrapperDiv; 

    appendMessageElement(stepWrapperDiv);
    scrollToBottomChat(); 
    console.log(`[ChatUI] Displayed Major Step Announcement: Step ${step_number}/${total_steps}`);
}

function displayToolOutputMessageUI(data) {
    // Removed the initial console.log for brevity, but kept the read_file specific one
    if (data && data.tool_name === 'read_file') {
        console.log(`[ChatUI DEBUG read_file] Tool: ${data.tool_name}, Input: ${data.tool_input_summary}, Content Length: ${data.tool_output_content?.length}, Artifact: ${data.artifact_filename}`);
    }

    const { tool_name, tool_input_summary, tool_output_content, artifact_filename, original_length } = data;

    const toolOutputWrapperDiv = document.createElement('div');
    toolOutputWrapperDiv.className = `message message-agent-tool-output ${getComponentClass('TOOL')}`;

    const topRowDiv = document.createElement('div'); 
    topRowDiv.className = 'tool-output-top-row'; 

    const labelDiv = document.createElement('div');
    labelDiv.className = 'tool-output-label clickable'; 
    labelDiv.innerHTML = `Tool Output: <strong>${escapeHTML(tool_name)}</strong> (Input: <em>${escapeHTML(tool_input_summary)}</em>)`;
    topRowDiv.appendChild(labelDiv);

    const copyButton = _createCopyButton(() => tool_output_content);
    topRowDiv.appendChild(copyButton);
    
    toolOutputWrapperDiv.appendChild(topRowDiv);

    const contentBoxDiv = document.createElement('div');
    contentBoxDiv.className = 'tool-output-content-box';

    const currentOriginalLength = typeof original_length === 'number' ? original_length : (tool_output_content ? tool_output_content.length : 0);
    const lines = tool_output_content ? tool_output_content.split('\n') : [];
    const isLongContent = currentOriginalLength > MAX_CHARS_TOOL_OUTPUT_PREVIEW || lines.length > MAX_LINES_TOOL_OUTPUT_PREVIEW;

    const previewDiv = document.createElement('div');
    previewDiv.className = 'tool-output-preview';
    
    const fullDiv = document.createElement('div');
    fullDiv.className = 'tool-output-full';

    if (isLongContent) {
        let previewText = lines.slice(0, MAX_LINES_TOOL_OUTPUT_PREVIEW).join('\n');
        if (previewText.length > MAX_CHARS_TOOL_OUTPUT_PREVIEW) {
            previewText = previewText.substring(0, MAX_CHARS_TOOL_OUTPUT_PREVIEW);
        }
        previewDiv.innerHTML = formatMessageContentInternal(previewText + (lines.length > MAX_LINES_TOOL_OUTPUT_PREVIEW || currentOriginalLength > MAX_CHARS_TOOL_OUTPUT_PREVIEW ? "\n..." : ""), true);
        contentBoxDiv.appendChild(previewDiv);

        fullDiv.style.display = 'none'; 
        fullDiv.innerHTML = formatMessageContentInternal(tool_output_content, true);
        contentBoxDiv.appendChild(fullDiv);
        labelDiv.classList.add('minimized'); 
    } else {
        previewDiv.style.display = 'none';
        contentBoxDiv.appendChild(previewDiv); 
        fullDiv.innerHTML = formatMessageContentInternal(tool_output_content || "(No output content)", true); 
        contentBoxDiv.appendChild(fullDiv);
    }
    toolOutputWrapperDiv.appendChild(contentBoxDiv);
    _addCopyButtonsToPreBlocks(contentBoxDiv);

    labelDiv.addEventListener('click', (e) => {
        e.stopPropagation();
        const isCurrentlyExpanded = toolOutputWrapperDiv.classList.toggle('expanded');
        if (isLongContent) { 
            previewDiv.style.display = isCurrentlyExpanded ? 'none' : 'block';
            fullDiv.style.display = isCurrentlyExpanded ? 'block' : 'none';
        } else { 
             previewDiv.style.display = 'none';
             fullDiv.style.display = 'block';
        }
    });

    if (artifact_filename) {
        const artifactLinkDiv = document.createElement('div');
        artifactLinkDiv.className = 'tool-output-artifact-link'; 
        artifactLinkDiv.innerHTML = `<em>References artifact: ${escapeHTML(artifact_filename)}</em>`;
        if (isLongContent) { 
            toolOutputWrapperDiv.appendChild(artifactLinkDiv); 
            artifactLinkDiv.style.display = toolOutputWrapperDiv.classList.contains('expanded') ? 'block' : 'none';
        } else { 
            toolOutputWrapperDiv.appendChild(artifactLinkDiv);
        }
    }
    
    const expandButton = toolOutputWrapperDiv.querySelector('.tool-output-expand-btn');
    if (expandButton) { 
        if (isLongContent) {
            expandButton.style.display = 'block'; 
            expandButton.textContent = toolOutputWrapperDiv.classList.contains('expanded') ? 'Collapse' : 'Expand';
            labelDiv.addEventListener('click', () => { 
                 expandButton.textContent = toolOutputWrapperDiv.classList.contains('expanded') ? 'Collapse' : 'Expand';
                 if (artifact_filename) {
                    const artifactLink = toolOutputWrapperDiv.querySelector('.tool-output-artifact-link');
                    if(artifactLink) artifactLink.style.display = toolOutputWrapperDiv.classList.contains('expanded') ? 'block' : 'none';
                 }
            });
        } else {
            expandButton.style.display = 'none';
        }
    }

    if (currentMajorStepDiv) {
        const subContentContainer = currentMajorStepDiv.querySelector('.sub-content-container');
        if (subContentContainer) {
            subContentContainer.appendChild(toolOutputWrapperDiv);
        } else {
            console.warn("[ChatUI] .sub-content-container not found in currentMajorStepDiv. Appending tool output to main chat.");
            appendMessageElement(toolOutputWrapperDiv); 
        }
    } else {
        appendMessageElement(toolOutputWrapperDiv); 
    }

    scrollToBottomChat(); 
    console.log(`[ChatUI] Displayed Tool Output: ${tool_name}`);
    return toolOutputWrapperDiv;
}

function addChatMessageToUI(messageData, type, options = {}, doScroll = true) {
    if (!chatMessagesContainerElement) {
        console.error("[ChatUI] Chat container missing! Cannot add message:", messageData);
        return null;
    }

    let baseMessageDiv; 
    let contentHolderDiv = null; // Initialize to null
    let messageProcessedInternally = false; // Flag to track if content is handled by a sub-function

    let textContent, componentHint; 
    if (typeof messageData === 'string') {
        textContent = messageData;
        componentHint = options.component_hint;
    } else if (typeof messageData === 'object' && messageData !== null) {
        textContent = messageData.content || messageData.text || (messageData.message ? String(messageData.message) : JSON.stringify(messageData));
        componentHint = messageData.component_hint || options.component_hint;
    } else {
        textContent = String(messageData);
        componentHint = options.component_hint;
    }
    const effectiveComponentHint = componentHint || 'SYSTEM';
    const originalTextContentForCopy = textContent; 

    if (type === 'user') {
        baseMessageDiv = document.createElement('div');
        baseMessageDiv.className = 'message message-user-wrapper'; 
        contentHolderDiv = document.createElement('div');
        contentHolderDiv.className = 'message-user'; 
        baseMessageDiv.appendChild(contentHolderDiv);
    } else if (type === 'propose_plan_for_confirmation') {
        baseMessageDiv = document.createElement('div');
        baseMessageDiv.className = 'message message-agent-wrapper'; 
        displayPlanConfirmationUI( // This function now appends to baseMessageDiv internally
            messageData.human_summary, 
            messageData.plan_id, 
            messageData.structured_plan,
            messageData.onConfirm, 
            messageData.onCancel,
            messageData.onViewDetails,
            baseMessageDiv 
        ); 
        messageProcessedInternally = true; 
    } else if (type === 'tool_result_for_chat') { 
        // displayToolOutputMessageUI handles its own creation and appending to the correct parent (main chat or sub-content)
        // It returns the element, but we don't need to re-append it here.
        baseMessageDiv = displayToolOutputMessageUI(messageData);
        messageProcessedInternally = true; 
    } else { 
        if (type === 'agent_message' || type === 'confirmed_plan_log') {
            baseMessageDiv = document.createElement('div');
            baseMessageDiv.className = 'message message-agent-wrapper'; 
        } else { 
            baseMessageDiv = document.createElement('div');
            baseMessageDiv.className = 'message message-outer-blue-line'; 
        }

        if (type === 'agent_message') { 
            currentMajorStepDiv = null; 
            const finalAnswerWrapper = document.createElement('div');
            finalAnswerWrapper.className = 'message-agent-final-content-wrapper';
            const avatarDiv = document.createElement('div');
            avatarDiv.className = 'agent-avatar';
            avatarDiv.textContent = 'RA'; 
            finalAnswerWrapper.appendChild(avatarDiv);
            contentHolderDiv = document.createElement('div');
            contentHolderDiv.className = 'message-agent-final-content';
            finalAnswerWrapper.appendChild(contentHolderDiv);
            baseMessageDiv.appendChild(finalAnswerWrapper);
            const copyBtnFinalAnswer = _createCopyButton(() => originalTextContentForCopy);
            baseMessageDiv.appendChild(copyBtnFinalAnswer);
        } else { 
            contentHolderDiv = document.createElement('div'); 
            baseMessageDiv.appendChild(contentHolderDiv);

            if (type === 'status_message') {
                contentHolderDiv.className = 'message-system-status-content';
                if (options.isError || String(textContent).toLowerCase().includes("error")) {
                    contentHolderDiv.classList.add('error-text'); 
                }
            } else if (type === 'confirmed_plan_log') { 
                // Removed the console.log for textContent here, as it's visible in the main console now.
                if (textContent) { 
                    contentHolderDiv.className = 'message-plan-proposal-content'; 
                    try {
                        const planData = JSON.parse(textContent); 
                        const planBlock = document.createElement('div');
                        planBlock.className = 'plan-proposal-block plan-confirmed-static'; 
                        const titleElement = document.createElement('h4');
                        titleElement.innerHTML = formatMessageContentInternal(planData.summary ? 'Confirmed Plan (from history):' : (planData.title || 'Plan Details (from history):'));
                        planBlock.appendChild(titleElement);

                        if (planData.summary) { 
                            const summaryElement = document.createElement('p');
                            summaryElement.className = 'plan-summary';
                            summaryElement.innerHTML = formatMessageContentInternal(planData.summary);
                            planBlock.appendChild(summaryElement);
                            _addCopyButtonsToPreBlocks(summaryElement);
                        }

                        const detailsDiv = document.createElement('div');
                        detailsDiv.className = 'plan-steps-details';
                        detailsDiv.style.display = 'block'; 
                        const ol = document.createElement('ol');
                        if (planData.steps && Array.isArray(planData.steps)) {
                            planData.steps.forEach(step => {
                                const li = document.createElement('li');
                                const stepDescription = `<strong>${step.step_id}. ${formatMessageContentInternal(step.description)}</strong>`;
                                const toolUsed = (step.tool_to_use && step.tool_to_use !== "None") ? `<br><span class="step-tool">Tool: ${formatMessageContentInternal(step.tool_to_use)}</span>` : '';
                                const inputHint = step.tool_input_instructions ? `<br><span class="step-tool">Input Hint: ${formatMessageContentInternal(step.tool_input_instructions)}</span>` : '';
                                const expectedOutcome = `<br><span class="step-expected">Expected: ${formatMessageContentInternal(step.expected_outcome)}</span>`;
                                li.innerHTML = stepDescription + toolUsed + inputHint + expectedOutcome;
                                ol.appendChild(li);
                            });
                        }
                        detailsDiv.appendChild(ol);
                        _addCopyButtonsToPreBlocks(detailsDiv);
                        planBlock.appendChild(detailsDiv); 
                        
                        let statusP = planBlock.querySelector('.plan-execution-status-confirmed');
                        if (!statusP) {
                            statusP = document.createElement('p');
                            statusP.className = 'plan-execution-status-confirmed';
                            planBlock.appendChild(statusP); 
                        }
                        const confirmedTime = planData.timestamp ? new Date(planData.timestamp).toLocaleTimeString() : "previously";
                        statusP.textContent = `Status: Confirmed & Executed ${confirmedTime}`; 

                        contentHolderDiv.appendChild(planBlock); 
                        messageProcessedInternally = true; // Content handled, textContent not needed for direct render
                    } catch (e) {
                        console.error("[ChatUI] Error parsing confirmed_plan_log data from history:", e, "Raw Data:", textContent);
                        textContent = `Error displaying confirmed plan from history.`; 
                        // Fall through to render textContent in contentHolderDiv
                    }
                } else {
                     console.warn("[ChatUI confirmed_plan_log] textContent was empty or null. Rendering error message.");
                     textContent = "Error: Plan data missing in history."; 
                }
            } else { 
                contentHolderDiv.className = 'message-content-text'; 
                if (!baseMessageDiv.classList.contains('message-agent-wrapper')) {
                    baseMessageDiv.classList.add(getComponentClass(effectiveComponentHint)); 
                }
            }
        }
    }
    
    if (contentHolderDiv && !messageProcessedInternally && textContent !== null) { 
        contentHolderDiv.innerHTML = formatMessageContentInternal(textContent);
        _addCopyButtonsToPreBlocks(contentHolderDiv);
    }
    
    // --- MODIFIED: Appending Logic ---
    // Always append baseMessageDiv if it was created and not already handled by a sub-function like displayToolOutputMessageUI
    if (baseMessageDiv && type !== 'tool_result_for_chat' && type !== 'propose_plan_for_confirmation') { 
        appendMessageElement(baseMessageDiv);
    } else if (baseMessageDiv && (type === 'propose_plan_for_confirmation')) {
        // displayPlanConfirmationUI already calls appendMessageElement with baseMessageDiv
        // So, no explicit append here for this type.
    }
    // For tool_result_for_chat, displayToolOutputMessageUI handles its own append.
    
    if (doScroll) {
        scrollToBottomChat();
    }
    return baseMessageDiv; 
}


function displayPlanConfirmationUI(humanSummary, planId, structuredPlan, onConfirm, onCancel, onViewDetails, baseWrapper) {
    if (!chatMessagesContainerElement || !baseWrapper) return null;

    chatMessagesContainerElement.querySelectorAll('.plan-confirmation-wrapper').forEach(ui => {
        if (ui.dataset.planId !== planId) ui.remove(); 
    });
    const existingPlanUI = chatMessagesContainerElement.querySelector(`.plan-confirmation-wrapper[data-plan-id="${planId}"]`);
    if (existingPlanUI) existingPlanUI.remove();

    baseWrapper.classList.add('plan-confirmation-wrapper'); 
    baseWrapper.dataset.planId = planId; 

    const planContentDiv = document.createElement('div');
    planContentDiv.className = 'message-plan-proposal-content'; 
    
    const planBlock = document.createElement('div'); 
    planBlock.className = 'plan-proposal-block';

    const titleElement = document.createElement('h4');
    titleElement.textContent = 'Agent Proposed Plan:'; 
    planBlock.appendChild(titleElement);

    const summaryElement = document.createElement('p');
    summaryElement.className = 'plan-summary';
    summaryElement.innerHTML = formatMessageContentInternal(humanSummary);
    planBlock.appendChild(summaryElement);
    _addCopyButtonsToPreBlocks(summaryElement); 

    const detailsDiv = document.createElement('div');
    detailsDiv.className = 'plan-steps-details';
    detailsDiv.style.display = 'none'; 

    const ol = document.createElement('ol');
    if (structuredPlan && Array.isArray(structuredPlan)) {
        structuredPlan.forEach(step => {
            const li = document.createElement('li');
            const stepDescription = `<strong>${step.step_id}. ${formatMessageContentInternal(step.description)}</strong>`;
            const toolUsed = (step.tool_to_use && step.tool_to_use !== "None") ? `<br><span class="step-tool">Tool: ${formatMessageContentInternal(step.tool_to_use)}</span>` : '';
            const inputHint = step.tool_input_instructions ? `<br><span class="step-tool">Input Hint: ${formatMessageContentInternal(step.tool_input_instructions)}</span>` : '';
            const expectedOutcome = `<br><span class="step-expected">Expected: ${formatMessageContentInternal(step.expected_outcome)}</span>`;
            li.innerHTML = stepDescription + toolUsed + inputHint + expectedOutcome;
            ol.appendChild(li);
        });
    } else {
        ol.innerHTML = "<li>Plan details not available.</li>"; 
    }
    detailsDiv.appendChild(ol);
    _addCopyButtonsToPreBlocks(detailsDiv); 
    planBlock.appendChild(detailsDiv);

    const viewDetailsBtn = document.createElement('button');
    viewDetailsBtn.className = 'plan-toggle-details-btn';
    viewDetailsBtn.textContent = 'View Details';
    viewDetailsBtn.title = `View detailed plan for proposal ${planId}`;
    viewDetailsBtn.onclick = (e) => { 
        e.stopPropagation();
        const isHidden = detailsDiv.style.display === 'none';
        detailsDiv.style.display = isHidden ? 'block' : 'none';
        viewDetailsBtn.textContent = isHidden ? 'Hide Details' : 'View Details';
        if (typeof onViewDetails === 'function') {
            onViewDetails(planId, isHidden);
        }
    };
    planBlock.appendChild(viewDetailsBtn);

    const actionsDiv = document.createElement('div');
    actionsDiv.className = 'plan-actions';
    const confirmBtn = document.createElement('button');
    confirmBtn.className = 'plan-confirm-btn';
    confirmBtn.textContent = 'Confirm & Run';
    confirmBtn.onclick = (e) => { e.stopPropagation(); if (typeof onConfirm === 'function') onConfirm(planId); };
    actionsDiv.appendChild(confirmBtn);
    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'plan-cancel-btn';
    cancelBtn.textContent = 'Cancel';
    cancelBtn.onclick = (e) => { e.stopPropagation(); if (typeof onCancel === 'function') onCancel(planId); };
    actionsDiv.appendChild(cancelBtn);
    planBlock.appendChild(actionsDiv);
    
    planContentDiv.appendChild(planBlock);
    baseWrapper.appendChild(planContentDiv); 

    appendMessageElement(baseWrapper); // This line was already here and correctly appends the baseWrapper.
    scrollToBottomChat();
    return baseWrapper; 
}


function transformToConfirmedPlanUI(planId) {
    if (!chatMessagesContainerElement) return;
    const planWrapper = chatMessagesContainerElement.querySelector(`.plan-confirmation-wrapper[data-plan-id="${planId}"]`);
    if (!planWrapper) {
        addChatMessageToUI(`Plan (ID: ${planId.substring(0,8)}...) confirmed. Executing steps...`, 'status_message', {component_hint: 'SYSTEM'});
        return;
    }
    
    const planBlock = planWrapper.querySelector('.plan-proposal-block');
    if (!planBlock) return;

    planBlock.classList.add('plan-confirmed-static'); 

    const titleElement = planBlock.querySelector('h4');
    if (titleElement) titleElement.textContent = 'Plan Confirmed:'; 
    
    const viewDetailsBtn = planBlock.querySelector('.plan-toggle-details-btn');
    if (viewDetailsBtn) viewDetailsBtn.remove();
    
    const actionsDiv = planBlock.querySelector('.plan-actions');
    if (actionsDiv) actionsDiv.remove();

    const detailsDiv = planBlock.querySelector('.plan-steps-details');
    if (detailsDiv) detailsDiv.style.display = 'block'; 

    let statusP = planBlock.querySelector('.plan-execution-status-confirmed');
    if (!statusP) {
        statusP = document.createElement('p');
        statusP.className = 'plan-execution-status-confirmed';
        planBlock.appendChild(statusP); 
    }
    statusP.textContent = `Status: Confirmed & Execution Started (at ${new Date().toLocaleTimeString()})`; 
    scrollToBottomChat();
}

function showAgentThinkingStatusInUI(show, statusUpdateObject = { message: "Thinking...", status_key: "DEFAULT_THINKING", component_hint: "SYSTEM", sub_type: null }) {
    if (!agentThinkingStatusElement || !chatMessagesContainerElement) return;

    let displayMessage = "Thinking...";
    let componentHint = statusUpdateObject?.component_hint || "SYSTEM";
    let statusKey = statusUpdateObject?.status_key || "UNKNOWN_STATUS";
    let subType = statusUpdateObject?.sub_type; 
    let originalMarkdownForCopy = ""; 

    if (typeof statusUpdateObject === 'string') { 
        displayMessage = statusUpdateObject;
        originalMarkdownForCopy = displayMessage;
    } else if (statusUpdateObject && typeof statusUpdateObject.message === 'string') {
        displayMessage = statusUpdateObject.message;
        originalMarkdownForCopy = displayMessage;
    } else if (statusUpdateObject && typeof statusUpdateObject.message === 'object' && subType === 'thought') {
        originalMarkdownForCopy = statusUpdateObject.message.content_markdown || "";
    } else if (statusUpdateObject && statusUpdateObject.message) {
         displayMessage = String(statusUpdateObject.message); 
         originalMarkdownForCopy = displayMessage;
    }

    const isFinalStateForBottomLine = ["IDLE", "CANCELLED", "ERROR", "PLAN_FAILED", "DIRECT_QA_COMPLETED", "DIRECT_QA_FAILED", "UNKNOWN_INTENT", "AWAITING_PLAN_CONFIRMATION", "PLAN_STOPPED", "PLAN_COMPLETED_ISSUES"].includes(statusKey);

    if (show && currentMajorStepDiv && (subType === 'sub_status' || subType === 'thought')) {
        const subContentContainer = currentMajorStepDiv.querySelector('.sub-content-container');
        if (subContentContainer) {
            let nestedMessageDiv;
            if (subType === 'sub_status') {
                nestedMessageDiv = document.createElement('div');
                nestedMessageDiv.className = `message message-agent-substatus ${getComponentClass(componentHint)}`;
                const contentEl = document.createElement('div');
                contentEl.className = 'content';
                contentEl.innerHTML = formatMessageContentInternal(`<em>${displayMessage}</em>`);
                nestedMessageDiv.appendChild(contentEl);
            } else if (subType === 'thought' && statusUpdateObject.message && typeof statusUpdateObject.message === 'object') {
                nestedMessageDiv = document.createElement('div');
                nestedMessageDiv.className = `message message-agent-thought ${getComponentClass(componentHint)}`;
                
                const thoughtTopRow = document.createElement('div');
                thoughtTopRow.className = 'thought-top-row';

                const labelEl = document.createElement('div');
                labelEl.className = 'thought-label';
                labelEl.innerHTML = formatMessageContentInternal(statusUpdateObject.message.label || `${componentHint} thought:`);
                thoughtTopRow.appendChild(labelEl);

                const copyBtnThought = _createCopyButton(() => originalMarkdownForCopy);
                thoughtTopRow.appendChild(copyBtnThought);
                nestedMessageDiv.appendChild(thoughtTopRow);

                const contentBoxEl = document.createElement('div');
                contentBoxEl.className = 'thought-content-box';
                contentBoxEl.innerHTML = formatMessageContentInternal(originalMarkdownForCopy, true); 
                nestedMessageDiv.appendChild(contentBoxEl);
                _addCopyButtonsToPreBlocks(contentBoxEl); 
            }

            if (nestedMessageDiv) {
                subContentContainer.appendChild(nestedMessageDiv); 
                agentThinkingStatusElement.style.display = 'none'; 
                scrollToBottomChat();
                return; 
            }
        } else {
            console.warn("[ChatUI] currentMajorStepDiv exists, but .sub-content-container not found within it. Appending to main chat.");
        }
    }
    
    if (show) {
        agentThinkingStatusElement.innerHTML = formatMessageContentInternal(displayMessage); 
        agentThinkingStatusElement.className = `message agent-thinking-status ${getComponentClass(componentHint)}`; 
        agentThinkingStatusElement.style.display = 'block';
        if (chatMessagesContainerElement.lastChild !== agentThinkingStatusElement) {
            chatMessagesContainerElement.appendChild(agentThinkingStatusElement);
        }
    } else { 
        if (isFinalStateForBottomLine) {
            agentThinkingStatusElement.innerHTML = formatMessageContentInternal(displayMessage); 
            agentThinkingStatusElement.className = `message agent-thinking-status ${getComponentClass(componentHint)}`;
            agentThinkingStatusElement.style.display = 'block';
            if (chatMessagesContainerElement.lastChild !== agentThinkingStatusElement) {
                chatMessagesContainerElement.appendChild(agentThinkingStatusElement);
            }
        } else {
            agentThinkingStatusElement.style.display = 'none';
        }
    }

    if (isFinalStateForBottomLine) {
        currentMajorStepDiv = null; 
    }
    scrollToBottomChat();
}


function clearChatMessagesUI() {
    if (chatMessagesContainerElement) {
        const thinkingStatus = agentThinkingStatusElement; 
        chatMessagesContainerElement.innerHTML = ''; 
        if (thinkingStatus) { 
            chatMessagesContainerElement.appendChild(thinkingStatus); 
            thinkingStatus.style.display = 'none'; 
            thinkingStatus.innerHTML = ''; 
        }
        currentMajorStepDiv = null; 
    }
}

function scrollToBottomChat() {
    if (chatMessagesContainerElement) {
        setTimeout(() => {
            chatMessagesContainerElement.scrollTop = chatMessagesContainerElement.scrollHeight;
        }, 0); 
    }
}

function appendMessageElement(messageElement) {
    if (!chatMessagesContainerElement || !messageElement) return;

    const thinkingStatusIsPresent = agentThinkingStatusElement.parentNode === chatMessagesContainerElement;
    if (thinkingStatusIsPresent) {
        chatMessagesContainerElement.insertBefore(messageElement, agentThinkingStatusElement);
    } else {
        chatMessagesContainerElement.appendChild(messageElement);
    }
}

function clearCurrentMajorStepUI() {
    currentMajorStepDiv = null;
}

