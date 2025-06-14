import { h } from 'preact';
import { useState, useEffect, useRef, useCallback } from 'preact/hooks';
import { ArchitectIcon, ChevronsLeftIcon, ChevronsRightIcon, ChevronDownIcon, EditorIcon, ForemanIcon, HandymanIcon, LoaderIcon, PencilIcon, PlusCircleIcon, RouterIcon, SlidersIcon, SupervisorIcon, Trash2Icon, UserIcon, FileIcon, ArrowLeftIcon, UploadCloudIcon, SendIcon } from './components/Icons';
import { ArchitectCard, SiteForemanCard, PlanApprovalCard, FinalAnswerCard } from './components/AgentCards';
import { ToggleButton, CopyButton } from './components/Common';

const PromptCard = ({ content }) => (
    <div class="mt-8 p-4 rounded-lg shadow-md bg-blue-900/50 border border-gray-700/50">
        <h3 class="font-bold text-sm text-gray-300 mb-2 capitalize flex items-center gap-2"><UserIcon class="h-5 w-5" /> You</h3>
        <p class="text-white whitespace-pre-wrap font-medium">{content}</p>
    </div>
);

const TaskItem = ({ task, isActive, onSelect, onRename, onDelete }) => {
    const [isEditing, setIsEditing] = useState(false);
    const [editText, setEditText] = useState(task.name);
    const inputRef = useRef(null);
    const handleStartEditing = (e) => { e.stopPropagation(); setIsEditing(true); };
    const handleSave = () => { if (editText.trim()) { onRename(task.id, editText.trim()); } setIsEditing(false); };
    const handleKeyDown = (e) => { if (e.key === 'Enter') handleSave(); else if (e.key === 'Escape') setIsEditing(false); };
    useEffect(() => { if (isEditing) inputRef.current?.focus(); }, [isEditing]);
    return (
        <div onClick={() => onSelect(task.id)} class={`group flex justify-between items-center p-3 mb-2 rounded-lg cursor-pointer transition-colors ${isActive ? 'bg-blue-600/50' : 'hover:bg-gray-700/50'}`}>
            {isEditing ? ( <input ref={inputRef} type="text" value={editText} onInput={(e) => setEditText(e.target.value)} onBlur={handleSave} onKeyDown={handleKeyDown} onClick={(e) => e.stopPropagation()} class="w-full bg-transparent text-white outline-none"/> ) : ( <p class="font-medium text-white truncate">{task.name}</p> )}
            {!isEditing && ( <div class="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity"> <button onClick={handleStartEditing} class="p-1 hover:text-white"><PencilIcon class="h-4 w-4" /></button> <button onClick={(e) => { e.stopPropagation(); onDelete(task.id); }} class="p-1 hover:text-red-400"><Trash2Icon class="h-4 w-4" /></button> </div> )}
        </div>
    );
};

const ModelSelector = ({ label, icon, onModelChange, models, selectedModel, roleKey }) => (
    <div class="mb-4 last:mb-0">
        <label class="block text-sm font-medium text-gray-400 mb-1 flex items-center gap-2">{icon}{label}</label>
        <div class="relative"> <select value={selectedModel} onChange={(e) => onModelChange(roleKey, e.target.value)} class="w-full p-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-blue-500 focus:outline-none appearance-none text-sm" disabled={!selectedModel || models.length === 0}> {models.map(model => <option key={model.id} value={model.id}>{model.name}</option>)} </select> <div class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-400"> <ChevronDownIcon class="h-4 w-4" /> </div> </div>
    </div>
);

const SettingsPanel = ({ models, selectedModels, onModelChange }) => {
    const [isExpanded, setIsExpanded] = useState(false);
    const agentRoles = [
        { key: 'ROUTER_LLM_ID', label: 'Router', icon: <RouterIcon className="h-4 w-4"/>, desc: "Classifies user requests." },
        { key: 'HANDYMAN_LLM_ID', label: 'Handyman', icon: <HandymanIcon className="h-4 w-4"/>, desc: "Handles simple tool commands." },
        { key: 'CHIEF_ARCHITECT_LLM_ID', label: 'Chief Architect', icon: <ArchitectIcon className="h-4 w-4"/>, desc: "Creates complex plans." },
        { key: 'SITE_FOREMAN_LLM_ID', label: 'Site Foreman', icon: <ForemanIcon className="h-4 w-4"/>, desc: "Prepares tool calls." },
        { key: 'PROJECT_SUPERVISOR_LLM_ID', label: 'Project Supervisor', icon: <SupervisorIcon className="h-4 w-4"/>, desc: "Validates step outcomes." },
        { key: 'EDITOR_LLM_ID', label: 'Editor', icon: <EditorIcon className="h-4 w-4"/>, desc: "Synthesizes all final reports." },
    ];
    return (
        <div class="mt-auto border-t border-gray-700 pt-4">
             <div class="flex items-center justify-between cursor-pointer" onClick={() => setIsExpanded(!isExpanded)}>
                <div class="flex items-center gap-2"> <SlidersIcon class="h-5 w-5 text-gray-400" /> <h3 class="text-lg font-semibold text-gray-200">Agent Models</h3> </div>
                <ChevronDownIcon class={`h-5 w-5 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
             </div>
             {isExpanded && ( <div class="mt-4 pl-2"> {agentRoles.map(role => ( <div key={role.key}> <ModelSelector label={role.label} roleKey={role.key} icon={role.icon} models={models} selectedModel={selectedModels[role.key]} onModelChange={onModelChange}/> <p class="text-xs text-gray-500 -mt-2 mb-4 pl-7">{role.desc}</p> </div> ))} </div> )}
        </div>
    )
}

export function App() {
    const [tasks, setTasks] = useState([]);
    const [activeTaskId, setActiveTaskId] = useState(null);
    const [isThinking, setIsThinking] = useState(false);
    const [inputValue, setInputValue] = useState("");
    const [connectionStatus, setConnectionStatus] = useState("Disconnected");
    const [isLeftSidebarVisible, setIsLeftSidebarVisible] = useState(true);
    const [isRightSidebarVisible, setIsRightSidebarVisible] = useState(true);
    const [workspaceFiles, setWorkspaceFiles] = useState([]);
    const [workspaceLoading, setWorkspaceLoading] = useState(false);
    const [workspaceError, setWorkspaceError] = useState(null);
    const [selectedFile, setSelectedFile] = useState(null);
    const [fileContent, setFileContent] = useState('');
    const [isFileLoading, setIsFileLoading] = useState(false);
    const [availableModels, setAvailableModels] = useState([]);
    const [selectedModels, setSelectedModels] = useState({});
    
    const ws = useRef(null);
    const messagesEndRef = useRef(null);
    const fileInputRef = useRef(null);
    const handlersRef = useRef();

    useEffect(() => { handlersRef.current = { fetchWorkspaceFiles, activeTaskId }; });
    useEffect(() => {
        const savedTasks = JSON.parse(localStorage.getItem('research_agent_tasks') || '[]');
        const savedActiveId = localStorage.getItem('research_agent_active_task_id');
        setTasks(savedTasks);
        if (savedActiveId && savedTasks.some(t => t.id === savedActiveId)) setActiveTaskId(savedActiveId);
        else if (savedTasks.length > 0) setActiveTaskId(savedTasks[0].id);
    }, []);
    useEffect(() => { localStorage.setItem('research_agent_tasks', JSON.stringify(tasks)); }, [tasks]);
    useEffect(() => { if (activeTaskId) localStorage.setItem('research_agent_active_task_id', activeTaskId); }, [activeTaskId]);
    
    const selectTask = (taskId) => { if (taskId !== activeTaskId) { setActiveTaskId(taskId); setIsThinking(false); } };
    const createNewTask = () => {
        const newTaskId = `task_${Date.now()}`;
        if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify({ type: 'task_create', task_id: newTaskId }));
            setTasks(prev => [...prev, { id: newTaskId, name: `New Task ${tasks.length + 1}`, history: [] }]);
            selectTask(newTaskId);
        }
    };
    const handleRenameTask = (taskId, newName) => { setTasks(prev => prev.map(t => t.id === taskId ? { ...t, name: newName } : t)); };
    const handleDeleteTask = (taskIdToDelete) => {
        if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify({ type: 'task_delete', task_id: taskIdToDelete }));
        const remainingTasks = tasks.filter(task => task.id !== taskIdToDelete);
        if (activeTaskId === taskIdToDelete) selectTask(remainingTasks[0]?.id || null);
        setTasks(remainingTasks);
    };
    const handleModelChange = (roleKey, modelId) => { setSelectedModels(prev => ({ ...prev, [roleKey]: modelId })); };
    
    const scrollToBottom = () => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); };
    
    const fetchWorkspaceFiles = useCallback(async (path) => { if (!path) return; setWorkspaceLoading(true); try { const r = await fetch(`http://localhost:8766/files?path=${path}`); if (!r.ok) throw new Error((await r.json()).error); setWorkspaceFiles((await r.json()).files || []); } catch (e) { setWorkspaceError(e.message); } finally { setWorkspaceLoading(false); } }, []);
    const fetchFileContent = useCallback(async (filename) => { if (!activeTaskId || !filename) return; setIsFileLoading(true); setSelectedFile(filename); try { const r = await fetch(`http://localhost:8766/file-content?path=${activeTaskId}&filename=${filename}`); if (!r.ok) throw new Error((await r.json()).error); setFileContent(await r.text()); } catch (e) { setFileContent(`Error: ${e.message}`); } finally { setIsFileLoading(false); } }, [activeTaskId]);
    const handleFileUpload = useCallback(async (e) => { const file = e.target.files[0]; if (!file || !activeTaskId) return; setWorkspaceLoading(true); const fd = new FormData(); fd.append('file', file); fd.append('workspace_id', activeTaskId); try { const r = await fetch('http://localhost:8766/upload', { method: 'POST', body: fd }); if (!r.ok) throw new Error((await r.json()).error); await fetchWorkspaceFiles(activeTaskId); } catch (e) { setWorkspaceError(`Upload failed: ${e.message}`); } finally { setWorkspaceLoading(false); if(fileInputRef.current) fileInputRef.current.value = ""; } }, [activeTaskId, fetchWorkspaceFiles]);

    useEffect(() => { fetchModels(); }, []);
    useEffect(() => { if (activeTaskId) fetchWorkspaceFiles(activeTaskId); }, [activeTaskId]);

    const fetchModels = async () => {
        try {
            const r = await fetch('http://localhost:8766/api/models');
            if (!r.ok) throw new Error('Failed to fetch models.');
            const config = await r.json();
            setAvailableModels(config.available_models);
            setSelectedModels(config.default_models);
        } catch (error) { console.error(error); }
    };

    useEffect(() => {
        function connect() {
            setConnectionStatus("Connecting...");
            const socket = new WebSocket("ws://localhost:8765");
            ws.current = socket;
            socket.onopen = () => setConnectionStatus("Connected");
            socket.onclose = () => { setConnectionStatus("Disconnected"); setTimeout(connect, 5000); };
            socket.onerror = (err) => { console.error("WS Error:", err); socket.close(); };
            socket.onmessage = (event) => handleSocketMessage(JSON.parse(event.data));
        }
        connect();
        return () => ws.current?.close();
    }, []);

    const handleSocketMessage = (msg) => {
        if (msg.task_id !== handlersRef.current.activeTaskId) return;
    
        setTasks(currentTasks => {
            const taskIndex = currentTasks.findIndex(t => t.id === msg.task_id);
            if (taskIndex === -1) return currentTasks;
    
            const newTasks = [...currentTasks];
            const task = { ...newTasks[taskIndex] };
            let history = [...task.history];
            let runContainer = history.length > 0 && history[history.length - 1]?.type === 'run_container' ? { ...history[history.length - 1] } : null;
    
            const ensureRunContainer = () => {
                if (!runContainer) {
                    runContainer = { type: 'run_container', children: [] };
                    history.push(runContainer);
                }
            };
    
            if (msg.type === 'final_answer') {
                ensureRunContainer();
                runContainer.children.push({ type: 'final_answer', content: msg.data });
                setIsThinking(false);
                handlersRef.current.fetchWorkspaceFiles(msg.task_id);
            } else if (msg.type === 'plan_approval_request') {
                ensureRunContainer();
                // Add an ArchitectCard to show the plan being proposed
                runContainer.children.push({ type: 'architect_plan', steps: msg.plan });
                // Then add the approval card itself
                runContainer.children.push({ type: 'plan_approval', plan: msg.plan });
                setIsThinking(false);
            } else if (msg.type === 'agent_event' && msg.name === 'Site_Foreman' && msg.event === 'on_chain_start') {
                ensureRunContainer();
                // When foreman starts, remove the approval card and add the execution card
                runContainer.children = runContainer.children.filter(c => c.type !== 'plan_approval');
                const plan = msg.data.input.plan || [];
                runContainer.children.push({ type: 'execution_plan', steps: plan.map(s => ({ ...s, status: 'pending' })) });
                setIsThinking(true); // Now we are executing
            } else if (msg.type === 'agent_event' && msg.name === 'Project_Supervisor' && msg.event === 'on_chain_end') {
                ensureRunContainer();
                const execPlan = runContainer.children.find(c => c.type === 'execution_plan');
                if (execPlan) {
                    const stepIndex = msg.data.input.current_step_index;
                    const stepStatus = msg.data.output.step_evaluation?.status === 'failure' ? 'failure' : 'completed';
                    const newSteps = [...execPlan.steps];
                    if (newSteps[stepIndex]) {
                        newSteps[stepIndex] = { 
                            ...newSteps[stepIndex], 
                            status: stepStatus, 
                            toolCall: msg.data.input.current_tool_call, 
                            toolOutput: msg.data.output.tool_output, 
                            evaluation: msg.data.output.step_evaluation 
                        };
                        execPlan.steps = newSteps;
                    }
                }
                 if (handlersRef.current.activeTaskId === msg.task_id) {
                    handlersRef.current.fetchWorkspaceFiles(msg.task_id);
                }
            } else if (msg.type === 'error') {
                alert(`Agent Error: ${msg.data}`);
                setIsThinking(false);
            }
    
            if (runContainer) {
                history[history.length - 1] = runContainer;
            }
            newTasks[taskIndex] = { ...task, history };
            return newTasks;
        });
    };
    

    const activeTask = tasks.find(t => t.id === activeTaskId);
    useEffect(() => { scrollToBottom(); }, [activeTask]);

    const handleSendMessage = (e) => {
        e.preventDefault();
        const message = inputValue.trim();
        const isReady = activeTask && connectionStatus === 'Connected' && !isThinking;
        if (!message || !isReady) return;
        setIsThinking(true);
        setTasks(prev => prev.map(t => t.id === activeTaskId ? { ...t, history: [...t.history, { type: 'prompt', content: message }] } : t));
        ws.current.send(JSON.stringify({ type: 'run_agent', prompt: message, llm_config: selectedModels, task_id: activeTaskId }));
        setInputValue("");
    };

    const handleSendFeedback = (feedback) => {
        if (!activeTask || connectionStatus !== 'Connected') return;
        setIsThinking(true);
        ws.current.send(JSON.stringify({ type: 'user_plan_feedback', feedback, task_id: activeTaskId }));
    };
    
    const lastRun = activeTask?.history[activeTask.history.length - 1];
    const isAwaitingFeedback = lastRun?.type === 'run_container' && lastRun.children.some(c => c.type === 'plan_approval');

    return (
        <div class="flex h-screen w-screen p-4 gap-4 bg-gray-900 text-gray-200 font-sans">
            {!isLeftSidebarVisible && <ToggleButton isVisible={isLeftSidebarVisible} onToggle={() => setIsLeftSidebarVisible(true)} side="left" />}
            {isLeftSidebarVisible && (
                <div class="h-full w-1/4 min-w-[300px] bg-gray-800/50 rounded-lg border border-gray-700/50 shadow-2xl flex flex-col">
                    <div class="flex justify-between items-center p-6 pb-4 border-b border-gray-700"> <h2 class="text-xl font-bold text-white">Tasks</h2> <div class="flex items-center gap-2"> <button onClick={createNewTask} class="p-1.5 rounded-md hover:bg-gray-700"><PlusCircleIcon class="h-5 w-5" /></button> <button onClick={() => setIsLeftSidebarVisible(false)} class="p-1.5 rounded-md hover:bg-gray-700"><ChevronsLeftIcon class="h-4 w-4" /></button> </div> </div>
                    <div class="flex flex-col flex-grow p-6 pt-4 min-h-0"> <div class="flex-grow overflow-y-auto pr-2"> {tasks.length > 0 ? <ul>{tasks.map(t => <TaskItem key={t.id} task={t} isActive={activeTaskId === t.id} onSelect={selectTask} onRename={handleRenameTask} onDelete={handleDeleteTask} />)}</ul> : <p class="text-gray-400 text-center mt-4">No tasks yet.</p>} </div> <SettingsPanel models={availableModels} selectedModels={selectedModels} onModelChange={handleModelChange} /> </div>
                </div>
            )}
            <div class="flex-1 flex flex-col h-full bg-gray-800/50 rounded-lg border border-gray-700/50 shadow-2xl min-w-0">
                <div class="flex items-center justify-between p-6 border-b border-gray-700"> <h1 class="text-2xl font-bold text-white">ResearchAgent</h1> <div class="flex items-center gap-2"> <span class="relative flex h-3 w-3"><span class={`absolute inline-flex h-full w-full rounded-full opacity-75 ${connectionStatus === 'Connected' ? 'animate-ping bg-green-400' : ''}`}></span><span class={`relative inline-flex rounded-full h-3 w-3 ${connectionStatus === 'Connected' ? 'bg-green-500' : 'bg-red-500'}`}></span></span><span class="text-sm text-gray-400">{connectionStatus}</span></div> </div>
                <div class="flex-1 overflow-y-auto p-6">
                    {activeTask?.history.map((item, index) => {
                        if (item.type === 'prompt') return <PromptCard key={index} content={item.content} />;
                        if (item.type === 'run_container') {
                            return (
                                <div key={index} class="relative mt-6 pl-8">
                                    <div class="absolute top-5 left-4 h-[calc(100%-2.5rem)] w-0.5 bg-gray-700/50" />
                                    <div class="space-y-4">
                                        {item.children.map((child, childIndex) => (
                                            <div key={childIndex} class="relative">
                                                <div class="absolute top-6 -left-4 h-0.5 w-4 bg-gray-700/50" />
                                                {child.type === 'architect_plan' && <ArchitectCard plan={child} />}
                                                {child.type === 'plan_approval' && <PlanApprovalCard plan={child.plan} onSendFeedback={handleSendFeedback} />}
                                                {child.type === 'execution_plan' && <SiteForemanCard plan={child} />}
                                                {child.type === 'final_answer' && <FinalAnswerCard answer={child.content} />}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            );
                        }
                        return null;
                    })}
                   {isThinking && (<div class="flex items-center gap-4 p-4 mt-6 pl-8"> <LoaderIcon class="h-5 w-5 text-yellow-400" /> <p class="text-gray-300 font-medium">Agent is thinking...</p> </div> )}
                   <div ref={messagesEndRef} />
                </div>
                <div class="p-6 border-t border-gray-700">
                    <form onSubmit={handleSendMessage} class="flex gap-3">
                        <textarea value={inputValue} onInput={e => setInputValue(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) handleSendMessage(e); }} class="flex-1 p-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-blue-500 outline-none resize-none" placeholder={activeTaskId ? (isAwaitingFeedback ? "Provide feedback on the plan above..." : "Send a message...") : "Please select or create a task."} rows="2" disabled={!activeTaskId || isThinking || isAwaitingFeedback} />
                        <button type="submit" class="px-4 py-2 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 disabled:bg-gray-500 flex items-center justify-center" disabled={!activeTaskId || isThinking || isAwaitingFeedback}> <SendIcon class="h-5 w-5"/> </button>
                    </form>
                </div>
            </div>
            {!isRightSidebarVisible && <ToggleButton isVisible={isRightSidebarVisible} onToggle={() => setIsRightSidebarVisible(true)} side="right" />}
            {isRightSidebarVisible && (
                <div class="h-full w-1/4 min-w-[300px] bg-gray-800/50 rounded-lg border border-gray-700/50 shadow-2xl flex flex-col">
                    <div class="flex justify-between items-center p-6 pb-4 border-b border-gray-700"> <h2 class="text-xl font-bold text-white">Workspace</h2> <button onClick={() => setIsRightSidebarVisible(false)} class="p-1.5 rounded-md hover:bg-gray-700"><ChevronsRightIcon class="h-4 w-4" /></button> </div>
                    <div class="flex flex-col flex-grow min-h-0 px-6 pb-6 pt-4"> {selectedFile ? ( <div class="flex flex-col h-full"> <div class="flex items-center justify-between gap-2 pb-2 mb-2 border-b border-gray-700"> <div class="flex items-center gap-2 min-w-0"> <button onClick={() => setSelectedFile(null)} class="p-1.5 rounded-md hover:bg-gray-700"><ArrowLeftIcon class="h-4 w-4" /></button> <span class="font-mono text-sm text-white truncate">{selectedFile}</span> </div> <CopyButton textToCopy={fileContent} /> </div> <div class="flex-grow bg-gray-900/50 rounded-md overflow-auto p-4"><pre class="h-full w-full text-sm text-gray-300 font-mono">{isFileLoading ? 'Loading...' : <code>{fileContent.trim()}</code>}</pre></div> </div> ) : ( <div class="flex flex-col flex-grow min-h-0"> <div class="flex justify-between items-center mb-2"> <div class="text-xs text-gray-500 truncate">{activeTaskId || 'No active workspace'}</div> <input type="file" ref={fileInputRef} onChange={handleFileUpload} class="hidden" /> <button onClick={() => fileInputRef.current?.click()} disabled={!activeTaskId || workspaceLoading} class="p-1.5 rounded-md hover:bg-gray-700 disabled:opacity-50"><UploadCloudIcon class="h-4 w-4" /></button> </div> <div class="flex-grow bg-gray-900/50 rounded-md p-4 text-sm text-gray-400 font-mono overflow-y-auto">{workspaceLoading ? <p>Loading...</p> : workspaceError ? <p class="text-red-400">Error: {workspaceError}</p> : workspaceFiles.length === 0 ? <p>// Workspace empty.</p> : ( <ul>{workspaceFiles.map(f => ( <li key={f} onClick={() => fetchFileContent(f)} class="flex items-center gap-2 mb-1 hover:text-white cursor-pointer"><FileIcon class="h-4 w-4 text-gray-500" />{f}</li> ))}</ul> )}</div> </div> )} </div>
                </div>
            )}
        </div>
    );
}