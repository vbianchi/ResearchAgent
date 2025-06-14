import { h } from 'preact';
import { useState } from 'preact/hooks';
import { ArchitectIcon, CheckCircleIcon, ChevronDownIcon, CircleDotIcon, EditorIcon, ForemanIcon, HandymanIcon, LibrarianIcon, LoaderIcon, SupervisorIcon, WorkerIcon, XCircleIcon, MessageSquareQuoteIcon, SendIcon, CheckIcon, XIcon } from './Icons';
import { CopyButton } from './Common';

// --- Re-usable Card Wrapper ---
const AgentResponseCard = ({ icon, title, children, cardClass = "bg-gray-800/50 border-gray-700/50" }) => (
    <div class={`p-4 rounded-lg shadow-md ${cardClass} border`}>
        <h3 class="font-bold text-sm text-gray-300 mb-3 capitalize flex items-center gap-2">{icon}{title}</h3>
        <div class="pl-1">{children}</div>
    </div>
);

// --- Step-level Cards for Complex View ---
const StepCard = ({ step }) => {
    const [isExpanded, setIsExpanded] = useState(true);
    const getStatusIcon = () => {
        switch (step.status) {
            case 'in-progress': return <LoaderIcon class="h-5 w-5 text-yellow-400" />;
            case 'completed': return <CheckCircleIcon class="h-5 w-5 text-green-400" />;
            case 'failure': return <XCircleIcon class="h-5 w-5 text-red-500" />;
            case 'pending': default: return <CircleDotIcon class="h-5 w-5 text-gray-500" />;
        }
    };
    return (
        <div class="bg-gray-900/50 rounded-lg border border-gray-700/50 mb-2 last:mb-0 transition-all">
             <div class="flex items-center gap-4 p-4 cursor-pointer" onClick={() => setIsExpanded(!isExpanded)}>
                {getStatusIcon()}
                <p class="text-gray-200 font-medium flex-1">{step.instruction}</p>
                <ChevronDownIcon class={`h-5 w-5 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
             </div>
             {isExpanded && (step.status === 'completed' || step.status === 'failure') && step.toolCall && (
                <div class="p-4 pt-0">
                    <div class="ml-9 pl-4 border-l-2 border-gray-700 space-y-4">
                        <div>
                           <div class="flex items-center gap-2 text-sm font-semibold text-gray-400"> <WorkerIcon class="h-4 w-4" /> <span>The Worker: Execute Step</span> </div>
                           <pre class="text-xs text-cyan-300 overflow-x-auto p-2 mt-1 ml-7 bg-black/20 rounded-md font-mono relative"> <CopyButton textToCopy={JSON.stringify(step.toolCall, null, 2)} className="absolute top-1 right-1" /> <code>{JSON.stringify(step.toolCall, null, 2)}</code> </pre>
                        </div>
                        <div>
                           <div class="flex items-center gap-2 text-sm font-semibold text-gray-400"> <SupervisorIcon class="h-4 w-4" /> <span>The Project Supervisor: Evaluation</span> </div>
                           <pre class="text-xs text-gray-300 mt-1 ml-7 whitespace-pre-wrap font-mono relative bg-black/20 p-2 rounded-md"> <CopyButton textToCopy={step.evaluation?.reasoning || 'No evaluation.'} className="absolute top-1 right-1" /> {step.evaluation?.reasoning || 'No evaluation provided.'} </pre>
                        </div>
                    </div>
                </div>
             )}
        </div>
    );
};

// --- Top-level Agent View Cards ---
export const ArchitectCard = ({ plan }) => (
    <AgentResponseCard icon={<ArchitectIcon class="h-5 w-5" />} title="The Chief Architect">
        <h4 class="text-sm font-bold text-gray-400 mb-2">Proposed Plan</h4>
        <ul class="list-decimal list-inside text-gray-300 space-y-1">
            {plan.steps.map(step => <li key={step.step_id}>{step.instruction}</li>)}
        </ul>
    </AgentResponseCard>
);

export const SiteForemanCard = ({ plan }) => (
    <AgentResponseCard icon={<ForemanIcon class="h-5 w-5" />} title="The Site Foreman">
        <h4 class="text-sm font-bold text-gray-400 mb-2">Execution Log</h4>
        {plan.steps.map(step => <StepCard key={step.step_id} step={step} />)}
    </AgentResponseCard>
);

// --- NEW/UPDATED: Card for plan approval ---
export const PlanApprovalCard = ({ plan, onSendFeedback }) => {
    const [feedbackInput, setFeedbackInput] = useState("");
    const handleSend = (feedbackType) => {
        onSendFeedback(feedbackType === 'feedback' ? feedbackInput : feedbackType);
        setFeedbackInput("");
    };

    return (
        <AgentResponseCard icon={<MessageSquareQuoteIcon class="h-5 w-5" />} title="Plan Approval Required" cardClass="bg-cyan-900/50 border-cyan-700/50">
            <div class="bg-gray-900/50 rounded-lg p-3 mb-3">
                <p class="text-sm font-semibold text-gray-300 mb-2">The Architect has proposed the following plan:</p>
                <ul class="list-decimal list-inside text-gray-300 space-y-1 text-sm pl-2">
                    {plan.map(step => <li key={step.step_id}>{step.instruction}</li>)}
                </ul>
            </div>
            <div class="flex flex-col gap-2">
                 <div class="flex items-center gap-2">
                    <input type="text" value={feedbackInput} onInput={(e) => setFeedbackInput(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && feedbackInput) handleSend('feedback'); }} placeholder="Request changes or use buttons below..." class="flex-1 p-2 bg-gray-800/60 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-cyan-500 focus:outline-none text-sm"/>
                    <button onClick={() => handleSend('feedback')} class="px-3 py-2 bg-cyan-600 text-white font-semibold rounded-md hover:bg-cyan-700 disabled:bg-gray-500 text-sm flex items-center gap-2" disabled={!feedbackInput}> <SendIcon class="h-4 w-4"/> Request Changes </button>
                </div>
                <div class="flex items-center justify-end gap-2 mt-1">
                    <button onClick={() => handleSend('approve')} class="px-3 py-1 bg-green-600 text-white font-semibold rounded-md hover:bg-green-700 text-sm flex items-center gap-2"> <CheckIcon class="h-4 w-4"/> Approve </button>
                    <button onClick={() => handleSend('abort')} class="px-3 py-1 bg-red-600 text-white font-semibold rounded-md hover:bg-red-700 text-sm flex items-center gap-2"> <XIcon class="h-4 w-4"/> Abort </button>
                </div>
            </div>
        </AgentResponseCard>
    );
};


// --- Final Output Cards from the Editor ---
export const FinalAnswerCard = ({ answer }) => {
    const parsedHtml = window.marked ? window.marked.parse(answer, { breaks: true, gfm: true }) : answer.replace(/\n/g, '<br />');
    return (
        <AgentResponseCard icon={<EditorIcon class="h-5 w-5" />} title="The Editor">
            <div class="prose prose-sm prose-invert max-w-none text-gray-200" dangerouslySetInnerHTML={{ __html: parsedHtml }}></div>
        </AgentResponseCard>
    );
};
