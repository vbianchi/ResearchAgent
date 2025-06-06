# Research Agent Graph Definition

Copy the code block below and paste it into a Mermaid.js viewer like [mermaid.live](https://mermaid.live) or a supporting Markdown editor to see the visual graph.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	intent_classifier(intent_classifier)
	direct_qa(direct_qa)
	planner(planner)
	controller(controller)
	executor(executor)
	step_evaluator(step_evaluator)
	overall_evaluator(overall_evaluator)
	increment_retry_count(increment_retry_count)
	advance_to_next_step(advance_to_next_step)
	__end__([<p>__end__</p>]):::last
	__start__ --> intent_classifier;
	advance_to_next_step --> controller;
	controller --> executor;
	direct_qa --> overall_evaluator;
	executor --> step_evaluator;
	increment_retry_count --> controller;
	intent_classifier -.-> direct_qa;
	intent_classifier -.-> overall_evaluator;
	intent_classifier -.-> planner;
	planner -.-> controller;
	planner -.-> overall_evaluator;
	step_evaluator -. &nbsp;next_step&nbsp; .-> advance_to_next_step;
	step_evaluator -. &nbsp;retry_step&nbsp; .-> increment_retry_count;
	step_evaluator -. &nbsp;evaluate_overall_plan&nbsp; .-> overall_evaluator;
	overall_evaluator --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```
