# LangGraph Refactoring Documentation

## Overview

This document describes the refactoring of the MedQG codebase to use LangGraph for workflow orchestration. The refactoring replaces manual while-loop orchestration with a stateful, graph-based approach that provides better visualization, state management, and maintainability.

## What is LangGraph?

LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of LangChain. It extends LangChain Expression Language with:

- **State Management**: TypedDict-based state that flows through graph nodes
- **Cyclic Graphs**: Support for loops and conditional branching
- **Persistence**: Built-in state persistence and checkpointing
- **Visualization**: Graph structure visualization capabilities
- **Human-in-the-loop**: Support for interrupting and resuming workflows

## Architecture Changes

### Before: Manual Orchestration

The original implementation used manual while-loops and procedural orchestration:

```python
# Original approach in run.py
n_attempts = 0
while n_attempts < max_attempts and check_stop(...):
    if n_attempts == 0:
        context, question, answer, distractors = task_init_lgc(...)
    else:
        context, question, answer, distractors = task_iterate(...)

    attempted_answer, reasoning = task_answer(...)
    feedback = task_feedback_lgc(...)
    n_attempts += 1
```

**Challenges:**
- Hard to visualize workflow
- State management scattered across variables
- Difficult to add persistence or human-in-the-loop
- No built-in support for parallel execution
- Tight coupling between orchestration and business logic

### After: LangGraph State Graphs

The refactored implementation uses LangGraph's StateGraph:

```python
# New approach in question_graph.py
workflow = StateGraph(QuestionGenerationState)
workflow.add_node("generate_question", self.generate_question_node)
workflow.add_node("generate_answer", self.generate_answer_node)
workflow.add_node("evaluate_feedback", self.evaluate_feedback_node)

workflow.add_edge("generate_question", "generate_answer")
workflow.add_edge("generate_answer", "evaluate_feedback")
workflow.add_conditional_edges(
    "evaluate_feedback",
    self.should_continue_decision,
    {"continue": "generate_question", "end": END}
)

graph = workflow.compile()
```

**Benefits:**
- Clear visual representation of workflow
- Centralized state management with TypedDict
- Built-in support for persistence and checkpointing
- Easy to add human-in-the-loop or monitoring
- Clean separation of orchestration and business logic
- Easier testing of individual nodes

## New File Structure

### Core LangGraph Files

```
src/usmle/
├── graph_state.py           # State schema definition
├── question_graph.py         # Question generation graph
├── pipeline_graph.py         # Complete end-to-end pipeline
└── run.py                    # Updated entry points (backward compatible)
```

### State Schema (`graph_state.py`)

Defines the state that flows through the graph:

```python
class QuestionGenerationState(TypedDict):
    # Input parameters
    clinical_note: str
    keypoint: str
    topic: str
    max_attempts: int

    # Generated components
    context: Optional[str]
    question: Optional[str]
    correct_answer: Optional[str]
    distractor_options: Optional[str]

    # Feedback and scores
    context_feedback: Optional[Dict[str, Any]]
    context_score: str
    # ... more feedback fields

    # Output tracking
    content_to_fb_ret: List[Dict[str, Any]]
    n_attempts: int
```

### Question Generation Graph (`question_graph.py`)

Implements the iterative question generation workflow:

**Nodes:**
1. `generate_question_node`: Initial generation or iteration based on feedback
2. `generate_answer_node`: Generate attempted answer and reasoning
3. `evaluate_feedback_node`: Evaluate all components and generate feedback

**Conditional Logic:**
- `should_continue_decision`: Checks if quality threshold met or max attempts reached

**Graph Flow:**
```
START → generate_question → generate_answer → evaluate_feedback
                ↑                                    ↓
                |← (if continue) ← should_continue? → END (if stop)
```

### Complete Pipeline Graph (`pipeline_graph.py`)

Orchestrates the full workflow: topics → keypoints → questions

**Nodes:**
1. `generate_topics_node`: Generate 5 relevant USMLE topics
2. `generate_keypoint_node`: Extract keypoint for current topic
3. `generate_question_node`: Generate question with refinement

**Graph Flow:**
```
START → generate_topics → generate_keypoint → generate_question
                              ↑                      ↓
                              |← (more topics?) ← continue?
                              |                      ↓
                              END ← (no more topics)
```

## Usage

### Basic Question Generation

```python
from src.usmle.question_graph import QuestionGenerationGraph

# Create graph
graph = QuestionGenerationGraph(engine="gpt-4")

# Generate question
results = graph.generate(
    clinical_note="An 84-year-old female with hypertension...",
    keypoint="pathophysiology of sepsis",
    topic="underlying processes/pathways",
    max_attempts=4
)

# Results contain all iterations with feedback
for iteration in results:
    print(f"Iteration {iteration['n_attempts']}")
    print(f"Question: {iteration['question']}")
    print(f"Scores: {iteration['question_score']}")
```

### Backward Compatible API

The refactored code maintains backward compatibility:

```python
from src.usmle.run import autofb_usmleqgen

# Same interface as before
results = autofb_usmleqgen(
    clinical_note="...",
    keypoint="...",
    topic="...",
    max_attempts=4
)
```

### Complete Pipeline

```python
from src.usmle.pipeline_graph import MedQGPipeline

# Create pipeline
pipeline = MedQGPipeline(engine="gpt-4")

# Run complete workflow
questions = pipeline.run(
    clinical_note="An 84-year-old female...",
    max_attempts=4
)

# Questions for all generated topics
for q in questions:
    print(f"Topic: {q['topic']}")
    print(f"Keypoint: {q['keypoint']}")
    print(f"Questions: {len(q['content_to_fb'])} iterations")
```

### Batch Processing

The batch processing interface remains unchanged:

```bash
# Question generation
python src/usmle/run.py batch-iter data/inputs/human_cn_t_kp_1-35.jsonl

# Topic generation
python src/usmle/topic_gen.py batch-iter <notes> data/prompt/usmle/topic_fewshot.jsonl

# Keypoint generation
python src/usmle/keypoint_gen.py batch-iter <notes> data/prompt/usmle/keypoint_fewshot.jsonl
```

## Advanced Features

### Graph Visualization

LangGraph graphs can be visualized using Mermaid or other tools:

```python
from IPython.display import Image, display

graph = QuestionGenerationGraph()
display(Image(graph.graph.get_graph().draw_mermaid_png()))
```

### State Inspection

Access state at any point:

```python
graph = QuestionGenerationGraph()

# Run with state tracking
for state in graph.graph.stream(initial_state):
    print(f"Current node: {state}")
    print(f"Attempts: {state['n_attempts']}")
    print(f"Scores: {state['question_score']}")
```

### Persistence (Future Enhancement)

LangGraph supports checkpointing for resumable workflows:

```python
from langgraph.checkpoint import MemorySaver

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# Run with checkpointing
config = {"configurable": {"thread_id": "1"}}
result = graph.invoke(initial_state, config=config)

# Resume from checkpoint later
result = graph.invoke(None, config=config)  # Resumes from last checkpoint
```

### Human-in-the-Loop (Future Enhancement)

Add breakpoints for human review:

```python
workflow.add_conditional_edges(
    "evaluate_feedback",
    self.should_continue_decision,
    {
        "continue": "generate_question",
        "review": "human_review",  # Pause for human input
        "end": END
    }
)
```

## Migration Guide

### For Existing Code

1. **No changes required** for batch processing scripts - they use the same API
2. **Import changes** if you're using the library programmatically:

```python
# Old
from src.usmle.run import autofb_usmleqgen

# New (both work, new is more explicit)
from src.usmle.question_graph import QuestionGenerationGraph
graph = QuestionGenerationGraph()
result = graph.generate(...)
```

### For New Development

1. **Use graph-based APIs** for new features
2. **Add new nodes** to extend functionality
3. **Use state for data flow** instead of function parameters

Example - Adding a new validation node:

```python
def validate_question_node(state: QuestionGenerationState) -> Dict[str, Any]:
    """Custom validation logic"""
    if not state['question'].endswith('?'):
        return {"question": state['question'] + '?'}
    return {}

# Add to graph
workflow.add_node("validate", validate_question_node)
workflow.add_edge("generate_question", "validate")
workflow.add_edge("validate", "generate_answer")
```

## Testing

### Unit Testing Nodes

Test individual nodes without running the full graph:

```python
def test_generate_question_node():
    graph = QuestionGenerationGraph()

    # Mock state
    state = {
        "clinical_note": "Test note",
        "keypoint": "Test keypoint",
        "topic": "Test topic",
        "n_attempts": 0,
        # ... other fields
    }

    # Test node directly
    result = graph.generate_question_node(state)

    assert result['context'] is not None
    assert result['question'] is not None
```

### Integration Testing

Test the full graph:

```python
def test_question_generation_graph():
    graph = QuestionGenerationGraph(engine="gpt-3.5-turbo")

    results = graph.generate(
        clinical_note="Test clinical note...",
        keypoint="test keypoint",
        topic="test topic",
        max_attempts=2
    )

    assert len(results) > 0
    assert len(results) <= 2
    assert all('question' in r for r in results)
```

## Performance Considerations

### Same Performance as Before

The LangGraph refactoring:
- ✅ Uses the same LLM calls as the original implementation
- ✅ Has minimal overhead (state dictionary management)
- ✅ Maintains the same retry and error handling logic
- ✅ Preserves the same caching behavior

### Potential Optimizations

LangGraph enables future optimizations:

1. **Parallel Execution**: Run multiple topics in parallel
2. **Streaming**: Stream results as they're generated
3. **Caching**: Cache intermediate states
4. **Distributed Execution**: Run nodes on different machines

## Troubleshooting

### Import Errors

If you see import errors for `langgraph`:

```bash
pip install -r requirements.txt
# or
pip install langgraph>=0.0.20
```

### State Type Errors

Ensure all state updates return dictionaries with correct types:

```python
# Good
return {"n_attempts": state['n_attempts'] + 1}

# Bad
return {"n_attempts": "1"}  # Wrong type
```

### Graph Execution Errors

Enable debug logging to see graph execution:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

The LangGraph refactoring enables several future enhancements:

1. **Multi-Agent Workflows**: Multiple LLMs with different roles
2. **Human-in-the-Loop**: Pause for human review/editing
3. **Persistent Sessions**: Save and resume long-running workflows
4. **Parallel Topic Processing**: Generate questions for multiple topics simultaneously
5. **Real-time Monitoring**: Track progress and metrics in real-time
6. **A/B Testing**: Compare different generation strategies
7. **Adaptive Iteration**: Dynamic max_attempts based on progress
8. **Quality Gates**: Stricter quality checks at different stages

## References

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [LangGraph Tutorials](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [State Management in LangGraph](https://python.langchain.com/docs/langgraph#stategraph)
- [Conditional Edges](https://python.langchain.com/docs/langgraph#conditional-edges)

## Summary

The LangGraph refactoring modernizes the MedQG codebase while maintaining full backward compatibility. The new graph-based architecture provides:

- ✅ Better code organization and maintainability
- ✅ Clear workflow visualization
- ✅ Easier testing and debugging
- ✅ Foundation for advanced features (human-in-the-loop, persistence, etc.)
- ✅ Same performance and behavior as the original implementation

All existing scripts and workflows continue to work without modification.
