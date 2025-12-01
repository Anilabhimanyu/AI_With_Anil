# LangGraph - Quick Reference Guide

## Overview
**LangGraph** is a library for building stateful, multi-actor applications with LLMs using graph-based workflows. Built on top of LangChain, it enables cyclic computation graphs for complex agent behaviors.

---

## LangChain vs LangGraph

| Feature | LangChain | LangGraph |
|---------|-----------|-----------|
| **Architecture** | Linear chains | Cyclic graphs (nodes + edges) |
| **Control Flow** | Sequential or branching | Arbitrary loops and cycles |
| **State Management** | Limited | Built-in state persistence |
| **Use Case** | Simple workflows | Complex, multi-step agent workflows |
| **Cycles** | Not supported | Native support |
| **Human-in-loop** | Manual implementation | Built-in checkpointing |
| **Best For** | Predictable pipelines | Agentic systems with loops |

---

## Core Concepts

| Concept | Description | Purpose |
|---------|-------------|---------|
| **State** | Shared data structure across nodes | Maintains context throughout execution |
| **Nodes** | Functions that process state | Represent actions (LLM calls, tools, logic) |
| **Edges** | Connections between nodes | Define execution flow |
| **Conditional Edges** | Dynamic routing based on state | Decision-making in workflow |
| **Checkpointing** | Automatic state persistence | Enable pause/resume, human-in-loop |
| **Graph** | Complete workflow structure | Orchestrates entire process |

---

## Key Components

### 1. State Management

| State Type | Description | Use Case |
|------------|-------------|----------|
| **TypedDict** | Structured state schema | Type-safe state definitions |
| **Reducers** | Functions to merge state updates | Handling list appends, counters |
| **Channels** | State keys with update logic | Managing specific state fields |

### 2. Node Types

| Type | Function | Example |
|------|----------|---------|
| **Action Node** | Performs computation | Call LLM, use tool, process data |
| **Router Node** | Makes decisions | Determine next step based on conditions |
| **Human Node** | Waits for input | Approval workflows, clarifications |
| **Tool Node** | Executes tools | Search, calculator, API calls |

### 3. Edge Types

| Edge Type | Behavior | When to Use |
|-----------|----------|-------------|
| **Normal Edge** | Fixed connection | Always go from A to B |
| **Conditional Edge** | Dynamic routing | Route based on state/output |
| **Entry Point** | Starting node | Define where graph begins |
| **Finish** | End execution | Terminal state reached |

---

## Graph Patterns

### 1. ReAct Agent Pattern

```
START → Agent → [Decision] → Tool → Agent → [Decision] → END
                    ↓                  ↑
                   END ←---------------┘
```

| Step | Node | Action |
|------|------|--------|
| 1 | Agent | Decide action based on state |
| 2 | Tool | Execute tool if needed |
| 3 | Agent | Process tool result |
| 4 | Decision | Continue or finish |

### 2. Human-in-the-Loop Pattern

```
START → Process → Review (Human) → Approve/Reject → END
                        ↓
                    Revise ← Reject
```

### 3. Multi-Agent Collaboration

```
START → Researcher → Writer → Reviewer → Editor → END
           ↓          ↓         ↓          ↓
        [Tools]   [Tools]   [Human]   [Tools]
```

---

## StateGraph Example Structure

### Basic Graph Setup

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Define state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # Reducer for appending
    next_action: str
    iteration: int

# Create graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tool", tool_node)

# Add edges
workflow.add_edge("tool", "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,  # Function returning next node
    {
        "continue": "tool",
        "end": END
    }
)

# Set entry point
workflow.set_entry_point("agent")

# Compile
app = workflow.compile()
```

---

## Checkpointing & Persistence

| Feature | Purpose | Benefit |
|---------|---------|---------|
| **Memory Checkpointer** | In-memory state storage | Testing, development |
| **SQLite Checkpointer** | Persistent local storage | Production single-machine |
| **Postgres Checkpointer** | Distributed persistence | Production multi-machine |
| **Thread ID** | Conversation identifier | Multi-user support |
| **State Snapshots** | Point-in-time state | Time travel, debugging |

### Checkpoint Usage

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# With checkpointing
memory = SqliteSaver.from_conn_string(":memory:")
app = workflow.compile(checkpointer=memory)

# Resume from checkpoint
config = {"configurable": {"thread_id": "conversation-1"}}
result = app.invoke(input, config)
```

---

## Advanced Features

### 1. Streaming

| Stream Type | What It Returns | Use Case |
|-------------|-----------------|----------|
| **values()** | Complete state after each node | See full state evolution |
| **updates()** | Only state changes | Efficient updates |
| **events()** | All internal events | Detailed debugging |

### 2. Subgraphs

| Concept | Description | Benefit |
|---------|-------------|---------|
| **Nested Graphs** | Graphs within nodes | Modular workflows |
| **State Mapping** | Map parent/child state | Isolation and reusability |
| **Hierarchical Design** | Multi-level orchestration | Complex systems |

### 3. Human-in-the-Loop

```python
# Add interrupt before node
workflow.add_node("human_review", human_node)
app = workflow.compile(checkpointer=memory, interrupt_before=["human_review"])

# Execute until interrupt
app.invoke(input, config)

# Resume after human input
app.invoke(None, config)  # Continues from checkpoint
```

---

## Common Use Cases

| Use Case | Pattern | Key Features |
|----------|---------|--------------|
| **Research Assistant** | Multi-agent with tools | Search → Analyze → Synthesize |
| **Code Generation** | Iterative refinement | Generate → Test → Fix → Repeat |
| **Content Moderation** | Classification + routing | Classify → Route to specialist |
| **Customer Support** | Stateful conversation | Context retention, escalation |
| **Data Processing Pipeline** | Sequential + conditional | Validate → Transform → Load |
| **Approval Workflows** | Human-in-loop | Request → Review → Approve/Reject |

---

## Interview Questions & Answers

### Q: "When would you use LangGraph over LangChain?"

**Key Points:**
- Need cycles/loops (retries, iterative refinement)
- Complex agent behavior with multiple decision points
- Human-in-the-loop workflows
- Need state persistence across sessions
- Multi-agent systems with coordination
- Fine-grained control over execution flow

### Q: "How does LangGraph handle state?"

**Answer:**
- State is a TypedDict passed between nodes
- Each node returns state updates
- Reducers define how updates merge (append, overwrite, custom)
- Checkpointers persist state at each step
- Enables time-travel debugging and resumption

### Q: "Explain conditional edges with an example"

**Answer:**
Agent decides next action based on reasoning:
- If needs information → route to "search" tool
- If needs calculation → route to "calculator" tool
- If has answer → route to END
- Conditional edge function examines state and returns next node name

---

## Architecture Comparison

### Simple Chain (LangChain)
```
Input → LLM → Output
```

### Agent with Tools (LangChain)
```
Input → Agent → [Tool Call?] → Output
                    ↓
                  Tool → Agent
```

### ReAct Loop (LangGraph)
```
Input → Agent → Should Continue?
         ↑           ↓
       Tool ←── Yes (use tool)
                    ↓
                   No → Output
```

---

## Performance Considerations

| Aspect | Consideration | Recommendation |
|--------|---------------|----------------|
| **State Size** | Large state = slow | Keep state minimal, paginate results |
| **Checkpointing** | I/O overhead | Use for production, skip for speed tests |
| **Cycles** | Infinite loops risk | Always have exit conditions |
| **Tool Calls** | Latency per call | Batch when possible |
| **Memory** | Checkpoint storage | Clean up old threads |

---

## Error Handling

| Strategy | Implementation | Use Case |
|----------|----------------|----------|
| **Retry Logic** | Loop back to same node | Transient failures |
| **Fallback Routes** | Conditional edge to backup | Service unavailable |
| **Error Nodes** | Dedicated error handling | Logging, cleanup |
| **Max Iterations** | Counter in state | Prevent infinite loops |

---

## Best Practices

1. **Design State Schema First**: Clear TypedDict with proper types
2. **Use Reducers Wisely**: Prevent state bloat
3. **Add Logging**: Track state transitions
4. **Set Max Iterations**: Prevent runaway loops
5. **Test Conditional Logic**: Cover all edge cases
6. **Checkpoint in Production**: Enable resumption
7. **Keep Nodes Focused**: Single responsibility
8. **Visualize Graphs**: Use built-in graph visualization

---

## Code Example: Complete ReAct Agent

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

class State(TypedDict):
    messages: list
    iterations: int

def agent_node(state: State):
    # LLM decides: tool call or final answer
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "iterations": state["iterations"] + 1
    }

def tool_node(state: State):
    # Execute tool based on last message
    tool_result = execute_tool(state["messages"][-1])
    return {"messages": [tool_result]}

def should_continue(state: State) -> Literal["tool", "end"]:
    last_message = state["messages"][-1]
    if state["iterations"] >= 5:
        return "end"
    if requires_tool(last_message):
        return "tool"
    return "end"

# Build graph
workflow = StateGraph(State)
workflow.add_node("agent", agent_node)
workflow.add_node("tool", tool_node)
workflow.add_edge("tool", "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.set_entry_point("agent")

app = workflow.compile()
```

---

## Key Differences Summary

| Aspect | LangChain Chains | LangGraph |
|--------|------------------|-----------|
| **Mental Model** | Pipeline | State machine |
| **Execution** | One-way flow | Cyclic computation |
| **State** | Implicit passing | Explicit central state |
| **Debugging** | Linear traces | Graph visualization |
| **Complexity** | Simple to medium | Medium to complex |
| **Learning Curve** | Lower | Higher |

---

## Alternatives to LangGraph

### Direct Competitors

| Framework | Company | Key Strengths | Best For | Limitations |
|-----------|---------|---------------|----------|-------------|
| **CrewAI** | CrewAI | - Role-based agents<br>- Simple API<br>- Built-in collaboration | Multi-agent teams with defined roles | Less flexible than LangGraph |
| **AutoGen** | Microsoft | - Multi-agent conversations<br>- Code execution<br>- Group chat patterns | Research, coding assistants | Steeper learning curve |
| **Semantic Kernel** | Microsoft | - Enterprise focus<br>- .NET/Python/Java<br>- Plugin architecture | Microsoft ecosystem | Less community adoption |
| **Haystack** | deepset | - Production-ready pipelines<br>- RAG focused<br>- Component library | NLP pipelines, search | Not agent-focused |
| **DSPy** | Stanford | - Optimized prompts<br>- Compiler for LLMs<br>- Automatic tuning | Research, prompt optimization | Early stage, academic |

### Orchestration & Workflow Tools

| Tool | Focus | Key Feature | Use Case |
|------|-------|-------------|----------|
| **Prefect** | Data workflows | Observable pipelines | Data engineering + LLM |
| **Temporal** | Microservices | Durable execution | Enterprise LLM workflows |
| **Airflow** | ETL/Data | Scheduling, DAGs | Batch LLM processing |
| **n8n** | Low-code automation | Visual workflow builder | Business automation with AI |
| **Zapier/Make** | No-code | Integration focus | Simple AI integrations |

### Agent Frameworks

| Framework | Approach | Unique Feature | Target User |
|-----------|----------|----------------|-------------|
| **LangGraph** | Graph-based | Cyclic workflows, state persistence | Python developers, complex agents |
| **CrewAI** | Role-based | Agents with roles, hierarchical teams | Quick multi-agent prototypes |
| **AutoGen** | Conversation-driven | Agent chat, code execution | Researchers, coding agents |
| **AgentGPT** | Autonomous | Browser-based, fully autonomous | Non-technical users |
| **BabyAGI** | Task-driven | Task creation/prioritization | Experimentation, learning |
| **SuperAGI** | Infrastructure | Agent monitoring, memory | Production agent deployment |
| **ix** | Visual | GUI for agent design | Visual thinkers |

### Custom/Minimal Approaches

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Raw API + State Machine** | Direct LLM calls with custom logic | Full control, no bloat | High development effort |
| **LangChain Agents** | Built-in agent executors | Simpler than LangGraph | Limited cycles, less control |
| **Marvin** | Lightweight AI functions | Minimal, Pythonic | Limited orchestration |
| **Guidance** | Constrained generation | Structured outputs | Not for complex workflows |

---

## Framework Selection Guide

### Choose **LangGraph** when you need:
- Complex cyclic workflows (retries, loops)
- Fine-grained control over execution
- Built-in state persistence
- Human-in-the-loop workflows
- Time-travel debugging
- Graph visualization

### Choose **CrewAI** when you need:
- Quick multi-agent prototype
- Role-based collaboration (researcher, writer, editor)
- Simpler API than LangGraph
- Pre-built agent patterns

### Choose **AutoGen** when you need:
- Conversational multi-agent systems
- Code generation and execution
- Group chat patterns
- Microsoft ecosystem integration

### Choose **Custom Solution** when you need:
- Simple linear workflows
- Maximum performance
- Minimal dependencies
- Full control over every aspect

### Choose **Haystack** when you need:
- Production RAG systems
- Document search and QA
- Enterprise NLP pipelines
- Not primarily agentic workflows

---

## Detailed Comparison: Top 3 Agent Frameworks

### LangGraph vs CrewAI vs AutoGen

| Feature | LangGraph | CrewAI | AutoGen |
|---------|-----------|--------|---------|
| **Architecture** | State graph | Role-based crew | Conversational agents |
| **Complexity** | High | Low-Medium | Medium-High |
| **Learning Curve** | Steep | Gentle | Moderate |
| **Flexibility** | Very high | Medium | High |
| **State Management** | Built-in, persistent | Automatic | Manual |
| **Cycles/Loops** | Native support | Limited | Supported |
| **Human-in-Loop** | Built-in checkpointing | Manual | Built-in |
| **Multi-Agent** | Custom coordination | Pre-built patterns | Chat-based coordination |
| **Code Execution** | Via tools | Via tools | Native support |
| **Debugging** | Graph visualization | Logs | Conversation logs |
| **Production Ready** | Yes | Growing | Yes |
| **Community** | Large (LangChain) | Growing fast | Large (Microsoft) |
| **Best For** | Complex workflows | Quick prototypes | Research, coding |

### Code Comparison: Simple Agent Task

**LangGraph:**
```python
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tool", tool_node)
workflow.add_conditional_edges("agent", router)
app = workflow.compile()
```

**CrewAI:**
```python
researcher = Agent(role="Researcher", goal="Find info")
writer = Agent(role="Writer", goal="Write report")
crew = Crew(agents=[researcher, writer], tasks=[task1, task2])
result = crew.kickoff()
```

**AutoGen:**
```python
assistant = AssistantAgent("assistant")
user_proxy = UserProxyAgent("user")
user_proxy.initiate_chat(assistant, message="Task here")
```

---

## Emerging Trends & Future

| Trend | Frameworks | Impact |
|-------|-----------|--------|
| **Agent Orchestration** | LangGraph, AutoGen | Standard way to coordinate agents |
| **Memory Systems** | Mem0, Zep | Long-term agent memory |
| **Multi-Modal Agents** | GPT-4V, Gemini | Vision + text agents |
| **Agent Marketplaces** | GPT Store, HuggingFace | Pre-built agent templates |
| **Observability** | LangSmith, Helicone | Production monitoring |
| **Security** | AgentOps, Invariant | Agent safety and guardrails |

---

## Resources & Tools

- **Graph Visualization**: Built-in Mermaid diagram generation
- **Debugging**: State inspection at each checkpoint
- **Testing**: Unit test individual nodes
- **Monitoring**: Track iterations, tool usage
- **LangSmith**: Production monitoring and tracing