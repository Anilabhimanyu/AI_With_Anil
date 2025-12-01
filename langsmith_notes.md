# LangSmith - Quick Reference Guide

## Overview
**LangSmith** is a developer platform for debugging, testing, evaluating, and monitoring LLM applications. Think of it as the "DevOps for LLMs" - providing observability and quality assurance for production AI systems.

**Key Value Proposition**: Bridge the gap between prototype and production by providing visibility into LLM application behavior.

---

## Core Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **Tracing** | Debug and visualize runs | - Full execution traces<br>- Token usage tracking<br>- Latency metrics<br>- Nested spans |
| **Datasets** | Test data management | - Input/output pairs<br>- Versioning<br>- Import/export<br>- Reusability |
| **Evaluation** | Assess performance | - Custom evaluators<br>- LLM-as-judge<br>- Metrics tracking<br>- A/B testing |
| **Monitoring** | Production observability | - Real-time dashboards<br>- Alerts<br>- Usage analytics<br>- Error tracking |
| **Prompts Hub** | Version control for prompts | - Prompt versioning<br>- Collaboration<br>- Rollback capability<br>- A/B testing |
| **Annotations** | Human feedback | - Label runs<br>- Collect corrections<br>- Build training data |

---

## LangSmith vs Alternatives

| Feature | LangSmith | Weights & Biases | MLflow | Helicone | Custom Logging |
|---------|-----------|------------------|---------|----------|----------------|
| **LLM-Specific** | Yes | Partial | No | Yes | No |
| **Tracing** | Built-in | Manual | Manual | Basic | Custom |
| **Evaluations** | Native | Yes | Yes | Limited | Custom |
| **Prompt Management** | Yes | No | No | No | No |
| **Integration** | LangChain native | Any framework | Any framework | API proxy | Any |
| **Pricing** | Usage-based | Usage-based | Free/Self-hosted | Usage-based | Infrastructure cost |
| **Setup Complexity** | Low | Medium | Medium | Low | High |
| **Best For** | LangChain apps | ML experiments | Traditional ML | API monitoring | Full control |

---

## Key Capabilities

### 1. Tracing & Debugging

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Run Tree** | Hierarchical trace visualization | See exact execution path |
| **Token Tracking** | Count tokens per call | Cost optimization |
| **Latency Metrics** | Time per component | Performance bottlenecks |
| **Input/Output Inspection** | See all intermediate data | Debug failures |
| **Error Tracking** | Capture exceptions | Identify failure patterns |
| **Metadata Tags** | Custom labels | Filter and organize runs |

**Example Trace Structure:**
```
Chain Run (5.2s, 1500 tokens, $0.045)
├─ LLM Call: Planning (2.1s, 800 tokens)
├─ Tool: Web Search (2.0s)
├─ LLM Call: Synthesis (1.1s, 700 tokens)
└─ Output Parsing (0.02s)
```

### 2. Datasets & Testing

| Dataset Type | Purpose | Example |
|--------------|---------|---------|
| **Key-Value** | Simple input → output | Q&A pairs |
| **Chat** | Conversation history | Multi-turn dialogues |
| **LLM** | Prompts with expected behavior | Prompt testing |

**Workflow:**
```
Create Dataset → Add Examples → Run Evaluations → Analyze Results → Iterate
```

### 3. Evaluation Framework

| Evaluator Type | Description | When to Use |
|----------------|-------------|-------------|
| **Exact Match** | Output == expected | Deterministic outputs |
| **String Distance** | Similarity metrics | Near-match validation |
| **LLM-as-Judge** | LLM evaluates quality | Subjective assessment |
| **Custom Function** | Your logic | Domain-specific criteria |
| **Embedding Distance** | Semantic similarity | Meaning preservation |
| **Regex/Contains** | Pattern matching | Format validation |

### 4. Monitoring & Observability

| Metric | What It Tracks | Why It Matters |
|--------|----------------|----------------|
| **Latency** | Response time | User experience |
| **Cost** | Token usage × price | Budget control |
| **Error Rate** | Failed runs / total | Reliability |
| **Token Usage** | Input + output tokens | Cost optimization |
| **Feedback Scores** | User ratings | Quality tracking |
| **Tool Usage** | Which tools called | Behavior analysis |

---

## Setup & Integration

### Basic Integration (Python)

```python
import os
from langsmith import Client
from langchain.callbacks import LangChainTracer

# Set API key
os.environ["LANGSMITH_API_KEY"] = "your-key"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "my-project"

# Automatic tracing with LangChain
chain = create_chain()
chain.invoke({"input": "Hello"})  # Auto-traced

# Manual tracing
from langsmith import traceable

@traceable
def my_function(input_text):
    # Your code
    return result
```

### Dataset Creation

```python
from langsmith import Client

client = Client()

# Create dataset
dataset = client.create_dataset("qa-dataset")

# Add examples
client.create_examples(
    dataset_id=dataset.id,
    inputs=[
        {"question": "What is LangSmith?"},
        {"question": "How does tracing work?"}
    ],
    outputs=[
        {"answer": "A platform for LLM observability"},
        {"answer": "It captures execution traces"}
    ]
)
```

### Running Evaluations

```python
from langsmith.evaluation import evaluate

# Define evaluator
def exact_match(outputs, reference_outputs):
    return outputs["answer"] == reference_outputs["answer"]

# Run evaluation
results = evaluate(
    target=my_chain,
    data="qa-dataset",
    evaluators=[exact_match],
    experiment_prefix="experiment-v1"
)

# View results in UI
```

---

## Production Workflows

### 1. Development → Production Pipeline

| Stage | LangSmith Feature | Action |
|-------|------------------|---------|
| **Development** | Local tracing | Debug logic, fix errors |
| **Testing** | Dataset evaluation | Validate against test cases |
| **Staging** | A/B testing | Compare prompt versions |
| **Production** | Monitoring | Track performance, errors |
| **Improvement** | Annotation | Collect feedback, fine-tune |

### 2. Prompt Engineering Workflow

```
1. Create prompt in Prompts Hub
2. Test with dataset
3. Evaluate results
4. Iterate on prompt
5. Compare versions
6. Deploy best version
7. Monitor in production
8. Collect feedback → Loop back to step 4
```

### 3. Error Investigation

| Step | Action | Tool |
|------|--------|------|
| 1 | Detect anomaly | Monitoring dashboard |
| 2 | Filter error runs | Run filters |
| 3 | Inspect traces | Trace viewer |
| 4 | Identify root cause | Nested spans |
| 5 | Reproduce locally | Dataset creation |
| 6 | Fix and validate | Evaluation |

---

## Advanced Features

### 1. LLM-as-Judge Evaluation

```python
from langchain.evaluation import load_evaluator

# Use LLM to judge quality
evaluator = load_evaluator(
    "criteria",
    criteria="helpfulness",
    llm=llm
)

# Automatic scoring
results = evaluate(
    target=chain,
    data="dataset",
    evaluators=[evaluator]
)
```

### 2. Custom Evaluators

```python
def custom_evaluator(run, example):
    """Custom evaluation logic"""
    prediction = run.outputs["answer"]
    reference = example.outputs["answer"]
    
    # Your logic
    score = calculate_score(prediction, reference)
    
    return {
        "key": "custom_metric",
        "score": score,
        "comment": "Explanation here"
    }
```

### 3. Feedback Collection

```python
# Programmatic feedback
client.create_feedback(
    run_id="run-uuid",
    key="user_rating",
    score=0.9,
    comment="Great response"
)

# In UI: thumbs up/down, star ratings
```

### 4. Prompt Versioning

| Feature | Benefit | Use Case |
|---------|---------|----------|
| **Version History** | Track all changes | Rollback bad prompts |
| **Tagging** | Label versions | Staging vs production |
| **Comparison** | Side-by-side testing | A/B testing |
| **Commit Messages** | Document changes | Team collaboration |

---

## Metrics & Analytics

### Key Metrics to Track

| Category | Metrics | Importance |
|----------|---------|------------|
| **Performance** | p50/p95/p99 latency | User experience |
| **Cost** | Total tokens, cost per user | Budget management |
| **Quality** | Success rate, feedback scores | Output quality |
| **Usage** | Requests per day, users | Growth tracking |
| **Errors** | Error rate, error types | Reliability |

### Dashboard Views

| View | What You See | When to Use |
|------|--------------|-------------|
| **Overview** | High-level KPIs | Daily health check |
| **Runs** | Individual traces | Debugging |
| **Datasets** | Test collections | Quality assurance |
| **Experiments** | Evaluation results | Improvement cycles |
| **Monitoring** | Real-time metrics | Production alerts |

---

## Common Use Cases

### 1. Debugging Production Issues

**Problem**: User reports incorrect answer

**Solution**:
1. Search runs by user ID or timestamp
2. View full trace with inputs/outputs
3. Identify failing component
4. Add to dataset for regression testing
5. Fix and re-evaluate

### 2. Optimizing Costs

**Problem**: High API bills

**Solution**:
1. Analyze token usage by component
2. Identify expensive calls
3. Test shorter prompts in datasets
4. Compare cost vs quality trade-offs
5. Deploy optimized version

### 3. Improving Quality

**Problem**: Inconsistent outputs

**Solution**:
1. Create dataset from production samples
2. Run evaluations with multiple metrics
3. Test prompt variations
4. Use LLM-as-judge for quality
5. A/B test in production

### 4. Team Collaboration

**Problem**: Multiple devs working on prompts

**Solution**:
1. Centralize prompts in Prompts Hub
2. Version all changes
3. Use datasets for validation
4. Review traces together
5. Annotate edge cases

---

## Best Practices

### For Development

| Practice | Why | How |
|----------|-----|-----|
| **Tag Runs** | Easy filtering | Add metadata like environment, user type |
| **Use Projects** | Organize by feature | Separate projects for each app component |
| **Create Datasets Early** | Test-driven development | Start with expected behaviors |
| **Trace Everything** | Full visibility | Enable for all environments |

### For Production

| Practice | Why | How |
|----------|-----|-----|
| **Sample Traces** | Control costs | Trace 10-20% of production traffic |
| **Set Alerts** | Proactive monitoring | Alert on error rate, latency spikes |
| **Collect Feedback** | Quality improvement | Thumbs up/down, corrections |
| **Regular Evaluations** | Prevent regressions | Run nightly evals on datasets |

### For Teams

| Practice | Why | How |
|----------|-----|-----|
| **Document Prompts** | Knowledge sharing | Add commit messages |
| **Share Datasets** | Consistency | Team-wide test cases |
| **Review Traces Together** | Learn from failures | Weekly trace reviews |
| **Standardize Tags** | Consistent filtering | Define tag taxonomy |

---

## Interview Questions & Answers

### Q: "How would you debug a RAG system in production?"

**Answer**:
1. **Trace Analysis**: View retrieval → generation pipeline in LangSmith
2. **Check Retrieval**: Inspect which documents were retrieved
3. **Validate Generation**: See exact prompt sent to LLM
4. **Token Analysis**: Check if context window is exceeded
5. **Create Test Case**: Add failing case to dataset
6. **Iterate**: Test fixes with evaluation
7. **Monitor**: Track fix effectiveness in production

### Q: "How do you measure LLM application quality?"

**Answer**:
- **Automated Metrics**: Exact match, semantic similarity, format validation
- **LLM-as-Judge**: GPT-4 evaluates on criteria (helpfulness, accuracy, safety)
- **Human Feedback**: Thumbs up/down, star ratings in production
- **Custom Evaluators**: Domain-specific logic (e.g., code compiles, math is correct)
- **Regression Testing**: Run evaluations on datasets with each change

### Q: "LangSmith vs building custom observability?"

**Strengths of LangSmith**:
- Pre-built LLM-specific features
- Integrated with LangChain ecosystem
- Low setup time
- Built-in evaluators

**When to build custom**:
- Need complete data control
- Unique compliance requirements
- Already have observability stack
- Very high volume (cost considerations)

### Q: "How do you do A/B testing for prompts?"

**Answer**:
1. **Create Variants**: Version prompts in Prompts Hub
2. **Dataset Testing**: Evaluate both on test dataset
3. **Compare Metrics**: Cost, latency, quality scores
4. **Production Split**: Deploy both versions to % of traffic
5. **Monitor**: Track real user feedback
6. **Analyze**: Statistical significance of differences
7. **Deploy Winner**: Roll out best performing variant

---

## Integration Ecosystem

### Works With

| Category | Tools | Integration Type |
|----------|-------|------------------|
| **Frameworks** | LangChain, LangGraph | Native |
| **LLM Providers** | OpenAI, Anthropic, Cohere | Auto-traced |
| **Vector DBs** | Pinecone, Weaviate, Chroma | Traced via LangChain |
| **Deployment** | Vercel, AWS Lambda, Docker | Environment variables |
| **CI/CD** | GitHub Actions, Jenkins | SDK in pipelines |

---

## Pricing Considerations

| Tier | Traces/Month | Cost Model | Best For |
|------|--------------|------------|----------|
| **Developer** | Limited free | Free | Learning, prototypes |
| **Team** | Higher limits | Per trace | Small production apps |
| **Enterprise** | Unlimited | Custom pricing | Large-scale deployments |

**Cost Optimization**:
- Sample production traces (10-20%)
- Use shorter retention periods
- Archive old projects
- Batch evaluations

---

## Limitations & Considerations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| **Vendor Lock-in** | Tied to LangSmith | Export data regularly |
| **Cost at Scale** | High trace volume = high cost | Sampling, selective tracing |
| **Data Privacy** | Traces sent to LangSmith | Self-hosted option (enterprise) |
| **LangChain Dependency** | Best with LangChain | Manual instrumentation possible |
| **Learning Curve** | Many features | Start with tracing only |

---

## Alternatives Comparison

| Tool | Best Feature | Limitation |
|------|-------------|------------|
| **LangSmith** | LangChain integration | LangChain-centric |
| **Weights & Biases** | Experiment tracking | Less LLM-specific |
| **Helicone** | Simple API proxy | Basic features |
| **Phoenix (Arize)** | Open source, local | Limited cloud features |
| **Langfuse** | Open source alternative | Smaller community |
| **HoneyHive** | Testing focus | Fewer monitoring features |

---

## Quick Start Checklist

- [ ] Create LangSmith account
- [ ] Set LANGSMITH_API_KEY environment variable
- [ ] Enable tracing in your app
- [ ] Create first dataset with 10-20 examples
- [ ] Run initial evaluation
- [ ] Set up monitoring dashboard
- [ ] Configure alerts for errors
- [ ] Document prompt versions
- [ ] Enable feedback collection
- [ ] Schedule regular evaluations

---

## Key Takeaways for Interviews

1. **LangSmith = Observability + Evaluation + Testing** for LLM apps
2. **Critical for production**: You can't improve what you can't measure
3. **Tracing is foundational**: See exactly what your LLM app is doing
4. **Datasets enable iteration**: Test-driven development for AI
5. **Evaluations prevent regressions**: Automated quality checks
6. **Monitoring catches issues**: Real-time production visibility
7. **Works best with LangChain**: But can trace any LLM app
8. **Bridge prototype to production**: Professional LLM application development