# LLM Evaluation - Comprehensive Reference Guide

## Overview
**LLM Evaluation** is the systematic assessment of language model performance across various dimensions including accuracy, safety, robustness, and efficiency. Unlike traditional ML where metrics are clear (accuracy, F1), LLM evaluation is complex due to open-ended generation.

**Core Challenge**: "How do you measure if generated text is 'good'?"

---

## Why LLM Evaluation is Hard

| Challenge | Description | Impact |
|-----------|-------------|--------|
| **Subjectivity** | No single "correct" answer | Hard to automate |
| **Context-dependent** | Quality varies by use case | Need task-specific metrics |
| **Multi-dimensional** | Accuracy ≠ helpfulness ≠ safety | Multiple metrics needed |
| **Expensive** | Human evaluation costly | Can't scale easily |
| **Moving target** | Models improve rapidly | Benchmarks saturate |
| **Gaming** | Models trained on benchmarks | Overfitting to tests |

---

## Evaluation Taxonomy

### Three Levels of Evaluation

| Level | What | When | Cost |
|-------|------|------|------|
| **1. Component-Level** | Individual capabilities (math, reasoning) | Development | Low |
| **2. Task-Level** | End-to-end performance (RAG accuracy) | Testing | Medium |
| **3. Production-Level** | Real-world metrics (user satisfaction) | Production | High |

### Four Evaluation Approaches

| Approach | Method | Pros | Cons | Use Case |
|----------|--------|------|------|----------|
| **Automated Metrics** | BLEU, ROUGE, perplexity | Fast, scalable, cheap | Don't capture quality | Initial screening |
| **LLM-as-Judge** | Use LLM to evaluate outputs | Scalable, flexible | Bias, cost | Development iteration |
| **Human Evaluation** | Expert reviewers | Gold standard | Expensive, slow | Final validation |
| **A/B Testing** | Real user feedback | True impact | Long duration | Production optimization |

---

## Automated Metrics

### Traditional NLP Metrics

| Metric | What It Measures | Range | Best For | Limitations |
|--------|------------------|-------|----------|-------------|
| **BLEU** | N-gram overlap with reference | 0-100 | Translation | Doesn't capture meaning |
| **ROUGE** | Recall of n-grams | 0-1 | Summarization | Favors longer outputs |
| **METEOR** | Alignment with synonyms | 0-1 | Translation | Still surface-level |
| **Perplexity** | Prediction confidence | Lower is better | Language modeling | Doesn't correlate with quality |
| **F1 Score** | Precision + Recall | 0-1 | Classification | Only for structured tasks |

#### BLEU (Bilingual Evaluation Understudy)

**Formula**: Geometric mean of n-gram precision + brevity penalty

**Example**:
```
Reference: "The cat sat on the mat"
Candidate: "The cat on the mat"
BLEU-1: 5/5 = 100% (unigrams match)
BLEU-2: 3/4 = 75% (bigrams)
```

**When to Use**: Translation, when references available
**Limitations**: Doesn't capture semantic similarity, favors exact matches

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**Variants**:
- **ROUGE-N**: N-gram recall
- **ROUGE-L**: Longest common subsequence
- **ROUGE-S**: Skip-bigrams

**When to Use**: Summarization evaluation
**Limitations**: Doesn't assess factuality or coherence

### Semantic Similarity Metrics

| Metric | Method | Pros | Cons |
|--------|--------|------|------|
| **BERTScore** | Embedding similarity | Captures semantics | Computationally expensive |
| **Embedding Distance** | Cosine similarity | Fast, intuitive | Model-dependent |
| **SacreBLEU** | Standardized BLEU | Reproducible | Still surface-level |

---

## LLM-as-Judge Evaluation

### Concept
Use a powerful LLM (GPT-4, Claude) to evaluate other LLM outputs based on criteria.

### Evaluation Patterns

#### 1. Pairwise Comparison
```
Which response is better?

Response A: [output 1]
Response B: [output 2]

Evaluate based on: accuracy, helpfulness, clarity
Answer: A or B
```

**Pros**: Easier than absolute scoring
**Cons**: Position bias (prefers A), needs many comparisons

#### 2. Criteria-Based Scoring
```
Evaluate this response on:
1. Accuracy (1-5): Is the information correct?
2. Helpfulness (1-5): Does it answer the question?
3. Clarity (1-5): Is it easy to understand?

Response: [output]
```

**Pros**: Multi-dimensional, interpretable
**Cons**: Score inflation, inconsistent scales

#### 3. Chain-of-Thought Evaluation
```
Evaluate this response step by step:
1. Identify key claims
2. Verify each claim
3. Check for logical consistency
4. Rate overall quality

Response: [output]
```

**Pros**: More reliable, explainable
**Cons**: Slower, more tokens

### Popular LLM-as-Judge Frameworks

| Framework | Key Feature | Best For |
|-----------|-------------|----------|
| **GPT-4 as Judge** | Most capable | High-stakes evaluation |
| **Claude as Judge** | Lower cost, good quality | Production use |
| **Prometheus** | Open-source judge | Budget-conscious |
| **Auto-evaluators** | Custom fine-tuned | Domain-specific |

### Best Practices for LLM-as-Judge

| Practice | Why | How |
|----------|-----|-----|
| **Use CoT** | More reliable | "Explain your reasoning before scoring" |
| **Calibrate** | Reduce bias | Compare against human labels |
| **Multiple judges** | Reduce variance | Use 3+ evaluations, take average |
| **Structured output** | Consistency | Use JSON schema for scores |
| **Randomize order** | Avoid position bias | Shuffle response order |
| **Reference checks** | Ground evaluation | Provide golden answers |

---

## Task-Specific Evaluation

### Question Answering (QA)

| Metric | What | Use Case |
|--------|------|----------|
| **Exact Match (EM)** | Answer == reference | Factoid QA |
| **F1 (token level)** | Token overlap | Open-domain QA |
| **Answer Relevance** | LLM judges relevance | Complex QA |
| **Faithfulness** | Grounded in context | RAG systems |

### Summarization

| Metric | What | Focus |
|--------|------|-------|
| **ROUGE-L** | Longest common sequence | Coverage |
| **Factual Consistency** | No hallucinations | Accuracy |
| **Compression Ratio** | Summary/original length | Conciseness |
| **Coherence** | Logical flow | Readability |

### Code Generation

| Metric | What | Example |
|--------|------|---------|
| **Pass@k** | % passing unit tests | Pass@1, Pass@10 |
| **Compilation Rate** | % that compile | Syntax correctness |
| **Code Quality** | Maintainability | Cyclomatic complexity |
| **Human Preference** | Developer ratings | Usability |

### Conversational AI

| Metric | What | Why |
|--------|------|-----|
| **Engagement** | Conversation length | User satisfaction |
| **Success Rate** | Task completion | Effectiveness |
| **Safety** | Harmful content rate | Trust |
| **Coherence** | Context maintenance | Quality |

---

## Comprehensive Evaluation Frameworks

### 1. RAGAS (RAG Assessment)

**Purpose**: Evaluate Retrieval-Augmented Generation systems

**Metrics**:

| Metric | Measures | Formula/Method |
|--------|----------|----------------|
| **Faithfulness** | Is answer grounded in context? | Claims supported / Total claims |
| **Answer Relevance** | Does answer address question? | LLM judges relevance |
| **Context Precision** | Are retrieved docs relevant? | Relevant docs in top-K |
| **Context Recall** | Are all relevant docs retrieved? | Ground truth docs retrieved |

**Code Example**:
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance

results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevance]
)
```

**Best For**: RAG systems, QA applications

### 2. DeepEval

**Purpose**: Unit testing for LLM applications

**Metrics**:

| Metric | Type | Use Case |
|--------|------|----------|
| **G-Eval** | LLM-based | Custom criteria |
| **Hallucination** | Factuality | RAG, knowledge tasks |
| **Answer Relevancy** | Relevance | QA systems |
| **Faithfulness** | Grounding | RAG systems |
| **Contextual Relevancy** | Retrieval quality | Search systems |
| **Bias** | Fairness | Production safety |
| **Toxicity** | Safety | Content moderation |

**Code Example**:
```python
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric

metric = AnswerRelevancyMetric(threshold=0.7)
assert_test({
    "input": "What is ML?",
    "actual_output": "Machine learning...",
    "expected_output": "ML is..."
}, metric)
```

**Best For**: CI/CD integration, regression testing

### 3. TruLens

**Purpose**: Observability and evaluation for LLM apps

**Key Features**:

| Feature | Purpose | Benefit |
|---------|---------|---------|
| **Feedback Functions** | Custom evaluators | Flexible metrics |
| **Guardrails** | Safety checks | Production safety |
| **Tracing** | Debug pipeline | Root cause analysis |
| **Dashboard** | Visualization | Monitor quality |

**Metrics**:
- Groundedness (RAG)
- Context Relevance
- Answer Relevance
- Moderation (safety)
- Custom feedback functions

**Best For**: Production monitoring, LLM app observability

### 4. LangSmith Evaluation

**Purpose**: LangChain-native evaluation

**Features**:

| Feature | Description |
|---------|-------------|
| **Datasets** | Test case management |
| **Evaluators** | Pre-built + custom |
| **Experiments** | A/B testing |
| **Feedback** | Production signals |

**Evaluator Types**:
- Exact match
- Embedding distance
- LLM-as-judge
- Custom Python functions

**Best For**: LangChain applications, prompt iteration

### 5. Weights & Biases (W&B)

**Purpose**: Experiment tracking for LLMs

**Features**:

| Feature | Use Case |
|---------|----------|
| **Prompt tracking** | Version control |
| **A/B testing** | Compare variants |
| **Human feedback** | Annotation |
| **Charts & reports** | Analysis |

**Best For**: Research, model comparison, team collaboration

### 6. Promptfoo

**Purpose**: CLI-based prompt testing

**Features**:
- Test prompts against multiple models
- Automated assertions
- Cost tracking
- Regression detection

**Example Config**:
```yaml
prompts:
  - "Summarize: {{text}}"
  
providers:
  - openai:gpt-4
  - anthropic:claude-3
  
tests:
  - vars:
      text: "Long article..."
    assert:
      - type: contains
        value: "key points"
```

**Best For**: Prompt engineering, model selection

---

## Academic Benchmarks

### General Capabilities

| Benchmark | Focus | Tasks | Difficulty |
|-----------|-------|-------|------------|
| **MMLU** | Multitask understanding | 57 subjects (math, history, etc.) | Hard |
| **HellaSwag** | Common sense reasoning | Sentence completion | Medium |
| **ARC** | Science QA | Grade school science | Medium |
| **TruthfulQA** | Truthfulness | Avoiding falsehoods | Hard |
| **GSM8K** | Math reasoning | Grade school math problems | Hard |

### Specialized Benchmarks

| Benchmark | Domain | Purpose |
|-----------|--------|---------|
| **HumanEval** | Code | Python function generation |
| **MBPP** | Code | Basic Python programming |
| **DROP** | Reading | Discrete reasoning |
| **BigBench** | General | 204 diverse tasks |
| **HELM** | Holistic | Multi-dimensional evaluation |

### Benchmark Limitations

| Issue | Description | Impact |
|-------|-------------|--------|
| **Data Contamination** | Test data in training set | Inflated scores |
| **Saturation** | Models near 100% | Not discriminative |
| **Narrow Coverage** | Missing real-world tasks | Poor generalization estimate |
| **Gaming** | Optimize for benchmark | Overfitting |

---

## Human Evaluation

### Evaluation Dimensions

| Dimension | Question | Scale |
|-----------|----------|-------|
| **Accuracy** | Is the information correct? | 1-5 |
| **Helpfulness** | Does it solve the problem? | 1-5 |
| **Harmfulness** | Is it safe/appropriate? | 1-5 |
| **Coherence** | Is it logical/consistent? | 1-5 |
| **Fluency** | Is it well-written? | 1-5 |

### Annotation Protocols

#### Side-by-Side Comparison
```
Which response is better overall?
Response A: [...]
Response B: [...]

○ A is much better
○ A is slightly better
○ Tie
○ B is slightly better
○ B is much better
```

**Pros**: Easier for annotators, more reliable
**Cons**: Needs many comparisons (O(n²))

#### Absolute Rating
```
Rate this response (1-5):
Response: [...]

Quality: ○1 ○2 ○3 ○4 ○5
```

**Pros**: Faster, direct scores
**Cons**: Inter-annotator variance, scale calibration

### Quality Assurance

| Practice | Purpose | Method |
|----------|---------|--------|
| **Multiple annotators** | Reduce bias | 3+ per sample |
| **Inter-annotator agreement** | Consistency | Cohen's Kappa, Fleiss' Kappa |
| **Gold samples** | Calibration | Known-quality examples |
| **Regular training** | Maintain quality | Weekly sessions |
| **Disagreement resolution** | Consensus | Expert arbitration |

---

## Production Metrics

### Business Metrics

| Metric | What | Why |
|--------|------|-----|
| **User Satisfaction** | Thumbs up/down, stars | Direct feedback |
| **Task Success Rate** | % completed tasks | Effectiveness |
| **Engagement** | Session length, messages | Stickiness |
| **Retention** | Return users | Long-term value |
| **Conversion** | Goal completion | Business impact |

### Operational Metrics

| Metric | What | Target |
|--------|------|--------|
| **Latency (p50)** | Median response time | < 1s |
| **Latency (p95)** | 95th percentile | < 2s |
| **Error Rate** | % failed requests | < 0.1% |
| **Token Usage** | Avg tokens per request | Cost control |
| **Cache Hit Rate** | % cached responses | > 50% |

### Safety Metrics

| Metric | What | Threshold |
|--------|------|-----------|
| **Harmful Content Rate** | % flagged by moderation | < 0.01% |
| **Refusal Rate** | % declined requests | Balance needed |
| **Policy Violations** | % breaking guidelines | 0% |
| **User Reports** | Manual flags | Track trend |

---

## Evaluation Strategies

### Development Phase

```
1. Unit Tests (DeepEval)
   - Test individual components
   - Assert expected behaviors
   
2. Dataset Evaluation (RAGAS, LangSmith)
   - Test on curated datasets
   - Measure core metrics
   
3. LLM-as-Judge (GPT-4)
   - Flexible quality assessment
   - Iterate on prompts
```

### Pre-Production

```
1. Benchmark Testing
   - Compare against baselines
   - Ensure no regression
   
2. Human Evaluation
   - Expert review
   - Edge case validation
   
3. Red Teaming
   - Adversarial testing
   - Safety validation
```

### Production

```
1. A/B Testing
   - Compare variants
   - Measure business impact
   
2. Monitoring (TruLens, LangSmith)
   - Track metrics
   - Detect anomalies
   
3. Continuous Feedback
   - User ratings
   - Support tickets
```

---

## Evaluation Best Practices

### Do's

| Practice | Why | How |
|----------|-----|-----|
| **Multi-metric** | Single metric insufficient | Track 5-10 metrics |
| **Representative data** | Real-world validity | Use production samples |
| **Version control** | Track changes | Git for prompts, datasets |
| **Statistical significance** | Avoid false conclusions | A/B test properly |
| **Regular evaluation** | Catch regressions | Nightly runs |
| **Human baselines** | Context for scores | Compare to human performance |

### Don'ts

| Anti-pattern | Why Bad | Instead |
|--------------|---------|---------|
| **Single metric** | Misses issues | Multi-dimensional |
| **Only automated** | Misses quality | Include human eval |
| **Cherry-picking** | False confidence | Systematic testing |
| **Ignoring edge cases** | Production failures | Test adversarial inputs |
| **Static datasets** | Models evolve | Update regularly |
| **Optimizing for benchmarks** | Overfitting | Focus on real use cases |

---

## Common Evaluation Patterns

### Pattern 1: RAG System Evaluation

```
Components to Test:
1. Retrieval Quality
   - Precision@K
   - Recall@K
   - MRR (Mean Reciprocal Rank)
   
2. Context Relevance (RAGAS)
   - Are retrieved docs relevant?
   
3. Answer Quality
   - Faithfulness (grounded in context?)
   - Answer Relevance (addresses question?)
   - Coherence (well-written?)
   
4. End-to-End
   - User satisfaction
   - Task success rate
```

### Pattern 2: Chatbot Evaluation

```
Dimensions:
1. Accuracy
   - Correct information
   - LLM-as-judge
   
2. Safety
   - Toxicity detection
   - Policy compliance
   
3. Helpfulness
   - Solves user problem
   - Follow-up questions
   
4. Engagement
   - Conversation length
   - User retention
```

### Pattern 3: Code Generation Evaluation

```
Metrics:
1. Functional Correctness
   - Pass@k (unit tests)
   - Compilation rate
   
2. Code Quality
   - Readability (linter scores)
   - Efficiency (complexity)
   
3. Human Preference
   - Developer surveys
   - Acceptance rate
```

---

## Tools Comparison

| Tool | Best For | Strengths | Limitations | Cost |
|------|----------|-----------|-------------|------|
| **RAGAS** | RAG evaluation | RAG-specific metrics | Limited to RAG | Free |
| **DeepEval** | Unit testing | CI/CD integration | Learning curve | Free |
| **TruLens** | Production monitoring | Real-time feedback | Python only | Free |
| **LangSmith** | LangChain apps | Native integration | LangChain-centric | Paid |
| **W&B** | Experiment tracking | Team collaboration | Complex setup | Freemium |
| **Promptfoo** | Prompt testing | CLI simplicity | Limited features | Free |
| **Human Eval** | Gold standard | Most accurate | Expensive, slow | $$$ |

---

## Interview Questions & Answers

### Q: "How do you evaluate an LLM application?"

**Answer**: Multi-level approach:

1. **Component-level**: Test individual capabilities (retrieval, generation) with automated metrics
2. **Task-level**: End-to-end performance on curated datasets using frameworks like RAGAS or DeepEval
3. **Production-level**: Real user feedback via A/B tests, satisfaction ratings

**Key insight**: No single metric is sufficient. Use combination of automated (BLEU, ROUGE), LLM-as-judge (GPT-4 evaluation), and human evaluation.

### Q: "What's the difference between BLEU and BERTScore?"

**Answer**:
- **BLEU**: N-gram overlap, surface-level matching. Fast but misses semantic similarity.
- **BERTScore**: Embedding-based, captures meaning. Slower but better quality assessment.

**Example**: "The cat sat" vs "A feline was seated"
- BLEU: Low (different words)
- BERTScore: High (same meaning)

**Use BLEU for**: Translation with references, fast iteration
**Use BERTScore for**: Semantic quality, paraphrasing tasks

### Q: "How do you evaluate RAG systems?"

**Answer**: Use **RAGAS framework** with 4 key metrics:

1. **Context Precision**: Are retrieved docs relevant to query?
2. **Context Recall**: Are all relevant docs retrieved?
3. **Faithfulness**: Is answer grounded in context (no hallucinations)?
4. **Answer Relevance**: Does answer address the question?

Also track: retrieval latency, user satisfaction, task success rate.

### Q: "What is LLM-as-Judge and when to use it?"

**Answer**: Using a powerful LLM (GPT-4, Claude) to evaluate outputs based on criteria.

**When to use**:
- Development iteration (fast feedback)
- Multi-dimensional quality (accuracy, helpfulness, clarity)
- No clear reference answer

**Best practices**:
- Use Chain-of-Thought ("explain your reasoning")
- Multiple judges to reduce variance
- Calibrate against human labels
- Watch for position bias

**Limitations**: Costs add up, may have biases, not as reliable as human eval

### Q: "How do you handle subjective evaluation?"

**Answer**:

1. **Multiple annotators**: 3+ per sample for reliability
2. **Clear rubrics**: Define scoring criteria explicitly
3. **Calibration samples**: Use gold standards
4. **Inter-annotator agreement**: Measure with Cohen's Kappa
5. **Expert arbitration**: Resolve disagreements

**For scale**: Use LLM-as-judge calibrated against human labels, validate periodically.

### Q: "What metrics for a customer support chatbot?"

**Answer**:

**Accuracy Metrics**:
- Answer correctness (LLM-as-judge)
- Factuality (grounded in knowledge base?)

**Effectiveness Metrics**:
- Resolution rate (% issues solved)
- Average handling time
- Escalation rate (to human)

**Quality Metrics**:
- User satisfaction (CSAT)
- Thumbs up/down ratio
- Conversation abandonment

**Safety Metrics**:
- Harmful content rate
- Policy violations
- User reports

### Q: "How do you detect model regression?"

**Answer**:

1. **Regression Test Suite**: Curated dataset with known-good outputs
2. **Automated Evaluation**: Run nightly on core metrics
3. **Alerts**: Notify if metrics drop > X%
4. **Version Control**: Track all changes (model, prompt, data)
5. **Rollback Plan**: Can revert quickly

**Tools**: DeepEval for unit tests, LangSmith for datasets, monitoring dashboards.

---

## Advanced Topics

### Calibration

**Problem**: LLM-as-judge scores don't match human judgment

**Solution**:
1. Collect human labels for subset (100-1000 samples)
2. Compare LLM scores vs human scores
3. Find calibration function: `human_score = f(llm_score)`
4. Apply calibration to all LLM evaluations

### Cost-Quality Trade-offs

| Approach | Cost | Quality | Use When |
|----------|------|---------|----------|
| **Automated only** | $ | 70% | Development |
| **LLM-as-judge** | $$ | 85% | Iteration |
| **Human review** | $$$$ | 95% | Production validation |
| **Hybrid** | $$$ | 90% | Production monitoring |

### Adversarial Testing (Red Teaming)

**Goal**: Find failure modes before production

**Methods**:
- Prompt injection attempts
- Offensive input handling
- Edge case exploration
- Multi-turn manipulation

**Tools**: 
- Manual testing by security team
- Automated adversarial prompts
- Bounty programs

---

## Evaluation Checklist

### Pre-Launch

- [ ] Unit tests pass (DeepEval)
- [ ] Core metrics on test set meet targets
- [ ] Human evaluation on 100+ samples
- [ ] Red team testing completed
- [ ] Safety guardrails validated
- [ ] Latency/cost within budget
- [ ] A/B test plan ready

### Post-Launch

- [ ] Real-time monitoring active
- [ ] User feedback collection setup
- [ ] Weekly metric review
- [ ] Monthly human evaluation
- [ ] Regression tests in CI/CD
- [ ] Incident response plan
- [ ] Continuous improvement pipeline

---

## Key Takeaways for Interviews

1. **No silver bullet**: Use multiple evaluation methods (automated, LLM-judge, human)
2. **Task-specific**: Different tasks need different metrics (RAG ≠ Code generation)
3. **RAGAS for RAG**: Standard framework for retrieval-augmented generation
4. **LLM-as-Judge**: Scalable but needs calibration against human labels
5. **Benchmarks saturate**: Focus on real use case performance, not leaderboard scores
6. **Production metrics matter most**: User satisfaction > benchmark scores
7. **Continuous evaluation**: Not one-time, ongoing monitoring required
8. **Multi-dimensional**: Track accuracy, safety, latency, cost together
9. **Human eval is gold standard**: But expensive, use strategically
10. **Version everything**: Models, prompts, eval datasets for reproducibility