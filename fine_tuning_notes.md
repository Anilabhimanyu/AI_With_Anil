# Fine-tuning (LoRA, QLoRA, PEFT) - Reference Guide

## Overview
**Fine-tuning** adapts pre-trained language models to specific tasks or domains by training on custom data. Modern techniques like LoRA and QLoRA enable efficient fine-tuning with minimal computational resources.

**Core Problem Solved**: Full fine-tuning requires massive compute (training 70B model = $100K+). Parameter-Efficient Fine-Tuning (PEFT) achieves similar results with <1% trainable parameters.

---

## Fine-tuning Landscape

### Types of Fine-tuning

| Type | Trainable Params | Memory | Time | Quality | Use Case |
|------|------------------|--------|------|---------|----------|
| **Full Fine-tuning** | 100% | Very High | Days-Weeks | Best | Large budgets, critical tasks |
| **LoRA** | 0.1-1% | Low | Hours | Excellent | Standard approach |
| **QLoRA** | 0.1-1% | Very Low | Hours | Excellent | Consumer GPUs |
| **Prefix Tuning** | <0.1% | Very Low | Hours | Good | Simple tasks |
| **Prompt Tuning** | <0.01% | Minimal | Minutes | Fair | Very simple tasks |
| **Adapter Layers** | 1-5% | Low | Hours | Good | Multi-task |

### When to Fine-tune vs Alternatives

| Approach | When | Cost | Quality | Flexibility |
|----------|------|------|---------|-------------|
| **Prompt Engineering** | Task is clear, model understands | $ | 70% | High |
| **RAG** | Need current/private data | $$ | 80% | High |
| **Fine-tuning** | Need specific style/domain | $$$ | 90% | Medium |
| **Pre-training** | New domain from scratch | $$$$ | 95% | Low |

**Decision Tree**:
```
Can prompting work? → Yes → Use prompts
    ↓ No
Need external knowledge? → Yes → Use RAG
    ↓ No
Need specific style/format? → Yes → Fine-tune
    ↓ No
Completely new domain? → Yes → Pre-train from scratch
```

---

## Full Fine-tuning

### Concept
Update all model parameters on your dataset.

### Process
1. Load pre-trained model
2. Replace/add task-specific head
3. Train on labeled data
4. Save entire model

### Memory Requirements

| Model | Parameters | FP32 Memory | Mixed Precision | Gradient Memory | Total |
|-------|------------|-------------|-----------------|-----------------|-------|
| **7B** | 7B | 28 GB | 14 GB | 28 GB | 70 GB |
| **13B** | 13B | 52 GB | 26 GB | 52 GB | 130 GB |
| **70B** | 70B | 280 GB | 140 GB | 280 GB | 700 GB |

**Formula**: `Memory = Model + Optimizer States + Gradients ≈ 3-4× model size`

### Pros & Cons

**Pros**:
- Maximum performance
- Full model adaptation
- No restrictions

**Cons**:
- Extremely expensive (70B = $100K+)
- Requires massive GPU memory
- Risk of catastrophic forgetting
- Slow iteration

**Best For**: Critical applications with large budgets, foundation model development

---

## Parameter-Efficient Fine-Tuning (PEFT)

### Core Idea
Train small number of additional parameters while keeping base model frozen.

### Benefits

| Benefit | Impact | Example |
|---------|--------|---------|
| **Memory Efficiency** | 10-100× less memory | 7B full: 70GB → LoRA: 8GB |
| **Training Speed** | 3-10× faster | Days → Hours |
| **Storage** | 1000× smaller | 14GB → 14MB (LoRA weights) |
| **Multi-task** | Switch adapters | One base + many adapters |
| **Lower Overfitting** | Fewer parameters | Better generalization |

---

## LoRA (Low-Rank Adaptation)

### Concept
Instead of updating weight matrix W, learn low-rank decomposition: **W' = W + BA**

**Visualization**:
```
Original: W (d × d)
         [Large matrix, millions of parameters]

LoRA: W + B × A
      W (frozen) + B(d × r) × A(r × d)
      where r << d (e.g., r=8, d=4096)
```

**Key Insight**: Weight updates are low-rank (most info in few dimensions)

### Mathematics

**Standard Fine-tuning**: 
- Update: `W' = W + ΔW`
- ΔW has d×d parameters

**LoRA**:
- Update: `W' = W + BA`
- B has d×r parameters
- A has r×d parameters
- Total: 2×d×r parameters (where r << d)

**Example**: For d=4096, r=8:
- Full: 4096×4096 = 16.7M parameters
- LoRA: 2×4096×8 = 65K parameters
- **Reduction: 256×**

### Hyperparameters

| Parameter | Description | Typical Values | Impact |
|-----------|-------------|----------------|--------|
| **r (rank)** | Bottleneck dimension | 8, 16, 32, 64 | Higher = more capacity, slower |
| **alpha** | Scaling factor | 16, 32 | Controls adaptation strength |
| **target_modules** | Which layers to adapt | q_proj, v_proj | More layers = more parameters |
| **dropout** | Regularization | 0.05, 0.1 | Prevents overfitting |

**Rank Selection**:
- **r=8**: Simple tasks, small datasets
- **r=16**: General purpose (default)
- **r=32-64**: Complex tasks, large datasets
- **r=128+**: Approaching full fine-tuning

### Code Example

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure LoRA
lora_config = LoraConfig(
    r=16,                          # Rank
    lora_alpha=32,                 # Scaling
    target_modules=["q_proj", "v_proj"],  # Which layers
    lora_dropout=0.05,             # Regularization
    bias="none",                   # Don't train bias
    task_type="CAUSAL_LM"          # Task type
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 4.2M || all params: 6.7B || trainable%: 0.06%
```

### Training

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,                     # Mixed precision
    save_steps=100,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

### Inference

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Load LoRA weights (small file, ~14MB)
model = PeftModel.from_pretrained(base_model, "./lora-model")

# Generate
outputs = model.generate(inputs)
```

### Pros & Cons

**Pros**:
- 99% parameter reduction
- Fast training (hours vs days)
- Multiple adapters (easy A/B testing)
- Small storage (MB vs GB)

**Cons**:
- Slightly lower performance than full fine-tuning (~2-5%)
- Requires careful hyperparameter tuning
- Extra inference latency (small)

**Best For**: Standard fine-tuning approach for most use cases

---

## QLoRA (Quantized LoRA)

### Concept
LoRA + 4-bit quantization = Fine-tune 70B models on consumer GPUs

**Innovation**: 
1. Quantize base model to 4-bit (NF4)
2. Keep LoRA adapters in FP16
3. Compute gradients in BF16

### Key Techniques

| Technique | Purpose | Benefit |
|-----------|---------|---------|
| **4-bit NormalFloat (NF4)** | Quantize base model | 4× memory reduction |
| **Double Quantization** | Quantize quantization constants | Extra 0.4 bits/param saved |
| **Paged Optimizers** | Handle memory spikes | Prevent OOM |

### Memory Comparison

| Model | Full FT | LoRA (FP16) | QLoRA (4-bit) | GPU Needed |
|-------|---------|-------------|---------------|------------|
| **7B** | 70 GB | 8 GB | 6 GB | RTX 3090 (24GB) |
| **13B** | 130 GB | 14 GB | 10 GB | RTX 4090 (24GB) |
| **33B** | 330 GB | 35 GB | 20 GB | A100 (40GB) |
| **70B** | 700 GB | 70 GB | 35 GB | A100 (80GB) |

**Key Insight**: QLoRA enables 70B fine-tuning on single A100, vs 8× A100s for full fine-tuning

### Code Example

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 4-bit quantization
    bnb_4bit_quant_type="nf4",              # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in BF16
    bnb_4bit_use_double_quant=True,         # Double quantization
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto",                      # Automatic device placement
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# LoRA config (same as before)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Train as normal
trainer.train()
```

### Quality

**Key Finding**: QLoRA achieves 99% of full fine-tuning quality

| Model | Task | Full FT | QLoRA | Difference |
|-------|------|---------|-------|------------|
| LLaMA-7B | MMLU | 45.2% | 44.8% | -0.4% |
| LLaMA-13B | GSM8K | 32.1% | 31.5% | -0.6% |
| LLaMA-70B | TruthfulQA | 51.3% | 50.9% | -0.4% |

### Pros & Cons

**Pros**:
- Fine-tune 70B on single GPU
- Same quality as full fine-tuning
- Democratizes large model training
- Minimal quality loss vs LoRA

**Cons**:
- Slower training than LoRA (quantization overhead)
- Requires recent GPU (Ampere+)
- More complex setup

**Best For**: Large models (33B+) on limited hardware

---

## Other PEFT Methods

### 1. Prefix Tuning

**Concept**: Add trainable prefix tokens to input

```
Input: [PREFIX_1, PREFIX_2, ..., PREFIX_K, actual_input]
       ↑________________________↑
       Trainable (frozen model processes all)
```

**Parameters**: K × hidden_dim (e.g., 20 × 4096 = 80K)

**Pros**: Very few parameters
**Cons**: Takes up context window

**Use Case**: Lightweight adaptation, multi-task learning

### 2. Prompt Tuning

**Concept**: Optimize continuous prompts (embedding space)

**Parameters**: Even fewer than prefix tuning

**Quality**: Lower than LoRA, suitable for simple tasks

### 3. Adapter Layers

**Concept**: Insert small bottleneck layers between transformer blocks

```
Transformer Layer
    ↓
Adapter (down-project → activation → up-project)
    ↓
Next Layer
```

**Parameters**: 1-5% of model

**Pros**: Modular, composable
**Cons**: Adds inference latency

**Use Case**: Multi-task learning (different adapter per task)

### 4. (IA)³ - Infused Adapter by Inhibiting and Amplifying Inner Activations

**Concept**: Learn scaling vectors for activations

**Parameters**: <0.01% of model

**Quality**: Good for small datasets

---

## PEFT Method Comparison

| Method | Params | Memory | Speed | Quality | Inference Cost |
|--------|--------|--------|-------|---------|----------------|
| **Full FT** | 100% | Very High | Slow | 100% | None |
| **LoRA** | 0.1-1% | Low | Fast | 97-99% | Minimal |
| **QLoRA** | 0.1-1% | Very Low | Medium | 97-99% | Some (quantization) |
| **Prefix** | <0.1% | Very Low | Fast | 85-95% | Context length |
| **Adapter** | 1-5% | Low | Fast | 90-95% | Extra layers |
| **Prompt** | <0.01% | Minimal | Very Fast | 70-85% | None |

---

## Fine-tuning Datasets

### Dataset Size Requirements

| Task Complexity | Minimum Samples | Recommended | Quality Matters |
|----------------|-----------------|-------------|-----------------|
| **Simple** (classification) | 100 | 1,000 | Medium |
| **Medium** (instruction following) | 1,000 | 10,000 | High |
| **Complex** (reasoning, coding) | 10,000 | 100,000 | Critical |

### Data Quality > Quantity

**Key Principle**: 1,000 high-quality samples > 100,000 low-quality

**Quality Checklist**:
- [ ] Diverse examples
- [ ] Correct labels/outputs
- [ ] Representative of use case
- [ ] Balanced distribution
- [ ] No duplicates

### Dataset Formats

#### Instruction-Following
```json
{
  "instruction": "Translate to French",
  "input": "Hello, how are you?",
  "output": "Bonjour, comment allez-vous?"
}
```

#### Chat Format
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."}
  ]
}
```

#### Completion Format
```json
{
  "prompt": "Question: What is AI?\nAnswer:",
  "completion": "AI stands for Artificial Intelligence..."
}
```

---

## Training Best Practices

### Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| **Learning Rate** | 1e-4 to 3e-4 | Higher for LoRA than full FT |
| **Batch Size** | 4-32 | With gradient accumulation |
| **Epochs** | 3-5 | More risks overfitting |
| **Warmup Steps** | 100-500 | Stabilizes training |
| **Weight Decay** | 0.01-0.1 | Regularization |
| **Max Grad Norm** | 1.0 | Prevent exploding gradients |

### Learning Rate Scheduling

| Schedule | When | Why |
|----------|------|-----|
| **Constant** | Small datasets | Simple |
| **Linear Decay** | Standard | Gradual reduction |
| **Cosine** | Long training | Smooth annealing |
| **Cosine with Restarts** | Multiple epochs | Escape local minima |

### Monitoring

| Metric | What | Alert If |
|--------|------|----------|
| **Training Loss** | Model fitting | Not decreasing |
| **Validation Loss** | Generalization | Increasing (overfitting) |
| **Perplexity** | Language modeling quality | Very high |
| **Gradient Norm** | Training stability | Exploding (>10) |
| **Learning Rate** | Current LR | Too high/low |

### Common Issues

| Problem | Symptom | Solution |
|---------|---------|----------|
| **Overfitting** | Val loss increases | More data, lower epochs, regularization |
| **Underfitting** | High train loss | More epochs, higher LR, larger rank |
| **Catastrophic Forgetting** | Loss of general abilities | Lower LR, fewer epochs, LoRA |
| **Slow Convergence** | Loss not decreasing | Higher LR, check data quality |
| **OOM** | Out of memory | Smaller batch, gradient checkpointing, QLoRA |

---

## Advanced Techniques

### 1. Multi-Task Fine-tuning

**Concept**: Train on multiple tasks simultaneously

**Benefits**:
- Better generalization
- Shared knowledge
- One model for many tasks

**Implementation**:
```python
# Mix datasets
mixed_dataset = concatenate([task1_data, task2_data, task3_data])

# Optional: Task-specific LoRA
lora_task1 = LoraConfig(...)  # Adapter for task 1
lora_task2 = LoraConfig(...)  # Adapter for task 2
```

### 2. Instruction Tuning

**Goal**: Make model follow instructions better

**Data**: Diverse instruction-response pairs

**Example Datasets**:
- Alpaca (52K instructions)
- Dolly (15K instructions)
- FLAN (1.8M instructions)

### 3. RLHF (Reinforcement Learning from Human Feedback)

**Process**:
1. Supervised fine-tuning (SFT)
2. Train reward model
3. RL optimization (PPO)

**Use Case**: Align model with human preferences (ChatGPT approach)

**Complexity**: High, requires specialized infrastructure

### 4. DPO (Direct Preference Optimization)

**Innovation**: Skip reward model, optimize preferences directly

**Benefits**:
- Simpler than RLHF
- More stable training
- Similar results

**Data Format**:
```json
{
  "prompt": "Explain quantum physics",
  "chosen": "Quantum physics studies matter at atomic scale...",
  "rejected": "Quantum is when things are small and weird."
}
```

### 5. Merge Adapters

**Concept**: Combine multiple LoRA adapters

**Methods**:
- **Linear**: Average weights
- **TIES**: Task-specific merging
- **DARE**: Drop and rescale

**Use Case**: Multi-skill models

---

## Practical Workflow

### Step-by-Step Process

```
1. Define Task & Metrics
   ↓
2. Collect/Prepare Dataset (1K-10K samples)
   ↓
3. Choose Base Model (7B, 13B, 70B)
   ↓
4. Select PEFT Method (LoRA/QLoRA)
   ↓
5. Configure Hyperparameters (rank, LR, etc.)
   ↓
6. Train & Monitor
   ↓
7. Evaluate on Test Set
   ↓
8. Iterate if Needed
   ↓
9. Deploy
```

### Iteration Strategy

| Iteration | Focus | What to Try |
|-----------|-------|-------------|
| **1st** | Baseline | r=16, default LR, 3 epochs |
| **2nd** | Data | More/better data, data cleaning |
| **3rd** | Hyperparameters | Try r=32, adjust LR |
| **4th** | Architecture | More target modules, larger model |

---

## Tools & Libraries

### HuggingFace PEFT

**Repository**: `huggingface/peft`

**Features**:
- LoRA, QLoRA, Prefix, Adapter, etc.
- Easy integration with Transformers
- Active development

**Installation**:
```bash
pip install peft transformers bitsandbytes accelerate
```

### Axolotl

**Purpose**: Configuration-based fine-tuning

**Benefits**:
- YAML config (no code)
- Best practices built-in
- Multi-GPU support

**Example Config**:
```yaml
base_model: meta-llama/Llama-2-7b-hf
model_type: AutoModelForCausalLM
tokenizer_type: LlamaTokenizer

load_in_4bit: true
adapter: lora
lora_r: 16
lora_alpha: 32

datasets:
  - path: my_dataset.jsonl
    type: alpaca

num_epochs: 3
micro_batch_size: 4
gradient_accumulation_steps: 4
```

### LLaMA-Factory

**Purpose**: Web UI for fine-tuning

**Features**:
- No-code fine-tuning
- Multiple models supported
- Built-in datasets

### TRL (Transformer Reinforcement Learning)

**Purpose**: RLHF/DPO training

**Features**:
- PPO trainer
- DPO trainer
- Reward modeling

---

## Cost Analysis

### Training Costs (Cloud GPUs)

| Model | Method | GPU | Time | Cost (AWS) |
|-------|--------|-----|------|------------|
| **7B** | Full FT | 8×A100 | 24h | $240 |
| **7B** | LoRA | 1×A100 | 4h | $12 |
| **7B** | QLoRA | 1×RTX4090 | 6h | $3 |
| **70B** | Full FT | 64×A100 | 48h | $3,840 |
| **70B** | LoRA | 8×A100 | 12h | $120 |
| **70B** | QLoRA | 1×A100 | 24h | $24 |

**Key Insight**: QLoRA reduces 70B fine-tuning cost from $3,840 to $24 (160× cheaper)

### Storage Costs

| Method | Model Size | Adapter Size | Storage |
|--------|------------|--------------|---------|
| **Full FT** | 7B FP16 | - | 14 GB |
| **LoRA** | Base 14GB | Adapter 14MB | 14 GB + 14 MB |
| **QLoRA** | Base 3.5GB (4-bit) | Adapter 14MB | 3.5 GB + 14 MB |

**Benefit**: Host 100 fine-tuned models = 1 base + 100 adapters (1.4 GB total for adapters)

---

## Interview Questions & Answers

### Q: "What is LoRA and how does it work?"

**Answer**: LoRA (Low-Rank Adaptation) is a PEFT technique that freezes the pre-trained model and trains low-rank decomposition matrices.

**Key Idea**: Weight updates are low-rank, so instead of updating W → W', we learn W' = W + BA where B and A are small matrices.

**Math**: 
- Full: d×d parameters (16M for d=4096)
- LoRA: 2×d×r parameters (65K for d=4096, r=8)
- **256× parameter reduction**

**Benefits**: 99% fewer parameters, 10× faster training, multiple adapters possible.

**Use Case**: Standard approach for fine-tuning 7B-70B models efficiently.

### Q: "Explain the difference between LoRA and QLoRA"

**Answer**:

**LoRA**:
- Base model in FP16
- LoRA adapters in FP16
- Memory: 7B model ≈ 14GB

**QLoRA**:
- Base model in 4-bit (NF4)
- LoRA adapters in FP16
- Memory: 7B model ≈ 6GB

**Key Innovation**: Quantize base model to 4-bit, keep adapters in FP16, compute gradients in BF16.

**Impact**: Fine-tune 70B on single A100 (vs 8× A100s for LoRA).

**Quality**: 99% of full fine-tuning performance, virtually no loss vs LoRA.

### Q: "When should you use full fine-tuning vs LoRA?"

**Answer**:

**Use Full Fine-tuning when**:
- Maximum performance critical
- Large budget available
- Creating foundation model
- Have 100K+ high-quality samples

**Use LoRA when**:
- Standard use case (most common)
- Limited compute budget
- Need fast iteration
- Want multiple adapters (A/B test)

**Use QLoRA when**:
- Large model (33B+) on limited hardware
- Consumer GPU (RTX 3090/4090)
- Cost-conscious

**My default**: Start with LoRA (r=16), only use full FT if LoRA insufficient.

### Q: "What is the 'rank' hyperparameter in LoRA?"

**Answer**: Rank (r) determines the bottleneck dimension in low-rank decomposition.

**Impact**:
- **Higher rank**: More capacity, more parameters, slower training
- **Lower rank**: Fewer parameters, faster, may underfit

**Typical Values**:
- r=8: Simple tasks, small datasets
- r=16: General purpose (default)
- r=32-64: Complex tasks
- r=128+: Approaching full fine-tuning

**Selection**: Start with r=16, increase if underfitting, decrease if overfitting.

**Parameters**: 2×d×r (so r=16 → r=32 doubles trainable parameters)

### Q: "How much data do you need for fine-tuning?"

**Answer**: Depends on task complexity:

**Guidelines**:
- **Simple** (classification, format): 100-1,000 samples
- **Medium** (instruction following): 1,000-10,000 samples
- **Complex** (reasoning, coding): 10,000-100,000 samples

**Key Principle**: Quality > Quantity. 1,000 high-quality > 100,000 low-quality.

**Data Quality Checklist**:
- Diverse examples
- Correct labels
- Representative of use case
- No duplicates
- Balanced distribution

**Red Flag**: If you need >100K samples, consider if prompting/RAG could work instead.

### Q: "What are common failure modes in fine-tuning?"

**Answer**:

1. **Catastrophic Forgetting**: Model loses general abilities
   - Solution: Lower LR, fewer epochs, use LoRA

2. **Overfitting**: Great on train, poor on validation
   - Solution: More data, regularization, fewer epochs

3. **Underfitting**: High training loss
   - Solution: Higher rank, more epochs, check data quality

4. **Mode Collapse**: Repetitive outputs
   - Solution: More diverse data, adjust temperature

5. **Data Leakage**: Test data in training set
   - Solution: Proper train/test split

**Prevention**: Monitor train vs val loss, early stopping, proper evaluation.

### Q: "How do you evaluate a fine-tuned model?"

**Answer**: Multi-level evaluation:

1. **Task-Specific Metrics**:
   - Classification: Accuracy, F1
   - Generation: ROUGE, human eval
   - Code: Pass@k

2. **General Capabilities**:
   - Test on benchmarks (MMLU, GSM8K)
   - Ensure no catastrophic forgetting

3. **Human Evaluation**:
   - Side-by-side comparison
   - Quality ratings

4. **Production Metrics**:
   - User satisfaction
   - Task success rate

**Best Practice**: Establish baseline (base model), compare fine-tuned, iterate.

### Q: "Explain DPO and how it differs from RLHF"

**Answer**:

**RLHF** (Reinforcement Learning from Human Feedback):
1. Supervised fine-tuning (SFT)
2. Train reward model on preferences
3. RL optimization (PPO)

**DPO** (Direct Preference Optimization):
1. SFT
2. Directly optimize on preference data (skip reward model)

**Key Difference**: DPO skips reward model, optimizes preferences directly using classification loss.

**Benefits of DPO**:
- Simpler (no RL complexity)
- More stable training
- Similar or better results
- Less compute intensive

**Use Case**: Aligning models with human preferences (helpful, harmless, honest).

---

## Key Takeaways for Interviews

1. **LoRA is the default**: 99% parameter reduction, 10× faster, 97-99% quality
2. **QLoRA for large models**: Fine-tune 70B on single GPU via 4-bit quantization
3. **Rank hyperparameter**: r=16 default, higher for complex tasks
4. **Quality > Quantity**: 1,000 good samples > 100,000 bad
5. **Catastrophic forgetting**: Risk with full FT, minimal with LoRA
6. **Multiple adapters**: One base model, many task-specific adapters
7. **Training time**: Hours (LoRA) vs Days (full FT)
8. **Cost**: LoRA 10-100× cheaper than full fine-tuning
9. **When to fine-tune**: Need specific style/domain that prompting/RAG can't achieve
10. **Evaluation critical**: Monitor train/val loss, test general capabilities

---

## Quick Reference: Decision Matrix

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| **7B model, single GPU** | LoRA (r=16) | Standard, efficient |
| **70B model, single A100** | QLoRA (r=16) | Only option |
| **Multiple tasks** | LoRA with adapters | Switch adapters per task |
| **Small dataset (<1K)** | LoRA (r=8) | Prevent overfitting |
| **Large dataset (>100K)** | LoRA (r=32-64) or Full FT | More capacity needed |
| **Cost-critical** | QLoRA | Cheapest option |
| **Performance-critical** | Full Fine-tuning | Maximum quality |
| **Fast iteration** | LoRA | Quick training |
| **Production serving** | Merge adapter to base | Single model |