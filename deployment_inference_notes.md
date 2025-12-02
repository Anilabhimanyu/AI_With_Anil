# LLM Deployment & Inference - Reference Guide

## Overview
**LLM Deployment & Inference** covers the techniques, tools, and architectures for serving large language models in production at scale. The goal is to optimize latency, throughput, cost, and resource utilization.

**Core Challenge**: LLMs are massive (7B-405B parameters), memory-intensive, and computationally expensive. Efficient inference is critical for production viability.

---

## Fundamental Concepts

### Inference Basics

| Concept | Description | Impact |
|---------|-------------|--------|
| **Latency** | Time from request to response | User experience |
| **Throughput** | Requests per second | System capacity |
| **TTFT** | Time To First Token | Perceived responsiveness |
| **TPOT** | Time Per Output Token | Generation speed |
| **Memory Bandwidth** | Data transfer rate | Bottleneck for large models |

### Key Metrics

| Metric | What | Target | Importance |
|--------|------|--------|------------|
| **Time To First Token (TTFT)** | Time until first token generated | < 200ms | User experience |
| **Throughput (tokens/sec)** | Generation speed | > 100 | Efficiency |
| **Latency (p50/p95/p99)** | Response time percentiles | p95 < 2s | Reliability |
| **GPU Utilization** | % GPU used | > 80% | Cost efficiency |
| **Requests/sec** | Concurrent request handling | Varies | Scale |
| **Cost per 1K tokens** | Operating cost | Minimize | Economics |

---

## The Inference Problem

### Why LLM Inference is Challenging

| Challenge | Description | Impact |
|-----------|-------------|--------|
| **Model Size** | 7B-405B parameters | Requires 14GB-810GB memory |
| **Memory Bandwidth Bound** | Loading weights from memory | Bottleneck, not compute |
| **KV Cache** | Storing attention keys/values | Memory grows with sequence length |
| **Sequential Generation** | Auto-regressive, can't parallelize | Inherently slow |
| **Batch Inefficiency** | Variable sequence lengths | GPU underutilization |

### Memory Requirements

**Formula**: `Memory = Parameters × Bytes_per_param + KV_cache + Activations`

| Model | Params | FP16 Memory | FP8 Memory | INT8 Memory |
|-------|--------|-------------|------------|-------------|
| **GPT-2** | 1.5B | 3 GB | 1.5 GB | 1.5 GB |
| **LLaMA-7B** | 7B | 14 GB | 7 GB | 7 GB |
| **LLaMA-13B** | 13B | 26 GB | 13 GB | 13 GB |
| **LLaMA-70B** | 70B | 140 GB | 70 GB | 70 GB |
| **GPT-3** | 175B | 350 GB | 175 GB | 175 GB |

*Note: Actual memory includes overhead for KV cache, activations*

---

## Optimization Techniques

### 1. Quantization

**Concept**: Reduce precision of model weights

| Precision | Bits | Memory | Accuracy Loss | Use Case |
|-----------|------|--------|---------------|----------|
| **FP32** | 32 | 4× | 0% (baseline) | Training |
| **FP16** | 16 | 2× | ~0% | Standard inference |
| **FP8** | 8 | 1× | <1% | Modern GPUs (H100) |
| **INT8** | 8 | 1× | 1-2% | CPU inference |
| **INT4** | 4 | 0.5× | 2-5% | Aggressive compression |

#### Popular Quantization Methods

| Method | Description | Quality | Speed |
|--------|-------------|---------|-------|
| **Post-Training Quantization (PTQ)** | Quantize after training | Good | Fast setup |
| **Quantization-Aware Training (QAT)** | Train with quantization | Better | Slower setup |
| **GPTQ** | One-shot weight quantization | Excellent | Medium setup |
| **AWQ** | Activation-aware quantization | Excellent | Medium setup |
| **GGUF** | Format for CPU inference | Good | Fast on CPU |

**Example - GPTQ**:
```python
from transformers import AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM

# Load quantized model
model = AutoGPTQForCausalLM.from_quantized(
    "model-name-gptq",
    device="cuda:0"
)
```

**Benefits**:
- 2-4× memory reduction
- 2-3× speed improvement
- Minimal accuracy loss

### 2. KV Cache Optimization

**Problem**: Attention keys and values grow linearly with sequence length

**Memory Formula**: `KV_cache = 2 × batch_size × seq_len × num_layers × hidden_dim × precision`

#### PagedAttention (vLLM)

**Concept**: Store KV cache in non-contiguous memory blocks (like OS paging)

**Benefits**:
- Near-zero waste (vs 20-40% waste)
- Dynamic memory allocation
- 2-4× higher throughput

**How It Works**:
```
Traditional: [█████████░░░░] (contiguous, wasteful)
PagedAttention: [███][██][████] (blocks, efficient)
```

#### Multi-Query Attention (MQA)

**Concept**: Share keys/values across attention heads

**Benefits**:
- Reduces KV cache size by num_heads
- Faster inference
- Used in: PaLM, Falcon

#### Grouped-Query Attention (GQA)

**Concept**: Share KV across groups of heads (compromise between MHA and MQA)

**Benefits**:
- Better quality than MQA
- More efficient than MHA
- Used in: LLaMA-2, Mistral

### 3. Batching Strategies

#### Static Batching

**Approach**: Wait for batch to fill, then process

**Problem**: Head-of-line blocking, padding waste

#### Continuous Batching

**Approach**: Add/remove sequences dynamically as they complete

**Benefits**:
- No waiting for batch to complete
- Better GPU utilization
- Higher throughput

**Example**: vLLM, TensorRT-LLM

#### Dynamic Batching

**Approach**: Adjust batch size based on load

### 4. Speculative Decoding

**Concept**: Use small "draft" model to predict tokens, verify with large model

**Process**:
1. Small model generates K tokens quickly
2. Large model verifies in parallel
3. Accept correct predictions, reject wrong ones

**Benefits**:
- 2-3× faster generation
- No quality loss
- Best for long sequences

**Trade-off**: Requires loading two models

---

## Inference Frameworks & Engines

### 1. vLLM

**Philosophy**: High-throughput LLM serving with PagedAttention

**Key Features**:

| Feature | Benefit | Technical |
|---------|---------|-----------|
| **PagedAttention** | 2-4× higher throughput | Efficient KV cache |
| **Continuous Batching** | Better GPU utilization | Dynamic scheduling |
| **Tensor Parallelism** | Scale to multiple GPUs | Model sharding |
| **Quantization** | Lower memory | AWQ, GPTQ support |

**Architecture**:
```
Requests → Scheduler → GPU Engine (PagedAttention) → Response
              ↓
         KV Cache Manager
```

**Code Example**:
```python
from vllm import LLM, SamplingParams

# Initialize
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Generate
prompts = ["Tell me about AI", "What is ML?"]
outputs = llm.generate(prompts, SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
))
```

**Performance**: 10-20× higher throughput than HuggingFace Transformers

**Best For**:
- High-throughput serving
- Multiple concurrent users
- Production deployments

**Limitations**:
- GPU only
- Less flexible than Transformers

### 2. TensorRT-LLM (NVIDIA)

**Philosophy**: Maximum performance on NVIDIA GPUs

**Key Features**:

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Kernel Fusion** | Combine operations | Faster execution |
| **In-flight Batching** | Continuous batching | Higher throughput |
| **FP8 Support** | H100 optimization | 2× faster |
| **Multi-GPU** | Tensor/Pipeline parallelism | Scale large models |

**Performance**: 
- 2-3× faster than PyTorch
- 4-8× faster than vanilla Transformers

**Code Example**:
```python
import tensorrt_llm

# Build engine (once)
tensorrt_llm.build(
    model="llama-7b",
    output_dir="llama-trt",
    max_batch_size=8
)

# Load and run
engine = tensorrt_llm.load("llama-trt")
outputs = engine.generate(prompts)
```

**Best For**:
- NVIDIA hardware (especially H100)
- Latency-critical applications
- Maximum performance needed

**Limitations**:
- NVIDIA only
- Complex setup
- Less community support

### 3. Text Generation Inference (TGI) - HuggingFace

**Philosophy**: Production-ready serving for HuggingFace models

**Key Features**:

| Feature | Description |
|---------|-------------|
| **Streaming** | Token-by-token response |
| **Continuous Batching** | Dynamic batching |
| **Quantization** | GPTQ, AWQ, bitsandbytes |
| **Flash Attention** | Optimized attention |
| **Safetensors** | Fast model loading |

**Deployment**:
```bash
docker run -p 8080:80 \
  -v $PWD/data:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-2-7b-hf
```

**Best For**:
- HuggingFace ecosystem
- Quick production deployment
- Docker-based serving

### 4. llama.cpp

**Philosophy**: CPU-friendly LLM inference

**Key Features**:

| Feature | Description | Benefit |
|---------|-------------|---------|
| **CPU Optimized** | SIMD, AVX instructions | Fast on CPU |
| **Quantization** | GGUF format (2-8 bit) | Low memory |
| **Apple Silicon** | Metal acceleration | Fast on Mac |
| **No Dependencies** | C++ only | Easy deployment |

**Use Cases**:
- Edge devices
- CPU-only servers
- Development machines
- Apple Silicon (M1/M2/M3)

**Quantization Formats (GGUF)**:
- Q2_K: 2.6 GB (very lossy)
- Q4_K_M: 4.1 GB (recommended)
- Q5_K_M: 5.0 GB (high quality)
- Q8_0: 7.2 GB (minimal loss)

### 5. CTranslate2

**Philosophy**: Fast CPU/GPU inference with quantization

**Key Features**:
- 4× faster than PyTorch
- INT8/INT16 quantization
- Dynamic batching
- Multi-GPU support

**Best For**:
- Translation models
- Encoder-decoder architectures
- CPU deployments

### 6. DeepSpeed-Inference

**Philosophy**: High-performance inference with kernel optimization

**Key Features**:
- Custom CUDA kernels
- ZeRO-Inference (sharding)
- Tensor parallelism
- INT8 quantization

**Best For**:
- Very large models (70B+)
- Research deployments
- Microsoft ecosystem

---

## Framework Comparison

| Framework | Speed | Ease | GPU | CPU | Quantization | Best For |
|-----------|-------|------|-----|-----|--------------|----------|
| **vLLM** | ★★★★★ | ★★★★ | ✅ | ❌ | ✅ | Production serving |
| **TensorRT-LLM** | ★★★★★ | ★★ | ✅ | ❌ | ✅ | Max performance |
| **TGI** | ★★★★ | ★★★★★ | ✅ | ❌ | ✅ | HuggingFace models |
| **llama.cpp** | ★★★ | ★★★★★ | ✅ | ✅ | ✅ | CPU/Edge |
| **CTranslate2** | ★★★★ | ★★★★ | ✅ | ✅ | ✅ | Translation |
| **Transformers** | ★★ | ★★★★★ | ✅ | ✅ | ✅ | Development |

---

## Model Parallelism

### Why Parallelism?

**Problem**: Model too large for single GPU

| Model Size | GPUs Needed (A100 80GB) |
|------------|-------------------------|
| 7B | 1 |
| 13B | 1 |
| 70B | 2-4 |
| 175B | 4-8 |

### Types of Parallelism

#### 1. Tensor Parallelism

**Concept**: Split individual layers across GPUs

**How It Works**:
```
Input
  ↓
Layer [GPU 1 | GPU 2 | GPU 3] ← Split weights
  ↓
All-Reduce (combine results)
  ↓
Next Layer
```

**Pros**:
- Low communication overhead
- All GPUs always active

**Cons**:
- Requires fast GPU interconnect (NVLink)
- Limited by GPU count per node

**Use Case**: Models up to 70B on 2-4 GPUs

#### 2. Pipeline Parallelism

**Concept**: Split layers across GPUs (vertical split)

**How It Works**:
```
GPU 1: Layers 1-8
  ↓
GPU 2: Layers 9-16
  ↓
GPU 3: Layers 17-24
```

**Pros**:
- Can use slower interconnects
- Scales to more GPUs

**Cons**:
- Pipeline bubbles (idle time)
- Complex implementation

**Use Case**: Very large models (175B+)

#### 3. Data Parallelism

**Concept**: Replicate model, split data

**How It Works**:
```
Batch 1 → GPU 1 (full model)
Batch 2 → GPU 2 (full model)
Batch 3 → GPU 3 (full model)
```

**Pros**:
- Simple to implement
- High throughput

**Cons**:
- Requires model fits on single GPU
- Not for large models

**Use Case**: Increase throughput, not model capacity

---

## Deployment Architectures

### 1. Single GPU Serving

**Setup**:
```
Load Balancer
    ↓
Single GPU Server
```

**Specs**: 7B-13B models on A100/H100

**Pros**: Simple, low latency
**Cons**: Limited throughput

**Cost**: ~$1-2/hour (cloud)

### 2. Multi-GPU Single Node

**Setup**:
```
Load Balancer
    ↓
Server with 4-8 GPUs (Tensor Parallel)
```

**Specs**: 70B models on 4×A100

**Pros**: High throughput, low latency
**Cons**: Expensive single node

**Cost**: ~$10-20/hour (cloud)

### 3. Multi-Node Cluster

**Setup**:
```
Load Balancer
    ↓
Multiple Nodes (8+ GPUs each)
```

**Specs**: 175B+ models

**Pros**: Massive scale
**Cons**: Complex, high latency

**Cost**: $50+/hour

### 4. Serverless (API)

**Providers**: OpenAI, Anthropic, Together.ai, Replicate

**Pros**: 
- Zero infrastructure
- Pay per token
- Auto-scaling

**Cons**:
- Vendor lock-in
- Less control
- Potential higher cost at scale

**When to Use**: Prototyping, low-medium volume

### 5. Hybrid Architecture

**Setup**:
```
Small requests → Serverless API
Large/specialized → Self-hosted
```

**Benefits**: Cost optimization, flexibility

---

## Optimization Strategies

### Memory Optimization

| Technique | Savings | Quality Loss | Complexity |
|-----------|---------|--------------|------------|
| **Quantization (INT8)** | 4× | 1-2% | Low |
| **KV Cache Optimization** | 20-40% | 0% | Medium |
| **Flash Attention** | 20-30% | 0% | Low |
| **Gradient Checkpointing** | 50% | 0% (training) | Low |

### Latency Optimization

| Technique | Improvement | Trade-off |
|-----------|-------------|-----------|
| **Tensor Parallelism** | 2-4× | Requires multiple GPUs |
| **Speculative Decoding** | 2-3× | Extra model memory |
| **Prompt Caching** | 10-100× | Stale responses |
| **Kernel Fusion** | 20-30% | Framework-specific |

### Throughput Optimization

| Technique | Improvement | How |
|-----------|-------------|-----|
| **Continuous Batching** | 2-4× | vLLM, TGI |
| **PagedAttention** | 2-4× | vLLM |
| **Larger Batches** | 2-5× | More memory needed |
| **Multi-GPU** | N× | Data parallelism |

---

## Production Best Practices

### Infrastructure

| Component | Recommendation | Why |
|-----------|----------------|-----|
| **GPU** | A100 80GB or H100 | Best price/performance |
| **Interconnect** | NVLink for tensor parallel | Low latency |
| **Storage** | NVMe SSD | Fast model loading |
| **Network** | 100Gbps+ | High throughput |
| **Memory** | 2× GPU memory in RAM | Safety buffer |

### Monitoring

| Metric | Tool | Alert Threshold |
|--------|------|-----------------|
| **Latency (p95)** | Prometheus | > 2s |
| **GPU Utilization** | nvidia-smi | < 70% |
| **Throughput** | Custom | < target |
| **Error Rate** | Prometheus | > 0.1% |
| **Queue Depth** | Custom | > 100 |
| **Memory Usage** | nvidia-smi | > 90% |

### Cost Optimization

| Strategy | Savings | Implementation |
|----------|---------|----------------|
| **Spot Instances** | 60-70% | Fault-tolerant system |
| **Auto-scaling** | 30-50% | Scale down off-peak |
| **Quantization** | 50% | INT8/INT4 |
| **Prompt Caching** | 40-80% | Cache common prompts |
| **Batch Requests** | 20-40% | Higher throughput |

---

## Deployment Checklist

### Pre-Deployment

- [ ] Model quantized (INT8 minimum)
- [ ] Benchmark latency (p50, p95, p99)
- [ ] Benchmark throughput (tokens/sec)
- [ ] Load testing completed
- [ ] Memory profiled (no OOM at peak)
- [ ] Error handling tested
- [ ] Monitoring setup (Prometheus, Grafana)
- [ ] Logging configured
- [ ] Auto-scaling policies defined

### Production

- [ ] Health checks active
- [ ] Circuit breakers configured
- [ ] Rate limiting enabled
- [ ] Caching strategy implemented
- [ ] Backup instances ready
- [ ] Rollback plan tested
- [ ] On-call rotation setup
- [ ] Documentation complete

---

## Interview Questions & Answers

### Q: "What is the main bottleneck in LLM inference?"

**Answer**: **Memory bandwidth**, not compute. 

**Explanation**: 
- LLM inference is memory-bound, not compute-bound
- Each token generation requires loading all model weights from memory
- Auto-regressive generation is sequential (can't parallelize output)

**Formula**: `Latency ≈ Model_Size / Memory_Bandwidth`

**Solutions**:
- Quantization (reduce model size)
- KV cache optimization (reduce memory access)
- Batching (amortize memory loads)

### Q: "Explain PagedAttention and why it matters"

**Answer**: PagedAttention (from vLLM) stores KV cache in non-contiguous memory blocks, like OS virtual memory.

**Traditional Problem**:
- Pre-allocate contiguous memory for max sequence length
- 20-40% wasted due to variable lengths
- Memory fragmentation

**PagedAttention Solution**:
- Allocate memory in small blocks (pages)
- Near-zero waste
- Dynamic allocation

**Impact**: 2-4× higher throughput by fitting more requests in memory.

### Q: "INT8 vs INT4 quantization trade-offs?"

**Answer**:

**INT8**:
- Memory: 4× reduction vs FP32
- Quality: 1-2% accuracy loss
- Speed: 2-3× faster
- Use case: Production standard

**INT4**:
- Memory: 8× reduction
- Quality: 2-5% accuracy loss
- Speed: 3-4× faster
- Use case: Resource-constrained, quality acceptable

**Recommendation**: Start with INT8, try INT4 if memory/cost critical.

### Q: "How do you choose between vLLM, TensorRT-LLM, and TGI?"

**Answer**:

**Choose vLLM when**:
- Need high throughput (many concurrent users)
- Want easy setup with great defaults
- GPU availability (works on various GPUs)

**Choose TensorRT-LLM when**:
- Have NVIDIA GPUs (especially H100)
- Latency is critical
- Can invest in complex setup

**Choose TGI when**:
- Using HuggingFace models
- Want Docker deployment
- Streaming responses needed

**My default**: vLLM for most production use cases.

### Q: "What is continuous batching?"

**Answer**: Dynamic batching where sequences are added/removed as they complete, rather than waiting for entire batch.

**Traditional batching**:
```
Batch: [Seq1, Seq2, Seq3]
Wait for ALL to complete → Process new batch
Problem: Long sequences block short ones
```

**Continuous batching**:
```
[Seq1, Seq2, Seq3]
Seq2 done → Add Seq4
[Seq1, Seq3, Seq4]
No blocking, better GPU utilization
```

**Impact**: 2-4× higher throughput. Used in vLLM, TGI, TensorRT-LLM.

### Q: "Tensor parallelism vs pipeline parallelism?"

**Answer**:

**Tensor Parallelism**:
- Split layers horizontally across GPUs
- All GPUs work on same layer
- Low latency, requires fast interconnect (NVLink)
- Best for: 2-8 GPUs, 70B models

**Pipeline Parallelism**:
- Split layers vertically (layer 1-10 GPU1, 11-20 GPU2)
- Sequential execution
- Pipeline bubbles (idle time)
- Best for: 8+ GPUs, 175B+ models

**In practice**: Often combined for very large models.

### Q: "How do you optimize inference costs?"

**Answer**: Multi-pronged approach:

1. **Model-level**:
   - Quantization (INT8 = 50% savings)
   - Smaller model if acceptable (7B vs 70B)
   - Distillation

2. **Infrastructure**:
   - Spot instances (60-70% cheaper)
   - Auto-scaling (scale down off-peak)
   - Right-size GPUs (don't over-provision)

3. **Request-level**:
   - Prompt caching (80% hit rate = 80% savings)
   - Batch requests
   - Short max_tokens limits

4. **Architecture**:
   - Hybrid (cheap model + expensive model)
   - Serverless for low volume

**Example**: 7B INT8 quantized on spot instances with caching = 85% cost reduction vs 70B FP16 on-demand.

### Q: "What causes high latency and how to fix?"

**Answer**:

**Common Causes**:

1. **Large model**: Use smaller model or quantization
2. **Long prompts**: Prompt compression or caching
3. **Cold start**: Keep models warm, use model caching
4. **Single GPU**: Add tensor parallelism
5. **Poor batching**: Use continuous batching
6. **Slow storage**: Use NVMe SSD, faster model loading

**Optimization Priority**:
1. Profile to find bottleneck
2. Quantize model (quick win)
3. Enable continuous batching
4. Add tensor parallelism if needed
5. Optimize prompt handling

---

## Advanced Topics

### Flash Attention

**Problem**: Standard attention is O(N²) in memory

**Solution**: Compute attention in blocks, never materialize full matrix

**Benefits**:
- 3-5× faster training
- 2-3× faster inference
- Enables longer contexts

**Adoption**: Built into vLLM, TGI, most frameworks

### Speculative Decoding

**Architecture**:
```
Draft Model (small, fast) generates K tokens
    ↓
Target Model (large, accurate) verifies in parallel
    ↓
Accept correct, reject wrong, continue
```

**Benefits**: 2-3× faster with no quality loss

**Trade-off**: 2× memory (two models)

### Multi-LoRA Serving

**Concept**: Serve base model + multiple LoRA adapters efficiently

**Use Case**: Personalization, multi-tenant

**Implementation**: vLLM, TGI support

---

## Key Takeaways for Interviews

1. **Memory bandwidth is bottleneck**: Not compute, optimize memory access
2. **Quantization is essential**: INT8 minimum for production (4× memory savings)
3. **vLLM for production**: PagedAttention + continuous batching = 10-20× faster
4. **Continuous batching critical**: 2-4× throughput improvement
5. **KV cache dominates memory**: Optimize with PagedAttention
6. **Tensor parallelism for large models**: 70B+ needs 2-4 GPUs
7. **Monitor everything**: Latency, throughput, GPU util, errors
8. **Cost optimization**: Quantization + spot instances + caching
9. **Framework choice**: vLLM (default), TensorRT-LLM (max perf), TGI (HF ecosystem)
10. **Start simple, optimize**: Profile first, then optimize bottlenecks