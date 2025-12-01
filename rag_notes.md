# RAG (Retrieval-Augmented Generation) - Quick Reference Guide

## Overview
**RAG** combines retrieval systems with large language models to provide accurate, grounded responses using external knowledge. Instead of relying solely on the LLM's training data, RAG retrieves relevant information and uses it to generate responses.

**Core Problem Solved**: LLMs have knowledge cutoffs, can hallucinate, and don't know your private data. RAG provides current, accurate, verifiable information.

---

## RAG vs Alternatives

| Approach | How It Works | Pros | Cons | Best For |
|----------|--------------|------|------|----------|
| **RAG** | Retrieve → Augment → Generate | - Current info<br>- Citable sources<br>- Cost-effective | - Retrieval quality matters<br>- Latency overhead | Most production use cases |
| **Fine-tuning** | Train model on your data | - Deep knowledge<br>- Fast inference | - Expensive<br>- Static knowledge<br>- Overfitting risk | Domain-specific language, style |
| **Long Context** | Put all docs in prompt | - Simple<br>- No retrieval | - Expensive tokens<br>- Lost-in-middle problem | Small, stable knowledge bases |
| **Prompt Engineering** | Clever prompts only | - Free<br>- Fast | - Limited by context<br>- No external data | General tasks |
| **Agents** | Dynamic tool use | - Flexible<br>- Multi-step | - Complex<br>- Slower | Complex queries needing reasoning |

---

## RAG Pipeline Architecture

### Basic Pipeline (5 Stages)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  1. Ingest  │ --> │  2. Index    │ --> │  3. Store   │
│  Documents  │     │  (Embeddings)│     │ (Vector DB) │
└─────────────┘     └──────────────┘     └─────────────┘
                                                 ↓
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ 5. Generate │ <-- │  4. Retrieve │ <-- │    Query    │
│  (LLM)      │     │  (Search)    │     │             │
└─────────────┘     └──────────────┘     └─────────────┘
```

### Stage Breakdown

| Stage | Components | Input | Output | Purpose |
|-------|------------|-------|--------|---------|
| **1. Ingest** | Document loaders | Raw files (PDF, HTML, etc.) | Text | Extract content |
| **2. Chunk** | Text splitters | Text | Chunks (e.g., 512 tokens) | Create searchable units |
| **3. Embed** | Embedding models | Chunks | Vectors | Convert to numeric representation |
| **4. Store** | Vector database | Vectors + metadata | Indexed data | Enable similarity search |
| **5. Retrieve** | Retriever | Query vector | Top-K chunks | Find relevant context |
| **6. Generate** | LLM | Query + context | Answer | Produce final response |

---

## Key Components Deep Dive

### 1. Document Loading

| Loader Type | Handles | Library | Use Case |
|-------------|---------|---------|----------|
| **PDF Loader** | PDF files | PyPDF2, pdfplumber | Reports, papers |
| **Web Loader** | URLs, HTML | BeautifulSoup, Selenium | Documentation, articles |
| **CSV Loader** | Structured data | pandas | Tables, databases |
| **Code Loader** | Source code | tree-sitter | Code repositories |
| **API Loader** | External APIs | requests | Live data |
| **Database Loader** | SQL databases | SQLAlchemy | Enterprise data |

### 2. Text Chunking Strategies

| Strategy | How It Works | Pros | Cons | Best For |
|----------|--------------|------|------|----------|
| **Fixed Size** | Split every N tokens | Simple, predictable | Breaks context | General purpose |
| **Recursive** | Split by hierarchy (¶ → sentence → word) | Preserves context | More complex | Natural text |
| **Semantic** | Split by meaning | Best context | Slow, model-dependent | High quality needs |
| **Document Structure** | Use headers, sections | Logical chunks | Needs structured docs | Technical docs |
| **Overlap** | Chunks share boundaries | Prevents context loss | More storage | Critical applications |

**Chunking Parameters:**

| Parameter | Typical Value | Impact |
|-----------|---------------|--------|
| **Chunk Size** | 256-1024 tokens | Larger = more context, worse retrieval |
| **Overlap** | 10-20% of size | Prevents boundary issues |
| **Separators** | ["\n\n", "\n", " "] | How to split text |

### 3. Embedding Models

| Model | Dimensions | Speed | Quality | Cost | Best For |
|-------|------------|-------|---------|------|----------|
| **OpenAI text-embedding-3-small** | 1536 | Fast | Good | Low | General purpose |
| **OpenAI text-embedding-3-large** | 3072 | Medium | Best | Medium | High accuracy needs |
| **Cohere embed-v3** | 1024 | Fast | Good | Low | Multilingual |
| **Sentence-Transformers** | 384-768 | Fast | Good | Free | Open source, local |
| **BGE** | 768-1024 | Fast | Very Good | Free | SOTA open source |
| **E5** | 768 | Fast | Good | Free | General purpose |
| **Instructor** | 768 | Medium | Good | Free | Task-specific |

**Key Considerations:**
- **Dimensions**: Higher = more nuanced, but slower + more storage
- **Domain**: Some models better for code, legal, medical
- **Multilingual**: Some support 100+ languages
- **Cost**: API calls vs local models

### 4. Vector Databases

| Database | Type | Strengths | Limitations | Best For |
|----------|------|-----------|-------------|----------|
| **Pinecone** | Managed | Fully managed, scalable | Paid only | Production, scale |
| **Weaviate** | Open source/Managed | GraphQL, hybrid search | Complex setup | Feature-rich needs |
| **Chroma** | Embedded/Server | Easy setup, local-first | Less mature | Development, small apps |
| **Qdrant** | Open source/Cloud | Fast, filters, Rust | Newer ecosystem | Performance-critical |
| **Milvus** | Open source/Cloud | Highly scalable | Complex deployment | Large scale |
| **FAISS** | Library | Facebook-backed, fast | In-memory, no server | Research, local |
| **pgvector** | PostgreSQL extension | SQL familiarity | Limited features | Existing Postgres users |
| **ElasticSearch** | Search engine | Mature, hybrid search | Not vector-native | Existing ES users |

**Selection Criteria:**

| Need | Recommendation |
|------|----------------|
| **Quick prototype** | Chroma (embedded) |
| **Production, managed** | Pinecone, Weaviate Cloud |
| **Self-hosted** | Qdrant, Milvus |
| **Existing Postgres** | pgvector |
| **Hybrid search** | Weaviate, ElasticSearch |
| **Budget-conscious** | Chroma, Qdrant (self-hosted) |

### 5. Retrieval Methods

| Method | How It Works | Pros | Cons | Use Case |
|--------|--------------|------|------|----------|
| **Semantic Search** | Vector similarity (cosine, dot product) | Meaning-based | Misses exact terms | General queries |
| **Keyword Search** | BM25, TF-IDF | Exact matches | Misses semantics | Specific terms, names |
| **Hybrid Search** | Combine semantic + keyword | Best of both | More complex | Production systems |
| **MMR** | Maximum Marginal Relevance | Diverse results | Slower | Avoid redundancy |
| **Reranking** | Two-stage (retrieve → rerank) | Higher precision | Extra latency | High quality needs |
| **Metadata Filtering** | Filter by attributes | Precise scoping | Requires metadata | Multi-tenant, time-based |

---

## Advanced RAG Techniques

### 1. Query Transformation

| Technique | Description | Benefit | Example |
|-----------|-------------|---------|---------|
| **Query Expansion** | Generate similar queries | Better recall | "Python errors" → add "Python exceptions", "debugging" |
| **Query Decomposition** | Break complex into sub-queries | Handle complexity | "Compare X and Y" → "What is X?", "What is Y?" |
| **HyDE** | Generate hypothetical answer, embed it | Better matching | Query → Fake answer → Retrieve similar |
| **Step-back Prompting** | Ask broader question first | More context | "Fix bug X" → "How does system Y work?" |

### 2. Retrieval Optimization

| Technique | Description | Benefit | When to Use |
|-----------|-------------|---------|-------------|
| **Parent-Child Chunks** | Retrieve small, return large | Precision + context | Long documents |
| **Sentence Window** | Retrieve sentence, return paragraph | Balance precision/context | Narrative text |
| **Auto-merging** | Merge adjacent retrieved chunks | Coherent context | Related sections |
| **Recursive Retrieval** | Multi-level retrieval | Handle hierarchy | Structured docs |

### 3. Context Enhancement

| Technique | Description | Trade-off | Use Case |
|-----------|-------------|-----------|----------|
| **Reranking** | Score and reorder results | Latency vs quality | High precision needs |
| **Contextual Compression** | Remove irrelevant parts | Tokens vs accuracy | Long contexts |
| **Fusion Retrieval** | Multiple retrievers, merge | Complexity vs recall | Critical applications |
| **Hypothetical Document** | Generate ideal doc, retrieve similar | Cost vs quality | Ambiguous queries |

### 4. Generation Improvements

| Technique | Description | Benefit |
|-----------|-------------|---------|
| **Citation Generation** | Return source references | Verifiability |
| **Self-RAG** | LLM decides when to retrieve | Efficiency |
| **FLARE** | Forward-looking active retrieval | Handles multi-hop |
| **Chain-of-Verification** | Generate → verify → correct | Reduces hallucination |

---

## RAG Evaluation

### Key Metrics

| Metric | What It Measures | How to Measure | Target |
|--------|------------------|----------------|--------|
| **Retrieval Precision** | % relevant in top-K | Manual review / LLM judge | >80% |
| **Retrieval Recall** | % of all relevant retrieved | Manual review | >90% |
| **Context Relevance** | How relevant is context to query | LLM judge | >4/5 |
| **Answer Faithfulness** | Answer grounded in context | LLM judge / NLI | >90% |
| **Answer Relevance** | Answers the actual question | LLM judge | >4/5 |
| **Hallucination Rate** | % fabricated information | Human review | <5% |
| **Latency** | Time to answer | System metrics | <2s p95 |

### Evaluation Frameworks

| Framework | Features | Best For |
|-----------|----------|----------|
| **RAGAS** | Automated LLM-based metrics | Quick evaluation |
| **TruLens** | Observability + evaluation | Production monitoring |
| **DeepEval** | Unit tests for LLM apps | CI/CD integration |
| **LangSmith** | Datasets + human review | LangChain users |
| **Custom** | Your domain logic | Specific needs |

---

## Common Architectures

### 1. Basic RAG

```python
# Pseudocode
query = "What is the refund policy?"
query_embedding = embed(query)
docs = vector_db.search(query_embedding, top_k=5)
context = format(docs)
prompt = f"Context: {context}\n\nQuestion: {query}"
answer = llm.generate(prompt)
```

**Pros**: Simple, fast
**Cons**: Limited quality control

### 2. Advanced RAG with Reranking

```
Query → Retrieve (top 50) → Rerank (top 5) → LLM → Answer
```

**Pros**: Better precision
**Cons**: Extra latency

### 3. Multi-Query RAG

```
Query → Generate 3 variants → Retrieve for each → Merge → Rerank → LLM
```

**Pros**: Better recall
**Cons**: More API calls

### 4. Agentic RAG

```
Query → Agent decides → [Retrieve | Web Search | Calculator] → Agent → Answer
```

**Pros**: Handles complex queries
**Cons**: Non-deterministic, slower

### 5. Corrective RAG (CRAG)

```
Query → Retrieve → Judge relevance → [Use | Web Search | Ignore] → Generate
```

**Pros**: Self-correcting
**Cons**: Complex logic

---

## RAG Problems & Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| **Poor Retrieval** | Bad embeddings, wrong chunks | Better chunking, hybrid search, reranking |
| **Hallucination** | LLM ignores context | Stronger prompts, citation requirement |
| **Outdated Info** | Stale index | Regular re-indexing, timestamp filtering |
| **Missing Info** | Not in knowledge base | Fallback to web search, "I don't know" |
| **Slow Retrieval** | Large index, inefficient DB | Optimize index, use filters, caching |
| **Context Window** | Too many chunks | Compression, summarization, parent-child |
| **Cost** | Too many tokens | Smaller chunks, fewer retrievals |
| **No Citation** | Can't verify sources | Return metadata, force citation format |

---

## Production Considerations

### Architecture Decisions

| Aspect | Options | Recommendation |
|--------|---------|----------------|
| **Embedding** | API vs Self-hosted | API for simplicity, self-host for scale |
| **Vector DB** | Managed vs Self-hosted | Managed for startups, self-host at scale |
| **Chunking** | Fixed vs Semantic | Start fixed, upgrade to semantic if needed |
| **Retrieval** | Semantic only vs Hybrid | Hybrid for production |
| **Reranking** | Yes/No | Yes for quality-critical apps |

### Scaling Strategies

| Challenge | Solution |
|-----------|----------|
| **Large corpus** | Sharding, distributed vector DB |
| **High QPS** | Caching, load balancing |
| **Update frequency** | Incremental indexing |
| **Multi-tenancy** | Metadata filtering per tenant |
| **Cost control** | Smaller models, caching, batching |

### Monitoring

| Metric | Why | Alert Threshold |
|--------|-----|-----------------|
| **Retrieval latency** | User experience | p95 > 500ms |
| **Generation latency** | User experience | p95 > 2s |
| **Error rate** | Reliability | >1% |
| **Relevance score** | Quality | Avg < 0.7 |
| **User feedback** | Satisfaction | <4/5 stars |

---

## Code Examples

### Basic RAG Setup (LangChain)

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 1. Load
loader = PyPDFLoader("docs.pdf")
documents = loader.load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)

# 3. Embed + Store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. Retrieve + Generate
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

# Query
answer = qa_chain.run("What is the main topic?")
```

### Advanced: Hybrid Search + Reranking

```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# Semantic retriever
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Keyword retriever
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 20

# Ensemble (hybrid)
hybrid_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

# Reranker
reranker = CohereRerank(top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid_retriever
)

# Use in QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever
)
```

---

## Interview Questions & Answers

### Q: "Explain the RAG pipeline"

**Answer**: RAG has two phases:
1. **Indexing** (offline): Load documents → Split into chunks → Create embeddings → Store in vector DB
2. **Retrieval** (online): User query → Embed query → Search vector DB → Retrieve top-K chunks → Augment prompt with context → Generate answer with LLM

**Key point**: Separates knowledge (in vector DB) from reasoning (LLM), enabling updates without retraining.

### Q: "RAG vs Fine-tuning: when to use each?"

**Answer**:

**Use RAG when**:
- Need current/updated information
- Want citable sources
- Have large, changing knowledge base
- Cost-conscious (fine-tuning expensive)

**Use Fine-tuning when**:
- Need specific writing style/format
- Domain-specific terminology
- Want faster inference (no retrieval)
- Knowledge is stable

**Use Both when**:
- Need domain expertise (fine-tune) + current facts (RAG)
- Example: Medical chatbot (fine-tune on medical reasoning, RAG for latest research)

### Q: "How do you handle 'document not found' scenarios?"

**Answer**:
1. **Confidence scoring**: Check retrieval scores, if too low → "I don't have information"
2. **Fallback mechanisms**: Web search, call another API
3. **Graceful degradation**: "Based on available docs, I can't find this. Can you rephrase?"
4. **Self-RAG**: LLM decides if retrieved docs are sufficient before answering
5. **User feedback**: "Was this helpful?" to catch misses

### Q: "How do you evaluate RAG quality?"

**Answer**:
- **Component-level**: Retrieval precision/recall (are right docs retrieved?)
- **End-to-end**: Answer relevance, faithfulness to context
- **Methods**: Human review (gold standard), LLM-as-judge (automated), user feedback
- **Tools**: RAGAS framework, LangSmith datasets
- **Production**: Monitor user thumbs up/down, conversation drops

### Q: "Chunk size: how do you decide?"

**Answer**:
**Trade-offs**:
- **Small chunks (256-512)**: Better retrieval precision, but may lack context
- **Large chunks (1024-2048)**: More context, but worse retrieval, expensive tokens

**Approach**:
1. Start with 512-1024 tokens, 10-20% overlap
2. Test on representative queries
3. Measure retrieval precision
4. Adjust based on document type (tweets vs research papers)
5. Consider parent-child: retrieve small, return large

### Q: "How do you prevent hallucination in RAG?"

**Solutions**:
1. **Prompt engineering**: "Only use provided context. If answer not in context, say 'I don't know'"
2. **Citation requirement**: Force model to cite sources
3. **Confidence scoring**: Return relevance scores, filter low-confidence
4. **Verification**: Second LLM call to verify answer against context
5. **Structured output**: JSON schema enforcement
6. **Human-in-loop**: Flag uncertain answers for review

---

## Advanced Topics

### Graph RAG

Instead of flat chunks, use knowledge graphs:
- **Nodes**: Entities (people, places, concepts)
- **Edges**: Relationships
- **Query**: Traverse graph for connected information
- **Benefit**: Better for multi-hop reasoning

### Multi-modal RAG

Retrieve and reason over multiple modalities:
- **Text + Images**: Document understanding
- **Text + Tables**: Financial reports
- **Text + Code**: Technical documentation
- **Example**: "Show me the chart about Q3 revenue" → retrieve image + describe

### Temporal RAG

Handle time-sensitive information:
- **Timestamp chunks**: Filter by date
- **Version control**: Track document changes
- **Recency weighting**: Prefer newer information
- **Use case**: News, legal documents

---

## Tools & Libraries

| Category | Tools |
|----------|-------|
| **Full Stack** | LangChain, LlamaIndex, Haystack |
| **Vector DBs** | Pinecone, Weaviate, Chroma, Qdrant |
| **Embeddings** | OpenAI, Cohere, Sentence-Transformers |
| **Evaluation** | RAGAS, TruLens, DeepEval |
| **Reranking** | Cohere Rerank, Cross-encoders |
| **Monitoring** | LangSmith, Weights & Biases |

---

## Key Takeaways

1. **RAG = Retrieval + Augmentation + Generation**: Three-stage pipeline
2. **Chunking matters**: Critical for retrieval quality
3. **Hybrid > Pure semantic**: Combine keyword + semantic search
4. **Reranking improves precision**: Two-stage retrieval
5. **Evaluation is essential**: Can't improve what you don't measure
6. **Citations build trust**: Always return sources
7. **RAG ≠ Silver bullet**: Use with fine-tuning, agents as needed
8. **Production requires optimization**: Caching, indexing, monitoring