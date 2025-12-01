# Vector Databases - Deep Dive Reference Guide

## Overview
**Vector Databases** are specialized databases optimized for storing, indexing, and searching high-dimensional vectors (embeddings). They enable semantic search, recommendation systems, and RAG applications by finding similar items based on meaning rather than exact matches.

**Core Problem Solved**: Traditional databases can't efficiently search by similarity in high-dimensional space. Vector databases make "find similar items" queries fast and scalable.

---

## Vectors & Embeddings Fundamentals

### What are Embeddings?

| Concept | Description | Example |
|---------|-------------|---------|
| **Embedding** | Numerical representation of data | Text → [0.2, -0.5, 0.8, ...] |
| **Dimensions** | Number of values in vector | 384, 768, 1536, 3072 |
| **Semantic Meaning** | Similar items → similar vectors | "cat" and "kitten" have close vectors |
| **Dense Vectors** | Most values are non-zero | [0.2, -0.5, 0.8, 0.1, ...] |

### Embedding Generation

| Source | Model | Dimensions | Use Case |
|--------|-------|------------|----------|
| **Text** | OpenAI text-embedding-3-small | 1536 | General text search |
| **Text** | Sentence-BERT | 384-768 | Open source text |
| **Text** | BGE, E5 | 768-1024 | SOTA open source |
| **Images** | CLIP | 512 | Image search, multimodal |
| **Code** | CodeBERT | 768 | Code search |
| **Audio** | Wav2Vec | 768 | Audio similarity |

---

## How Vector Databases Work

### Core Concepts

| Concept | Description | Analogy |
|---------|-------------|----------|
| **Vector** | Point in N-dimensional space | GPS coordinates, but 100s of dimensions |
| **Similarity** | Distance between vectors | How "close" two points are |
| **Index** | Data structure for fast search | Like a book index, but for vectors |
| **Query** | Find K nearest neighbors | "Find 10 most similar items" |
| **Metadata** | Additional attributes | Tags, dates, categories |

### Similarity Metrics

| Metric | Formula | Range | When to Use | Properties |
|--------|---------|-------|-------------|------------|
| **Cosine Similarity** | cos(θ) = A·B / (‖A‖‖B‖) | [-1, 1] | Text, normalized vectors | Ignores magnitude |
| **Euclidean Distance** | √Σ(aᵢ-bᵢ)² | [0, ∞] | Images, coordinates | Considers magnitude |
| **Dot Product** | A·B = Σ(aᵢ×bᵢ) | [-∞, ∞] | Pre-normalized vectors | Fastest computation |
| **Manhattan Distance** | Σ\|aᵢ-bᵢ\| | [0, ∞] | Sparse vectors | L1 norm |

**Most Common**: Cosine similarity for text (measures angle, not magnitude)

### Search Process

```
1. Query Input: "machine learning tutorials"
2. Embedding: [0.1, -0.3, 0.7, ...] (1536 dims)
3. Index Search: Find K nearest vectors
4. Similarity Scoring: Calculate distances
5. Ranking: Sort by similarity
6. Results: Top-K most similar items
7. Metadata Filter: Apply constraints (optional)
```

---

## Vector Indexing Algorithms

### Algorithm Comparison

| Algorithm | Type | Speed | Accuracy | Memory | Best For |
|-----------|------|-------|----------|--------|----------|
| **Flat (Brute Force)** | Exact | Slow | 100% | High | Small datasets (<10K) |
| **IVF (Inverted File)** | Approximate | Fast | 90-95% | Medium | Medium datasets (10K-10M) |
| **HNSW** | Graph-based | Very Fast | 95-99% | High | Large datasets (10M+) |
| **LSH (Locality Sensitive)** | Hash-based | Fast | 80-90% | Low | Very large, lower accuracy OK |
| **Product Quantization** | Compression | Fast | 85-95% | Very Low | Memory-constrained |
| **ScaNN** | Google's hybrid | Very Fast | 95-99% | Medium | Production scale |

### Deep Dive: HNSW (Hierarchical Navigable Small World)

**How It Works**:
- Multi-layer graph structure
- Higher layers: Coarse navigation (long jumps)
- Lower layers: Fine navigation (precise search)
- Search: Start top layer, descend while getting closer

**Parameters**:
| Parameter | Description | Impact |
|-----------|-------------|--------|
| **M** | Connections per node | Higher = better recall, more memory |
| **efConstruction** | Build-time search depth | Higher = better quality, slower build |
| **efSearch** | Query-time search depth | Higher = better recall, slower search |

**Pros**:
- Excellent recall (95-99%)
- Fast queries (sub-millisecond)
- Good for high-dimensional data

**Cons**:
- High memory usage
- Slow indexing
- No easy updates (rebuild often needed)

### Deep Dive: IVF (Inverted File Index)

**How It Works**:
- Partition space into clusters (Voronoi cells)
- Each cluster has a centroid
- Search: Find nearest centroids, search within those clusters
- Trade-off: Search fewer clusters = faster but less accurate

**Parameters**:
| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| **nlist** | Number of clusters | √N to N/30 |
| **nprobe** | Clusters to search | 1-100 |

**Pros**:
- Good balance of speed and accuracy
- Memory efficient
- Scales well

**Cons**:
- Clustering overhead
- Sensitive to nprobe tuning

---

## Vector Database Comparison

### Major Players

| Database | Type | Strengths | Weaknesses | Best For |
|----------|------|-----------|------------|----------|
| **Pinecone** | Managed | Fully managed, scalable, easy | Paid only, vendor lock-in | Production, no ops team |
| **Weaviate** | Hybrid | Hybrid search, GraphQL, multimodal | Complex setup | Feature-rich apps |
| **Chroma** | Embedded/Server | Easy setup, local-first, Python | Less mature, limited scale | Development, small apps |
| **Qdrant** | OSS/Cloud | Fast (Rust), filters, growing | Newer ecosystem | Performance-critical |
| **Milvus** | OSS/Cloud | Highly scalable, CNCF | Complex deployment | Enterprise scale |
| **FAISS** | Library | Facebook-backed, research-grade | No server, in-memory | Research, prototyping |
| **pgvector** | Postgres Ext | SQL familiarity, ACID | Limited vector features | Existing Postgres users |
| **ElasticSearch** | Search Engine | Mature ecosystem, hybrid | Not vector-native | Existing ES users |
| **Redis** | In-memory | Ultra-fast, familiar | Memory limitations | Real-time apps |
| **Vespa** | Yahoo | ML features, production-ready | Steep learning curve | Advanced ML pipelines |

### Detailed Comparison

#### Pinecone

| Aspect | Details |
|--------|---------|
| **Architecture** | Managed, serverless |
| **Index Types** | Pod-based, Serverless |
| **Similarity** | Cosine, Euclidean, Dot Product |
| **Filtering** | Metadata filters (pre/post) |
| **Pricing** | Pay per query + storage |
| **Scale** | Billions of vectors |
| **Pros** | Zero ops, auto-scaling, consistent performance |
| **Cons** | Expensive at scale, vendor lock-in |
| **Best For** | Startups, production apps without DevOps |

**Use Case**: SaaS RAG application, need reliability without managing infrastructure

#### Weaviate

| Aspect | Details |
|--------|---------|
| **Architecture** | Open source, cloud managed option |
| **Index Types** | HNSW |
| **Similarity** | Cosine, Euclidean, Dot, Manhattan |
| **Special Features** | Hybrid search, GraphQL, modules |
| **Filtering** | Pre-filtering with where clause |
| **Scale** | Billions of vectors |
| **Pros** | Feature-rich, hybrid search, multimodal |
| **Cons** | Complex configuration |
| **Best For** | Apps needing hybrid semantic + keyword search |

**Use Case**: E-commerce with text + image search, complex filtering

#### Chroma

| Aspect | Details |
|--------|---------|
| **Architecture** | Embedded (SQLite-like) or client-server |
| **Index Types** | HNSW via hnswlib |
| **Similarity** | Cosine, L2, IP |
| **Filtering** | Metadata filtering |
| **Scale** | Millions of vectors |
| **Pros** | Easiest setup, great DX, local-first |
| **Cons** | Less mature, scaling limitations |
| **Best For** | Development, MVPs, small production apps |

**Use Case**: Prototyping RAG, personal projects, small production

#### Qdrant

| Aspect | Details |
|--------|---------|
| **Architecture** | Rust-based, OSS or cloud |
| **Index Types** | HNSW, custom implementations |
| **Similarity** | Cosine, Euclidean, Dot Product |
| **Special Features** | Powerful filtering, recommendations |
| **Filtering** | Pre-filtering, complex conditions |
| **Scale** | Billions of vectors |
| **Pros** | Fast (Rust), great filters, active development |
| **Cons** | Smaller ecosystem than Pinecone |
| **Best For** | Performance-critical, complex filtering needs |

**Use Case**: Recommendation system with complex user preferences

#### Milvus

| Aspect | Details |
|--------|---------|
| **Architecture** | Cloud-native, distributed |
| **Index Types** | Flat, IVF, HNSW, DiskANN |
| **Similarity** | L2, IP, Cosine, Hamming |
| **Special Features** | Multiple indexes per collection |
| **Scale** | 10+ billion vectors |
| **Pros** | Highly scalable, flexible, CNCF project |
| **Cons** | Complex deployment, steep learning curve |
| **Best For** | Enterprise scale, multi-tenancy |

**Use Case**: Large enterprise with billions of vectors, multiple teams

#### FAISS (Facebook AI Similarity Search)

| Aspect | Details |
|--------|---------|
| **Architecture** | C++ library (Python bindings) |
| **Index Types** | Flat, IVF, HNSW, PQ, many more |
| **Similarity** | L2, IP, custom |
| **Special Features** | GPU support, research-grade |
| **Scale** | Billions (in-memory) |
| **Pros** | Highly optimized, flexible, free |
| **Cons** | No server, no persistence, DIY everything |
| **Best For** | Research, custom solutions, understanding |

**Use Case**: Research experiments, building custom vector DB

#### pgvector

| Aspect | Details |
|--------|---------|
| **Architecture** | PostgreSQL extension |
| **Index Types** | IVF (ivfflat), HNSW |
| **Similarity** | L2, Cosine, IP |
| **Special Features** | SQL, ACID, joins with relational data |
| **Scale** | Millions of vectors |
| **Pros** | Familiar SQL, ACID guarantees, existing infra |
| **Cons** | Slower than purpose-built, limited features |
| **Best For** | Existing Postgres users, need ACID + vectors |

**Use Case**: E-commerce with products (relational) + embeddings (semantic search)

---

## Key Features Comparison

### Filtering Capabilities

| Database | Pre-filtering | Post-filtering | Complex Logic | Performance |
|----------|---------------|----------------|---------------|-------------|
| **Pinecone** | ✅ Yes | ✅ Yes | Limited | Good |
| **Weaviate** | ✅ Yes | ✅ Yes | GraphQL queries | Excellent |
| **Qdrant** | ✅ Yes | ✅ Yes | Must/should/not | Excellent |
| **Milvus** | ✅ Yes | ✅ Yes | Boolean expressions | Good |
| **Chroma** | ✅ Yes | ❌ No | Basic | Fair |
| **FAISS** | ❌ No | Manual | Manual | N/A |

**Pre-filtering**: Apply filters before vector search (faster, more accurate)
**Post-filtering**: Filter results after vector search (simpler but less accurate)

### Hybrid Search

| Database | Keyword Search | Hybrid Ranking | Implementation |
|----------|----------------|----------------|----------------|
| **Weaviate** | ✅ BM25 | ✅ Built-in | Native |
| **ElasticSearch** | ✅ Native | ✅ Script score | Native |
| **Qdrant** | ✅ Via payload | Manual | Custom |
| **Pinecone** | ❌ No | Manual | External |
| **Milvus** | ❌ No | Manual | External |

**Hybrid Search**: Combine semantic (vector) + keyword (BM25) search for best results

### Multi-tenancy

| Database | Approach | Performance | Best For |
|----------|----------|-------------|----------|
| **Pinecone** | Namespaces | Good | Simple isolation |
| **Weaviate** | Multi-tenancy feature | Excellent | Many tenants |
| **Qdrant** | Collections per tenant | Good | Moderate tenants |
| **Milvus** | Partitions | Excellent | Large scale |

---

## Performance Characteristics

### Query Latency (Approximate)

| Database | Dataset Size | Latency (p50) | Latency (p95) |
|----------|--------------|---------------|---------------|
| **Pinecone** | 1M vectors | 10-30ms | 50-100ms |
| **Weaviate** | 1M vectors | 10-50ms | 100-200ms |
| **Chroma** | 100K vectors | 5-20ms | 50-100ms |
| **Qdrant** | 1M vectors | 5-15ms | 30-60ms |
| **FAISS (GPU)** | 1M vectors | 1-5ms | 10-20ms |

*Note: Highly dependent on dimensionality, index type, hardware*

### Indexing Speed

| Database | Vectors/Second | Notes |
|----------|----------------|-------|
| **FAISS** | 10K-100K | CPU/GPU dependent |
| **Qdrant** | 5K-50K | Depends on config |
| **Pinecone** | Varies | Managed, auto-scaled |
| **Milvus** | 10K-100K | Distributed |
| **Chroma** | 1K-10K | Single node |

### Memory Consumption

| Index Type | Memory per Vector | Example (1M x 768D) |
|------------|-------------------|---------------------|
| **Flat** | dims × 4 bytes | 3 GB |
| **HNSW** | dims × 4 + graph | 4-6 GB |
| **IVF + PQ** | Compressed | 500 MB - 1 GB |
| **Product Quantization** | dims × 0.5 bytes | 384 MB |

---

## Advanced Concepts

### Product Quantization (PQ)

**Purpose**: Compress vectors to reduce memory

**How It Works**:
1. Split vector into sub-vectors: [a, b, c, d, e, f] → [a,b], [c,d], [e,f]
2. Cluster each sub-vector space
3. Replace sub-vectors with cluster IDs
4. Store codebook for decompression

**Trade-offs**:
- **Compression**: 8-32x smaller
- **Accuracy**: 85-95% recall
- **Speed**: Slightly slower queries

**Use Case**: Billions of vectors, memory-constrained

### Scalar Quantization

**Purpose**: Reduce precision (float32 → int8)

**Benefits**:
- 4x memory reduction
- Faster distance computation (integer arithmetic)
- 95-99% accuracy

**Trade-offs**: Minimal quality loss for most use cases

### Approximate Nearest Neighbor (ANN)

**Why ANN?**: Exact search is O(N) - too slow for large datasets

**Approaches**:
1. **Graph-based** (HNSW): Navigate graph structure
2. **Tree-based** (KD-tree): Partition space hierarchically
3. **Hashing** (LSH): Hash similar items to same buckets
4. **Clustering** (IVF): Search within clusters

**Accuracy vs Speed Trade-off**:
| Accuracy | Speed | Use Case |
|----------|-------|----------|
| 99% | Slow | Critical applications |
| 95% | Medium | Production standard |
| 90% | Fast | Real-time systems |
| 80% | Very Fast | Recommendations, suggestions |

### Sharding & Distribution

**Horizontal Sharding**:
- Split vectors across machines by ID
- Load balancing
- Each shard independently searchable

**Replication**:
- Multiple copies for fault tolerance
- Higher query throughput
- Consistency challenges

**Query Routing**:
- Broadcast to all shards → Merge results
- OR route by metadata filter

---

## Implementation Patterns

### Basic Vector Search (Python)

#### Pinecone
```python
import pinecone

# Initialize
pinecone.init(api_key="key")
index = pinecone.Index("my-index")

# Insert
index.upsert([
    ("id1", [0.1, 0.2, ...], {"category": "tech"}),
    ("id2", [0.3, 0.4, ...], {"category": "sports"})
])

# Query
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=10,
    filter={"category": "tech"}
)
```

#### Weaviate
```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# Insert
client.data_object.create(
    {"content": "Machine learning tutorial"},
    "Article",
    vector=[0.1, 0.2, ...]
)

# Query (hybrid search)
result = client.query.get("Article", ["content"])\
    .with_hybrid(query="machine learning", alpha=0.5)\
    .with_limit(10)\
    .do()
```

#### Chroma
```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_collection")

# Insert
collection.add(
    documents=["Machine learning tutorial"],
    metadatas=[{"category": "tech"}],
    ids=["id1"]
)

# Query
results = collection.query(
    query_texts=["AI tutorials"],
    n_results=10,
    where={"category": "tech"}
)
```

#### Qdrant
```python
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

# Insert
client.upsert(
    collection_name="my_collection",
    points=[
        {"id": 1, "vector": [0.1, 0.2, ...], "payload": {"category": "tech"}}
    ]
)

# Query with filter
results = client.search(
    collection_name="my_collection",
    query_vector=[0.1, 0.2, ...],
    limit=10,
    query_filter={"must": [{"key": "category", "match": {"value": "tech"}}]}
)
```

---

## Use Case Patterns

### Semantic Search

**Architecture**:
```
User Query → Embed → Vector DB Search → Return Results
```

**Optimization**:
- Pre-filter by metadata (date, category)
- Hybrid search for better recall
- Reranking for precision

### Recommendation System

**Architecture**:
```
User Profile/History → Embed → Find Similar Items → Filter/Rank → Recommend
```

**Considerations**:
- Collaborative filtering (user-user similarity)
- Content-based (item-item similarity)
- Cold start problem
- Diversity vs relevance

### RAG (Retrieval-Augmented Generation)

**Architecture**:
```
Query → Embed → Vector Search → Retrieved Docs → LLM Context → Generate Answer
```

**Vector DB Role**:
- Store document embeddings
- Fast retrieval (< 100ms)
- Metadata filtering (date, source, etc.)
- Hybrid search for accuracy

### Duplicate Detection

**Architecture**:
```
New Item → Embed → Search Similar → If similarity > threshold → Flag Duplicate
```

**Threshold Tuning**:
- 0.95+: Nearly identical
- 0.85-0.95: Very similar
- 0.70-0.85: Somewhat similar

### Anomaly Detection

**Architecture**:
```
New Data Point → Embed → Find K Nearest → If average distance > threshold → Anomaly
```

**Use Cases**:
- Fraud detection
- Network intrusion
- Manufacturing defects

---

## Production Considerations

### Scaling Strategies

| Challenge | Solution |
|-----------|----------|
| **Query Load** | Replicas, caching, load balancing |
| **Data Size** | Sharding, compression (PQ) |
| **Indexing Speed** | Batch inserts, async indexing |
| **Cost** | Quantization, cheaper indices |
| **Latency** | Pre-filtering, faster index (HNSW) |

### Monitoring Metrics

| Metric | Why | Alert Threshold |
|--------|-----|-----------------|
| **Query Latency (p95)** | User experience | > 200ms |
| **Recall** | Quality | < 90% |
| **Index Size** | Memory | > 80% capacity |
| **Query Rate** | Load | Near max capacity |
| **Error Rate** | Reliability | > 0.1% |

### Cost Optimization

| Strategy | Savings | Trade-off |
|----------|---------|-----------|
| **Quantization** | 50-75% | 5% accuracy loss |
| **Smaller embeddings** | 30-50% | 10% accuracy loss |
| **IVF instead of HNSW** | 30-50% memory | 5-10% slower |
| **Caching frequent queries** | 40-60% cost | Stale results |
| **Batch operations** | 20-40% | Higher latency |

### Data Management

**Updating Vectors**:
- **HNSW**: Expensive, often requires rebuild
- **IVF**: Moderate, can update incrementally
- **Strategy**: Batch updates, off-peak rebuilds

**Deleting Vectors**:
- Soft delete (mark deleted)
- Hard delete (rebuild index)
- Lazy deletion (periodic cleanup)

**Versioning**:
- Separate collections per version
- Gradual rollover
- A/B testing

---

## Interview Questions & Answers

### Q: "What is a vector database and why use it?"

**Answer**: A vector database is specialized for storing and searching high-dimensional vectors (embeddings). Unlike traditional databases that search by exact matches, vector databases find "similar" items using distance metrics like cosine similarity. They're essential for semantic search, RAG, recommendations - any use case where you need "find items like this" based on meaning, not keywords.

### Q: "How does HNSW work?"

**Answer**: HNSW (Hierarchical Navigable Small World) is a graph-based index with multiple layers. Think of it like a highway system:
- **Top layers**: Long-distance jumps (coarse navigation)
- **Bottom layers**: Local streets (fine-grained search)
- **Search**: Start at top, greedily move to closer nodes, descend layers

**Benefits**: 95-99% recall, very fast queries
**Trade-offs**: High memory, slow indexing, hard to update

### Q: "Pinecone vs self-hosted (Qdrant/Milvus)?"

**Answer**:

**Choose Pinecone when**:
- Need zero ops/maintenance
- Startup without DevOps team
- Want predictable performance
- Value time-to-market

**Choose self-hosted when**:
- Have DevOps expertise
- Cost-sensitive at scale (Pinecone expensive)
- Need complete control/customization
- Data residency requirements

**My recommendation**: Start Pinecone, migrate to self-hosted if cost/control becomes issue.

### Q: "How do you choose similarity metric?"

**Answer**:
- **Cosine similarity**: Text embeddings (most common). Measures angle, ignores magnitude. Use when vector length doesn't matter.
- **Euclidean distance**: Images, coordinates. Measures actual distance. Use when magnitude matters.
- **Dot product**: Pre-normalized vectors. Fastest. Use for speed when vectors are normalized.

**Default**: Cosine for text, Euclidean for images/spatial data.

### Q: "What's the difference between pre-filtering and post-filtering?"

**Answer**:

**Pre-filtering**: Apply filters BEFORE vector search
- Search only within filtered subset
- More accurate (searches exact subset)
- Better performance (fewer vectors to search)
- Requires index support

**Post-filtering**: Vector search first, THEN filter results
- Search all vectors, filter after
- Less accurate (may miss results)
- Simpler implementation
- Works with any index

**Best practice**: Use pre-filtering when database supports it (Weaviate, Qdrant, Milvus).

### Q: "How do you handle updates in production?"

**Answer**:

**Strategies**:
1. **Incremental updates** (if supported): Add/update without full rebuild
2. **Batch updates**: Collect changes, update off-peak
3. **Dual indices**: Build new index while serving from old, swap
4. **Soft deletes**: Mark deleted, filter at query time, periodic cleanup

**HNSW challenge**: Updates expensive, often need rebuild
**Solution**: Batch daily/weekly, use IVF for frequent updates, or accept rebuild cost

### Q: "How do you optimize for cost?"

**Answer**:

**Strategies**:
1. **Quantization**: Float32 → Int8 (4x smaller, minimal quality loss)
2. **Product Quantization**: 8-32x compression, 5-10% accuracy loss
3. **Smaller embeddings**: 384D vs 1536D (4x cheaper)
4. **IVF instead of HNSW**: Less memory, slightly slower
5. **Caching**: Cache frequent queries
6. **Self-hosting**: Cheaper at scale than managed services

**Trade-offs**: Balance cost vs accuracy vs latency

### Q: "Hybrid search: how and why?"

**Answer**:

**Why**: Combines semantic (meaning) + keyword (exact term) search
- Semantic alone: Misses specific terms/names
- Keyword alone: Misses synonyms/context
- Together: Best recall

**How**:
1. Run semantic search (vector)
2. Run keyword search (BM25)
3. Combine scores (weighted average): `α * semantic + (1-α) * keyword`
4. α = 0.5 is common starting point

**Implementation**: Weaviate (native), ElasticSearch (native), others (manual)

---

## Database Selection Decision Tree

```
Start: What's your scale?

< 100K vectors → Chroma (embedded mode)
  ↓
100K - 1M vectors → Do you need hybrid search?
  ↓                    ↓
  Yes → Weaviate     No → Need managed?
                           ↓         ↓
                         Yes        No
                           ↓         ↓
                       Pinecone   Qdrant

1M - 10M vectors → Have DevOps team?
  ↓                    ↓
  Yes → Qdrant      No → Pinecone
  (self-hosted)

10M+ vectors → Multi-tenant?
  ↓                ↓
  Yes → Milvus   No → Need ACID?
                      ↓        ↓
                    Yes       No
                      ↓        ↓
                  pgvector  Qdrant/Milvus

Existing Postgres? → pgvector
Research/Experimentation? → FAISS
```

---

## Emerging Trends

| Trend | Description | Impact |
|-------|-------------|--------|
| **Multi-modal** | Vectors for text + images + audio | Unified search |
| **Sparse + Dense** | Combine sparse and dense vectors | Better accuracy |
| **GPU Acceleration** | GPU-based indices | 10-100x faster |
| **Serverless** | Pay per query, zero ops | Lower barrier to entry |
| **Quantization** | Better compression algorithms | Lower costs |
| **Streaming Updates** | Real-time index updates | Fresher data |

---

## Key Takeaways for Interviews

1. **Vector DB ≠ Traditional DB**: Optimized for similarity search, not exact match
2. **Index matters**: HNSW (accuracy), IVF (balance), Flat (exact, slow)
3. **Cosine for text**: Most common similarity metric
4. **Pre-filtering > Post-filtering**: When supported
5. **Hybrid search**: Combine semantic + keyword for best results
6. **Trade-offs everywhere**: Speed vs accuracy, cost vs quality
7. **Start managed**: Pinecone/Chroma, self-host at scale
8. **Quantization**: 4-32x smaller, minimal quality loss
9. **HNSW vs IVF**: HNSW faster queries, IVF easier updates
10. **Production needs**: Monitoring, scaling, cost optimization

---

## Quick Reference: When to Use Which

| Use Case | Database Choice | Index Type | Why |
|----------|----------------|------------|-----|
| **RAG (< 1M docs)** | Chroma | HNSW | Easy setup, good enough |
| **RAG (production)** | Pinecone / Qdrant | HNSW | Managed / Performance |
| **Hybrid search** | Weaviate | HNSW | Native hybrid support |
| **Existing Postgres** | pgvector | IVF/HNSW | Leverage existing infra |
| **Recommendations** | Qdrant | HNSW | Great filtering |
| **Image search** | Weaviate | HNSW | Multi-modal support |
| **Research** | FAISS | Various | Flexibility, free |
| **Enterprise scale** | Milvus | Multiple | Proven at scale |
| **Real-time** | Redis | Flat/HNSW | Ultra-fast |