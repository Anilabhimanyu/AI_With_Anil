# LangChain - Quick Reference Guide

## Overview
**LangChain** is a framework for building LLM-powered applications with modular components and orchestration capabilities.

---

## Core Components

| Component | Purpose | Key Features | Example Use Case |
|-----------|---------|--------------|------------------|
| **Models** | LLM integrations | - Supports OpenAI, Anthropic, Cohere, etc.<br>- Unified interface for different providers<br>- Streaming support | Switching between GPT-4 and Claude without code changes |
| **Prompts** | Template management | - Prompt templates with variables<br>- Few-shot examples<br>- Output parsers | Creating reusable prompt structures |
| **Chains** | Sequential operations | - Combine multiple LLM calls<br>- Data transformation pipelines<br>- Error handling | Document summarization → Q&A |
| **Agents** | Autonomous decision-making | - Tool selection<br>- ReAct framework<br>- Self-correction | AI that decides when to search vs calculate |
| **Memory** | Context retention | - Conversation buffer<br>- Summary memory<br>- Vector store memory | Chatbots remembering conversation history |
| **Retrievers** | Data access | - Vector store integration<br>- Document loaders<br>- Text splitters | RAG systems, semantic search |

---

## Key Concepts for Interviews

### 1. Chains vs Agents

| Feature | Chains | Agents |
|---------|--------|--------|
| **Control Flow** | Predetermined sequence | Dynamic, LLM-decided |
| **Flexibility** | Fixed steps | Adaptive based on input |
| **Use Case** | Predictable workflows | Complex, multi-step reasoning |
| **Example** | Load doc → Summarize → Translate | "Answer this question using whatever tools needed" |

### 2. RAG (Retrieval-Augmented Generation)

```
Pipeline: Query → Retrieve relevant docs → Augment prompt → Generate response
```

| Step | LangChain Component | Purpose |
|------|---------------------|---------|
| 1. Indexing | Document Loaders, Text Splitters | Prepare documents |
| 2. Storage | Vector Stores (Pinecone, Chroma) | Store embeddings |
| 3. Retrieval | Retrievers | Find relevant chunks |
| 4. Generation | LLM + Prompt Template | Generate answer |

### 3. Memory Types

| Type | Description | Best For | Limitation |
|------|-------------|----------|------------|
| **ConversationBufferMemory** | Stores all messages | Short conversations | Token limit issues |
| **ConversationSummaryMemory** | Summarizes old messages | Long conversations | May lose details |
| **ConversationBufferWindowMemory** | Keeps last N messages | Medium conversations | Loses older context |
| **VectorStoreMemory** | Semantic similarity search | Relevant context retrieval | Complex setup |

---

## Common Interview Questions

### Q: "How would you build a chatbot that answers from company docs?"

**Answer Structure:**
1. **Data Ingestion**: Load documents using DocumentLoaders
2. **Chunking**: Split with RecursiveCharacterTextSplitter
3. **Embedding**: Create vector embeddings
4. **Storage**: Store in vector database (Pinecone/Chroma)
5. **Retrieval**: Use RetrievalQA chain
6. **Memory**: Add ConversationBufferMemory for context
7. **Response**: Generate with LLM using retrieved context

### Q: "Chains vs direct LLM API calls?"

**LangChain Advantages:**
- Modular components (easier maintenance)
- Built-in memory management
- Tool integration (web search, calculators)
- Logging and debugging utilities
- Production-ready patterns

**Direct API Advantages:**
- Less overhead
- Full control
- Simpler for basic use cases
- Fewer dependencies

---

## Code Pattern Examples

### Basic Chain
```python
from langchain import PromptTemplate, LLMChain

template = "Translate {text} to {language}"
chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(template))
result = chain.run(text="Hello", language="Spanish")
```

### RAG Pattern
```python
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

vectorstore = Chroma.from_documents(documents, embeddings)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)
answer = qa_chain.run("What is the refund policy?")
```

### Agent Pattern
```python
from langchain.agents import initialize_agent, Tool

tools = [
    Tool(name="Search", func=search_tool),
    Tool(name="Calculator", func=calculator_tool)
]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
agent.run("What's 25% of the GDP of France in 2023?")
```

---

## Key Terminology

| Term | Definition |
|------|------------|
| **Embeddings** | Vector representations of text for semantic similarity |
| **Vector Store** | Database optimized for similarity search (FAISS, Pinecone, Chroma) |
| **Few-shot prompting** | Providing examples in prompts to guide LLM behavior |
| **ReAct** | Reasoning + Acting framework for agents |
| **Hallucination** | LLM generating incorrect/made-up information |
| **Grounding** | Using retrieved facts to reduce hallucinations |
| **Token** | Basic unit of text (roughly 0.75 words) |

---

## Alternatives to LangChain

| Framework | Key Difference |
|-----------|----------------|
| **LlamaIndex** | Focused specifically on RAG and data indexing |
| **Haystack** | Production-focused, enterprise-ready NLP pipelines |
| **Semantic Kernel** | Microsoft's orchestration framework |
| **Custom Solutions** | Direct API calls with custom orchestration |

---

## Pro Tips for Interviews

1. **Know when NOT to use LangChain**: Simple use cases don't need it
2. **Understand the tradeoffs**: Abstraction vs control
3. **RAG is crucial**: Most companies need this for production
4. **Agents are powerful but tricky**: Harder to make deterministic
5. **Cost management**: Token usage matters in production
6. **Evaluation matters**: How do you measure quality?