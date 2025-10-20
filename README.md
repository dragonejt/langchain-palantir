# LangChain on Palantir AIP

langchain-palantir is a library that allows users to call Palantir-provided LLMs in the LangChain framework. It provides LangChain wrappers around Palantir-provided LLMs, and can be used anywhere Palantir-provided LLMs can be used.

## Installation

Install langchain-palantir like any other Palantir conda package, with the Libraries left sidebar in a Code Repository or Code Workspace. langchain-palantir currently requires Python version 3.11 or later, and LangChain v1.0.

## Supported Features

### Chat Models

| Chat Model              | Model Family     | Chat Completions | Tool Calling | Image Input |
| ----------------------- | ---------------- | ---------------- | ------------ | ----------- |
| `PalantirChatOpenAI`    | OpenAI GPT       | ✅               | ✅           | ✅          |
| `PalantirChatAnthropic` | Anthropic Claude | ✅               | ✅           | ✅          |
| `PalantirChatGeneric`   | Meta Llama, etc.  | ✅               | ❌           | ❌          |

### Embedding Models

| Embedding Model             | Embed Query | Embed Documents |
| --------------------------- | ----------- | --------------- |
| `PalantirGenericEmbeddings` | ✅          | ✅              |

## Usage

langchain-palantir can be used like any other LangChain extension. Examples use LangChain v1.0.

### Basic Agentic Workflow

```python
model = OpenAiGptChatLanguageModel.get("GPT_4_1")
messages = [
  HumanMessage("Using the date_time tool, what is today's date?")
]

@tool
def date_time() -> datetime:
  """
  Returns the current datetime in python.
  Parameters: None
  """

  return datetime.now(timezone.utc)

agent = create_agent(model=self.llm, tools=[date_time])
answer = agent.invoke({"messages": messages})
```

### Basic Multimodal Workflow

```python
model = AnthropicClaudeLanguageModel.get("AnthropicClaude_4_Sonnet")
image_data = b64encode(pizza_jpg.read()).decode("utf-8")
messages = [
  HumanMessage("What is in the following image?"),
  HumanMessage(
    [
      {
        "type": "image",
        "source_type": "base64",
        "data": image_data,
        "mime_type": "image/jpeg",
      }
    ]
  ),
]

llm = PalantirChatAnthropic(model=model)

answer = llm.invoke(messages)
```

### Using Palantir Embedding Models

```python
model = GenericEmbeddingModel.get("Text_Embedding_3_Small")
texts = ["Hello World", "Hello AI"]
embedding = PalantirGenericEmbeddings(model=model)

embeddings = embedding.embed_documents(texts)
```
