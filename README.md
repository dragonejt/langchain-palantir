# LangChain on Palantir AIP

langchain-palantir is a library that allows users to call Palantir-provided LLMs in the LangChain framework. It provides LangChain wrappers around Palantir-provided LLMs, and can be used anywhere Palantir-provided LLMs can be used.

## Installation
Install langchain-palantir like any other Palantir conda package, with the Libraries left sidebar in a Code Repository or Code Workspace. langchain-palantir currently requires Python version 3.12.10 or later.

## Usage
langchain-palantir can be used like any other LangChain extension.
### Basic Tool Calling Workflow
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

tools = {"date_time": date_time}

llm = PalantirChatOpenAI(model=model)
llm_with_tools = llm.bind_tools(tools.values())
answer = llm_with_tools.invoke(messages)
messages.append(answer)

for tool_call in answer.tool_calls:
  messages.append(tools[tool_call["name"]].invoke(tool_call))

final_answer = llm_with_tools.invoke(messages)
```