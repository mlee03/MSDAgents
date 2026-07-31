
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

llm = "mistral-nemo:latest"

system_prompt = """You are a funny assistant,  Be polite and respectful"""

@tool
def get_weather(query: str) -> str:
    """Return the weather."""
    return "It's thunderstorming!"


def run_ollama_agent():
    #https://docs.langchain.com/oss/python/integrations/chat/ollama
    model = ChatOllama(model=llm).bind_tools([get_weather])
    while True:
        query = input("User: ")
        if query.lower() == "exit":
            break
        response = model.invoke(query)
        print(f"Assistant: {response}\n")
        for tool_call in response.tool_calls:
            print(f". Tool call: {tool_call}")
        #for chunk in model.stream(query): print(chunk, end="", flush=True)


def run_create_agent():
    #https://docs.langchain.com/oss/python/langchain/tools
    #https://reference.langchain.com/python/langchain/agents/factory/create_agent
    #https://docs.langchain.com/oss/python/langchain/agents
    """
    Creates and runs an agent that can call tools.
    
    Example response:
    {
        'messages': [
            HumanMessage(
                content="How's the weather right now?",
                additional_kwargs={},
                response_metadata={},
                id='df5b2f9b-fbe6-4d50-9f81-bbbdb237b303'
            ),
            AIMessage(
                content='',
                additional_kwargs={},
                response_metadata={
                    'model': 'mistral-nemo:latest',
                    'created_at': '2026-07-06T15:28:26.577271313Z',
                    'done': True,
                    'done_reason': 'stop',
                    'total_duration': 3348988406,
                    'load_duration': 183534100,
                    'prompt_eval_count': 68,
                    'prompt_eval_duration': 147511026,
                    'eval_count': 23,
                    'eval_duration': 2976714358,
                    'logprobs': None,
                    'model_name': 'mistral-nemo:latest',
                    'model_provider': 'ollama'
                },
                id='lc_run--019f380b-76bb-7e30-9129-ea8d33af3e38-0',
                tool_calls=[
                    {
                        'name': 'get_weather',
                        'args': {'query': 'Abuja'},
                        'id': 'e3599371-a783-4d6c-82e1-8c79b5d96c13',
                        'type': 'tool_call'
                    }
                ],
                invalid_tool_calls=[],
                usage_metadata={'input_tokens': 68, 'output_tokens': 23, 'total_tokens': 91}
            ),
            ToolMessage(
                content="It's thunderstorming!",
                name='get_weather',
                id='b326c387-d3e0-4566-82aa-95a81e27ac5e',
                tool_call_id='e3599371-a783-4d6c-82e1-8c79b5d96c13'
            ),
            AIMessage(
                content="It's currently a thunderstorm in Abuja.",
                additional_kwargs={},
                response_metadata={
                    'model': 'mistral-nemo:latest',
                    'created_at': '2026-07-06T15:28:28.587837599Z',
                    'done': True,
                    'done_reason': 'stop',
                    'total_duration': 2006056995,
                    'load_duration': 203354746,
                    'prompt_eval_count': 44,
                    'prompt_eval_duration': 427582314,
                    'eval_count': 11,
                    'eval_duration': 1352452519,
                    'logprobs': None,
                    'model_name': 'mistral-nemo:latest',
                    'model_provider': 'ollama'
                },
                id='lc_run--019f380b-83d4-78a0-8c51-62acfee7059a-0',
                tool_calls=[],
                invalid_tool_calls=[],
                usage_metadata={'input_tokens': 44, 'output_tokens': 11, 'total_tokens': 55}
            )
        ]
    }
    """
    model = ChatOllama(model=llm).bind_tools([get_weather])
    agent = create_agent(model=model, tools=[get_weather], system_prompt=system_prompt)
    while True:
        query = input("User: ")
        if query.lower() == "exit":
            break
        response = agent.invoke({"messages": [{"role": "user", "content": query}]})
        print(f"Assistant: {response}")

run_ollama_agent()
#run_create_agent()