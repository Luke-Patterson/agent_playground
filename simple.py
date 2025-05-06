from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from air_genai_helper import initialize_urls_and_key
from langgraph.prebuilt import create_react_agent


vault_url, api_key, azure_endpoint = initialize_urls_and_key("ts", "eastus2")


def init_llm():
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",  # or your deployment
        api_version="2024-06-01",
        azure_endpoint=azure_endpoint,
        api_key=api_key,
    )
    return llm


def check_weather(location: str) -> str:
    '''Return the weather forecast for the specified location.'''
    return f"It's always sunny in {location}"

tools = [check_weather]
model = init_llm()
graph = create_react_agent(model, tools=tools)
inputs = {"messages": [("user", "what is the weather in sf")]}
for s in graph.stream(inputs, stream_mode="values"):
    message = s["messages"][-1]
    if isinstance(message, tuple):
        print(message)
    else:
        message.pretty_print()