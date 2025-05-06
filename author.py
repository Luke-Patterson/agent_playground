import os
from typing import Dict, TypedDict, Annotated, List
import operator

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from air_genai_helper import initialize_urls_and_key
from langgraph.graph import StateGraph, END


# Define our state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    essay: str
    critique: str
    iterations: int
    max_iterations: int
    complete: bool


# Initialize Azure OpenAI model
vault_url, api_key, azure_endpoint = initialize_urls_and_key("ts", "eastus2")


def init_llm():
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",  # or your deployment
        api_version="2024-06-01",
        azure_endpoint=azure_endpoint,
        api_key=api_key,
    )
    return llm


# Get LLM instance
model = init_llm()

# Essay Writer Agent
essay_writer_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a creative essay writer. Your task is to write an insightful and engaging essay about why Rick and Morty is a great show.

Your essay should:
- Be approximately 500-700 words
- Include specific examples from the show
- Discuss themes, character development, humor, and cultural impact
- Have a clear introduction, body, and conclusion
- Be written in an engaging, thoughtful style

If you receive critique feedback, incorporate it to improve your essay in the next iteration."""),
    MessagesPlaceholder(variable_name="messages"),
])

essay_writer_chain = essay_writer_prompt | model | StrOutputParser()

# Essay Critic Agent
essay_critic_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a thoughtful literary critic. Your task is to provide constructive criticism on an essay about Rick and Morty.

In your critique:
- Evaluate the essay's arguments, structure, and clarity
- Point out specific strengths to maintain
- Identify areas for improvement with specific suggestions
- Assess whether the essay effectively conveys why Rick and Morty is a great show
- Be constructive but honest in your assessment

Your goal is to help improve the essay, not to completely rewrite it."""),
    MessagesPlaceholder(variable_name="messages"),
])

essay_critic_chain = essay_critic_prompt | model | StrOutputParser()


# Define the nodes for our graph
def writer_node(state: AgentState) -> AgentState:
    """Essay writer node that creates or refines the essay"""
    messages = state["messages"].copy()

    # If there's a critique and we're not done yet, include it for the writer to consider
    if state["critique"] and state["iterations"] < state["max_iterations"]:
        messages.append(HumanMessage(
            content=f"Here is criticism of your previous essay. Please revise and improve your essay based on this feedback:\n\n{state['critique']}"))
    # If this is the first iteration, ask for a new essay
    elif state["iterations"] == 0:
        messages.append(HumanMessage(content="Please write an essay on why Rick and Morty is a great show."))

    # Get the essay from the model
    essay = essay_writer_chain.invoke({"messages": messages})

    # Update and return the state
    return {
        **state,
        "essay": essay,
        "messages": state["messages"] + [HumanMessage(content="Write an essay on why Rick and Morty is a great show."),
                                         AIMessage(content=essay)],
        "iterations": state["iterations"] + 1
    }


def critic_node(state: AgentState) -> AgentState:
    """Essay critic node that provides feedback on the essay"""
    messages = state["messages"].copy()

    # Add the essay for the critic to review
    messages.append(
        HumanMessage(content=f"Please critique the following essay about Rick and Morty:\n\n{state['essay']}"))

    # Get the critique from the model
    critique = essay_critic_chain.invoke({"messages": messages})

    # Update and return the state
    return {
        **state,
        "critique": critique,
        "messages": state["messages"] + [
            HumanMessage(content=f"Please critique this essay about Rick and Morty:\n\n{state['essay']}"),
            AIMessage(content=critique)]
    }


def should_continue(state: AgentState) -> str:
    """Determine if we should continue iterating or finish"""
    if state["iterations"] >= state["max_iterations"]:
        return "complete"
    else:
        return "continue"


# Define the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("writer", writer_node)
graph.add_node("critic", critic_node)

# Define edges
graph.add_edge("writer", "critic")
graph.add_conditional_edges(
    "critic",
    should_continue,
    {
        "continue": "writer",
        "complete": END
    }
)

# Set the entry point
graph.set_entry_point("writer")

# Compile the graph
essay_critique_graph = graph.compile()


# Function to run the graph
def create_and_critique_essay(max_iterations=2):
    """Run the essay creation and critique process"""

    # Initialize the state
    initial_state = {
        "messages": [],
        "essay": "",
        "critique": "",
        "iterations": 0,
        "max_iterations": max_iterations,
        "complete": False
    }

    # Execute the graph
    final_state = essay_critique_graph.invoke(initial_state)

    # Return the final state
    return {
        "final_essay": final_state["essay"],
        "final_critique": final_state["critique"],
        "iterations": final_state["iterations"]
    }


# Example usage
if __name__ == "__main__":
    result = create_and_critique_essay(max_iterations=2)

    print("=== FINAL ESSAY ===")
    print(result["final_essay"])
    print("\n=== FINAL CRITIQUE ===")
    print(result["final_critique"])
    print(f"\nCompleted in {result['iterations']} iterations")