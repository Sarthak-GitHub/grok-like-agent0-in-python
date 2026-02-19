from langchain_community.llms import LlamaCpp
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langchain.utilities import PythonREPL
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

def load_llm(model_path: str):
    return LlamaCpp(
        model_path=model_path,
        n_gpu_layers=-1,
        n_batch=512,
        temperature=0.75,
        max_tokens=768,
        verbose=True
    )

def create_tools():
    search = DuckDuckGoSearchRun()
    repl = PythonREPL()
    return [
        Tool(
            name="web_search",
            func=search.run,
            description="Search the web for current information."
        ),
        Tool(
            name="python_repl",
            func=repl.run,
            description="Execute Python code. Input must be valid Python."
        )
    ]

def create_grok_agent(llm, tools):
    system_prompt = """
You are Grok-like — witty, helpful, a bit sarcastic, built in the spirit of xAI.
Answer naturally. Use Hindi/Marathi if user asks in Indic language.
Use tools when you need facts, code execution, or current info.
Always think step-by-step before acting.
Put final answer in <final> tags.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6
    )

    return executor
