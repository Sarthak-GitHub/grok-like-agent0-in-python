from agent_core import load_llm, create_tools, create_grok_agent

MODEL_PATH = "./models/llama-3.1-8b.Q5_K_M.gguf"

llm = load_llm(MODEL_PATH)
tools = create_tools()
agent = create_grok_agent(llm, tools)

while True:
    query = input("\nYou: ")
    if query.lower() in ["exit", "quit", "q"]:
        break
    response = agent.invoke({"input": query})
    print("Agent:", response["output"])
