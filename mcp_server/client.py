# mcp_server/client.py
import datetime
import inspect
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from models.ollama_model import OllamaLLM
from models.bedrock_model import BedrockLLM
from tools.promptGen import assemble_prompt

mcp_client = None
temporary_memory = []

KNOWLEDGE_CUTOFF = "2024-06-01"


def _to_lc_messages(messages_dict_list):
    out = []
    for m in messages_dict_list:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        elif role == "system":
            out.append(SystemMessage(content=content))
    return out


async def agents(llm_model, llm_provider, question):
    global mcp_client, temporary_memory

    # Select model
    if llm_provider == "aws":
        model = BedrockLLM(llm_model).get_llm()
    elif llm_provider == "ollama":
        model = OllamaLLM(llm_model).get_llm()
    else:
        raise ValueError("Unsupported LLM provider")

    # MCP servers
    servers = {
        "serpSearch": {
            "url": "http://localhost:8003/mcp/",
            "transport": "streamable_http",
        },
        "weather": {
            "url": "http://localhost:8004/mcp/",
            "transport": "streamable_http",
        },
    }

    # Initialize MCP client and load tools
    mcp_client = MultiServerMCPClient(servers)
    tools = await mcp_client.get_tools()

    # Feature-detect whether this create_react_agent supports `state_modifier`
    sig = inspect.signature(create_react_agent)
    supports_state_modifier = "state_modifier" in sig.parameters

    if supports_state_modifier:
        # Newer LangGraph: use state_modifier (preferred)
        agent = create_react_agent(
            model=model["llm_model"],
            tools=tools,
            state_modifier=lambda state: assemble_prompt(
                now=datetime.datetime.now().astimezone(),
                knowledge_cutoff=KNOWLEDGE_CUTOFF,
            ),
        )
        # Build call messages (system is injected by state_modifier)
        history_msgs = _to_lc_messages(temporary_memory)
        call_messages = history_msgs + [HumanMessage(question)]
    else:
        # Older LangGraph: prepend SystemMessage at call time
        agent = create_react_agent(model=model["llm_model"], tools=tools)
        system_msg = SystemMessage(
            content=assemble_prompt(
                now=datetime.datetime.now().astimezone(),
                knowledge_cutoff=KNOWLEDGE_CUTOFF,
            )
        )
        history_msgs = _to_lc_messages(temporary_memory)
        call_messages = [system_msg, *history_msgs, HumanMessage(question)]

    # Invoke agent
    response = await agent.ainvoke({"messages": call_messages})

    # Extract and persist turn
    final_content = (
        response["messages"][-1].content if response.get("messages") else str(response)
    )
    temporary_memory.append({"role": "user", "content": question})
    temporary_memory.append({"role": "assistant", "content": final_content})
    return final_content
