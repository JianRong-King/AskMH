from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent import nodes
from agent.nodes import State


@lru_cache(maxsize=1)
def build_graph(checkpointer=None):
    g = StateGraph(State)
    for name in ["router", "greet", "retrieve", "grade", "generate", "decline", "handoff"]:
        g.add_node(name, getattr(nodes, name))

    g.add_edge(START, "router")
    g.add_conditional_edges("router", nodes.after_router, ["greet", "handoff", "retrieve", "decline"])
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", nodes.after_grade, {"generate": "generate", "decline": "decline", "end": END})
    # generate can still refuse if the articles turn out not to answer the question,
    # so it escalates on the same rule as decline.
    for name in ["decline", "generate"]:
        g.add_conditional_edges(name, nodes.after_failure, {"handoff": "handoff", "end": END})
    for name in ["greet", "handoff"]:
        g.add_edge(name, END)
    return g.compile(checkpointer=checkpointer or MemorySaver())
