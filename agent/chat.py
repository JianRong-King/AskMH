import argparse
import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import build_graph


def main():
    ap = argparse.ArgumentParser(description="Chat with AskMH in the terminal")
    ap.add_argument("--debug", action="store_true", help="show route, question and decision")
    args = ap.parse_args()

    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print("AskMH (Ctrl+C or empty line to quit)")
    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        if args.debug:
            print(f"You: {text}")
        t0 = time.time()
        before = len(graph.get_state(config).values.get("messages", []))
        state = graph.invoke({"messages": [HumanMessage(text)]}, config)
        for m in state["messages"][before + 1:]:
            if isinstance(m, AIMessage):
                print(f"\nAskMH: {m.content}")
        if args.debug:
            print(f"  [route={state.get('route')} decision={state.get('decision')} "
                  f"failures={state.get('failures', 0)} q={state.get('question')!r} {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
