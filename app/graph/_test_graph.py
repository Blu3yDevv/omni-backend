# app/graph/_test_graph.py

from app.graph.workflow import run_omni_graph


def main():
    # Simple debug message to send through the pipeline
    user_message = "Explain what OmniAI is, in 2–3 short sentences."

    # chat_history can be extended later; keep empty for now
    state = run_omni_graph(user_message=user_message, chat_history=[])

    # NOTE: run_omni_graph currently returns an OmniState object,
    # not a dict, so we use attributes instead of .get(...)

    print("\n=== PLAN ===")
    print(state.plan)

    print("\n=== RESEARCH ===")
    print(state.research)

    print("\n=== DRAFT ANSWER ===")
    print(state.draft_answer)

    print("\n=== TESTER ISSUES ===")
    if state.tester_issues:
        for issue in state.tester_issues:
            print("-", issue)
    else:
        print("(none)")

    print("\n=== TESTER FIXES ===")
    if state.tester_fixes:
        for fix in state.tester_fixes:
            print("-", fix)
    else:
        print("(none)")

    print("\n=== SAFETY FLAGS ===")
    if state.safety_flags:
        for flag in state.safety_flags:
            print("-", flag)
    else:
        print("(none)")

    print("\n=== FINAL ANSWER ===")
    print(state.final_answer or "(empty)")


if __name__ == "__main__":
    main()
