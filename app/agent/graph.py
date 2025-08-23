# app/agent/graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .nodes import (
    ingest_input,
    guardrails_node,
    precheck_rule_node,
    extract_slots_node,
    rag_node,
    reason_and_decide_node,
    self_reflect_node,
    explain_node,
)

def build_graph():
    sg = StateGraph(dict)

    # 节点注册
    sg.add_node("ingest", ingest_input)
    sg.add_node("guardrails", guardrails_node)            # 红旗/安全检查
    sg.add_node("extract", extract_slots_node)   
    sg.add_node("precheck_rule", precheck_rule_node)      # 旧项目 compute_eligibility
    sg.add_node("rag", rag_node)                          # 旧项目 rag_answer
    sg.add_node("reason", reason_and_decide_node)         # 归纳推理 → 决策 JSON
    sg.add_node("reflect", self_reflect_node)             # 自反思 → 修正
    sg.add_node("explain", explain_node)                  # 收尾解释 + 引用

    # 执行流
    sg.set_entry_point("ingest")
    sg.add_edge("ingest", "guardrails")
    sg.add_edge("guardrails", "extract")              
    sg.add_edge("extract", "precheck_rule")           
    sg.add_edge("precheck_rule", "rag")
    sg.add_edge("rag", "reason")
    sg.add_edge("reason", "reflect")
    sg.add_edge("reflect", "explain")
    sg.add_edge("explain", END)
    # 开记忆（需要调用时传 thread_id）
    return sg.compile(checkpointer=MemorySaver())

GRAPH = build_graph()
