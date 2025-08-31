#!/usr/bin/env python3
"""
Minimal OpenAI agent example (function/tool calling) using environment key.

Prereqs:
- pip install openai
- Set OPENAI_API_KEY in your environment.

Usage:
- python agent_example.py "What is 3*(7+5), and what time is it?"
- python agent_example.py   # starts an interactive REPL
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "The 'openai' package is required. Install with: pip install openai"
    )


# ------------------------
# Tool implementations
# ------------------------

def tool_get_current_time(_: Dict[str, Any]) -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _safe_eval_arith(expr: str) -> float:
    """Safely evaluate a simple arithmetic expression.

    Supports numbers, +, -, *, /, **, parentheses. No names or calls.
    """
    import ast
    import operator as op

    allowed_ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.UAdd: lambda x: x,
    }

    def _eval(node):
        if isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return node.n
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        raise ValueError("Disallowed expression in calculator")

    tree = ast.parse(expr, mode="eval")
    return float(_eval(tree))


def tool_calculator(args: Dict[str, Any]) -> str:
    """Evaluate a basic arithmetic expression and return the numeric result.

    Args schema: {"expression": "3*(7+5)"}
    """
    expr = args.get("expression", "")
    if not isinstance(expr, str) or not expr:
        raise ValueError("Missing 'expression' string for calculator")
    value = _safe_eval_arith(expr)
    # Return as string for tool message content
    return str(value)


def tool_search(args: Dict[str, Any]) -> str:
    """A stub search tool that returns a canned result list.

    Replace with a real search API in your environment if desired.
    Args schema: {"query": "..."}
    """
    query = args.get("query", "")
    results = [
        {"title": "Example Result A", "url": "https://example.com/a"},
        {"title": "Example Result B", "url": "https://example.com/b"},
    ]
    return json.dumps({"query": query, "results": results})


# Registry mapping tool names to callables
TOOL_REGISTRY = {
    "get_current_time": tool_get_current_time,
    "calculator": tool_calculator,
    "search": tool_search,
}


# Tool specifications for the model
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current UTC time as ISO 8601 string.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web and return JSON results (stub).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Use tools when they help produce accurate answers. "
    "If you used tools, cite their results succinctly."
)


def run_agent(user_input: str, model: str = "gpt-4o-mini") -> str:
    """Run a simple tool-using agent for a single user input.

    Returns the assistant's final message content as text.
    """
    # Ensure API key is available (OpenAI client reads OPENAI_API_KEY by default)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set in environment")

    client = OpenAI()

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    # First call: let the model decide whether to invoke tools
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOL_SPECS,
        tool_choice="auto",
        temperature=0.2,
    )

    msg = resp.choices[0].message
    messages.append(msg.dict(exclude_none=True))  # keep assistant msg with tool_calls, if any

    # If the assistant requested tools, execute them and send results back
    if msg.tool_calls:
        for call in msg.tool_calls:
            name = call.function.name
            args_json = call.function.arguments or "{}"
            try:
                args = json.loads(args_json)
            except json.JSONDecodeError:
                args = {"_raw": args_json}

            if name not in TOOL_REGISTRY:
                tool_output = f"Tool '{name}' not implemented."
            else:
                try:
                    tool_output = TOOL_REGISTRY[name](args)
                except Exception as e:  # pragma: no cover
                    tool_output = f"Error in tool '{name}': {e}"

            # Append tool result message linked via tool_call_id
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": name,
                    "content": tool_output,
                }
            )

        # Second call: provide tool outputs for the final answer
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )

    final_msg = resp.choices[0].message
    # Handle potential structured content parts; join to text
    content = final_msg.content or ""
    return content


def main() -> None:
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(run_agent(query))
        return

    # Simple interactive loop
    print("OpenAI Agent REPL. Press Ctrl+C to exit.")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    while True:
        try:
            user_input = input("You> ").strip()
            if not user_input:
                continue
            answer = run_agent(user_input, model=model)
            print(f"Agent> {answer}")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break


if __name__ == "__main__":
    main()

