"""HYBRID (neuro-symbolic) pipeline:

    problem text --(LLM formalizer)--> IR --(validation)--> Z3 solver --(presentation)--> answer

This is the main thing your team builds. The plumbing is here; the parts marked TODO(team) are yours.
Suggested order of work:
    1. write FORMALIZER_PROMPT and get formalize() returning sensible IR for one simple problem;
    2. implement validate_ir();
    3. add ONE retry with feedback in solve();
    4. fix run_solver() so that "unique" and "multiple" can be told apart;
    5. (extension) replace the LLM-based present() with deterministic code for some problem types.

See docs/ir.md for the IR format.
"""
import json
from typing import Any, Optional

from pydantic import BaseModel, Field

import ir_solver
import llm
from answer import Answer


class HybridError(Exception):
    """The hybrid pipeline could not produce an answer (recorded by run.py as an error, not as status "none")."""


class IRModel(BaseModel):
    """JSON schema the formalizer must follow. Keep it in sync with docs/ir.md."""
    variables: dict[str, Any]
    all_different: list[list[str]] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    objective: Optional[dict[str, str]] = None


# Compact IR description for the model. You may edit it, but keep it consistent with ir_solver.py.
IR_SPEC = """IR format (JSON):
{
  "variables": {"<name>": [lo, hi]  or  {"values": [v1, v2, ...]}},     integer variables and their domains
  "all_different": [["<name>", ...], ...],                                 groups whose values must all differ
  "constraints": ["<expression>", ...],                                    every expression must be true
  "objective": {"maximize": "<expression>"}  or  {"minimize": "<expression>"}   ONLY for optimisation problems
}
Expressions may use: integer numbers, variable names, + - * // %, == != < <= > >=, and, or, not, abs(), parentheses.
Inside arithmetic a comparison counts as 1 (true) or 0 (false), e.g. "(A == 1) + (B == 1) == 1".
Variable names must be identifiers: letters, digits and underscores only, no spaces."""

SYSTEM = "You translate logic puzzles into a formal constraint representation. You answer only with JSON."


FORMALIZER_PROMPT = """Translate the puzzle below into the IR format. Do not solve the puzzle.

{ir_spec}

CRITICAL ENCODING RULES:
1. VARIABLES & DOMAINS: 
   - Map ordered categories like days (Monday=1, ..., Friday=5) or times (9:00=9) to integers.
   - For optimization/subset selection (e.g., choosing features), use binary [0, 1] domains for each candidate (1 = selected, 0 = not). Convert textual names into valid identifiers (e.g., "Offline mode" -> Offline_mode).
2. ALL_DIFFERENT: If multiple entities belong to the same logical category (like people taking days off) and must be unique, group their variable names in "all_different". Do not use this for binary optimization variables.
3. RELATIVE POSITIONS & TIME:
   - "X is earlier than Y" / "X is before Y" -> "X < Y"
   - "X is immediately after Y" -> "X == Y + 1"
4. ROUND TABLE (CIRCULAR ORDER): For N chairs numbered 1 to N clockwise:
   - "X sits next to Y" -> "(abs(X - Y) == 1) or (abs(X - Y) == N - 1)"
   - "X sits directly opposite Y" -> "abs(X - Y) == (N // 2)"
   - "X sits immediately counter-clockwise from Y" -> "(Y == X + 1) or (X == N and Y == 1)"
   - "X sits immediately clockwise from Y" -> "(X == Y + 1) or (Y == N and X == 1)"
5. OPTIMIZATION & SUBSETS (Variables are 0 or 1):
   - "A can only be chosen if B is chosen" / "If A then B" -> "A <= B"
   - "A and B cannot both be chosen" -> "A + B <= 1"
   - "At least two of A, B, C" -> "A + B + C >= 2"
   - Capacity/Budget limits -> "cost1*A + cost2*B ... <= limit"
   - Define the "objective" key to "maximize" or "minimize" the total value: "val1*A + val2*B ...".
6. KNIGHTS AND KNAVES: Define domains as [0, 1] where Knave=0, Knight=1. 
   - If X makes a statement S, encode it as "X == (S)". 
   - "At least one of X, Y, Z is a knave" -> "(X == 0) or (Y == 0) or (Z == 0)".
7. SYNTAX: Variable names must be valid identifiers (letters, digits, underscores only; NO spaces).

Puzzle:
{text}
{feedback}"""

def formalize(problem: dict, feedback: Optional[str] = None) -> dict:
    """Ask the model for an IR encoding of the problem. `feedback` = error messages from a previous attempt."""
    fb = f"\nYour previous encoding was rejected:\n{feedback}\nFix these problems.\n" if feedback else ""
    prompt = FORMALIZER_PROMPT.format(ir_spec=IR_SPEC, text=problem["text"], feedback=fb)
    parsed, raw = llm.ask_json(prompt, IRModel, system=SYSTEM)
    return parsed.model_dump(exclude_none=True)


def validate_ir(ir: dict) -> list:
    errors = []
    
    variables = ir.get("variables", {})
    if not isinstance(variables, dict):
        errors.append("'variables' must be a dictionary.")
    else:
        for name, domain in variables.items():
            if isinstance(domain, list):
                if len(domain) != 2 or not isinstance(domain[0], int) or not isinstance(domain[1], int):
                    errors.append(f"Domain for '{name}' must be a list of two integers [lo, hi].")
                elif domain[0] > domain[1]:
                    errors.append(f"Domain for '{name}' has lo > hi: {domain}.")
            elif isinstance(domain, dict):
                if "values" not in domain or not isinstance(domain["values"], list) or not all(isinstance(v, int) for v in domain["values"]):
                    errors.append(f"Domain for '{name}' dict must contain 'values' with a list of integers.")
            else:
                errors.append(f"Domain format for '{name}' is invalid.")

    all_different = ir.get("all_different", [])
    if not isinstance(all_different, list):
        errors.append("'all_different' must be a list of lists.")
    else:
        for group in all_different:
            if not isinstance(group, list):
                errors.append(f"Item in 'all_different' must be a list, got {type(group).__name__}.")
            else:
                for var in group:
                    if isinstance(variables, dict) and var not in variables:
                        errors.append(f"Variable '{var}' in 'all_different' is not declared in 'variables'.")

    try:
        ir_solver.build(ir)
    except ValueError as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(f"Error building constraints: {str(e)}")

    return errors


def run_solver(ir: dict) -> dict:
    result = ir_solver.solve(ir, limit=2)
    
    status = result["status"]
    solutions = result["solutions"]
    
    if status != "optimal":
        if not solutions:
            result["status"] = "none"
        elif len(solutions) == 1:
            result["status"] = "unique"
        elif len(solutions) > 1:
            result["status"] = "multiple"
            
    return result


PRESENT_PROMPT = """A puzzle has been solved by a constraint solver. Rewrite the solver's result in the required
answer format. Do NOT solve the puzzle again - only translate the solver's assignment.

Puzzle:
{text}

Encoding used by the solver:
{ir}

Solver assignment (variable -> value):
{assignment}

{answer_format}

IMPORTANT: You MUST return a complete JSON object that includes BOTH the "solution" object and the "status" key.
Use exactly this status: "status": "{status}".
"""

def present(problem: dict, ir: dict, result: dict) -> dict:
    """Turn the solver result into the required answer format."""
    status = result["status"]
    if status == "none":
        return Answer(status="none").model_dump()
        
    prompt = PRESENT_PROMPT.format(
        text=problem["text"], 
        ir=json.dumps(ir, ensure_ascii=False),
        assignment=json.dumps(result["solutions"][0], ensure_ascii=False),
        answer_format=problem["answer_format"],
        status=status  
    )
    parsed, raw = llm.ask_json(prompt, Answer)
    
    return Answer(
        status=status, 
        solution=parsed.solution, 
        objective=result.get("objective")
    ).model_dump()


def solve(problem: dict) -> dict:
    """Returns {"answer": <Answer as dict>, "trace": {...}}. Raise HybridError if no answer can be produced."""
    trace = {"attempts": []}

    ir = formalize(problem)
    errors = validate_ir(ir)
    trace["attempts"].append({"ir": ir, "errors": errors})

    if errors:
        ir = formalize(problem, feedback="\n".join(errors))
        errors = validate_ir(ir)
        trace["attempts"].append({"ir": ir, "errors": errors})
        
        if errors:
            raise HybridError("invalid IR: " + "; ".join(errors))

    try:
        result = run_solver(ir)
    except ValueError as e:
        raise HybridError(f"solver rejected the IR: {e}") from e
        
    trace["solver"] = {
        "status": result["status"], 
        "n_solutions": len(result["solutions"]),
        "objective": result.get("objective"),
        "first_solution": result["solutions"][0] if result["solutions"] else None
    }

    answer = present(problem, ir, result)
    return {"answer": answer, "trace": trace}