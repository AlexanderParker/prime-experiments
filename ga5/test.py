import csv
import sys
import evolve


def load_best_expression_from_seeds(seeds_filename: str = "seeds.csv") -> str:
    """Load the best expression from seeds.csv."""
    with open(seeds_filename, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        if not rows:
            raise ValueError("No expressions found in seeds.csv")

        best_row = min(rows, key=lambda x: float(x["fitness"]))
        return best_row["expression"]


def expression_to_ast(expression: str) -> evolve.ASTNode:
    """Convert a string expression back to an AST."""
    expression = expression.strip()

    def parse_expression(expr: str) -> evolve.ASTNode:
        expr = expr.strip()

        if expr in evolve.CONSTANTS:
            return evolve.ASTNode(op="named_const", value=expr)

        if expr == "n":
            return evolve.ASTNode(op="var", value="n")

        try:
            value = int(expr)
            return evolve.ASTNode(op="const", value=value)
        except ValueError:
            pass

        try:
            value = float(expr)
            return evolve.ASTNode(op="const", value=value)
        except ValueError:
            pass

        unary_ops = [
            "floor",
            "ceil",
            "sqrt",
            "abs",
            "sin",
            "cos",
            "tan",
            "log",
            "log2",
            "log10",
            "factorial",
            "sinh",
            "cosh",
            "tanh",
            "asin",
            "acos",
            "atan",
            "neg",
        ]

        for op in unary_ops:
            if expr.startswith(f"{op}(") and expr.endswith(")"):
                inner = expr[len(op) + 1 : -1]
                return evolve.ASTNode(op=op, left=parse_expression(inner))

        if expr.startswith("nroot(") and expr.endswith(")"):
            inner = expr[6:-1]
            parts = split_on_comma(inner)
            if len(parts) == 2:
                return evolve.ASTNode(op="nroot", left=parse_expression(parts[0]), right=parse_expression(parts[1]))

        if expr.startswith("log_{") and "}" in expr:
            close_brace = expr.index("}")
            base = expr[5:close_brace]
            if expr[close_brace + 1] == "(" and expr.endswith(")"):
                inner = expr[close_brace + 2 : -1]
                return evolve.ASTNode(op="logbase", left=parse_expression(inner), right=parse_expression(base))

        if expr.startswith("(") and expr.endswith(")"):
            expr = expr[1:-1]

            binary_ops = ["<<", ">>", "**", "//", "+", "-", "*", "/", "%", "&", "|", "^", "min", "max"]

            for op in binary_ops:
                parts = split_on_operator(expr, op)
                if len(parts) == 2:
                    return evolve.ASTNode(op=op, left=parse_expression(parts[0]), right=parse_expression(parts[1]))

        raise ValueError(f"Could not parse expression: {expr}")

    def split_on_comma(expr: str):
        parts = []
        current = ""
        depth = 0

        for char in expr:
            if char == "," and depth == 0:
                parts.append(current.strip())
                current = ""
            else:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                current += char

        if current:
            parts.append(current.strip())

        return parts

    def split_on_operator(expr: str, op: str):
        depth = 0
        i = 0

        while i < len(expr):
            if depth == 0 and expr[i : i + len(op)] == op:
                left = expr[:i].strip()
                right = expr[i + len(op) :].strip()
                return [left, right]

            if expr[i] == "(":
                depth += 1
            elif expr[i] == ")":
                depth -= 1

            i += 1

        return []

    return parse_expression(expression)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <n>")
        sys.exit(1)

    try:
        max_n = int(sys.argv[1])
    except ValueError:
        print("Error: argument must be an integer")
        sys.exit(1)

    expression = load_best_expression_from_seeds()
    print(f"Testing expression: {expression}\n")

    ast = expression_to_ast(expression)

    print(f"{'n':<5} {'result':<10} {'expected':<10} {'diff':<10} {'is_prime':<10}")
    print("-" * 55)

    for n in range(1, max_n + 1):
        result = evolve.evaluate_ast(ast, n)
        expected = evolve.get_nth_prime(n)

        if result is None:
            diff = "N/A"
            is_prime = "N/A"
        else:
            diff = result - expected
            is_prime = evolve.is_prime(result)

        print(f"{n:<5} {str(result):<10} {expected:<10} {str(diff):<10} {str(is_prime):<10}")
