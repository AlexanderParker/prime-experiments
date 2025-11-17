import random
import math
import sys
from typing import Any, Tuple, Union, List, Dict, Optional
from dataclasses import dataclass

sys.setrecursionlimit(50000)


def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def get_nth_prime(n: int) -> int:
    """Get the nth prime number (1-indexed)."""
    count = 0
    num = 2
    while count < n:
        if is_prime(num):
            count += 1
            if count == n:
                return num
        num += 1
    return num


@dataclass
class ASTNode:
    """Represents a node in the mathematical AST."""

    op: str
    left: Any = None
    right: Any = None
    value: Any = None


CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "phi": (1 + math.sqrt(5)) / 2,
    "euler_gamma": 0.5772156649,
    "catalan": 0.915965594,
    "c": 299792458,
    "h": 6.62607015e-34,
    "G": 6.67430e-11,
    "avogadro": 6.02214076e23,
    "boltzmann": 1.380649e-23,
}


def create_random_ast(depth: int, max_depth: int, n_var: str = "n") -> ASTNode:
    """Create a random mathematical AST."""
    if depth >= max_depth or (depth > 0 and random.random() < 0.3):
        choice = random.random()
        if choice < 0.5:
            return ASTNode(op="var", value=n_var)
        else:
            const_name = random.choice(list(CONSTANTS.keys()))
            return ASTNode(op="named_const", value=const_name)

    binary_ops = ["+", "-", "*", "/", "//", "%", "**", "&", "|", "^", "<<", ">>", "nroot", "logbase", "min", "max"]
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

    if random.random() < 0.7:
        op = random.choice(binary_ops)
        left = create_random_ast(depth + 1, max_depth, n_var)
        right = create_random_ast(depth + 1, max_depth, n_var)
        return ASTNode(op=op, left=left, right=right)
    else:
        op = random.choice(unary_ops)
        left = create_random_ast(depth + 1, max_depth, n_var)
        return ASTNode(op=op, left=left)


def get_tree_depth(node: ASTNode) -> int:
    """Calculate the depth of the tree iteratively to avoid recursion issues."""
    if node is None:
        return 0

    max_depth = 0
    stack = [(node, 1)]

    while stack:
        current, depth = stack.pop()
        max_depth = max(max_depth, depth)

        if current.op in ["var", "const", "named_const"]:
            continue
        elif current.op in [
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
            "sinh",
            "cosh",
            "tanh",
            "asin",
            "acos",
            "atan",
            "neg",
        ]:
            if current.left:
                stack.append((current.left, depth + 1))
        else:
            if current.left:
                stack.append((current.left, depth + 1))
            if current.right:
                stack.append((current.right, depth + 1))

    return max_depth


def count_nodes(node: ASTNode) -> int:
    """Count the number of nodes in the AST."""
    if node.op in ["var", "const", "named_const"]:
        return 1
    if node.op in [
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
    ]:
        return 1 + count_nodes(node.left)
    return 1 + count_nodes(node.left) + count_nodes(node.right)


def has_variable(node: ASTNode) -> bool:
    """Check if the AST contains the variable n."""
    if node.op == "var":
        return True
    if node.op in ["const", "named_const"]:
        return False
    if node.op in [
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
    ]:
        return has_variable(node.left)
    return has_variable(node.left) or has_variable(node.right)


def is_trivial_solution(node: ASTNode) -> bool:
    """Check if the solution is trivial (constant or n+1)."""
    if not has_variable(node):
        return True

    if node.op == "+":
        if node.left.op == "var" and node.right.op == "const" and node.right.value == 1:
            return True
        if node.right.op == "var" and node.left.op == "const" and node.left.value == 1:
            return True

    return False


def evaluate_ast(node: ASTNode, n: int, is_root: bool = True) -> Union[int, float, None]:
    """Evaluate the AST with a given value of n."""
    try:
        if node.op == "var":
            return n
        if node.op == "const":
            return node.value
        if node.op == "named_const":
            const_val = CONSTANTS[node.value]
            return const_val

        if node.op in [
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
        ]:
            left_val = evaluate_ast(node.left, n, is_root=False)

            if left_val is None:
                return None

            if not isinstance(left_val, (int, float)):
                return None

            if node.op == "floor":
                result = math.floor(left_val)
            elif node.op == "ceil":
                result = math.ceil(left_val)
            elif node.op == "sqrt":
                if left_val < 0:
                    return None
                result = math.sqrt(left_val)
            elif node.op == "abs":
                result = abs(left_val)
            elif node.op == "sin":
                result = math.sin(left_val)
            elif node.op == "cos":
                result = math.cos(left_val)
            elif node.op == "tan":
                result = math.tan(left_val)
            elif node.op == "sinh":
                result = math.sinh(left_val)
            elif node.op == "cosh":
                result = math.cosh(left_val)
            elif node.op == "tanh":
                result = math.tanh(left_val)
            elif node.op == "asin":
                if left_val < -1 or left_val > 1:
                    return None
                result = math.asin(left_val)
            elif node.op == "acos":
                if left_val < -1 or left_val > 1:
                    return None
                result = math.acos(left_val)
            elif node.op == "atan":
                result = math.atan(left_val)
            elif node.op == "log":
                if left_val <= 0:
                    return None
                result = math.log(left_val)
            elif node.op == "log2":
                if left_val <= 0:
                    return None
                result = math.log2(left_val)
            elif node.op == "log10":
                if left_val <= 0:
                    return None
                result = math.log10(left_val)
            elif node.op == "factorial":
                if not isinstance(left_val, int) or left_val < 0 or left_val > 20:
                    return None
                result = math.factorial(left_val)
            elif node.op == "neg":
                result = -left_val

            if abs(result) > 10**9:
                return None

            if is_root:
                result = math.floor(result)
                if not isinstance(result, int):
                    return None

            return result

        left_val = evaluate_ast(node.left, n, is_root=False)
        right_val = evaluate_ast(node.right, n, is_root=False)

        if left_val is None or right_val is None:
            return None

        if not isinstance(left_val, (int, float)) or not isinstance(right_val, (int, float)):
            return None

        if node.op == "+":
            result = left_val + right_val
        elif node.op == "-":
            result = left_val - right_val
        elif node.op == "*":
            result = left_val * right_val
        elif node.op == "/":
            if right_val == 0:
                return None
            result = left_val / right_val
        elif node.op == "//":
            if right_val == 0:
                return None
            result = left_val // right_val
        elif node.op == "%":
            if right_val == 0:
                return None
            result = left_val % right_val
        elif node.op == "**":
            if right_val > 100 or right_val < 0:
                return None
            if abs(left_val) > 1000:
                return None
            result = left_val**right_val
        elif node.op == "&":
            if not isinstance(left_val, int) or not isinstance(right_val, int):
                return None
            result = left_val & right_val
        elif node.op == "|":
            if not isinstance(left_val, int) or not isinstance(right_val, int):
                return None
            result = left_val | right_val
        elif node.op == "^":
            if not isinstance(left_val, int) or not isinstance(right_val, int):
                return None
            result = left_val ^ right_val
        elif node.op == "<<":
            if not isinstance(left_val, int) or not isinstance(right_val, int):
                return None
            if right_val < 0 or right_val > 30:
                return None
            result = left_val << right_val
        elif node.op == ">>":
            if not isinstance(left_val, int) or not isinstance(right_val, int):
                return None
            if right_val < 0 or right_val > 30:
                return None
            result = left_val >> right_val
        elif node.op == "nroot":
            if right_val <= 0 or right_val > 20:
                return None
            if left_val < 0 and right_val % 2 == 0:
                return None
            if left_val < 0:
                result = -(abs(left_val) ** (1.0 / right_val))
            else:
                result = left_val ** (1.0 / right_val)
        elif node.op == "logbase":
            if left_val <= 0 or right_val <= 0 or right_val == 1:
                return None
            result = math.log(left_val) / math.log(right_val)
        elif node.op == "min":
            result = min(left_val, right_val)
        elif node.op == "max":
            result = max(left_val, right_val)
        else:
            return None

        if abs(result) > 10**9:
            return None

        if is_root:
            result = math.floor(result)
            if not isinstance(result, int):
                return None

        return result
    except (OverflowError, ValueError, ZeroDivisionError, TypeError):
        return None


def calculate_fitness(
    node: ASTNode, max_test: int = 20, stop_limit: int = None, match_weight_factor: float = 1.0
) -> Tuple[float, int, int, float]:
    """Calculate fitness based on matches and complexity."""
    if is_trivial_solution(node):
        return float("inf"), 0, count_nodes(node), 0.0

    matches = 0
    match_score = 0.0
    test_limit = stop_limit if stop_limit is not None else max_test

    for n in range(1, test_limit + 1):
        result = evaluate_ast(node, n)
        if result is None:
            break
        expected_prime = get_nth_prime(n)
        if result == expected_prime:
            matches += 1
            match_score += n**match_weight_factor
        else:
            break

    complexity = count_nodes(node)
    constant_count = count_constants(node)
    non_constant_nodes = complexity - constant_count

    if match_score == 0:
        fitness = float("inf")
    else:
        fitness = -match_score * 10 + complexity - non_constant_nodes

    return fitness, matches, complexity, match_score


def count_constants(node: ASTNode) -> int:
    """Count the number of named constants in the AST."""
    if node.op == "named_const":
        return 1
    if node.op in ["var", "const"]:
        return 0
    if node.op in [
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
    ]:
        return count_constants(node.left)
    return count_constants(node.left) + count_constants(node.right)


def ast_to_string(node: ASTNode) -> str:
    """Convert AST to string representation."""
    if node.op == "var":
        return node.value
    if node.op == "const":
        return str(node.value)
    if node.op == "named_const":
        return node.value

    if node.op in [
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
    ]:
        left_str = ast_to_string(node.left)
        return f"{node.op}({left_str})"

    left_str = ast_to_string(node.left)
    right_str = ast_to_string(node.right)

    if node.op == "nroot":
        return f"nroot({left_str}, {right_str})"

    if node.op == "logbase":
        return f"log_{{{right_str}}}({left_str})"

    return f"({left_str} {node.op} {right_str})"


def copy_ast(node: ASTNode) -> ASTNode:
    """Create a deep copy of an AST."""
    if node.op in ["var", "const", "named_const"]:
        return ASTNode(op=node.op, value=node.value)
    if node.op in [
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
    ]:
        return ASTNode(op=node.op, left=copy_ast(node.left))
    return ASTNode(op=node.op, left=copy_ast(node.left), right=copy_ast(node.right))


def get_all_nodes(node: ASTNode) -> List[ASTNode]:
    """Get a list of all nodes in the tree."""
    nodes = [node]
    if node.op in [
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
    ]:
        nodes.extend(get_all_nodes(node.left))
    elif node.op not in ["var", "const", "named_const"]:
        nodes.extend(get_all_nodes(node.left))
        nodes.extend(get_all_nodes(node.right))
    return nodes


def replace_node_at_path(node: ASTNode, path: List[int], replacement: ASTNode) -> ASTNode:
    """Replace a node at a specific path with a replacement node."""
    if not path:
        return copy_ast(replacement)

    node = copy_ast(node)
    current = node

    for i, direction in enumerate(path[:-1]):
        if direction == 0:
            current = current.left
        else:
            current = current.right

    if path[-1] == 0:
        current.left = copy_ast(replacement)
    else:
        current.right = copy_ast(replacement)

    return node


def get_node_at_path(node: ASTNode, path: List[int]) -> ASTNode:
    """Get the node at a specific path."""
    current = node
    for direction in path:
        if direction == 0:
            current = current.left
        else:
            current = current.right
    return current


def get_all_paths(node: ASTNode, current_path: List[int] = None) -> List[List[int]]:
    """Get all paths to nodes in the tree."""
    if current_path is None:
        current_path = []

    paths = [current_path[:]]

    if node.op in [
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
    ]:
        paths.extend(get_all_paths(node.left, current_path + [0]))
    elif node.op not in ["var", "const", "named_const"]:
        paths.extend(get_all_paths(node.left, current_path + [0]))
        paths.extend(get_all_paths(node.right, current_path + [1]))

    return paths


def crossover(parent1: ASTNode, parent2: ASTNode, max_depth: int = 10) -> ASTNode:
    """Perform crossover by swapping subtrees from both parents, respecting max depth."""
    paths1 = get_all_paths(parent1)
    paths2 = get_all_paths(parent2)

    max_attempts = 20
    for attempt in range(max_attempts):
        path1 = random.choice(paths1)
        path2 = random.choice(paths2)

        subtree_from_parent2 = get_node_at_path(parent2, path2)
        subtree_depth = get_tree_depth(subtree_from_parent2)

        if not path1:
            if subtree_depth <= max_depth:
                return copy_ast(subtree_from_parent2)
        else:
            subtree_to_replace = get_node_at_path(parent1, path1)
            old_subtree_depth = get_tree_depth(subtree_to_replace)

            parent1_depth = get_tree_depth(parent1)
            estimated_new_depth = parent1_depth - old_subtree_depth + subtree_depth

            if estimated_new_depth <= max_depth:
                result = replace_node_at_path(parent1, path1, subtree_from_parent2)
                return result

    return copy_ast(parent1)


def mutate_ast(
    node: ASTNode,
    mutation_rate: float = 0.1,
    prune_rate: float = 0.2,
    duplicate_mutate_rate: float = 0.15,
    max_depth: int = 4,
) -> ASTNode:
    """Mutate an AST with improved mutation strategies including local and branch pruning."""
    current_depth = get_tree_depth(node)

    node = copy_ast(node)

    if random.random() < mutation_rate:
        rand_val = random.random()

        if rand_val < prune_rate:
            mutation_type = "prune"
        elif rand_val < prune_rate + duplicate_mutate_rate:
            mutation_type = "duplicate_mutate"
        elif rand_val < prune_rate + duplicate_mutate_rate + (1 - prune_rate - duplicate_mutate_rate) * 0.33:
            mutation_type = "operator_change"
        elif rand_val < prune_rate + duplicate_mutate_rate + (1 - prune_rate - duplicate_mutate_rate) * 0.66:
            mutation_type = "full_replacement"
        else:
            mutation_type = "subtree_replacement"

        if mutation_type == "prune":
            if random.random() < 0.5:
                if node.op in [
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
                ]:
                    return copy_ast(node.left)
                elif node.op not in ["var", "const", "named_const"]:
                    return copy_ast(node.left) if random.random() < 0.5 else copy_ast(node.right)
            else:
                paths = get_all_paths(node)
                non_leaf_paths = []
                for path in paths:
                    if not path:
                        continue
                    node_at_path = get_node_at_path(node, path)
                    if node_at_path.op not in ["var", "const", "named_const"]:
                        non_leaf_paths.append(path)

                if non_leaf_paths:
                    path = random.choice(non_leaf_paths)
                    node_to_prune = get_node_at_path(node, path)

                    if node_to_prune.op in [
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
                    ]:
                        replacement = node_to_prune.left
                    else:
                        replacement = node_to_prune.left if random.random() < 0.5 else node_to_prune.right

                    node = replace_node_at_path(node, path, replacement)

        elif mutation_type == "duplicate_mutate":
            paths = get_all_paths(node)
            if paths:
                path = random.choice(paths)
                subtree = get_node_at_path(node, path)

                subtree_copy = copy_ast(subtree)
                mutated_copy = mutate_ast(
                    subtree_copy,
                    mutation_rate=1.0,
                    prune_rate=prune_rate,
                    duplicate_mutate_rate=0.0,
                    max_depth=max_depth,
                )

                binary_ops = ["+", "-", "*", "/", "//", "%", "**", "&", "|", "^", "<<", ">>", "min", "max"]
                chosen_op = random.choice(binary_ops)

                new_subtree = ASTNode(op=chosen_op, left=copy_ast(subtree), right=mutated_copy)
                new_subtree_depth = get_tree_depth(new_subtree)

                if not path:
                    if new_subtree_depth <= max_depth:
                        return new_subtree
                else:
                    old_subtree_depth = get_tree_depth(subtree)
                    estimated_new_depth = current_depth - old_subtree_depth + new_subtree_depth

                    if estimated_new_depth <= max_depth:
                        node = replace_node_at_path(node, path, new_subtree)

        elif mutation_type == "operator_change":
            if node.op in ["var", "const", "named_const"]:
                choice = random.random()
                if choice < 0.5:
                    return ASTNode(op="var", value="n")
                else:
                    const_name = random.choice(list(CONSTANTS.keys()))
                    return ASTNode(op="named_const", value=const_name)
            elif node.op in [
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
            ]:
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
                node.op = random.choice(unary_ops)
            else:
                binary_ops = [
                    "+",
                    "-",
                    "*",
                    "/",
                    "//",
                    "%",
                    "**",
                    "&",
                    "|",
                    "^",
                    "<<",
                    ">>",
                    "nroot",
                    "logbase",
                    "min",
                    "max",
                ]
                node.op = random.choice(binary_ops)

        elif mutation_type == "full_replacement":
            return create_random_ast(0, max_depth)

        elif mutation_type == "subtree_replacement":
            paths = get_all_paths(node)
            if len(paths) > 1:
                path = random.choice(paths[1:])
                old_subtree = get_node_at_path(node, path)
                old_subtree_depth = get_tree_depth(old_subtree)
                replacement = create_random_ast(0, max_depth)
                replacement_depth = get_tree_depth(replacement)

                estimated_new_depth = current_depth - old_subtree_depth + replacement_depth
                if estimated_new_depth <= max_depth:
                    node = replace_node_at_path(node, path, replacement)
                else:
                    return node

    if node.op in [
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
    ]:
        child_depth = get_tree_depth(node.left)
        if child_depth < max_depth:
            node.left = mutate_ast(node.left, mutation_rate, prune_rate, duplicate_mutate_rate, max_depth)
    elif node.op not in ["var", "const", "named_const"]:
        left_depth = get_tree_depth(node.left)
        right_depth = get_tree_depth(node.right)
        if left_depth < max_depth:
            node.left = mutate_ast(node.left, mutation_rate, prune_rate, duplicate_mutate_rate, max_depth)
        if right_depth < max_depth:
            node.right = mutate_ast(node.right, mutation_rate, prune_rate, duplicate_mutate_rate, max_depth)

    return node


def genetic_algorithm(
    population_size: int = 100,
    generations: int = 1000,
    max_depth: int = 4,
    stop_limit: int = None,
    keep_pct: float = 0.2,
    crossover_pct: float = 0.6,
    random_pct: float = 0.2,
    match_weight_factor: float = 1.0,
    mutation_rate: float = 0.1,
    prune_rate: float = 0.2,
    duplicate_mutate_rate: float = 0.15,
    verbose: bool = True,
    seed_ast: Optional[ASTNode] = None,
    seed_asts: Optional[List[ASTNode]] = None,
) -> Dict[str, Any]:
    """Run the genetic algorithm and return results."""
    if abs(keep_pct + crossover_pct + random_pct - 1.0) > 0.001:
        raise ValueError("Percentages must sum to 1.0")

    population = []

    if seed_asts is not None:
        for ast in seed_asts:
            population.append(copy_ast(ast))
    elif seed_ast is not None:
        population.append(copy_ast(seed_ast))

    while len(population) < population_size:
        population.append(create_random_ast(0, max_depth))

    best_ever = None
    best_fitness_ever = float("inf")
    best_matches_ever = 0
    best_match_score_ever = 0.0
    best_complexity_ever = 0

    for generation in range(generations):
        fitness_scores = []
        for individual in population:
            fitness, matches, complexity, match_score = calculate_fitness(
                individual, stop_limit=stop_limit, match_weight_factor=match_weight_factor
            )
            fitness_scores.append((fitness, matches, complexity, match_score, individual))

        fitness_scores.sort(key=lambda x: x[0])

        if fitness_scores[0][0] < best_fitness_ever:
            best_fitness_ever = fitness_scores[0][0]
            best_ever = fitness_scores[0][4]
            best_matches_ever = fitness_scores[0][1]
            best_match_score_ever = fitness_scores[0][3]
            best_complexity_ever = fitness_scores[0][2]

            if verbose:
                print(
                    f"Generation {generation}: Best fitness = {best_fitness_ever:.4f}, "
                    f"Matches = {best_matches_ever}, Match Score = {best_match_score_ever:.2f}, Complexity = {best_complexity_ever}"
                )
                print(f"Expression: {ast_to_string(best_ever)}")
                print()

        keep_count = int(population_size * keep_pct)
        crossover_count = int(population_size * crossover_pct)
        random_count = population_size - keep_count - crossover_count

        survivors = [ind for _, _, _, _, ind in fitness_scores[:keep_count]]

        new_population = survivors[:]

        for _ in range(crossover_count):
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            child = crossover(parent1, parent2, max_depth=max_depth)
            child = mutate_ast(
                child,
                mutation_rate=mutation_rate,
                prune_rate=prune_rate,
                duplicate_mutate_rate=duplicate_mutate_rate,
                max_depth=max_depth,
            )
            new_population.append(child)

        for _ in range(random_count):
            new_population.append(create_random_ast(0, max_depth))

        population = new_population

    return {
        "best_ast": best_ever,
        "best_fitness": best_fitness_ever,
        "best_matches": best_matches_ever,
        "best_match_score": best_match_score_ever,
        "best_complexity": best_complexity_ever,
        "expression": ast_to_string(best_ever) if best_ever else None,
    }


if __name__ == "__main__":
    stop_limit = 1000
    match_weight_factor = 1.0

    print(f"Starting genetic algorithm with stop_limit={stop_limit}, match_weight_factor={match_weight_factor}")
    print()

    results = genetic_algorithm(
        population_size=200,
        generations=2000,
        max_depth=4,
        stop_limit=stop_limit,
        keep_pct=0.2,
        crossover_pct=0.6,
        random_pct=0.2,
        match_weight_factor=match_weight_factor,
        mutation_rate=0.1,
        prune_rate=0.2,
        duplicate_mutate_rate=0.15,
        verbose=True,
    )

    print("=" * 80)
    print("Final best solution:")
    print(f"Expression: {results['expression']}")
    print(f"Fitness: {results['best_fitness']:.4f}")
    print(f"Matches: {results['best_matches']}")
    print(f"Match Score: {results['best_match_score']:.2f}")
    print(f"Complexity: {results['best_complexity']}")
    print()

    print("Testing first 10 values:")
    for n in range(1, 11):
        result = evaluate_ast(results["best_ast"], n)
        expected = get_nth_prime(n)
        print(f"n={n}: result={result}, expected={expected}, match={result == expected}")
