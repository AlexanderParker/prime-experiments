import random
import math
import sys
from typing import Any, Tuple, Union, List, Dict, Optional
from dataclasses import dataclass

sys.setrecursionlimit(50000)

_PRIME_CACHE = []


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
    """Get the nth prime number (1-indexed) with caching."""
    global _PRIME_CACHE

    # If we already have this prime cached, return it
    if n <= len(_PRIME_CACHE):
        return _PRIME_CACHE[n - 1]

    # Otherwise, compute primes up to the nth one
    if len(_PRIME_CACHE) == 0:
        current = 2
    else:
        current = _PRIME_CACHE[-1] + 1

    while len(_PRIME_CACHE) < n:
        if is_prime(current):
            _PRIME_CACHE.append(current)
        current += 1

    return _PRIME_CACHE[n - 1]


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
}

INTEGER_CONSTANTS = list(range(1, 11))


def create_random_ast(depth: int, max_depth: int, n_var: str = "n") -> ASTNode:
    """Create a random mathematical AST."""
    if depth >= max_depth or (depth > 0 and random.random() < 0.3):
        choice = random.random()
        if choice < 0.5:
            return ASTNode(op="var", value=n_var)
        else:
            const_name = random.choice(list(CONSTANTS.keys()))
            return ASTNode(op="named_const", value=const_name)

    binary_ops = ["+", "-", "*", "/", "**", "nroot", "logbase"]
    unary_ops = ["abs", "neg", "sin", "cos", "tan", "asin", "acos", "atan"]

    if random.random() < 0.7:
        op = random.choice(binary_ops)
        left = create_random_ast(depth + 1, max_depth, n_var)

        if op in ["logbase", "**", "nroot"] and random.random() < 0.3:
            right = ASTNode(op="int_const", value=random.choice(INTEGER_CONSTANTS))
        else:
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

        if current.op in ["var", "const", "named_const", "int_const"]:
            continue
        elif current.op in ["abs", "neg", "sin", "cos", "tan", "asin", "acos", "atan"]:
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
    if node.op in ["var", "const", "named_const", "int_const"]:
        return 1
    if node.op in ["abs", "neg", "sin", "cos", "tan", "asin", "acos", "atan"]:
        return 1 + count_nodes(node.left)
    return 1 + count_nodes(node.left) + count_nodes(node.right)


def count_named_constants(node: ASTNode) -> int:
    """Count the number of named constants in the AST."""
    if node.op == "named_const":
        return 1
    if node.op in ["var", "const", "int_const"]:
        return 0
    if node.op in ["abs", "neg", "sin", "cos", "tan", "asin", "acos", "atan"]:
        return count_named_constants(node.left)
    return count_named_constants(node.left) + count_named_constants(node.right)


def count_int_constants(node: ASTNode) -> int:
    """Count the number of integer constants in the AST."""
    if node.op == "int_const":
        return 1
    if node.op in ["var", "const", "named_const"]:
        return 0
    if node.op in ["abs", "neg", "sin", "cos", "tan", "asin", "acos", "atan"]:
        return count_int_constants(node.left)
    return count_int_constants(node.left) + count_int_constants(node.right)


def has_variable(node: ASTNode) -> bool:
    """Check if the AST contains the variable n."""
    if node.op == "var":
        return True
    if node.op in ["const", "named_const", "int_const"]:
        return False
    if node.op in ["abs", "neg", "sin", "cos", "tan", "asin", "acos", "atan"]:
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


def clean_float(value: float) -> Union[int, float]:
    """Round values very close to integers back to integers."""
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return value


def evaluate_ast(node: ASTNode, n: int, is_root: bool = True) -> Union[int, float, None]:
    """Evaluate the AST with a given value of n."""
    try:
        if node.op == "var":
            return n
        if node.op == "const":
            return node.value
        if node.op == "named_const":
            return clean_float(CONSTANTS[node.value])
        if node.op == "int_const":
            return node.value

        if node.op in ["abs", "neg", "sin", "cos", "tan", "asin", "acos", "atan"]:
            left_val = evaluate_ast(node.left, n, is_root=False)

            if left_val is None:
                return None

            if not isinstance(left_val, (int, float)):
                return None

            if node.op == "abs":
                result = abs(left_val)
            elif node.op == "neg":
                result = -left_val
            elif node.op == "sin":
                result = math.sin(left_val)
            elif node.op == "cos":
                result = math.cos(left_val)
            elif node.op == "tan":
                result = math.tan(left_val)
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

            if abs(result) > 10**9:
                return None

            return clean_float(result)

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
        elif node.op == "**":
            if right_val > 100 or right_val < 0:
                return None
            if abs(left_val) > 1000:
                return None
            result = left_val**right_val
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
        else:
            return None

        if abs(result) > 10**9:
            return None

        return clean_float(result)
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
    first_mismatch_penalty = 0
    lookahead_matches = 0
    nan_result = False
    test_limit = stop_limit if stop_limit is not None else max_test

    # Find sequential matches
    for n in range(1, test_limit + 1):
        result = evaluate_ast(node, n)
        if result is None:
            break
        if not isinstance(result, (int, float)):
            break

        expected_prime = get_nth_prime(n)

        if abs(result - round(result)) < 1e-9:
            rounded_result = int(round(result))
            if rounded_result == expected_prime:
                matches += 1
                match_score += n**match_weight_factor
            else:
                first_mismatch_penalty = abs(rounded_result - expected_prime)
                break
        else:
            rounded_result = int(round(result))
            first_mismatch_penalty = abs(rounded_result - expected_prime)
            break

    # Lookahead scoring: check (matches * 2) values after last sequential match
    if matches > 0:
        lookahead_start = matches + 1
        lookahead_count = matches * 2
        lookahead_end = min(lookahead_start + lookahead_count, test_limit + 1)

        for n in range(lookahead_start, lookahead_end):
            result = evaluate_ast(node, n)
            if result is None:
                continue
            if not isinstance(result, (int, float)):
                nan_result = True
                continue

            if abs(result - round(result)) < 1e-9:
                rounded_result = int(round(result))
                expected_prime = get_nth_prime(n)
                if rounded_result == expected_prime:
                    lookahead_matches += 1

    complexity = count_nodes(node)
    named_const_count = count_named_constants(node)
    int_const_count = count_int_constants(node)
    longrange_check = evaluate_ast(node, n ** n)

    if not isinstance(longrange_check, (int, float)) or nan_result:
        fitness = 0.0
    elif match_score == 0:
        fitness = float("inf")
    else:
        fitness = (
            -match_score * 10
            - lookahead_matches
            + complexity
            + named_const_count
            + (int_const_count * 3)
            + first_mismatch_penalty
        )

    return fitness, matches, complexity, match_score


def ast_to_string(node: ASTNode) -> str:
    """Convert AST to string representation."""
    if node.op == "var":
        return node.value
    if node.op == "const":
        return str(node.value)
    if node.op == "named_const":
        return node.value
    if node.op == "int_const":
        return str(node.value)

    if node.op in ["abs", "neg", "sin", "cos", "tan", "asin", "acos", "atan"]:
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
    if node.op in ["var", "const", "named_const", "int_const"]:
        return ASTNode(op=node.op, value=node.value)
    if node.op in ["abs", "neg", "sin", "cos", "tan", "asin", "acos", "atan"]:
        return ASTNode(op=node.op, left=copy_ast(node.left))
    return ASTNode(op=node.op, left=copy_ast(node.left), right=copy_ast(node.right))


def get_all_nodes(node: ASTNode) -> List[ASTNode]:
    """Get a list of all nodes in the tree."""
    nodes = [node]
    if node.op in ["abs", "neg", "sin", "cos", "tan", "asin", "acos", "atan"]:
        nodes.extend(get_all_nodes(node.left))
    elif node.op not in ["var", "const", "named_const", "int_const"]:
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

    if node.op in ["abs", "neg", "sin", "cos", "tan", "asin", "acos", "atan"]:
        paths.extend(get_all_paths(node.left, current_path + [0]))
    elif node.op not in ["var", "const", "named_const", "int_const"]:
        paths.extend(get_all_paths(node.left, current_path + [0]))
        paths.extend(get_all_paths(node.right, current_path + [1]))

    return paths


def validate_int_constants(node: ASTNode) -> bool:
    """Check if all int_const nodes are only used in allowed operators."""
    if node.op in ["var", "const", "named_const", "int_const"]:
        return True

    if node.op in ["abs", "neg", "sin", "cos", "tan", "asin", "acos", "atan"]:
        return validate_int_constants(node.left)

    # Binary operators
    left_valid = validate_int_constants(node.left)
    right_valid = validate_int_constants(node.right)

    if not left_valid or not right_valid:
        return False

    # Check if right child is int_const and operator allows it
    if node.right and node.right.op == "int_const":
        if node.op not in ["**", "nroot", "logbase"]:
            return False

    # Check if left child is int_const (shouldn't happen in valid expressions)
    if node.left and node.left.op == "int_const":
        return False

    return True


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
                result = copy_ast(subtree_from_parent2)
                if validate_int_constants(result):
                    return result
        else:
            subtree_to_replace = get_node_at_path(parent1, path1)
            old_subtree_depth = get_tree_depth(subtree_to_replace)

            parent1_depth = get_tree_depth(parent1)
            estimated_new_depth = parent1_depth - old_subtree_depth + subtree_depth

            if estimated_new_depth <= max_depth:
                result = replace_node_at_path(parent1, path1, subtree_from_parent2)
                if validate_int_constants(result):
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
                if node.op in ["abs", "neg", "sin", "cos", "tan", "asin", "acos", "atan"]:
                    return copy_ast(node.left)
                elif node.op not in ["var", "const", "named_const", "int_const"]:
                    return copy_ast(node.left) if random.random() < 0.5 else copy_ast(node.right)
            else:
                paths = get_all_paths(node)
                non_leaf_paths = []
                for path in paths:
                    if not path:
                        continue
                    node_at_path = get_node_at_path(node, path)
                    if node_at_path.op not in ["var", "const", "named_const", "int_const"]:
                        non_leaf_paths.append(path)

                if non_leaf_paths:
                    path = random.choice(non_leaf_paths)
                    node_to_prune = get_node_at_path(node, path)

                    if node_to_prune.op in ["abs", "neg", "sin", "cos", "tan", "asin", "acos", "atan"]:
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

                binary_ops = ["+", "-", "*", "/", "**"]
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
            if node.op in ["var", "const", "named_const", "int_const"]:
                choice = random.random()
                if choice < 0.5:
                    return ASTNode(op="var", value="n")
                else:
                    const_name = random.choice(list(CONSTANTS.keys()))
                    return ASTNode(op="named_const", value=const_name)
            elif node.op in ["abs", "neg", "sin", "cos", "tan", "asin", "acos", "atan"]:
                unary_ops = ["abs", "neg", "sin", "cos", "tan", "asin", "acos", "atan"]
                node.op = random.choice(unary_ops)
            else:
                has_int_const_right = node.right and node.right.op == "int_const"

                if has_int_const_right:
                    allowed_ops = ["**", "nroot", "logbase"]
                else:
                    allowed_ops = ["+", "-", "*", "/", "**", "nroot", "logbase"]

                node.op = random.choice(allowed_ops)

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

    if node.op in ["abs", "neg", "sin", "cos", "tan", "asin", "acos", "atan"]:
        child_depth = get_tree_depth(node.left)
        if child_depth < max_depth:
            node.left = mutate_ast(node.left, mutation_rate, prune_rate, duplicate_mutate_rate, max_depth)
    elif node.op not in ["var", "const", "named_const", "int_const"]:
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
        max_depth=20,
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
