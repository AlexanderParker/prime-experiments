# Updated best.py with unique expression handling and email notifications
import math
import random
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import evolve
import csv
import os
from datetime import datetime
import time
import sys
import signal
import email_notifier


@dataclass
class HyperParams:
    """Hyperparameters for the genetic algorithm."""

    population_size: int
    generations: int
    max_depth: int
    keep_pct: float
    crossover_pct: float
    random_pct: float
    match_weight_factor: float
    mutation_rate: float
    prune_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "population_size": self.population_size,
            "generations": self.generations,
            "max_depth": self.max_depth,
            "keep_pct": self.keep_pct,
            "crossover_pct": self.crossover_pct,
            "random_pct": self.random_pct,
            "match_weight_factor": self.match_weight_factor,
            "mutation_rate": self.mutation_rate,
            "prune_rate": self.prune_rate,
        }


def load_best_hyperparams_from_csv(csv_filename: str) -> HyperParams:
    """Load the best hyperparameters from a hyperparameter evolution CSV file."""
    best_fitness = float("inf")
    best_params = None

    with open(csv_filename, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            fitness = float(row["fitness"])
            if fitness < best_fitness:
                best_fitness = fitness
                best_params = HyperParams(
                    population_size=int(row["population_size"]),
                    generations=int(row["generations"]),
                    max_depth=int(row["max_depth"]),
                    keep_pct=float(row["keep_pct"]),
                    crossover_pct=float(row["crossover_pct"]),
                    random_pct=float(row["random_pct"]),
                    match_weight_factor=float(row["match_weight_factor"]),
                    mutation_rate=float(row["mutation_rate"]),
                    prune_rate=float(row["prune_rate"]),
                )

    if best_params is None:
        raise ValueError(f"No valid hyperparameters found in {csv_filename}")

    return best_params


def load_seeds_from_csv(seeds_filename: str = "seeds.csv") -> List[Tuple[float, int, str]]:
    """Load unique seeds from seeds.csv file."""
    seeds = []
    seen_expressions = set()

    with open(seeds_filename, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            expression = row["expression"]
            if expression not in seen_expressions:
                fitness = float(row["fitness"])
                matches = int(row["matches"])
                seeds.append((fitness, matches, expression))
                seen_expressions.add(expression)

    seeds.sort(key=lambda x: x[0])
    return seeds


def save_seeds_to_csv(seeds: List[Tuple[float, int, str]], seeds_filename: str = "seeds.csv"):
    """Save only unique seeds to seeds.csv file."""
    seen_expressions = set()
    unique_seeds = []

    for fitness, matches, expression in seeds:
        if expression not in seen_expressions:
            unique_seeds.append((fitness, matches, expression))
            seen_expressions.add(expression)

    with open(seeds_filename, "w", newline="") as csvfile:
        fieldnames = ["fitness", "matches", "expression"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for fitness, matches, expression in unique_seeds:
            writer.writerow({"fitness": fitness, "matches": matches, "expression": expression})


def load_top_expressions_from_csv(csv_filename: str) -> List[Tuple[float, int, str]]:
    """Load the top scoring unique expressions from a hyperparameter evolution CSV file."""
    rows = []
    seen_expressions = set()

    with open(csv_filename, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            expression = row["expression"]
            if expression not in seen_expressions:
                fitness = float(row["fitness"])
                matches = int(row["matches"])
                rows.append((fitness, matches, expression))
                seen_expressions.add(expression)

    rows.sort(key=lambda x: x[0])
    return rows


def expression_to_ast(expression: str) -> Optional[evolve.ASTNode]:
    """Convert a string expression back to an AST."""
    try:
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

        def split_on_comma(expr: str) -> List[str]:
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

        def split_on_operator(expr: str, op: str) -> List[str]:
            parts = []
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

    except Exception:
        return None


def find_latest_hyperparameter_csv() -> str:
    """Find the most recent hyperparameter_evolution CSV file."""
    files = [f for f in os.listdir(".") if f.startswith("hyperparameter_evolution_") and f.endswith(".csv")]

    if not files:
        raise FileNotFoundError("No hyperparameter_evolution CSV files found in current directory")

    files.sort(reverse=True)
    return files[0]


def evaluate_hyperparams(
    params: HyperParams, stop_limit: int = 100, seed_asts: List[evolve.ASTNode] = None
) -> Tuple[float, int, str, evolve.ASTNode, float]:
    """Evaluate hyperparameters by running the genetic algorithm with seed ASTs."""
    start_time = time.time()

    results = evolve.genetic_algorithm(
        population_size=params.population_size,
        generations=params.generations,
        max_depth=params.max_depth,
        stop_limit=stop_limit,
        keep_pct=params.keep_pct,
        crossover_pct=params.crossover_pct,
        random_pct=params.random_pct,
        match_weight_factor=params.match_weight_factor,
        mutation_rate=params.mutation_rate,
        prune_rate=params.prune_rate,
        verbose=False,
        seed_asts=seed_asts,
    )

    elapsed_time = time.time() - start_time

    return results["best_fitness"], results["best_matches"], results["expression"], results["best_ast"], elapsed_time


def update_seeds(
    seeds: List[Tuple[float, int, str]], new_fitness: float, new_matches: int, new_expression: str
) -> Tuple[List[Tuple[float, int, str]], bool]:
    """Update seeds list with new result if it's unique. Returns updated seeds and whether it was added."""
    existing_expressions = {expr for _, _, expr in seeds}

    if new_expression in existing_expressions:
        return seeds, False

    seeds.append((new_fitness, new_matches, new_expression))
    seeds.sort(key=lambda x: x[0])

    return seeds, True


if __name__ == "__main__":

    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal, terminating...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    seeds_filename = "seeds.csv"
    seeds = []

    try:
        if os.path.exists(seeds_filename):
            print(f"Loading seeds from: {seeds_filename}")
            seeds = load_seeds_from_csv(seeds_filename)
            print(f"Loaded {len(seeds)} unique seeds")
        else:
            print(f"{seeds_filename} not found, loading from hyperparameter evolution file")
            csv_filename = find_latest_hyperparameter_csv()
            print(f"Loading from: {csv_filename}")
            seeds = load_top_expressions_from_csv(csv_filename)
            print(f"Loaded {len(seeds)} unique expressions")

            print(f"Saving initial seeds to: {seeds_filename}")
            save_seeds_to_csv(seeds, seeds_filename)

        csv_filename = find_latest_hyperparameter_csv()
        print(f"Loading best hyperparameters from: {csv_filename}")
        best_params = load_best_hyperparams_from_csv(csv_filename)

        print("\n" + "=" * 80)
        print("TOP SEED ASTs")
        print("=" * 80)
        for i, (fitness, matches, expr) in enumerate(seeds[:10]):
            print(f"{i+1}. Fitness: {fitness:.4f}, Matches: {matches}, Expression: {expr}")
        if len(seeds) > 10:
            print(f"... and {len(seeds) - 10} more")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("BEST HYPERPARAMETERS LOADED")
    print("=" * 80)
    print(f"population_size: {best_params.population_size}")
    print(f"generations: {best_params.generations}")
    print(f"max_depth: {best_params.max_depth}")
    print(f"keep_pct: {best_params.keep_pct:.3f}")
    print(f"crossover_pct: {best_params.crossover_pct:.3f}")
    print(f"random_pct: {best_params.random_pct:.3f}")
    print(f"match_weight_factor: {best_params.match_weight_factor:.3f}")
    print(f"mutation_rate: {best_params.mutation_rate:.3f}")
    print(f"prune_rate: {best_params.prune_rate:.3f}")

    print("\n" + "=" * 80)
    print("RUNNING INDEFINITELY WITH BEST HYPERPARAMETERS")
    print("Press Ctrl+C to stop")
    print("=" * 80)

    run_count = 0
    best_ever_fitness = float("inf")
    best_ever_matches = 0
    best_ever_expression = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    infinite_csv_filename = f"infinite_runs_{timestamp}.csv"

    with open(infinite_csv_filename, "w", newline="") as csvfile:
        fieldnames = ["run", "fitness", "matches", "elapsed_time_seconds", "expression", "is_new_best"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        try:
            while True:
                run_count += 1
                print(f"\n{'='*80}")
                print(f"Run #{run_count} with best hyperparameters")

                seeds = load_seeds_from_csv(seeds_filename)

                if len(seeds) > 0:
                    # Randomly choose split method
                    split_method = random.choice(["thirds", "quarters"])

                    if split_method == "thirds":
                        third = max(1, len(seeds) // 3)
                        tiers = {"top": seeds[:third], "middle": seeds[third : third * 2], "bottom": seeds[third * 2 :]}
                    else:  # quarters
                        quarter = max(1, len(seeds) // 4)
                        tiers = {
                            "top": seeds[:quarter],
                            "upper_mid": seeds[quarter : quarter * 2],
                            "lower_mid": seeds[quarter * 2 : quarter * 3],
                            "bottom": seeds[quarter * 3 :],
                        }

                    # Pick random tier from available
                    tier_name = random.choice(list(tiers.keys()))
                    selected_tier = tiers[tier_name]

                    # Handle empty tier
                    if not selected_tier:
                        selected_tier = seeds
                        tier_name = "all"

                    # Uniformly sample from selected tier
                    num_seeds_to_use = min(50, len(selected_tier))
                    sampled_seeds = random.sample(selected_tier, num_seeds_to_use)

                    print(f"Split: {split_method}, Tier: {tier_name} ({len(selected_tier)} available)")
                else:
                    sampled_seeds = []
                    tier_name = "none"

                seed_asts = []
                for fitness, matches, expr in sampled_seeds:
                    ast = expression_to_ast(expr)
                    if ast is not None:
                        seed_asts.append(ast)

                print(f"Seeding with {len(seed_asts)} ASTs from {tier_name} tier out of {len(seeds)} total seeds")
                print(f"{'='*80}\n")

                fitness, matches, expression, best_ast, elapsed_time = evaluate_hyperparams(
                    best_params, stop_limit=1000, seed_asts=seed_asts
                )

                print(f"Fitness: {fitness:.4f}, Matches: {matches}")
                print(f"Elapsed time: {elapsed_time:.2f} seconds")
                print(f"Expression: {expression}")

                is_new_best = False
                seeds, was_added = update_seeds(seeds, fitness, matches, expression)

                if was_added:
                    print(f"\n*** NEW UNIQUE SEED ADDED TO seeds.csv ***")
                    save_seeds_to_csv(seeds, seeds_filename)

                if fitness < best_ever_fitness:
                    best_ever_fitness = fitness
                    best_ever_matches = matches
                    best_ever_expression = expression
                    is_new_best = True
                    print(f"\n*** NEW OVERALL BEST FITNESS: {best_ever_fitness:.4f} ***")
                    print(f"*** MATCHES: {best_ever_matches} ***")
                    print(f"*** EXPRESSION: {best_ever_expression} ***")

                    # Send email notification for new unique best match
                    if was_added:
                        print("Sending email notification...")
                        email_notifier.send_new_match_notification(
                            matches=best_ever_matches,
                            fitness=best_ever_fitness,
                            expression=best_ever_expression,
                            run=run_count,
                        )

                writer.writerow(
                    {
                        "run": run_count,
                        "fitness": fitness,
                        "matches": matches,
                        "elapsed_time_seconds": elapsed_time,
                        "expression": expression,
                        "is_new_best": is_new_best,
                    }
                )
                csvfile.flush()

                print(f"\nBest ever fitness: {best_ever_fitness:.4f}")
                print(f"Best ever matches: {best_ever_matches}")
                print(f"Best ever expression: {best_ever_expression}")

        except KeyboardInterrupt:
            print("\n\n" + "=" * 80)
            print("STOPPED BY USER")
            print("=" * 80)
            print(f"Total runs completed: {run_count}")
            print(f"\nBest ever fitness: {best_ever_fitness:.4f}")
            print(f"Best ever matches: {best_ever_matches}")
            print(f"Best ever expression: {best_ever_expression}")
            print(f"\nInfinite runs log saved to: {infinite_csv_filename}")

# .env file example for email configuration
# SMTP_SERVER=smtp.gmail.com
# SMTP_PORT=587
# SENDER_EMAIL=your_email@gmail.com
# SENDER_PASSWORD=your_app_password
# RECIPIENT_EMAIL=recipient@example.com
