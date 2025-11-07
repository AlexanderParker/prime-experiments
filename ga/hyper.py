import random
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import evolve
import csv
import os
from datetime import datetime
import time
import multiprocessing as mp
import signal
import sys


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
        }


def create_random_hyperparams() -> HyperParams:
    """Generate random hyperparameters within sensible ranges."""
    keep_pct = random.uniform(0.1, 1.0)
    crossover_pct = random.uniform(0.0, 1.0 - keep_pct)
    random_pct = 1.0 - keep_pct - crossover_pct

    return HyperParams(
        population_size=random.choice([50, 100, 200, 300, 500, 1000, 1500, 2000]),
        generations=random.choice([500, 1000, 1500, 2000, 3000, 4000, 5000]),
        max_depth=random.choice([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        keep_pct=keep_pct,
        crossover_pct=crossover_pct,
        random_pct=random_pct,
        match_weight_factor=random.uniform(1.0, 10.0),
        mutation_rate=random.uniform(0.05, 1.0),
    )


def mutate_hyperparams(params: HyperParams, mutation_rate: float = 0.3) -> HyperParams:
    """Mutate hyperparameters."""
    new_params = HyperParams(**params.to_dict())

    if random.random() < mutation_rate:
        choice = random.randint(0, 7)
        if choice == 0:
            new_params.population_size = random.choice([50, 100, 200, 300, 500, 1000, 1500, 2000])
        elif choice == 1:
            new_params.generations = random.choice([500, 1000, 1500, 2000, 3000, 4000, 5000])
        elif choice == 2:
            new_params.max_depth = random.choice([3, 4, 5, 6, 7, 8, 9, 10])
        elif choice == 3:
            new_params.match_weight_factor = random.uniform(1.0, 10.0)
        elif choice == 4:
            new_params.mutation_rate = random.uniform(0.05, 1.0)
        else:
            keep_pct = random.uniform(0.1, 1.0)
            crossover_pct = random.uniform(0.0, 1.0 - keep_pct)
            random_pct = 1.0 - keep_pct - crossover_pct
            new_params.keep_pct = keep_pct
            new_params.crossover_pct = crossover_pct
            new_params.random_pct = random_pct

    return new_params


def evaluate_hyperparams(
    params: HyperParams, stop_limit: int = 100, seed_ast: evolve.ASTNode = None
) -> Tuple[float, int, str, evolve.ASTNode, float]:
    """Evaluate hyperparameters by running the genetic algorithm."""
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
        verbose=False,
        seed_ast=seed_ast,
    )

    elapsed_time = time.time() - start_time

    return results["best_fitness"], results["best_matches"], results["expression"], results["best_ast"], elapsed_time


def evaluate_hyperparams_wrapper(args):
    """Wrapper for parallel evaluation."""
    params, stop_limit, index = args
    fitness, matches, expression, best_ast, elapsed_time = evaluate_hyperparams(params, stop_limit, seed_ast=None)
    return index, fitness, matches, expression, best_ast, elapsed_time


def hyperparameter_evolution(
    population_size: int = 20,
    generations: int = 50,
    stop_limit: int = 100,
    keep_best: int = 5,
    use_multiprocessing: bool = True,
) -> Tuple[HyperParams, float]:
    """Evolve hyperparameters to find optimal settings."""
    population = [create_random_hyperparams() for _ in range(population_size)]

    best_params = None
    best_fitness = float("inf")
    best_matches = 0
    best_expression = None

    num_cores = max(1, (mp.cpu_count() // 2) - 2) if use_multiprocessing else 1
    total_logical_cores = mp.cpu_count()
    estimated_physical_cores = total_logical_cores // 2
    print(f"Detected {total_logical_cores} logical cores (~{estimated_physical_cores} physical cores)")
    print(f"Using {num_cores} cores for parallel processing")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"hyperparameter_evolution_{timestamp}.csv"

    with open(csv_filename, "w", newline="") as csvfile:
        fieldnames = [
            "generation",
            "individual",
            "population_size",
            "generations",
            "max_depth",
            "keep_pct",
            "crossover_pct",
            "random_pct",
            "match_weight_factor",
            "mutation_rate",
            "fitness",
            "matches",
            "elapsed_time_seconds",
            "expression",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for generation in range(generations):
            print(f"\n{'='*80}")
            print(f"Hyperparameter Generation {generation + 1}/{generations}")
            print(f"{'='*80}\n")

            try:
                if use_multiprocessing and num_cores > 1:
                    pool = mp.Pool(processes=num_cores)

                    # Submit all jobs and track them
                    pending_results = []
                    for i, params in enumerate(population):
                        args = (params, stop_limit, i)
                        async_result = pool.apply_async(evaluate_hyperparams_wrapper, (args,))
                        pending_results.append((async_result, i, params))

                    pool.close()

                    # Process results as they complete
                    fitness_scores = []
                    completed = 0
                    while pending_results:
                        for i, (async_result, index, params) in enumerate(pending_results):
                            if async_result.ready():
                                try:
                                    result_index, fitness, matches, expression, best_ast, elapsed_time = (
                                        async_result.get(timeout=0.1)
                                    )
                                    completed += 1
                                    fitness_scores.append(
                                        (fitness, matches, expression, best_ast, elapsed_time, params)
                                    )

                                    print(
                                        f"\nCompleted hyperparameter set {result_index + 1}/{len(population)} ({completed}/{len(population)} total)"
                                    )
                                    print(
                                        f"  population_size={params.population_size}, generations={params.generations}, max_depth={params.max_depth}"
                                    )
                                    print(
                                        f"  keep_pct={params.keep_pct:.3f}, crossover_pct={params.crossover_pct:.3f}, random_pct={params.random_pct:.3f}"
                                    )
                                    print(
                                        f"  match_weight_factor={params.match_weight_factor:.3f}, mutation_rate={params.mutation_rate:.3f}"
                                    )
                                    print(f"  Fitness: {fitness:.4f}, Matches: {matches}")
                                    print(f"  Elapsed time: {elapsed_time:.2f} seconds")
                                    print(f"  Expression: {expression}")

                                    writer.writerow(
                                        {
                                            "generation": generation + 1,
                                            "individual": result_index + 1,
                                            "population_size": params.population_size,
                                            "generations": params.generations,
                                            "max_depth": params.max_depth,
                                            "keep_pct": params.keep_pct,
                                            "crossover_pct": params.crossover_pct,
                                            "random_pct": params.random_pct,
                                            "match_weight_factor": params.match_weight_factor,
                                            "mutation_rate": params.mutation_rate,
                                            "fitness": fitness,
                                            "matches": matches,
                                            "elapsed_time_seconds": elapsed_time,
                                            "expression": expression,
                                        }
                                    )
                                    csvfile.flush()

                                    if fitness < best_fitness:
                                        best_fitness = fitness
                                        best_params = params
                                        best_matches = matches
                                        best_expression = expression
                                        print(f"  *** NEW BEST FITNESS: {best_fitness:.4f} ***")

                                    pending_results.pop(i)
                                    break
                                except mp.TimeoutError:
                                    continue

                        time.sleep(0.1)

                    pool.join()

                else:
                    fitness_scores = []
                    for i, params in enumerate(population):
                        print(f"Evaluating hyperparameter set {i + 1}/{len(population)}...")
                        print(
                            f"  population_size={params.population_size}, generations={params.generations}, max_depth={params.max_depth}"
                        )
                        print(
                            f"  keep_pct={params.keep_pct:.3f}, crossover_pct={params.crossover_pct:.3f}, random_pct={params.random_pct:.3f}"
                        )
                        print(
                            f"  match_weight_factor={params.match_weight_factor:.3f}, mutation_rate={params.mutation_rate:.3f}"
                        )

                        fitness, matches, expression, best_ast, elapsed_time = evaluate_hyperparams(
                            params, stop_limit=stop_limit
                        )
                        fitness_scores.append((fitness, matches, expression, best_ast, elapsed_time, params))

                        print(f"  Fitness: {fitness:.4f}, Matches: {matches}")
                        print(f"  Elapsed time: {elapsed_time:.2f} seconds")
                        print(f"  Expression: {expression}")

                        writer.writerow(
                            {
                                "generation": generation + 1,
                                "individual": i + 1,
                                "population_size": params.population_size,
                                "generations": params.generations,
                                "max_depth": params.max_depth,
                                "keep_pct": params.keep_pct,
                                "crossover_pct": params.crossover_pct,
                                "random_pct": params.random_pct,
                                "match_weight_factor": params.match_weight_factor,
                                "mutation_rate": params.mutation_rate,
                                "fitness": fitness,
                                "matches": matches,
                                "elapsed_time_seconds": elapsed_time,
                                "expression": expression,
                            }
                        )
                        csvfile.flush()

                        if fitness < best_fitness:
                            best_fitness = fitness
                            best_params = params
                            best_matches = matches
                            best_expression = expression
                            print(f"  *** NEW BEST FITNESS: {best_fitness:.4f} ***")

            except KeyboardInterrupt:
                if use_multiprocessing and num_cores > 1:
                    print("\n\nInterrupting workers...")
                    pool.terminate()
                    pool.join()
                print("\nInterrupted by user during hyperparameter evolution")
                print(f"Completed {generation} full generations")
                if best_params is not None:
                    print(f"Best fitness so far: {best_fitness:.4f}")
                    print(f"Best matches so far: {best_matches}")
                csvfile.close()
                exit(0)

            fitness_scores.sort(key=lambda x: x[0])

            print(f"\nGeneration {generation + 1} Summary:")
            print(f"Best fitness this generation: {fitness_scores[0][0]:.4f}")
            print(f"Best matches this generation: {fitness_scores[0][1]}")
            print(f"Best fitness overall: {best_fitness:.4f}")
            print(f"Best matches overall: {best_matches}")

            survivors = [params for _, _, _, _, _, params in fitness_scores[:keep_best]]

            new_population = survivors[:]

            while len(new_population) < population_size:
                parent = random.choice(survivors)
                child = mutate_hyperparams(parent)
                new_population.append(child)

            population = new_population

    print(f"\nHyperparameter evolution log saved to: {csv_filename}")
    return best_params, best_fitness


if __name__ == "__main__":

    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal, terminating...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("Starting hyperparameter evolution")
    print("This will take a while as each evaluation runs a full genetic algorithm")
    print()

    try:
        best_params, best_fitness = hyperparameter_evolution(
            population_size=20, generations=50, stop_limit=100, keep_best=5, use_multiprocessing=True
        )
    except KeyboardInterrupt:
        print("\nHyperparameter evolution interrupted by user")
        exit(0)

    print("\n" + "=" * 80)
    print("FINAL BEST HYPERPARAMETERS")
    print("=" * 80)
    print(f"population_size: {best_params.population_size}")
    print(f"generations: {best_params.generations}")
    print(f"max_depth: {best_params.max_depth}")
    print(f"keep_pct: {best_params.keep_pct:.3f}")
    print(f"crossover_pct: {best_params.crossover_pct:.3f}")
    print(f"random_pct: {best_params.random_pct:.3f}")
    print(f"match_weight_factor: {best_params.match_weight_factor:.3f}")
    print(f"mutation_rate: {best_params.mutation_rate:.3f}")
    print(f"\nBest fitness achieved: {best_fitness:.4f}")

    print("\n" + "=" * 80)
    print("RUNNING INDEFINITELY WITH BEST HYPERPARAMETERS")
    print("Press Ctrl+C to stop")
    print("=" * 80)

    run_count = 0
    best_ever_fitness = float("inf")
    best_ever_matches = 0
    best_ever_expression = None
    best_ever_ast = None

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
                if best_ever_ast is not None:
                    print("Seeding with previous best solution")
                print(f"{'='*80}\n")

                fitness, matches, expression, best_ast, elapsed_time = evaluate_hyperparams(
                    best_params, stop_limit=1000, seed_ast=best_ever_ast
                )

                print(f"Fitness: {fitness:.4f}, Matches: {matches}")
                print(f"Elapsed time: {elapsed_time:.2f} seconds")
                print(f"Expression: {expression}")

                is_new_best = False
                if fitness < best_ever_fitness:
                    best_ever_fitness = fitness
                    best_ever_matches = matches
                    best_ever_expression = expression
                    best_ever_ast = best_ast
                    is_new_best = True
                    print(f"\n*** NEW OVERALL BEST FITNESS: {best_ever_fitness:.4f} ***")
                    print(f"*** MATCHES: {best_ever_matches} ***")
                    print(f"*** EXPRESSION: {best_ever_expression} ***")

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
