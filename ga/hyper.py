import random
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import evolve


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
        max_depth=random.choice([3, 4, 5, 6, 7, 8, 9, 10]),
        keep_pct=keep_pct,
        crossover_pct=crossover_pct,
        random_pct=random_pct,
        match_weight_factor=random.uniform(0.5, 10.0),
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
            new_params.match_weight_factor = random.uniform(0.5, 10.0)
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


def evaluate_hyperparams(params: HyperParams, stop_limit: int = 100) -> Tuple[float, int, str]:
    """Evaluate hyperparameters by running the genetic algorithm."""
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
    )

    return results["best_fitness"], results["best_matches"], results["expression"]


def hyperparameter_evolution(
    population_size: int = 20,
    generations: int = 50,
    stop_limit: int = 100,
    keep_best: int = 5,
) -> Tuple[HyperParams, float]:
    """Evolve hyperparameters to find optimal settings."""
    population = [create_random_hyperparams() for _ in range(population_size)]

    best_params = None
    best_fitness = float("inf")
    best_matches = 0
    best_expression = None

    for generation in range(generations):
        print(f"\n{'='*80}")
        print(f"Hyperparameter Generation {generation + 1}/{generations}")
        print(f"{'='*80}\n")

        fitness_scores: List[Tuple[float, int, str, HyperParams]] = []

        for i, params in enumerate(population):
            print(f"Evaluating hyperparameter set {i + 1}/{len(population)}...")
            print(
                f"  population_size={params.population_size}, generations={params.generations}, max_depth={params.max_depth}"
            )
            print(
                f"  keep_pct={params.keep_pct:.3f}, crossover_pct={params.crossover_pct:.3f}, random_pct={params.random_pct:.3f}"
            )
            print(f"  match_weight_factor={params.match_weight_factor:.3f}, mutation_rate={params.mutation_rate:.3f}")

            fitness, matches, expression = evaluate_hyperparams(params, stop_limit=stop_limit)
            fitness_scores.append((fitness, matches, expression, params))

            print(f"  Fitness: {fitness:.4f}, Matches: {matches}")
            print(f"  Expression: {expression}")

            if fitness < best_fitness:
                best_fitness = fitness
                best_params = params
                best_matches = matches
                best_expression = expression
                print(f"  *** NEW BEST FITNESS: {best_fitness:.4f} ***")

        fitness_scores.sort(key=lambda x: x[0])

        print(f"\nGeneration {generation + 1} Summary:")
        print(f"Best fitness this generation: {fitness_scores[0][0]:.4f}")
        print(f"Best matches this generation: {fitness_scores[0][1]}")
        print(f"Best fitness overall: {best_fitness:.4f}")
        print(f"Best matches overall: {best_matches}")

        survivors = [params for _, _, _, params in fitness_scores[:keep_best]]

        new_population = survivors[:]

        while len(new_population) < population_size:
            parent = random.choice(survivors)
            child = mutate_hyperparams(parent)
            new_population.append(child)

        population = new_population

    return best_params, best_fitness


if __name__ == "__main__":
    print("Starting hyperparameter evolution")
    print("This will take a while as each evaluation runs a full genetic algorithm")
    print()

    best_params, best_fitness = hyperparameter_evolution(
        population_size=20, generations=50, stop_limit=100, keep_best=5
    )

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

    try:
        while True:
            run_count += 1
            print(f"\n{'='*80}")
            print(f"Run #{run_count} with best hyperparameters")
            print(f"{'='*80}\n")

            fitness, matches, expression = evaluate_hyperparams(best_params, stop_limit=1000)

            print(f"Fitness: {fitness:.4f}, Matches: {matches}")
            print(f"Expression: {expression}")

            if fitness < best_ever_fitness:
                best_ever_fitness = fitness
                best_ever_matches = matches
                best_ever_expression = expression
                print(f"\n*** NEW OVERALL BEST FITNESS: {best_ever_fitness:.4f} ***")
                print(f"*** MATCHES: {best_ever_matches} ***")
                print(f"*** EXPRESSION: {best_ever_expression} ***")

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
