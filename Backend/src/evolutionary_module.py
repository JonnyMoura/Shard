import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter, IsoLineEmitter
from fitness import automatic_fitness

archive = GridArchive(
    solution_dim=11,
    dims=[6, 6, 5],
    ranges=[
        (300, 5000),  # Centroid - frequency content
        (300, 4000),  # Spectral Flux - temporal patterns
        (800, 6000),  # Spectral Rolloff - frequency balance
    ],
)

general_bounds = (
    np.array([0.1, 0.5, -0.05, 1000, 0.01, 0.1, 0.05, 2, 40, 400, 0.01]),
    np.array([1.0, 10.0, 0.05, 8000000, 0.4, 0.4, 0.3, 8, 3000, 12000, 0.3])
)


def create_emitters():
    emitters = []

    for _ in range(4):
        emitters.extend(
            [
                EvolutionStrategyEmitter(
                    archive,
                    x0=np.random.uniform(low=general_bounds[0], high=general_bounds[1]),
                    sigma0=0.7,
                    batch_size=4,
                ),
                IsoLineEmitter(
                    archive,
                    x0=np.random.uniform(low=general_bounds[0], high=general_bounds[1]),
                    batch_size=4,
                ),
                EvolutionStrategyEmitter(
                    archive,
                    x0=np.random.uniform(low=general_bounds[0], high=general_bounds[1]),
                    sigma0=0.7,
                    batch_size=4,
                ),
                EvolutionStrategyEmitter(
                    archive,
                    x0=np.random.uniform(low=general_bounds[0], high=general_bounds[1]),
                    sigma0=0.7,
                    batch_size=4,
                ),
            ]
        )

    return emitters


emitters = create_emitters()

