from pathlib import Path
import numpy as np
import pandas as pd

from connectome.graph import build_adjacency, identify_pools
from interface.motor_decoder import ACTIONS, MotorDecoder
from interface.sensory_encoder import SensoryEncoder
from sim.lif import LIFNetwork, LIFParams
from training.train_interface import (
    SITUATIONS,
    flatten_params,
    run_es,
    ESConfig,
    evaluate_situations,
)

data_dir = Path(__file__).resolve().parent.parent / "data" / "hemibrain"

neurons_df = pd.read_parquet(data_dir / "neurons.parquet")
conn_df = pd.read_parquet(data_dir / "connections.parquet")

weights, neuron_meta = build_adjacency(neurons_df, conn_df)

in_strength = np.abs(weights).sum(axis=1).A.flatten()
in_strength[in_strength == 0] = 1.0

norm_weights = weights.multiply(
    1.0 / in_strength[:, None]
).tocsr()

input_idx, output_idx = identify_pools(
    neuron_meta,
    r"^(LC|LT|LPLC|MeTu)",
    r"^DN"
)

lif_params = LIFParams(
    syn_scale=1500,
    b_adapt=0.4,
    tau_adapt=40.0,
    depression_frac=0.5,
    tau_depression=30.0,
)

encoder = SensoryEncoder(input_idx, seed=0)
decoder = MotorDecoder(output_idx, seed=0)

theta0 = flatten_params(encoder, decoder)

def fitness_fn(theta):
    return evaluate_situations(
        theta,
        encoder,
        decoder,
        norm_weights,
        lif_params,
    )

print("training...")
result = run_es(
    theta0,
    fitness_fn,
    ESConfig(
        generations=25,
        population_size=15,
        sigma=0.5,
        lr=0.3,
        seed=0,
    ),
)

print("\n=== DIAGNOSTIC ===")

# Put the best parameters into the encoder/decoder.
from training.train_interface import unflatten_params
unflatten_params(result.theta, encoder, decoder)

for i, (observation, target) in enumerate(SITUATIONS):
    net = LIFNetwork(norm_weights, lif_params)

    ext_current = encoder.encode(observation, net.n)

    spike_window = np.zeros((10, net.n))

    for t in range(10):
        spike_window[t] = net.step(ext_current)

    predicted = decoder.decode(spike_window)

    print(f"\nSituation {i + 1}")
    print(f"target:    {target}")
    print(f"predicted: {predicted}")

    for action in ACTIONS:
        if predicted[action] != target[action]:
            print(
                f"  WRONG: {action} "
                f"(expected {target[action]}, got {predicted[action]})"
            )