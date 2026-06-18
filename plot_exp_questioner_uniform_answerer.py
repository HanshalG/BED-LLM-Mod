from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TITLE = "Gemma-4-4B (Exp Q, Uniform A)"
OUTPUT = Path("plots/results_gemma4_4b_exp_questioner_uniform_answerer.png")
RESULTS_DIR = Path("results/cluster-exp-questioner-uniform-answerer")


def load_curve(filename: str) -> np.ndarray:
    return np.load(RESULTS_DIR / filename) * 100.0


def main() -> None:
    naive = load_curve(
        "84741_naive_Q:google_gemma-4-E4B-it__thinking-off,"
        "A:google_gemma-4-31B-it__thinking-off_categorical_depth-1_0_animals.npy"
    )
    naive_belief = load_curve(
        "84741_naive+belief_Q:google_gemma-4-E4B-it__thinking-off,"
        "A:google_gemma-4-31B-it__thinking-off_categorical_depth-1_0_animals.npy"
    )
    eig_1_step = load_curve(
        "84741_EIG_Q:google_gemma-4-E4B-it__thinking-off,"
        "A:google_gemma-4-31B-it__thinking-off_categorical_depth-1_0_animals.npy"
    )
    eig_2_step = np.mean(
        [
            load_curve(
                "84742_EIG_Q:google_gemma-4-E4B-it__thinking-off,"
                "A:google_gemma-4-31B-it__thinking-off_categorical_depth-2_0_animals.npy"
            ),
            load_curve(
                "84743_EIG_Q:google_gemma-4-E4B-it__thinking-off,"
                "A:google_gemma-4-31B-it__thinking-off_categorical_depth-2_0_animals.npy"
            ),
        ],
        axis=0,
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    x = np.arange(1, len(naive) + 1)

    ax.plot(x, naive, linewidth=1.7, color="C2", label="Naive")
    ax.plot(x, naive_belief, linewidth=1.7, color="C1", label="Naive+Belief")
    ax.plot(x, eig_1_step, linewidth=1.7, color="C0", label="1-Step EIG")
    ax.plot(x, eig_2_step, linewidth=1.7, color="C4", label="2-Step EIG")

    ax.set_title(TITLE)
    ax.set_xlabel("# Questions")
    ax.set_ylabel("% correct guesses")
    ax.set_xticks([5, 10, 15, 20])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(-5, 102)
    ax.legend(loc="upper left")
    fig.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200)
    plt.close(fig)

    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
