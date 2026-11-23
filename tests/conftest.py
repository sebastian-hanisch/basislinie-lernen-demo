"""Gemeinsame Fixtures: das Handbeispiel (fünf Orte im Rechteck 4 x 3 mit Mitte, das Mini-Netz d = 2 mit 22 Parametern), an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b).

Orte: 0 = (0, 0), 1 = (4, 0), 2 = (4, 3), 3 = (0, 3), 4 = (2, 1,5). Abstände: Seiten 4, 3, 4, 3, Diagonalen 5, Mitte zu jeder Ecke 2,5; Optimum 15."""
# ruff: noqa: E402

import itertools
import math
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bsl_instances as I  # noqa: E402
import bsl_net as N  # noqa: E402

HAND_PUNKTE = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0), (2.0, 1.5)]


@pytest.fixture
def hand_x():
    """Eingabe des Mini-Netzes: die Orte geteilt durch 4, Form (5, 2)."""
    return I.hand_instance()


@pytest.fixture
def hand_P():
    return N.hand_params()


@pytest.fixture
def hand_D():
    """Abstandsmatrix des Handbeispiels in den Einheiten des Rechtecks (Liste von Listen)."""
    return [[math.hypot(a[0] - b[0], a[1] - b[1]) for b in HAND_PUNKTE] for a in HAND_PUNKTE]


def brute_force(D) -> float:
    """Exaktes Optimum durch Aufzählen aller Rundtouren (n bis 9), unabhängig von bsl_exact."""
    n = len(D)
    best = 1e300
    for perm in itertools.permutations(range(1, n)):
        if perm[0] > perm[-1]:
            continue
        best = min(best, D[0][perm[0]] + D[perm[-1]][0] + sum(D[a][b] for a, b in zip(perm, perm[1:])))
    return best


def zufallsinstanz(n: int, seed: int) -> np.ndarray:
    """Kleine Zufallsinstanz für Tests (numpy.random nur hier in den Tests, nie im Kern)."""
    return np.random.default_rng(seed).random((n, 2))
