"""Bau-Vormessung VB5: Zeit je Trainingsschritt je Form, Zeit der Standard-Läufe des Live-Trainings und Speicherspitze in der Näherung der Cloud (ein BLAS-Thread, ein Prozess).

Aufruf: tools/bsl_kalibrierung.py (Test-venv). Gibt Zahlen aus, die in README und BAU_2.md als Richtwert stehen; die Streamlit Cloud selbst ist damit NICHT gemessen. Zeit ist kein Befund
über das Verfahren, nur eine Größenordnung für die Obergrenzen der App. Speicher: Spitze der numpy-Zuteilungen (tracemalloc)."""
# ruff: noqa: E402
import os
import sys
import time
import tracemalloc
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bsl_constants as C
import bsl_estimator as E1
import bsl_learn as LE


def ms_je_schritt(n: int, form: str, schritte: int = 100) -> float:
    t0 = time.perf_counter()
    LE.train_form(n, form, schritte, eval_every=0)
    return 1000 * (time.perf_counter() - t0) / schritte


def main() -> None:
    for n in C.N_MESS:
        zeiten = {f: ms_je_schritt(n, f) for f in C.FORMEN}
        print(f"n = {n}: ms je Schritt " + ", ".join(f"{f} {z:.1f}" for f, z in zeiten.items()) + f"; alle vier zusammen {sum(zeiten.values()):.1f}")
        print(f"n = {n}: Live-Voreinstellung ({C.KURZ_SCHRITTE[n]} Schritte, alle vier Formen) rund {sum(zeiten.values()) * C.KURZ_SCHRITTE[n] / 1000:.0f} s plus Validierung")
    tracemalloc.start()
    LE.train_form(20, "rollout", 50, eval_every=0)
    _, spitze = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Training n = 20, Rollout (50 Schritte): Speicherspitze {spitze / 1e6:.0f} MB")
    P = LE.initial_params(32, 0)
    pool = E1.pool_instanzen(0)
    for B in C.E1_B:
        t0 = time.perf_counter()
        E1.tabelle(P, B, reps=C.E1_REPS, seed=0, pool=pool)
        print(f"Experiment 1 (B = {B}, {C.E1_REPS} Wiederholungen, ungelernt): {time.perf_counter() - t0:.1f} s")
    tracemalloc.start()
    E1.exakter_gradient(P, pool, alle_starts=True)
    _, spitze = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Experiment 1, exakter Gradient mit 720 Touren je Instanz: Speicherspitze {spitze / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
