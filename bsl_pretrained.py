"""Rückfallmodelle: Laden der Parametervektoren (float32) aus data/bsl_pretrained.npz.

Schlüssel `kurz_n{n}_{form}_s{zufalls-seed}` (Modell des kurzen Laufs von Experiment 2) und `lang_n10_mittel_s0` (Hauptreihe, Netzzustand „vorgerechnet“ in Experiment 1).
Ein Wörterbuch der Meta-Angaben (Schritte, Lernrate, gemessene Lücken) steht in data/bsl_pretrained_meta.json."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import bsl_constants as C
import bsl_learn as LE

DATA = Path(__file__).resolve().parent / "data"
NPZ = DATA / "bsl_pretrained.npz"
META = DATA / "bsl_pretrained_meta.json"


def schluessel(art: str, n: int, form: str, seed: int) -> str:
    return f"{art}_n{n}_{form}_s{seed}"


@lru_cache(maxsize=1)
def meta() -> dict | None:
    if not META.exists():
        return None
    return json.loads(META.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _vektoren() -> dict | None:
    if not NPZ.exists():
        return None
    with np.load(NPZ) as z:
        return {k: z[k] for k in z.files}


def verfuegbar(art: str, n: int, form: str, seed: int) -> bool:
    v = _vektoren()
    return v is not None and schluessel(art, n, form, seed) in v


def seeds_kurz(n: int) -> list[int]:
    """Zufalls-Seeds, für die Rückfallmodelle der kurzen Läufe vorhanden sind (alle vier Formen)."""
    v = _vektoren() or {}
    out = []
    for k in v:
        if k.startswith(f"kurz_n{n}_keiner_s"):
            out.append(int(k.rsplit("_s", 1)[1]))
    return sorted(out)


def lade(art: str, n: int, form: str, seed: int) -> dict | None:
    """Parameter (float64) oder None, wenn nicht vorhanden. Eine falsche Länge löst ValueError aus."""
    v = _vektoren()
    k = schluessel(art, n, form, seed)
    if v is None or k not in v:
        return None
    return LE.unpack_params(v[k].astype(np.float64), C.D_STANDARD)


def meta_von(art: str, n: int, form: str, seed: int) -> dict | None:
    m = meta()
    return None if m is None else m["modelle"].get(schluessel(art, n, form, seed))
