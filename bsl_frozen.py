"""Eingefrorene Daten: Laden und Prüfen der JSON-Dateien, Mengen mit exakter Wahrheit, ein Messlauf (Training plus Auswertung) als Einheit.

Dateien im Ordner `data/` (alle erzeugt von `tools/bsl_make_frozen.py` und `tools/bsl_make_pretrained.py`):
* `bsl_exact_ref.json`: exakte Längen (Held-Karp) und die Längen von Nächstem Nachbarn und Nächstem Nachbarn mit 2-opt für Test- und Validierungsmenge je n.
* `bsl_frozen_v2.json`: Hauptreihe (Experimente 3, 5, 6): je n, Form und Zufalls-Seed ein Lauf.
* `bsl_frozen_lr.json`: Abstimmung der Lernrate (Experiment 4).
* `bsl_frozen_cross.json`: gekreuzte Läufe Gewichts-Seed mal Daten-Seed (Experiment 6).
* `bsl_frozen_short.json`: kurze Läufe wie das Live-Training von Experiment 2 (Rückfall und Preset-Tests).
* `bsl_frozen_cost.json`: Trainingsposten und Annahmen je n und Form (Übergabe an Stück 6).
* `bsl_pretrained.npz`, `bsl_pretrained_meta.json`: Parametervektoren float32 (Rückfallmodelle).
Die Läufe sind nur als Band wiederholbar (Plattform, letztes Bit)."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import bsl_constants as C
import bsl_instances as I
import bsl_learn as LE
import bsl_train as T

DATA = Path(__file__).resolve().parent / "data"
DATEIEN = {
    "exact": "bsl_exact_ref.json", "v2": "bsl_frozen_v2.json", "lr": "bsl_frozen_lr.json", "cross": "bsl_frozen_cross.json",
    "short": "bsl_frozen_short.json", "cost": "bsl_frozen_cost.json",
}


def pfad(name: str) -> Path:
    return DATA / DATEIEN[name]


@lru_cache(maxsize=None)
def lade(name: str) -> dict | None:
    """Eine eingefrorene Datei (None, wenn sie fehlt). Der Aufrufer verändert das Ergebnis nicht."""
    p = pfad(name)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def testsatz(n: int) -> np.ndarray:
    """Testinstanzen (Zufalls-Seed 0 bis TEST_N - 1), Form (m, n, 2)."""
    return I.test_set(n, C.TEST_N[n])


@lru_cache(maxsize=None)
def valsatz(n: int) -> np.ndarray:
    """Validierungsinstanzen (Zufalls-Seed ab 1000), Form (m, n, 2)."""
    return I.val_set(n, C.VAL_N[n])


def exakte_laengen(n: int, art: str) -> np.ndarray:
    """Exakte Längen (Held-Karp) der Test- bzw. Validierungsmenge aus der eingefrorenen Datei."""
    ref = lade("exact")
    if ref is None:
        raise FileNotFoundError("data/bsl_exact_ref.json fehlt")
    return np.asarray(ref[str(n)][f"{art}_opt"], dtype=np.float64)


def messlauf(n: int, form: str, steps: int, lr: float, init_seed: int, data_seed: int, eval_every: int = C.MESS_ALLE, test: bool = True,
             max_seconds: float | None = None, progress=None):
    """Ein Lauf: Training mit Validierungskurve (alle `eval_every` Schritte), danach Lücke des Endmodells auf der Validierungs- und (wenn `test`) der Testmenge.

    Rückgabe (Datensatz, P). Der Datensatz ist ein JSON-fähiges Wörterbuch; die Parameter bleiben beim Aufrufer."""
    xv, ov = valsatz(n), exakte_laengen(n, "val")
    P, hist, counts = LE.train_form(n, form, steps, lr=lr, init_seed=init_seed, data_seed=data_seed, val=(xv, ov), eval_every=eval_every, max_seconds=max_seconds, progress=progress)
    rec = {
        "n": n, "form": form, "lr": lr, "steps": steps, "batch": C.BATCH, "init_seed": init_seed, "data_seed": data_seed,
        "val_kurve": [[h["step"], round(h["val_gap"], 6)] for h in hist],
        "val_gap_ende": round(T.gap_greedy(P, xv, ov), 6),
        "instanzen": counts["instanzen"], "touren": counts["touren"], "baseline_touren": counts["baseline_touren"], "kodierer": counts["kodierer"],
        "pruefungen": counts["pruefungen"], "ersetzungen": counts["ersetzungen"], "schritte": counts["schritte"], "abgebrochen": counts["abgebrochen"],
        "sekunden": round(counts["sekunden"], 2),
    }
    if test:
        rec["test_gap"] = round(T.gap_greedy(P, testsatz(n), exakte_laengen(n, "test")), 6)
    return rec, P


def nn_gaps(n: int, art: str = "test") -> dict:
    """Mittlere Lücken des Nächsten Nachbarn und des Nächsten Nachbarn mit 2-opt auf der Menge (aus der eingefrorenen Datei)."""
    ref = lade("exact")[str(n)]
    opt = np.asarray(ref[f"{art}_opt"], dtype=np.float64)
    nn = np.asarray(ref[f"{art}_nn"], dtype=np.float64)
    nn2 = np.asarray(ref[f"{art}_nn2"], dtype=np.float64)
    out = {"nn": float(np.mean(nn / opt - 1) * 100), "nn_2opt": float(np.mean(nn2 / opt - 1) * 100)}
    if art == "test":
        out["zugauswertungen_mittel"] = float(np.mean(ref["test_k2"]))
        out["zugauswertungen_min"], out["zugauswertungen_max"] = int(min(ref["test_k2"])), int(max(ref["test_k2"]))
    return out


# ------------------------------------------------------------------------------------------------------------------ Prüfung der Schemata
def pruefe_exact(d: dict) -> list[str]:
    """Fehlermeldungen (leer = in Ordnung) für `bsl_exact_ref.json`."""
    fehler = []
    for n in C.N_MESS:
        e = d.get(str(n))
        if e is None:
            fehler.append(f"n = {n} fehlt")
            continue
        for art, m in (("test", C.TEST_N[n]), ("val", C.VAL_N[n])):
            for feld in ("opt", "nn", "nn2"):
                if len(e.get(f"{art}_{feld}", [])) != m:
                    fehler.append(f"n = {n}: {art}_{feld} hat nicht {m} Werte")
        if len(e.get("test_k2", [])) != C.TEST_N[n]:
            fehler.append(f"n = {n}: test_k2 hat nicht {C.TEST_N[n]} Werte")
    return fehler


def pruefe_lauf(rec: dict) -> list[str]:
    fehler = []
    for feld in ("n", "form", "lr", "steps", "init_seed", "data_seed", "val_kurve", "val_gap_ende", "instanzen", "touren", "baseline_touren", "kodierer", "ersetzungen"):
        if feld not in rec:
            fehler.append(f"Feld {feld} fehlt")
    if rec.get("form") not in C.FORMEN:
        fehler.append(f"Form {rec.get('form')!r} unbekannt")
    return fehler
