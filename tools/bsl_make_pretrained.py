"""Erzeugt das Rückfallmodell data/bsl_pretrained.npz (+ bsl_pretrained_meta.json) aus den Zwischendateien von bsl_make_frozen.py.

Enthalten: die Modelle der kurzen Läufe (Experiment 2: n = 10 nach 300, n = 20 nach 250 Schritten) für zwei Zufalls-Seeds je n und alle vier Formen, dazu das Modell der Hauptreihe
n = 10, Form Mittel, Zufalls-Seed 0 (Netzzustand „vorgerechnet“ von Experiment 1). Parameter als Vektor float32 (12 480 Werte), Schlüssel `kurz_n{n}_{form}_s{seed}` und
`lang_n10_mittel_s0`. Aufruf: tools/bsl_make_pretrained.py --seeds10 3 7 --seeds20 1 4"""
# ruff: noqa: E402
import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

import bsl_constants as C
import bsl_pretrained as PT

ZW = ROOT / "tools" / "_zwischen"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds10", type=int, nargs=2, required=True)
    ap.add_argument("--seeds20", type=int, nargs=2, required=True)
    a = ap.parse_args()
    arrays, meta = {}, {"format": C.PRETRAINED_FORMAT, "d": C.D_STANDARD, "n_params": 12480, "modelle": {}}
    for n, seeds in ((10, a.seeds10), (20, a.seeds20)):
        for s in seeds:
            for form in C.FORMEN:
                rid = f"kurz_n{n}_{form}_lr{C.LERNRATE:g}_w{C.GEWICHT_BASIS + s}_d{C.DATEN_BASIS + s}_s{C.KURZ_SCHRITTE[n]}"
                rec = json.loads((ZW / f"{rid}.json").read_text(encoding="utf-8"))
                key = PT.schluessel("kurz", n, form, s)
                arrays[key] = np.load(ZW / f"{rid}.npy").astype(np.float32)
                meta["modelle"][key] = {"n": n, "form": form, "zufalls_seed": s, "schritte": rec["steps"], "lr": rec["lr"], "test_gap": rec["test_gap"], "val_gap_ende": rec["val_gap_ende"]}
    rid = f"main_n10_mittel_lr{C.LERNRATE:g}_w{C.GEWICHT_BASIS}_d{C.DATEN_BASIS}_s{C.SCHRITTE_MESS[10]}"
    rec = json.loads((ZW / f"{rid}.json").read_text(encoding="utf-8"))
    key = PT.schluessel("lang", 10, "mittel", 0)
    arrays[key] = np.load(ZW / f"{rid}.npy").astype(np.float32)
    meta["modelle"][key] = {"n": 10, "form": "mittel", "zufalls_seed": 0, "schritte": rec["steps"], "lr": rec["lr"], "test_gap": rec["test_gap"], "val_gap_ende": rec["val_gap_ende"]}
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "bsl_pretrained.npz", **arrays)
    (out / "bsl_pretrained_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(arrays)} Modelle, {(out / 'bsl_pretrained.npz').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
