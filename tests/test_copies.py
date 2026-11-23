"""Kopier-Tests (DEMO-PLAYBOOK §11a Punkt 11): jede Kopie gleicht ihrer Quelle bis auf das Präfix (znz_ gegen bsl_); ohne Quelle (CI ohne Nachbar-Ordner) übersprungen.

Quelle ist zeigernetz-demo (Stück 1); `bsl_rng.py` ist die unveränderte Vorlage aus dem Website-Repo, `modul_frisch.py` die Datei aus zeigernetz-demo.
Keine Docstring-Korrekturen in Kopien: eine Änderung gehört in die Quelle und wird neu kopiert."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
QUELLE = ROOT.parent / "zeigernetz-demo"
VORLAGE = ROOT.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"
KOPIEN = ["rngblock", "rngvec", "scenario", "instances", "net", "train", "exact", "gegner", "cost", "stats"]


def _lf(p: Path) -> bytes:
    return p.read_bytes().replace(b"\r\n", b"\n")


@pytest.mark.parametrize("name", KOPIEN)
def test_kopie_gleicht_der_quelle_nach_ersetzung(name):
    if not (QUELLE / f"znz_{name}.py").exists():
        pytest.skip("Quelle zeigernetz-demo nicht vorhanden")
    assert _lf(ROOT / f"bsl_{name}.py") == _lf(QUELLE / f"znz_{name}.py").replace(b"znz_", b"bsl_")


def test_rng_ist_die_unveraenderte_vorlage():
    if not VORLAGE.exists():
        pytest.skip("Vorlage nicht vorhanden (CI ohne Website-Repo)")
    assert _lf(ROOT / "bsl_rng.py") == _lf(VORLAGE)


def test_modul_frisch_gleicht_der_quelle():
    if not (QUELLE / "modul_frisch.py").exists():
        pytest.skip("Quelle zeigernetz-demo nicht vorhanden")
    assert _lf(ROOT / "modul_frisch.py") == _lf(QUELLE / "modul_frisch.py")


def test_kopien_importieren_nur_eigene_module():
    for name in KOPIEN:
        text = (ROOT / f"bsl_{name}.py").read_text(encoding="utf-8")
        assert "znz_" not in text, name
