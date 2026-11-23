"""Zeilen der Befundtabelle der README, gebildet aus den eingefrorenen Daten mit denselben Funktionen wie die App.

Die README enthält jede dieser Zeilen wörtlich; `tests/test_claims.py` prüft das und zusätzlich Bänder um die Zahlen der Vormessung. Wer die Daten neu erzeugt, erzeugt die Zeilen neu
(Aufruf `python -m bsl_befunde` gibt die Tabelle aus)."""

from functools import lru_cache

import numpy as np

import bsl_baseline as BL
import bsl_constants as C
import bsl_estimator as E1
import bsl_evaluation as V
import bsl_frozen as FZ
import bsl_posten as PO
import bsl_ranktest as RT

de, zahl = V.de, V.zahl


def ns_mit_messung() -> list[int]:
    return [n for n in C.N_MESS if V.hat_messung(n)]


def _paar(ranks, a, b):
    p = next(p for p in ranks["paare"] if {p["a"], p["b"]} == {a, b})
    s = 1 if (p["a"], p["b"]) == (a, b) else -1
    return s * p["diff_mittel"], p["p_holm"], (p["a_besser"] if s == 1 else p["b_besser"]), p["n_paare"]


def gegner(n: int) -> str:
    g = FZ.nn_gaps(n, "test")
    return (f"n = {n}: Nächster Nachbar {de(g['nn'])} %, Nächster Nachbar + 2-opt {de(g['nn_2opt'])} % ({de(g['zugauswertungen_mittel'], 1)} Zugauswertungen je Instanz im Mittel, "
            f"kleinster und größter Wert {zahl(g['zugauswertungen_min'])} und {zahl(g['zugauswertungen_max'])}), Testsatz {zahl(C.TEST_N[n])} Instanzen")


def keiner(n: int) -> str:
    t = V.seed_tabelle(n, "endgap", "gemeinsam")
    z = t["formen"]["keiner"]
    return (f"n = {n}: Mittel {de(z['mittel'])} %, Median {de(z['median'])} %, kleinster und größter Wert {de(z['min'])} und {de(z['max'])} %, "
            f"{z['unter_nn']} von {z['s']} Läufen unter dem Nächsten Nachbarn ({de(t['nn'])} %), {z['unter_2opt']} von {z['s']} unter NN + 2-opt")


def referenzwerte(n: int, form: str, modus: str = "gemeinsam") -> str:
    t = V.seed_tabelle(n, "endgap", modus)
    z = t["formen"][form]
    return (f"n = {n}: Mittel {de(z['mittel'])} %, Median {de(z['median'])} %, Standardabweichung {de(z['sd'])}, kleinster und größter Wert {de(z['min'])} und {de(z['max'])} %, "
            f"{z['unter_nn']} von {z['s']} unter dem Nächsten Nachbarn, {z['unter_2opt']} von {z['s']} unter NN + 2-opt")


def rangtests(n: int, modus: str = "gemeinsam") -> str:
    t = V.seed_tabelle(n, "endgap", modus)
    r = V.rang_tests(t)
    teile = []
    for a, b in (("mittel", "rollout"), ("mittel", "pomo"), ("rollout", "pomo")):
        d, ph, ab, np_ = _paar(r, a, b)
        teile.append(f"{C.FORM_KURZ[a]} gegen {C.FORM_KURZ[b]} {V.de_vz(d)} Prozentpunkte (Holm-p {de(ph, 4)}, {ab} von {np_} Läufen mit der ersten Form besser)")
    fr, f3 = r["friedman"], r["friedman_drei"]
    return (f"n = {n}: Friedman über die vier Formen einschließlich „keiner“ χ² = {de(fr['chi2'], 2)}, p = {de(fr['p'], 6)}; "
            f"Friedman nur über Mittel, Rollout und POMO χ² = {de(f3['chi2'], 2)}, p = {de(f3['p'], 4)}; " + "; ".join(teile))


def lernrate(n: int) -> str:
    lt = V.lr_tabelle(n)
    teile = []
    for f, z in lt["formen"].items():
        teile.append(f"{C.FORM_KURZ[f]} beste Lernrate {V.lr_text(z['beste'])} (Validierungslücke {de(z['lr'][z['beste']]['mittel'])} %)")
    k = lt["formen"]["keiner"]
    gem = k["lr"][C.LERNRATE]
    return f"n = {n}, {zahl(lt['schritte'])} Schritte, Nächster Nachbar {de(lt['nn'])} %: " + "; ".join(teile) + f"; „keiner“ bei der Lernrate 0,001 in {gem['unter']} von {gem['von']} Abstimmungs-Seeds unter dem Nächsten Nachbarn"


def kosten(n: int) -> str:
    kv = V.kosten_kurven(n, "ops", "gemeinsam")
    teile = []
    ref = kv["formen"]["mittel"]["erreicht_x"]
    for f in ("mittel", "rollout", "pomo", "keiner"):
        x = kv["formen"][f]["erreicht_x"]
        schritt = kv["formen"][f]["erreicht_schritt"]
        if x is None:
            teile.append(f"{C.FORM_KURZ[f]} erreicht es nicht")
        else:
            teile.append(f"{C.FORM_KURZ[f]} nach {zahl(schritt)} Schritten und {zahl(x)} Operationen" + ("" if f == "mittel" or not ref else f" ({de(x / ref, 2)}-fach der Operationen des Mittels)"))
    return f"n = {n}, Ziel {de(kv['ziel'])} %: " + "; ".join(teile)


def streuung(n: int) -> str:
    cd = V.cross_daten(n)
    t = np.array(cd["test_gap"], dtype=float)
    sh = RT.quadratsummen(t)
    teile = []
    for k in (1, 5):
        rs = V.rauschen(n, k)
        teile.append((f"letzte {k} Messpunkte" if k > 1 else "letzter Messpunkt") + f": Streuung zwischen den Läufen {de(rs['zwischen_sd'])}" + (f", innerhalb eines Laufs {de(rs['innen_sd'])}" if k > 1 else ""))
    return (f"n = {n} ({t.shape[0]} × {t.shape[1]} Läufe): Gewichts-Seed {de(sh['gewichte'], 1)} %, Daten-Seed {de(sh['daten'], 1)} %, Rest {de(sh['rest'], 1)} % der Quadratsumme; "
            + "; ".join(teile) + f" Prozentpunkte (Standardabweichungen der Validierungslücke der {V.rauschen(n, 1)['s']} Läufe der Hauptreihe mit dem Mittel, Lernrate 0,001; nicht aus den gekreuzten Läufen)")


@lru_cache(maxsize=8)
def experiment1_werte(B: int = C.E1_B_STANDARD, stand: str = "ungelernt") -> dict:
    if stand == "ungelernt":
        import bsl_net as N
        from bsl_rngvec import UStream
        from bsl_scenario import derive
        P = N.init_params(C.D_STANDARD, UStream(derive(C.ZWECK_E1_NETZ, 0)), layers=1, attn=True, ffn=True)
    else:
        import bsl_pretrained as PT
        P = PT.lade("lang", 10, "mittel", 0)
    return {r["form"]: r["rel_streuung"] for r in E1.tabelle(P, B, reps=C.E1_REPS, seed=0)}


def experiment1(B: int = C.E1_B_STANDARD, stand: str = "ungelernt") -> str:
    z = experiment1_werte(B, stand)
    return f"B = {B}, Netz {stand}: " + ", ".join(f"{C.FORM_KURZ[f]} {de(z[f], 2)}" for f in C.FORMEN)


def kurze_laeufe(n: int) -> str:
    d = FZ.lade("short")
    g = FZ.nn_gaps(n, "test")["nn"]
    rs = d[str(n)]
    zeilen = []
    for f in C.FORMEN:
        w = np.array([r["test_gap"] for r in rs if r["form"] == f])
        zeilen.append(f"{C.FORM_KURZ[f]} {int((w < g).sum())} von {len(w)} unter dem Nächsten Nachbarn (Mittel {de(float(w.mean()))} %)")
    return f"n = {n}, {C.KURZ_SCHRITTE[n]} Schritte, Nächster Nachbar {de(g)} %: " + "; ".join(zeilen)


def austausche(n: int) -> str:
    z = V.seed_tabelle(n, "endgap", "gemeinsam")["formen"]["rollout"]
    return f"n = {n}: {min(z['ersetzungen'])} bis {max(z['ersetzungen'])} von {z['posten']['pruefungen']} Prüfungen"


def hook() -> str:
    """Der Befund der Kernfrage, nur Zählungen und Zahlen (kein Urteilswort), jede Zahl mit ihrer Einstellung."""
    teile = []
    for n in ns_mit_messung():
        t = V.seed_tabelle(n, "endgap", "gemeinsam")
        f = t["formen"]
        k = f["keiner"]
        rest = [f"{C.FORM_KURZ[x]} {de(f[x]['mittel'])} % ({f[x]['unter_nn']} von {f[x]['s']} Läufen unter dem Nächsten Nachbarn)" for x in ("mittel", "rollout", "pomo")]
        teile.append(f"n = {n} ({zahl(t['steps'])} Schritte, {zahl(PO.posten_je_lauf('mittel', n, t['steps'])['touren'])} gezogene Touren je Lauf, {k['s']} Zufalls-Seeds): ohne Referenzwert enden {k['unter_nn']} von {k['s']} Läufen unter dem Nächsten Nachbarn "
                     f"(Median {de(k['median'])} %, Mittel {de(k['mittel'])} %; Nächster Nachbar {de(t['nn'])} %); mit Referenzwert: " + ", ".join(rest))
    return ("**Befund** (Voraussetzungen: Rundtour, gleichverteilte Orte im Einheitsquadrat, Greedy ab Ort 0, Zeigernetz mit d = 32 und einer Schicht, Batch 120, gemeinsame Lernrate 0,001, "
            "eigene numpy-Umsetzung, Mittel der Lücke je Instanz über den Testsatz, Lücke des Endmodells): " + "; ".join(teile) + ".")


def befund_saetze() -> list[str]:
    """Sätze unter der Befundtabelle: jede Zahl aus den Daten, jeder Satz mit seiner Voraussetzung (Test: wörtlich in der README)."""
    s = []
    for n in ns_mit_messung():
        t = V.seed_tabelle(n, "endgap", "gemeinsam")
        f = t["formen"]
        s.append(f"Bei n = {n} liegt die mittlere Lücke ohne Referenzwert {de(f['keiner']['mittel'] - f['mittel']['mittel'])} Prozentpunkte über der mit dem Mittel ({de(f['keiner']['mittel'])} % gegen {de(f['mittel']['mittel'])} %; "
                 f"gemeinsame Lernrate 0,001, {f['keiner']['s']} Zufalls-Seeds).")
    for n in ns_mit_messung():
        a = V.seed_tabelle(n, "endgap", "abgestimmt")
        g = V.seed_tabelle(n, "endgap", "gemeinsam")
        k, m = a["formen"]["keiner"], a["formen"]["mittel"]
        s.append(f"Mit der je Form abgestimmten Lernrate (keiner {V.lr_text(k['lr'])}, Mittel {V.lr_text(m['lr'])}) enden bei n = {n} {k['unter_nn']} von {k['s']} Läufen ohne Referenzwert unter dem Nächsten Nachbarn "
                 f"(Mittel {de(k['mittel'])} % statt {de(g['formen']['keiner']['mittel'])} % bei Lernrate 0,001) und liegen {de(k['mittel'] - m['mittel'])} Prozentpunkte hinter dem Mittel mit seiner abgestimmten Lernrate ({de(m['mittel'])} %).")
    for n in ns_mit_messung():
        rg = V.rang_tests(V.seed_tabelle(n, "endgap", "gemeinsam"))
        ra = V.rang_tests(V.seed_tabelle(n, "endgap", "abgestimmt"))
        if any((pg["p_holm"] < C.HOLM_SCHWELLE) != (pa["p_holm"] < C.HOLM_SCHWELLE) for pg, pa in zip(rg["paare"], ra["paare"])):
            def _txt(r):
                return "; ".join(f"{C.FORM_KURZ[a]} gegen {C.FORM_KURZ[b]} {V.de_vz(_paar(r, a, b)[0])} Prozentpunkte (Holm-p {de(_paar(r, a, b)[1], 4)})" for a, b in (("mittel", "rollout"), ("mittel", "pomo"), ("rollout", "pomo")))
            s.append(f"Bei n = {n} hängt die Rangfolge der drei Referenzwerte an der Lernrate: mit der gemeinsamen Lernrate 0,001 ({_txt(rg)}) und mit der je Form abgestimmten Lernrate ({_txt(ra)}; "
                     f"Lernraten: " + ", ".join(f"{C.FORM_KURZ[f]} {V.lr_text(V.lernrate_der_form(n, f, 'abgestimmt'))}" for f in ("mittel", "rollout", "pomo")) + "; 2 Zufalls-Seeds Abstimmung, Hauptreihe mit anderen Zufalls-Seeds).")
    for stand in ("ungelernt", "vorgerechnet"):
        z = experiment1_werte(C.E1_B_STANDARD, stand)
        beste = min(("mittel", "rollout", "pomo"), key=lambda f: z[f])
        fk = {f: BL.expected_gradient_scale(f, C.E1_B_STANDARD, C.E1_N) for f in ("mittel", "rollout", "pomo")}                      # Erwartungsfaktoren: (B - 1) / B, 1, (N - 1) / N mit N = n
        hoch = {f: z[f] / fk[f] ** 2 for f in fk}
        beste_hoch = min(hoch, key=hoch.get)
        s.append(f"Am Netz im Zustand „{stand}“ ({C.E1_N} Orte, Pool aus {C.E1_POOL} Instanzen, B = {C.E1_B_STANDARD}, {C.E1_REPS} Wiederholungen) streut der Schritt ohne Referenzwert etwa das {de(z['keiner'] / z[beste], 0)}-Fache "
                 f"des besten der drei anderen ({C.FORM_KURZ[beste]}; Verhältnis der Varianzen ohne Hochrechnung auf gleiche Erwartung; mit Hochrechnung auf gleiche Erwartung etwa das {de(z['keiner'] / hoch[beste_hoch], 0)}-Fache, bester dann {C.FORM_KURZ[beste_hoch]}; "
                 f"Schätzwert aus {C.E1_REPS} Wiederholungen, nicht auf die letzte Stelle belastbar); die Rangfolge der drei untereinander hängt am Netzzustand und gilt nur für diesen Pool.")
    for n in C.N_MESS:
        ek = PO.einsatz_kosten(n, FZ.nn_gaps(n, "test")["zugauswertungen_mittel"])
        s.append(f"Einsatzkosten einer Tour bei n = {n} (K Kandidatenbewertungen / O grobe Operationen; K begünstigt das Netz, O benachteiligt es): Netz {zahl(ek['netz']['k'])} / {zahl(ek['netz']['o'])}, "
                 f"Nächster Nachbar {zahl(ek['nn']['k'])} / {zahl(ek['nn']['o'])}, Nächster Nachbar + 2-opt {de(ek['nn_2opt']['k'], 1)} / {zahl(ek['nn_2opt']['o'])} (Zugauswertungen im Mittel der Messung, 8 Operationen je Bewertung als Annahme).")
    unter2, gesamt, doppelt = 0, 0, 0
    for n in ns_mit_messung():
        for modus in ("gemeinsam", "abgestimmt"):
            tt = V.seed_tabelle(n, "endgap", modus)
            for fo in C.FORMEN:
                if modus == "abgestimmt" and V.lernrate_der_form(n, fo, "abgestimmt") == C.LERNRATE:
                    doppelt += tt["formen"][fo]["s"]                                    # dieselben Läufe wie in der gemeinsamen Spalte
                    continue
                unter2 += tt["formen"][fo]["unter_2opt"]
                gesamt += tt["formen"][fo]["s"]
    s.append(f"Kein Lauf der Hauptreihe liegt unter dem Nächsten Nachbarn mit 2-opt ({unter2} von {gesamt} verschiedenen Läufen, alle vier Formen, beide Lernraten-Spalten, n = 10 und n = 20; "
             f"die {gesamt + doppelt} Zellen der Tabelle enthalten {doppelt} Läufe doppelt, weil bei n = 20 die abgestimmte Lernrate von Rollout und POMO gleich der gemeinsamen ist): "
             "das Netz schlägt in dieser Messung den Nächsten Nachbarn (mit Referenzwert), nicht die einfachste lokale Suche dahinter.")
    for n in ns_mit_messung():
        kv = V.kosten_kurven(n, "ops", "gemeinsam")
        ref = kv["formen"]["mittel"]["erreicht_x"]
        x = {fo: kv["formen"][fo]["erreicht_x"] for fo in ("rollout", "pomo")}
        if ref and all(x.values()):
            km = kv["formen"]["mittel"]
            i0 = km["schritte"].index(km["erreicht_schritt"])
            spaeter = next((st for st, g in zip(km["schritte"][i0 + 1:], km["median"][i0 + 1:]) if g > kv["ziel"]), None)
            n_spaeter = sum(g > kv["ziel"] for g in km["median"][i0 + 1:])
            nicht_mono = (f" (beim Mittel liegt die Median-Kurve bei Schritt {zahl(spaeter)} wieder über dem Ziel" + (f", insgesamt an {n_spaeter} späteren Messpunkten" if n_spaeter > 1 else "") + ")") if spaeter else ""
            s.append(f"Bei n = {n} braucht POMO mit der gemeinsamen Lernrate 0,001 bis zum Ziel das {de(x['pomo'] / ref, 2)}-Fache der groben Operationen des Mittels, der Rollout das {de(x['rollout'] / ref, 2)}-Fache "
                     f"(Annahme: Rückwärtspass doppelt so teuer wie Vorwärtspass; Messpunkte alle {C.MESS_ALLE} Schritte; die Faktoren hängen am ersten Messpunkt, an dem der Median das Ziel erreicht{nicht_mono}, "
                     "belastbar ist nur die Größenordnung, nicht die zweite Stelle).")
    for n in ns_mit_messung():
        lt = V.lr_tabelle(n)
        if lt:
            s.append(V.verdict_lr(lt)[1])
    for n in C.N_MESS:
        d = FZ.lade("short")
        g = FZ.nn_gaps(n, "test")["nn"]
        rs = d[str(n)]
        kn = [r["test_gap"] for r in rs if r["form"] == "keiner"]
        andere = [r["test_gap"] for r in rs if r["form"] != "keiner"]
        s.append(f"Nach den kurzen Läufen des Live-Trainings (n = {n}, {C.KURZ_SCHRITTE[n]} Schritte) liegt „keiner“ in {sum(w > g for w in kn)} von {len(kn)} Zufalls-Seeds über dem Nächsten Nachbarn ({de(g)} %), "
                 f"die drei anderen Formen in {sum(w > g for w in andere)} von {len(andere)} Läufen.")
    return s


def tabelle_zeilen() -> list[tuple[str, str, str]]:
    """(Frage, Ergebnis, Testverweis) der Befundtabelle, eine Zeile je Größe n."""
    ns = ns_mit_messung()
    z = []
    for n in C.N_MESS:
        z.append(("Vergleichsverfahren auf dem Testsatz", gegner(n), "`test_claims.py::test_befund_gegner`"))
    for n in ns:
        z.append(("Ohne Referenzwert („keiner“), gemeinsame Lernrate 0,001, Lücke des Endmodells", keiner(n), "`test_claims.py::test_befund_keiner`"))
    for form in ("mittel", "rollout", "pomo"):
        for n in ns:
            z.append((f"{C.FORM_ANZEIGE[form]}, gemeinsame Lernrate 0,001", referenzwerte(n, form), "`test_claims.py::test_befund_referenzwerte`"))
    for n in ns:
        z.append(("Rangtests über die Zufalls-Seeds, gemeinsame Lernrate 0,001 (Friedman, exaktes Wilcoxon, Holm über 6 Vergleiche, drei davon gegen „keiner“)", rangtests(n), "`test_claims.py::test_befund_rangtests`"))
    for n in ns:
        z.append(("Rangtests über die Zufalls-Seeds, je Form abgestimmte Lernrate (wie in der Hauptreihe mit abgestimmter Lernrate)", rangtests(n, "abgestimmt"), "`test_claims.py::test_befund_rangtests`"))
    for n in C.N_MESS:
        if V.lr_tabelle(n):
            z.append(("Lernrate je Form auf eigenen Abstimmungs-Seeds", lernrate(n), "`test_claims.py::test_befund_lernrate`"))
    for form in C.FORMEN:
        for n in ns:
            z.append((f"Hauptreihe mit abgestimmter Lernrate, {C.FORM_KURZ[form]}", referenzwerte(n, form, "abgestimmt") + f" (Lernrate {V.lr_text(V.lernrate_der_form(n, form, 'abgestimmt'))})",
                      "`test_claims.py::test_befund_abgestimmte_hauptreihe`"))
    for n in ns:
        z.append(("Kosten bis zum Ziel in groben Operationen (Annahme, gemeinsame Lernrate 0,001)", kosten(n), "`test_claims.py::test_befund_kosten`"))
    for n in ns:
        if V.cross_daten(n):
            z.append(("Streuung des Endergebnisses (gekreuzt, Referenzwert Mittel)", streuung(n), "`test_claims.py::test_befund_streuung`"))
    for stand in ("ungelernt", "vorgerechnet"):
        z.append((f"Streuung des Schätzers am Netz, 6 Orte (relative Streuung, Schätzwerte aus {C.E1_REPS} Wiederholungen)", experiment1(C.E1_B_STANDARD, stand), "`test_claims.py::test_befund_experiment1`"))
    for n in C.N_MESS:
        z.append(("Kurze Läufe wie das Live-Training", kurze_laeufe(n), "`test_claims.py::test_befund_kurze_laeufe`"))
    for n in ns:
        z.append(("Austausche des Referenznetzes beim Rollout", austausche(n), "`test_claims.py::test_befund_austausche`"))
    return z


if __name__ == "__main__":
    for f, e, t in tabelle_zeilen():
        print(f"| {f} | {e} | {t} |")
