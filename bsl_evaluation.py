"""Auswertung: Zugriff auf die eingefrorenen Messreihen, Tabellen, Rangtests, Kostenkurven und alle Verdikte als reine Funktionen.

Jedes Verdikt gibt (art, text) zurück (art: "success", "info", "warning"), nennt Voraussetzung, Zahl der Läufe und Zähleinheit und sagt nur, was die Messung zeigt.
Schwellen stehen in `bsl_constants` (Festlegungen, nicht abgeleitet): Holm-p 0,05, Rauschgrenze 0,5 Prozentpunkte, „deutlich kleiner“ ab Faktor 2."""

import math

import numpy as np

import bsl_constants as C
import bsl_frozen as FZ
import bsl_posten as PO
import bsl_ranktest as RT
import bsl_tabular as TB


# ---------------------------------------------------------------------------------------------------------------- Formate
def de(x: float, stellen: int = 2) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{stellen}f}".replace(".", ",")


def de_vz(x: float, stellen: int = 2) -> str:
    """Zahl mit Dezimalkomma und Vorzeichen."""
    return f"{x:+.{stellen}f}".replace(".", ",")


def zahl(n) -> str:
    """Ganzzahl mit Leerzeichen als Tausendertrenner."""
    return f"{int(round(n)):,}".replace(",", " ")


def lr_text(lr: float) -> str:
    """Lernrate als Dezimalzahl: 0,0001 / 0,0003 / 0,001 / 0,003."""
    return f"{lr:.4f}".rstrip("0").replace(".", ",")


# ---------------------------------------------------------------------------------------------------------------- Zugriff auf die Hauptreihe
def hat_messung(n: int) -> bool:
    v2 = FZ.lade("v2")
    return bool(v2) and str(n) in v2 and all(v2[str(n)]["formen"][f]["gemeinsam"] for f in C.FORMEN)


def laeufe(n: int, form: str, modus: str = "gemeinsam") -> list[dict]:
    """Läufe einer Form, nach Gewichts-Seed sortiert; bei „abgestimmt“ die Läufe mit der abgestimmten Lernrate, sonst (oder wenn sie fehlen) die gemeinsame."""
    e = FZ.lade("v2")[str(n)]["formen"][form]
    runs = e["abgestimmt"]["laeufe"] if modus == "abgestimmt" and "abgestimmt" in e else e["gemeinsam"]
    return sorted(runs, key=lambda r: r["init_seed"])


def lernrate_der_form(n: int, form: str, modus: str) -> float:
    e = FZ.lade("v2")[str(n)]["formen"][form]
    return e["abgestimmt"]["lr"] if modus == "abgestimmt" and "abgestimmt" in e else C.LERNRATE


def seed_zahl(n: int) -> int:
    return len(FZ.lade("v2")[str(n)]["formen"]["keiner"]["gemeinsam"])


def mittel_letzte(kurve, k: int = 5) -> float:
    """Mittel der letzten k Messpunkte einer Validierungskurve [[Schritt, Lücke], ...]."""
    return float(np.mean([g for _, g in kurve][-k:]))


def endwert(r: dict, metrik: str) -> float:
    return r["test_gap"] if metrik == "endgap" else mittel_letzte(r["val_kurve"], 5)


def gegner_luecken(n: int, metrik: str) -> dict:
    """Mittlere Lücke des Nächsten Nachbarn und des Nächsten Nachbarn mit 2-opt auf der Menge, die zur Metrik gehört (Test bei Endlücke, Validierung sonst)."""
    g = FZ.nn_gaps(n, "test" if metrik == "endgap" else "val")
    return {"nn": g["nn"], "nn_2opt": g["nn_2opt"]}


def seed_tabelle(n: int, metrik: str = "endgap", modus: str = "gemeinsam") -> dict:
    """Je Form die Endwerte der Läufe (nach Gewichts-Seed) und die Kennzahlen: Mittel, Median, Standardabweichung, kleinster und größter Wert, Zahl unter dem Nächsten Nachbarn
    und unter dem Nächsten Nachbarn mit 2-opt, Trainingsposten."""
    gg = gegner_luecken(n, metrik)
    zeilen = {}
    for form in C.FORMEN:
        runs = laeufe(n, form, modus)
        w = np.array([endwert(r, metrik) for r in runs])
        steps = runs[0]["schritte"]
        zeilen[form] = {
            "form": form, "werte": w.tolist(), "seeds": [r["init_seed"] - C.GEWICHT_BASIS for r in runs], "lr": lernrate_der_form(n, form, modus), "s": len(w),
            "mittel": float(w.mean()), "median": float(np.median(w)), "sd": float(w.std(ddof=1)) if len(w) > 1 else 0.0, "min": float(w.min()), "max": float(w.max()),
            "unter_nn": int((w < gg["nn"]).sum()), "unter_2opt": int((w < gg["nn_2opt"]).sum()),
            "posten": PO.posten_je_lauf(form, n, steps), "ops": PO.ops_je_lauf(form, n, steps), "ersetzungen": [r["ersetzungen"] for r in runs],
        }
    return {"n": n, "metrik": metrik, "modus": modus, "nn": gg["nn"], "nn_2opt": gg["nn_2opt"], "formen": zeilen, "steps": runs[0]["schritte"]}


def rang_tests(tab: dict) -> dict:
    """Friedman über die vier Formen (einschließlich „keiner“, sein Ergebnis hängt am Abstand von „keiner“), Friedman nur über Mittel, Rollout und POMO und paarweise exaktes Wilcoxon mit Holm
    über alle sechs Paare (Paarung über den Gewichts-Zufalls-Seed; bei POMO sind die Instanzströme andere)."""
    cols = {f: tab["formen"][f]["werte"] for f in C.FORMEN}
    if len({len(v) for v in cols.values()}) != 1:
        raise ValueError("Die Formen haben verschieden viele Läufe")
    fr = RT.friedman(np.array([cols[f] for f in C.FORMEN]).T)
    drei = RT.friedman(np.array([cols[f] for f in ("mittel", "rollout", "pomo")]).T)                    # nur die drei Referenzwerte, ohne „keiner“
    return {"friedman": fr, "friedman_drei": drei, "paare": RT.paarweise(cols), "s": len(cols["keiner"])}


def streuung_zwischen_seeds(tab: dict, formen=("mittel", "rollout", "pomo")) -> float:
    """Mittlere Standardabweichung der Endwerte über die Zufalls-Seeds, Mittel über die genannten Formen."""
    return float(np.mean([tab["formen"][f]["sd"] for f in formen]))


# ---------------------------------------------------------------------------------------------------------------- Verdikte
def verdict_tabular(zeilen: list[dict], B: int, politik: str) -> tuple[str, str]:
    """Exakter Schritt (vier Orte): Erwartung als Vielfaches des wahren Gradienten und Streuung je Zeile; Vergleich keiner gegen Mittel auf gleiche Erwartung hochgerechnet."""
    z = {r["form"]: r for r in zeilen}
    teile = []
    for r in zeilen:
        name = C.FORM_KURZ[r["form"]]
        f = r["faktor"]
        if math.isnan(f):
            teile.append(f"{name}: Erwartungsfaktor {de(r['faktor_formel'], 3)} aus der Formel (der wahre Gradient ist hier null, der Faktor ist nicht messbar), Streuung {de(r['gesamtstreuung'], 3)}")
        else:
            teile.append(f"{name}: Erwartung {de(f, 3)}-mal der wahre Gradient, Streuung {de(r['gesamtstreuung'], 3)}")
    mittel_h = TB.hochgerechnete_streuung(z["mittel"])
    ohne = z["keiner"]["gesamtstreuung"]
    quot = ohne / mittel_h if mittel_h > 0 else float("inf")
    satz = (f"Die Streuung ohne Referenzwert ist das {de(quot, 1)}-Fache der Streuung des Mittels, auf gleiche Erwartung hochgerechnet "
            f"(Politik „{C.POLITIK_ANZEIGE[politik]}“, B = {B} Touren je Schritt, alle {6 ** B} Kombinationen aufgezählt).")
    art = "success" if quot >= C.STREUUNG_DEUTLICH else "info"
    if art == "info":
        satz += " Bei dieser Politik ist der Unterschied nicht deutlich (Schwelle: Faktor 2)."
    return art, "; ".join(teile) + ". " + satz


def verdict_estimator(zeilen: list[dict], n_reps: int, B: int, stand: str) -> tuple[str, str]:
    """Experiment 1: Verhältnis der relativen Streuung ohne Referenzwert zu der des besten der drei anderen."""
    z = {r["form"]: r for r in zeilen}
    beste = min((f for f in C.FORMEN if f != "keiner"), key=lambda f: z[f]["rel_streuung"])
    rahmen = f"gilt für diesen Pool aus {C.E1_POOL} Instanzen mit {C.E1_N} Orten, das Netz im Zustand „{stand}“, B = {B} Touren je Schritt und {n_reps} Wiederholungen"
    if z[beste]["rel_streuung"] <= 0:
        return "info", f"Mit {C.FORM_KURZ[beste]} streut der Schritt auf diesem Pool gar nicht (Streuung 0), das Verhältnis ist nicht definiert; {rahmen}."
    r = z["keiner"]["rel_streuung"] / z[beste]["rel_streuung"]
    if r >= C.STREUUNG_DEUTLICH:
        return "success", (f"Ohne Referenzwert streut der Schritt auf diesem Netzzustand etwa {de(r, 0)}-mal so stark wie mit dem besten der drei anderen ({C.FORM_KURZ[beste]}; Schätzwert, nicht auf die letzte Stelle belastbar); {rahmen}.")
    return "info", f"Auf diesem Netzzustand und diesem Pool kein deutlicher Unterschied (Verhältnis {de(r, 1)}, bestes der drei anderen: {C.FORM_KURZ[beste]}); {rahmen}."


def verdict_live(gaps: dict, nn_gap: float, nn2_gap: float, n: int, steps: int, seed: int, abgebrochen: dict | None = None) -> tuple[str, str]:
    """Experiment 2: je Form, ob dieser eine Lauf unter oder über dem Nächsten Nachbarn endet (Lücke auf dem Testsatz)."""
    teile = []
    for f in C.FORMEN:
        if f in gaps:
            g = gaps[f]
            richtung = "endet unter" if g < nn_gap else "endet über"
            teile.append(f"{C.FORM_KURZ[f]} {richtung} dem Nächsten Nachbarn ({de(g)} % gegen {de(nn_gap)} %)")
    text = "; ".join(teile) + f". Der Nächste Nachbar mit 2-opt liegt bei {de(nn2_gap)} %. "
    text += f"Das ist ein einzelner Lauf (Zufalls-Seed {seed}, {steps} Schritte, n = {n}); die Streuung zeigt Experiment 3 mit mehreren Zufalls-Seeds."
    if abgebrochen:
        text += " Abgebrochen (Zeitgrenze): " + ", ".join(f"{C.FORM_KURZ[f]} bei Schritt {k}" for f, k in abgebrochen.items()) + "."
    n_unter = sum(1 for f in gaps if gaps[f] < nn_gap)
    return ("success" if n_unter == len(gaps) else "info"), text


def verdict_seeds(tab: dict, ranks: dict) -> tuple[str, str]:
    """Experiment 3: (A) „keiner“ gegen den Nächsten Nachbarn, (B) die drei Referenzwerte untereinander, (C) fester Zusatz mit den Voraussetzungen."""
    n, s = tab["n"], ranks["s"]
    k = tab["formen"]["keiner"]
    a = (f"Ohne Referenzwert enden {k['unter_nn']} von {k['s']} Läufen unter dem Nächsten Nachbarn (Median {de(k['median'])} %, größter Wert {de(k['max'])} %, "
         f"Nächster Nachbar {de(tab['nn'])} %).")
    drei = {"mittel", "rollout", "pomo"}
    paare = [p for p in ranks["paare"] if p["a"] in drei and p["b"] in drei]
    unter_drei = [tab["formen"][f]["unter_nn"] for f in ("mittel", "rollout", "pomo")]
    a += f" Mit Mittel, Rollout und POMO: {unter_drei[0]}, {unter_drei[1]} und {unter_drei[2]} von {s} Läufen unter dem Nächsten Nachbarn."
    kleinst = min(p["p_holm"] for p in paare)
    mittel_werte = [tab["formen"][f]["mittel"] for f in ("mittel", "rollout", "pomo")]
    spanne = max(mittel_werte) - min(mittel_werte)
    sd = streuung_zwischen_seeds(tab)
    if kleinst >= C.HOLM_SCHWELLE:
        lage = "unter" if spanne < sd else "über"
        moeglich = min(1.0, 6 * 2 / 2 ** s)
        b = (f" Unter den drei Referenzwerten kein nachweisbarer Unterschied (kleinster Holm-p {de(kleinst, 4)}, bei {s} Paaren und 6 Vergleichen ist {de(moeglich, 4)} der kleinste mögliche Wert); "
             f"die Spanne der Mittel ({de(spanne)} Prozentpunkte) liegt {lage} der mittleren Streuung zwischen Zufalls-Seeds ({de(sd)}).")
    else:
        teile = []
        for p in sorted(paare, key=lambda p: p["p_holm"]):
            if p["p_holm"] < C.HOLM_SCHWELLE:
                besser, schlechter = (p["a"], p["b"]) if p["diff_mittel"] < 0 else (p["b"], p["a"])
                teile.append(f"{C.FORM_KURZ[besser]} liegt {de(abs(p['diff_mittel']))} Prozentpunkte unter {C.FORM_KURZ[schlechter]} (Holm-p {de(p['p_holm'], 4)}, {p['n_paare']} Paare)")
        b = " Unter den drei Referenzwerten: " + "; ".join(teile) + ". Die Paarung gilt nur für die Anfangsgewichte (die Instanzströme von POMO sind andere)."
    f3 = ranks.get("friedman_drei")
    if f3 is not None:
        b += (f" Der Friedman-Test über alle vier Formen wird von „keiner“ bestimmt; nur über Mittel, Rollout und POMO: χ² = {de(f3['chi2'], 2)}, p = {de(f3['p'], 4)}.")
    c = (f" Gilt für n = {n}, {zahl(tab['steps'])} Schritte, Lernrate je Spalte der Tabelle, {s} Zufalls-Seeds und diese Metrik; ein Referenzwert, der hier nicht auffällt, "
         "kann bei n = 50 anders abschneiden (nicht gemessen).")
    return "info", a + b + c


def lr_tabelle(n: int) -> dict | None:
    """Experiment 4: je Form und Lernrate die mittlere Validierungslücke des Endmodells über die Abstimmungs-Seeds und die Zahl unter dem Nächsten Nachbarn (auf der Validierungsmenge)."""
    d = FZ.lade("lr")
    if not d or str(n) not in d:
        return None
    gg = FZ.nn_gaps(n, "val")
    out = {"n": n, "schritte": d[str(n)]["schritte"], "nn": gg["nn"], "formen": {}}
    for form in C.LR_FORMEN[n]:
        je = {}
        for lr in C.LERNRATEN:
            werte = [z["val_gap_ende"] for z in d[str(n)]["zellen"] if z["form"] == form and abs(z["lr"] - lr) < 1e-12]
            if werte:
                je[lr] = {"mittel": float(np.mean(werte)), "werte": werte, "unter": int(sum(w < gg["nn"] for w in werte)), "von": len(werte)}
        if je:
            out["formen"][form] = {"lr": je, "beste": min(je, key=lambda lr: (je[lr]["mittel"], lr))}
    return out


def verdict_lr(tab: dict) -> tuple[str, str]:
    """Experiment 4: Fassung A (keine Lernrate bringt „keiner“ unter den Nächsten Nachbarn) oder Fassung B (mit Lernrate x liegt „keiner“ in allen Abstimmungs-Seeds darunter)."""
    k = tab["formen"]["keiner"]
    m = tab["formen"]["mittel"]
    schritte = zahl(tab["schritte"])
    von = k['lr'][next(iter(k['lr']))]['von']
    rahmen = (f"Abgestimmt auf {von} Zufalls-Seeds, {schritte} Schritte, bewertet auf {C.VAL_N[tab['n']]} Validierungsinstanzen mit n = {tab['n']}; die Zufalls-Seeds der Hauptreihe sind andere. "
              f"Der Abstand zum Mittel bleibt bei dieser groben Abstimmung ({len(C.LERNRATEN)} Lernraten, {von} Zufalls-Seeds, kürzere Läufe) bestehen.")
    mit_lr = [lr for lr, z in k["lr"].items() if z["unter"] == z["von"]]
    delta = k["lr"][k["beste"]]["mittel"] - m["lr"][m["beste"]]["mittel"]
    if not mit_lr:
        return "info", (f"Keine der vier Lernraten bringt „keiner“ in allen Abstimmungs-Seeds unter den Nächsten Nachbarn ({de(tab['nn'])} %); die Lücke bleibt hinter dem Mittel bei Lernrate "
                        f"{lr_text(m['beste'])} (bester Wert von „keiner“ bei Lernrate {lr_text(k['beste'])}: {de(delta)} Prozentpunkte). " + rahmen)
    x = min(mit_lr, key=lambda lr: k["lr"][lr]["mittel"])
    gemeinsam = k["lr"].get(C.LERNRATE)
    dazu = f"; mit der gemeinsamen Lernrate {lr_text(C.LERNRATE)} lag „keiner“ in {gemeinsam['unter']} von {gemeinsam['von']} Abstimmungs-Seeds darunter" if gemeinsam else ""
    return "info", (f"Mit Lernrate {lr_text(x)} liegt „keiner“ in allen Abstimmungs-Seeds unter dem Nächsten Nachbarn und bleibt {de(k['lr'][x]['mittel'] - m['lr'][m['beste']]['mittel'])} Prozentpunkte "
                       f"hinter dem Mittel bei dessen bester Lernrate ({lr_text(m['beste'])}){dazu}. " + rahmen)


# ---------------------------------------------------------------------------------------------------------------- Kosten
def einheit_x(form: str, n: int, einheit: str, step: int) -> float:
    """Zähleinheit nach `step` Schritten: Schritte, gezogene Touren, Kodierer-Durchläufe oder grobe Operationen (Annahme)."""
    if einheit == "schritte":
        return float(step)
    p = PO.posten_je_lauf(form, n, step)
    if einheit == "touren":
        return float(p["touren"])
    if einheit == "kodierer":
        return float(p["kodierer"])
    return float(PO.ops_je_lauf(form, n, step))


def kosten_kurven(n: int, einheit: str, modus: str = "gemeinsam") -> dict:
    """Median der Validierungslücke über die Zufalls-Seeds je Messpunkt mit Band (kleinster bis größter Wert), x in der gewählten Einheit; Ziel = Median des Mittels am Ende plus 0,5 Prozentpunkte
    und der erste Messpunkt, an dem der Median das Ziel erreicht."""
    kurven = {}
    for form in C.FORMEN:
        runs = laeufe(n, form, modus)
        steps = [s for s, _ in runs[0]["val_kurve"]]
        m = np.array([[g for _, g in r["val_kurve"]] for r in runs])
        kurven[form] = {"schritte": steps, "median": np.median(m, axis=0).tolist(), "min": m.min(axis=0).tolist(), "max": m.max(axis=0).tolist(),
                        "x": [einheit_x(form, n, einheit, s) for s in steps]}
    ziel = kurven["mittel"]["median"][-1] + C.ZIEL_AUFSCHLAG
    for form, k in kurven.items():
        idx = next((i for i, g in enumerate(k["median"]) if g <= ziel), None)
        k["erreicht_schritt"] = None if idx is None else k["schritte"][idx]
        k["erreicht_x"] = None if idx is None else k["x"][idx]
    return {"n": n, "einheit": einheit, "ziel": ziel, "formen": kurven, "s": len(runs)}


def verdict_cost(kv: dict) -> tuple[str, str]:
    """Experiment 5: nach wie vielen Einheiten erreicht jede Form das Ziel, das Mittel als Vergleich; auf Zehnerpotenzen belastbar."""
    e, kf = kv["einheit"], kv["formen"]
    ref = kf["mittel"]["erreicht_x"]
    teile = []
    for f in ("pomo", "rollout", "keiner"):
        x = kf[f]["erreicht_x"]
        if x is None:
            teile.append(f"{C.FORM_KURZ[f]} erreicht das Ziel in keinem Messpunkt")
        elif ref:
            teile.append(f"{C.FORM_KURZ[f]} nach {zahl(x)} (Faktor {de(ref / x, 2)} gegenüber dem Mittel)")
        else:
            teile.append(f"{C.FORM_KURZ[f]} nach {zahl(x)}")
    kopf = f"In der Einheit „{C.EINHEIT_ANZEIGE[e]}“ erreicht das Mittel das Ziel ({de(kv['ziel'])} % Median der Validierungslücke) nach {zahl(ref) if ref else 'keinem Messpunkt'}; "
    return "info", (kopf + "; ".join(teile) + f". Median über {kv['s']} Zufalls-Seeds, Messpunkte alle {C.MESS_ALLE} Schritte; die Werte hängen am ersten Messpunkt, an dem der Median das Ziel erreicht; belastbar ist nur die Größenordnung, nicht die zweite Stelle.")


# ---------------------------------------------------------------------------------------------------------------- Streuung des Endergebnisses
def cross_daten(n: int) -> dict | None:
    d = FZ.lade("cross")
    return d[str(n)] if d and str(n) in d else None


def rauschen(n: int, k: int, modus: str = "gemeinsam") -> dict:
    """Streuung innerhalb eines Laufs (Standardabweichung der letzten k Messpunkte, Mittel über die Läufe der Form Mittel) gegen die Streuung zwischen den Läufen
    (Standardabweichung des Mittels der letzten k Messpunkte über die Zufalls-Seeds)."""
    runs = laeufe(n, "mittel", modus)
    innen = [float(np.std([g for _, g in r["val_kurve"]][-k:], ddof=1)) if k > 1 else float("nan") for r in runs]
    ende = [mittel_letzte(r["val_kurve"], k) for r in runs]
    return {"k": k, "innen_sd": float(np.nanmean(innen)) if k > 1 else float("nan"), "zwischen_sd": float(np.std(ende, ddof=1)), "s": len(runs)}


def verdict_spread(shares: dict, noise_sd: float, seed_sd: float, n: int, grid: tuple[int, int], k: int) -> tuple[str, str]:
    """Experiment 6: Anteile der Quadratsumme (Gewichts-Seed, Daten-Seed, Rest) und Endpunkt-Rauschen gegen die Streuung zwischen Zufalls-Seeds."""
    satz = (f"Der Gewichts-Seed erklärt {de(shares['gewichte'], 1)} %, der Daten-Seed {de(shares['daten'], 1)} %, der Rest {de(shares['rest'], 1)} % der Quadratsumme "
            f"(Referenzwert Mittel, n = {n}, {grid[0]} × {grid[1]} Läufe). Die Standardabweichungen unten stammen nicht aus diesem gekreuzten Raster, sondern aus den Läufen der Hauptreihe mit dem Mittel (Lernrate 0,001). ")
    if k > 1 and not math.isnan(noise_sd):
        rel = "so groß wie" if abs(noise_sd - seed_sd) <= 0.5 * max(noise_sd, seed_sd) else ("kleiner als" if noise_sd < seed_sd else "größer als")
        satz += f"Das Rauschen am Ende eines Laufs (Standardabweichung {de(noise_sd)} Prozentpunkte über die letzten {k} Messpunkte) ist {rel} die Streuung zwischen Zufalls-Seeds (Standardabweichung {de(seed_sd)}). "
    else:
        satz += f"Die Streuung zwischen Zufalls-Seeds (Mittel der letzten {k} Messpunkte) hat die Standardabweichung {de(seed_sd)} Prozentpunkte. "
    satz += (f"Bei reinem Rauschen wären für den Gewichts-Seed schon {de(100 * (grid[0] - 1) / (grid[0] * grid[1] - 1), 1)} % der Quadratsumme zu erwarten (Freiheitsgrade {grid[0] - 1} von {grid[0] * grid[1] - 1}); "
             "Anteile verschiedener Raster sind deshalb nicht direkt vergleichbar. Der Rest enthält Wechselwirkung und Rauschen, ohne Wiederholungszellen nicht trennbar.")
    return "info", satz


def kurz_tabelle(n: int, seed: int) -> dict | None:
    """Kurze Läufe (wie das Live-Training) eines Zufalls-Seeds: je Form Verlauf und Testlücke; None, wenn nicht vorhanden."""
    d = FZ.lade("short")
    if not d or str(n) not in d:
        return None
    runs = {r["form"]: r for r in d[str(n)] if r["init_seed"] == C.GEWICHT_BASIS + seed}
    return runs or None
