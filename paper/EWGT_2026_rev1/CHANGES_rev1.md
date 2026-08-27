# EWGT 2026 — Revision rev1: Änderungsprotokoll

> **Fortsetzung:** Die Änderungen der Modell-Revision 2026-08 (universelle
> Tour-Regel, zwei Kostenlinsen, Operator-Polish, BundleHead) stehen in
> [`docs/CHANGES_rev1.md`](../../docs/CHANGES_rev1.md) — eine Zeile je
> geänderter Aussage (alt → neu → Beleg). Dieses Dokument bleibt das Protokoll
> der Reviewer-Runde und wird nicht zurückgezogen.

Basis: eingereichte Version (`TRPRO_EWGT2026.tex` aus der Submission-Zip).
Bearbeitet: Elsevier-Master unter `elsevier_source/` (gitignored, leak-sicher).
Kompiliert: 9 S. (tectonic-Proxy) = unverändert zur Baseline → in MiKTeX/Elsevier voraussichtlich weiter 8 S. (**finale Abnahme durch Lasse**).

## Reviewer-Punkte → Umsetzung im Paper

| # | Reviewer-Punkt | Umsetzung | Status |
|---|---|---|---|
| 1 | „klassisches Problem"/Novelty | Intro: 3 explizite Beiträge (hybrid Surrogat / skalierbare diff. TBC-Optimierung / empirische Evidenz) | ✅ |
| 2 | Off-the-shelf-Tools | „our own HAGRID demand model" | ✅ (Teil) |
| 3 | Akronyme | MAPE + pp im Abstract definiert; „Monday–Saturday" statt Mo–Sa; „Daganzo-LightGBM hybrid (Daganzo-LGB-Hybrid)" bei Erstnutzung | ⚠️ HAGRID-Vollform fehlt (siehe unten) |
| 4 | Praktische Bedeutung | Managerial Implications in die Conclusion eingewoben | ✅ |
| 5 | Simulation vs. Optimierung | Intro-Satz: „surrogate-assisted delivery-schedule optimization, not a simulation study" | ✅ |
| 6 | Per-LSP vs. systemweit | Klarstellung: Optimierung je Provider-Hub-Netz, keine anbieterübergreifende Teilung; Balancing „within each provider"; „system-wide" entschärft | ✅ |
| 7 | Nicht reproduzierbar | Repo-Fußnote erweitert (Parameter, Hyperparameter, Seeds, 39-Muster); keine Tabelle (Platz) | ✅ |
| 8 | Einzelnes α | §3.1-Satz mit Daten: per-LSP 3.2% vs global 2.9% OOF-MAPE, LGB absorbiert Skala | ✅ |
| 9 | 39 Muster total/je Zelle | „39 distinct admissible patterns; same candidate set for every cell" | ✅ |
| 10 | Gl. (3) = Nash | §2.2: CD auf gemeinsamem Hub-Ziel = Nash eines identical-interest game; Restart-Spread <1e-12; keine globale Optimalitätsgarantie | ✅ |
| 11 | Cluster in Abb. 3 | §3.1-Satz: ⌈p/Q⌉-Tour-Bänder + same-PLZ-Augmentationsfamilien; fold-sicher gruppiert | ✅ |

## Zahlen-Updates (Stage-3-konsistent)

- Abstract CV: „up to 60%" → „up to 54% in that range and up to 78% at the most consolidated points".
- Abstract Konservativität: „1.3–2.1 pp" → „0.9–2.7 pp across four validated operating points".
- §3.3 CV-Klausel: „54% at (0.5,1) and up to 71% at the most consolidated points" → „54% at (0.5,1) and up to 78% across the grid" (der frühere Extrempunkt (0,1) ist jetzt 44.3%, kein Bug mehr).
- §3.3 Wait: 0.45 → 0.46 d bei P=0.25 (marginal).
- §3.4 Validierung: 3 → 4 Punkte, VROOM-Actual 23.7/19.8/15.6/13.0% vs Prognose 22.8/18.5/13.5/10.2%, Band 0.9–2.7 pp.
- Conclusion: 22.8% sauber entkoppelt („13.5–18.5% in the efficient range, up to 22.8% at the cost-optimal extreme P=0").
- Fleet-Totals (-10/-8.3/-5.9%, +4.6% Wachstum), Peak 12.9% @ (0.5,1): unverändert bestätigt (Smoothing frequenzerhaltend).

## Figuren

- Fig 5 (`fig_grid_heatmap_6.pdf`) → Stage-3-Version (Panels b–f), Label „after per-hub balancing and system smoothing".
- Fig 6 (`fig_structural_grid_6.pdf`) → Stage-3-Version.
- Fig 4 (`fig_SM_mix_pct_8P.pdf`) → neu gerendert, inhaltlich identisch (frequenzinvariant).
- Fig 1/2/3 unverändert. S1-Cluster-Beleg bleibt im Repo (nicht im Paper).

## Metrik-Audit 2026-08-25 — zwei Reporting-Bugs behoben (ECHTE Funde)

Ausloeser: die 3,6-%-Ersparnis bei (P=10, theta=0,1) wirkte implausibel. Die
Untersuchung fand zwei Fehler — **beide nur im Reporting, nicht in der
Optimierung** (der Strafterm nutzt bereits `local_willing`, der Kostenpfad
buendelt bereits). Kein Neu-Lauf noetig; `scripts/revision/50_recompute_fleet_wait_fixed.py`
rechnet beide Metriken ueber alle 88 Zellen neu.

**BUG A — Flotten-Doppelzaehlung.** `veh_3d` weist JEDER nicht-liefernden Zelle
an JEDEM Tag >= 1 Fahrzeug zu (costs.py: `veh_3d[active] = max(1, nr)`, `active`
enthaelt `express_demand` an Nicht-Liefertagen). Die KOSTEN derselben Pakete
werden aber als EINE gebuendelte Hub-Tour gerechnet (`_hub_express_day_ml`).
Flotte und Kosten widersprachen sich. FIX: Flotte(hub, tag) = dd-Fahrzeuge der
liefernden Zellen + ceil(gebuendelte Express-Menge / Q).

**BUG B — Wartezeit-Uebergewichtung.** Berichtet wurde
`sum(sched_wait * ALLE Pakete) / sum(ALLE Pakete)`, obwohl nur der willige
Anteil wartet. FIX: `sum(sched_wait * willige Pakete) / sum(ALLE Pakete)`.

**Beide sind bei theta = 1 mathematisch No-Ops** (dort warten alle Pakete, und
es gibt keine Standard-Tour). Verifiziert: alle theta=1-Werte bit-identisch —
Wartezeit 0,975/0,228, Flotte peak/total/CV, damit ALLE Headline-Zahlen
(22,8/18,5/13,5 %, VROOM-Validierung, 54,1 % CV) unveraendert.

Geaenderte Aussagen (nur theta < 1):
- Abstract-CV: „up to 54 % in that range and up to 78 % at the most consolidated
  points" -> „drops by 54–78 % in that range". Grund: das Grid-Maximum 78,2 %
  liegt nach dem Fix bei (P=0,25, theta=1) statt bei (0, 0,8) — also im
  effizienten Bereich, was die Aussage sogar staerkt.
- §3.3-Flottensatz komplett neu. **Die Behauptung „total weekly fleet can grow
  by up to 4,6 % at intermediate theta" ist widerlegt**: nach dem Fix sinkt die
  Wochenflotte ueber das GANZE Grid (min 0,0 %), schon bei theta=0,1 um
  6,3–6,8 %. Auch die strukturelle Begruendung („redistribution rather than an
  absolute reduction") war ein Artefakt der Doppelzaehlung und ist ersetzt durch
  den ehrlichen Hinweis, dass der Flottengewinn bei niedrigem theta zu einem
  guten Teil aus der **raeumlichen Buendelung** der Standard-Pakete stammt, nicht
  aus der Zeitkonsolidierung.
- Peak-Fleet in der cost-friendly region: 8–14 % -> 11,9–18,7 %.
- Panel (c) der Abb. 5: Wartezeit steigt jetzt proportional zur Adoption
  (0,06 -> 0,98 d bei P=0) statt fast flach (0,48 -> 0,98 d).

Ergaenzender Befund (dokumentiert, nicht im Paper): bei niedrigem theta stammt
auch die KOSTEN-Ersparnis ueberwiegend aus raeumlicher Buendelung. Test:
Ersparnis je konsolidiertem Paket betraegt dort 2,08 € (Obergrenze; mit echter
lokaler Willigkeit ~14 €) gegenueber Gesamtkosten von 1,51 €/Paket — unmoeglich
aus Konsolidierung erklaerbar. Bei theta=1 sind es plausible 0,34 €/Paket.
Gegenprobe: Zellen unter 1000 Paketen/Woche zwangsweise taeglich zu liefern
aendert die Ersparnis nur von 3,81 % auf 3,77 % — es gibt also keinen
Outlier-Cluster, den man herausrechnen koennte (min 774, Median 2172 Pakete/Woche).

## Mathe-Audit 2026-08-19 — Gl. (1) korrigiert (ECHTER Fund) + Präzisierungen

**Gl. (1) beschrieb NICHT die implementierte Formel.** Produktions-Backbone ist
`daganzo_vrp_cost_v0` (legacy/daganzo.py:42): `Ĉ = m·(c_f + c_d·(2r + β·√((n/m)·A)))`,
m = ⌈p/Q⌉. Zwei Abweichungen im gedruckten Paper:
1. Lokalterm: Paper hatte klassisch `β√(nA)` (implizite Flächen-Partition je Tour);
   Code rechnet **je Tour volle Fläche** mit n/m Stopps → total `β√(m·n·A)`.
   Bei Ø ~3 Touren/Zelle materiell (Faktor √m).
2. Phantom-Zeitterm `c_t(D̂/v + s·n)`: existiert im Backbone nicht (c_t=0,
   Arbeit in c_f enthalten; v und s werden in v0 nirgends benutzt).
FIX: Gl. (1) jetzt exakt in Per-Tour-Form `D̂ = m(2r + β√((n/m)A))`,
`Ĉ = c_f·m + c_d·D̂`, Legende bereinigt (c_t, v, s raus; c_f "incl. driver labor"),
ein Klartext-Satz erklärt die Struktur. WICHTIG: α=1.34, −26 % Rohbias und
9.7 % MAPE waren stets gegen die CODE-Formel gerechnet — die Zahlen stimmten
immer, nur die gedruckte Formel nicht. Fürs Rebuttal transparent benennen
("formula now stated in the exact per-tour form implemented").

Weitere Präzisierungen aus dem Audit (beide Fassungen):
- Gl.(3)-Underbraces ("total hub routing cost" / "penalty on waiting") — selbsterklärend.
- F-Beschreibung Tag-genau: geteilte Tour nimmt nur Standard-Pakete der Zellen
  **ohne eigene Tour an dem Tag** (deckt _hub_express_day-Semantik exakt).
- Restart-Claim: "within 10^{-12}" → "within a **relative** 10^{-12}"
  (gemessen 1.7e-15 relativ; absolut wären es ~6e-10 €, Claim war angreifbar).
- Redundanten Kopplungssatz entfernt (steht jetzt in der F-Definition) →
  Master wieder 9 S. im tectonic-Proxy.
Verifiziert korrekt (Code-Abgleich): Gl. (2) (α·Ĉ+g, Median-α), Penalty-Term
P·θ_z·p_z·w̄ (== pen_mx), w̄ zyklisches 6-Tage-Mittel, m=⌈p/Q⌉, 39 Muster, 8×11-Grid.

## Gl. (3) verschärft (Reviewer #10, Nash) — in BEIDEN Fassungen

Der ursprüngliche Nash-Fix war rein textlich; die Gleichung selbst zeigte noch
`\sum_d \tilde{C}_{z,d}` (zell-eigen), was den Best-Response-/Nash-Eindruck nährte.
Jetzt zeigt die Gleichung explizit die **Hub-Zielfunktion** mit Unterklammern
(selbsterklärend): `\sigma_z^* = argmin_\sigma [ F_{h(z)}(\sigma|\sigma_{-z})
[total hub routing cost] + P θ_z p_z w̄(\sigma) [penalty on waiting] ]`.

WICHTIGE KORREKTUR ggü. dem ersten Versuch: `F` NICHT als `\sum_{z'} \sum_d
\tilde{C}_{z',d}` definieren — diese Summe ist **separierbar** und würde die
Kopplung wegdefinieren. Korrekt: `F_{h(z)}` = Gesamt-Wochen-Surrogatkosten des
Hubs = die gebündelten Liefertagstouren aller Zellen **plus die eine geteilte
Tour** für die nicht-konsolidierten (Standard-)Pakete. Genau diese geteilte Tour
koppelt die Zellen (hängt vom gemeinsamen Hub-Fahrplan ab); ein σ_z-Wechsel
verschiebt nur die eigene Liefertagstour + die geteilte Tour. Formel = Code
(bewertet Hub-Total inkl. gebündeltem Express). Beide Fassungen kompilieren
(Elsevier-Master unverändert 9 S. im tectonic-Proxy).

## Mirror in die Preprint-Fassung — 2026-08-18 erledigt

Der öffentliche Preprint (`tbc_preprint_main.tex`) trug bis dahin noch den
Submission-Stand. Gespiegelt wurden alle 11 Reviewer-Punkte plus die
Zahlen-Updates; Textähnlichkeit zum Elsevier-Master steigt damit von 0.768 auf
Deckungsgleichheit in allen inhaltlichen Blöcken. Verifiziert über 29
Fixed-String-Checks (22 müssen vorhanden sein, 7 veraltete müssen weg).

Dabei zusätzlich gefunden und behoben — jeweils Kopien des Submission-Stands in
einem Ordner, der die Revision hätte enthalten sollen:

- `figures/fig4,5,6` waren md5-identisch mit `../EWGT_2026/figures/`, also die
  **alten** Renders. Jetzt Stage-3 (md5-geprüft gegen
  `results/revision_2026_07/figures/`).
- **Alle sechs** `tables/*.csv` waren md5-identisch mit der Submission. Vier
  hatten Stage-3-Gegenstücke; `tab_op_kpi_{per_day,weekly}.csv` existierten auf
  Stage 3 nicht und werden jetzt von
  `scripts/revision/41_op_kpi_tables_smoothed.py` erzeugt.
- `EWGT26_Full_Paper_LB_preprint.pdf` war md5-identisch mit der Submission —
  ein Revisionsordner mit unrevidiertem PDF. Jetzt der frische rev1-Build
  (tectonic, 10 S. im Preprint-Layout).
- `MANIFEST.md` war eine unveränderte Kopie und behauptete „frozen at
  submission" samt veralteter Kennzahlen (1.3–2.1 pp, up to 60 %). Neu
  geschrieben für den Revisionsstand.
- Beim Bau der KPI-Tabellen fiel auf, dass zwei VROOM-Zellen (DHL, PLZ 30855,
  Tag 0 und 3 bei P=0) `PARTIAL` zurückgeben. Sie **müssen** mitgezählt werden:
  die publizierte Validierung (1 457 294,20 € = 23,69 %) enthält sie. Ohne sie
  ergäbe sich 24,92 % und 2 058 km weniger. `41_` hat jetzt ein Gate, das die
  aggregierten Summen exakt gegen
  `tab_savings_pred_vs_actual_smoothed.csv` prüft.

## OFFEN — braucht Lasse

1. **HAGRID-Akronym-Vollform** (Reviewer #3): Ich habe „our own HAGRID demand model" gesetzt, aber die Ausschreibung nicht erfunden. Bitte Vollform liefern → wird bei Erstnutzung ergänzt (in **beiden** Fassungen).
2. **Finale 8-Seiten-Abnahme** in MiKTeX/Elsevier-Umgebung. Der Preprint-Build hier ist 10 S. — anderes Layout (`preprint,12pt` statt `3p,times,procedia`), für die Seitenbegrenzung also nicht aussagekräftig.
3. Stale `%TODO`-Kommentare am Kopf von `TRPRO_EWGT2026.tex` entfernen (no./avg.-Check).
4. Inhaltliche Durchsicht des gespiegelten Preprints — die Spiegelung ist maschinell verifiziert, aber nicht von einem Menschen gelesen.
