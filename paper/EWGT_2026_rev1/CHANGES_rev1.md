# EWGT 2026 — Revision rev1: Änderungsprotokoll

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
