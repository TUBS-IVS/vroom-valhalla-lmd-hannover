"""Baue eine globale PLZ-Cluster-Definition aus den merge_maps aller 7 Provider.

Hintergrund:
  merge_small_plz() in io/demand.py merged PLZ mit < MIN_PLZ_JOBS_MERGE=75
  unique str_idx in ihren raeumlich naechsten grossen Nachbarn. Das passiert
  pro Provider mit teilweise unterschiedlichen Ergebnissen (z.B. UPS merged
  mehr PLZ als DHL, weil weniger MATSim-Sites pro UPS-Adresse).

  Fuer die Paper-Auswertungen wollen wir eine *einheitliche* Cluster-
  Definition, damit Provider apples-to-apples vergleichbar sind. Dieses
  Skript bildet die Connected-Components ueber die Vereinigung aller
  merge-Kanten der 7 Provider.

Output:
  data/geodata/plz_clusters.csv:
    cluster_id, member_plz_list, n_members, providers_with_merge
"""
from __future__ import annotations

import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from batch_delivery.config.constants import PROVIDERS  # noqa: E402
from batch_delivery.io import demand  # noqa: E402

import pandas as pd  # noqa: E402

OUT_CSV = ROOT / "data" / "geodata" / "plz_clusters.csv"
OUT_AUDIT = ROOT / "results" / "audits" / "plz_clusters_report.md"


def collect_merge_maps() -> dict[str, dict[str, str]]:
    """Run stage-1 merge logic for every provider and return its merge_map."""
    out = {}
    for prov in PROVIDERS:
        daily_gdfs, b2c, b2b = demand.load_daily_demand(lsp_prefix=prov)
        gdf_unified = demand.build_unified_gdf(daily_gdfs)
        gdf_plz = demand.load_plz_areas()
        _, _, _, merge_map = demand.merge_small_plz(gdf_unified, gdf_plz, b2c.copy(), b2b.copy())
        out[prov] = merge_map
    return out


def build_clusters(merge_maps: dict[str, dict[str, str]], all_plz: list[str]):
    """Connected components over union of per-provider merge edges."""
    edges = set()
    edge_provenance = defaultdict(set)
    for prov, m in merge_maps.items():
        for src, tgt in m.items():
            edges.add((src, tgt))
            edge_provenance[(src, tgt)].add(prov)

    adj = defaultdict(set)
    for s, t in edges:
        adj[s].add(t)
        adj[t].add(s)

    visited = set()
    components = []
    for p in sorted(all_plz):
        if p in visited:
            continue
        if p not in adj:
            components.append([p])
            visited.add(p)
            continue
        stack = [p]
        comp = []
        while stack:
            q = stack.pop()
            if q in visited:
                continue
            visited.add(q)
            comp.append(q)
            for n in adj[q]:
                if n not in visited:
                    stack.append(n)
        components.append(sorted(comp))

    # Choose cluster_id: the PLZ that appears as a target (the "kept" one).
    # If multiple, prefer the alphanumerically smallest target.
    rows = []
    for comp in components:
        targets_in = {t for (s, t) in edges if t in comp and s in comp}
        if targets_in:
            cluster_id = sorted(targets_in)[0]
        else:
            cluster_id = comp[0]
        provs_per_member = {}
        for m_plz in comp:
            provs = set()
            for prov, mp in merge_maps.items():
                if m_plz in mp or m_plz == cluster_id and any(t == cluster_id for t in mp.values()):
                    provs.add(prov)
            provs_per_member[m_plz] = sorted(provs)
        rows.append(
            {
                "cluster_id": cluster_id,
                "member_plz_list": ",".join(comp),
                "n_members": len(comp),
                "is_merged": len(comp) > 1,
            }
        )
    return pd.DataFrame(rows), edges, edge_provenance


def main():
    print("Collecting merge_maps for", PROVIDERS)
    merge_maps = collect_merge_maps()
    for prov, m in merge_maps.items():
        print(f"  {prov:8s}: {len(m)} PLZ merged")

    geo = pd.read_csv(ROOT / "data" / "geodata" / "plz_areas.csv", dtype={"plz": str})
    geo["plz"] = geo["plz"].str.zfill(5)
    all_plz = sorted(geo["plz"].unique())

    df, edges, prov_per_edge = build_clusters(merge_maps, all_plz)
    df_sorted = df.sort_values(["is_merged", "cluster_id"], ascending=[False, True])
    df_sorted.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    print(f"\n=== Cluster summary ===")
    n_clusters = len(df_sorted)
    n_multi = int(df_sorted["is_merged"].sum())
    n_in_multi = int(df_sorted.loc[df_sorted["is_merged"], "n_members"].sum())
    print(f"  total clusters     : {n_clusters}")
    print(f"  multi-PLZ clusters : {n_multi}")
    print(f"  PLZ inside multi-clusters: {n_in_multi}")
    print(f"  PLZ-ID reduction   : {n_in_multi - n_multi} (those PLZ are now represented by a cluster_id)")

    # Markdown report
    lines = ["# PLZ-Cluster Definition (globale Vereinigung der Provider-Merges)\n"]
    lines.append(f"Cluster total: **{n_clusters}**  ({n_multi} multi-PLZ, davon {n_in_multi} Mitglieder)\n")
    lines.append("## Multi-PLZ-Cluster\n")
    lines.append("| cluster_id | members | size |")
    lines.append("|---|---|---:|")
    for _, r in df_sorted[df_sorted["is_merged"]].iterrows():
        lines.append(f"| {r['cluster_id']} | {r['member_plz_list']} | {r['n_members']} |")
    lines.append("")
    lines.append("## Provider-spezifische Merge-Counts\n")
    lines.append("| Provider | # PLZ merged |")
    lines.append("|---|---:|")
    for prov in PROVIDERS:
        lines.append(f"| {prov} | {len(merge_maps[prov])} |")
    lines.append("")
    OUT_AUDIT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_AUDIT}")


if __name__ == "__main__":
    main()
