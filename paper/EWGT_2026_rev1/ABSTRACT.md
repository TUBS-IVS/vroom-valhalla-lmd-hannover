# EWGT 2026 — Submitted Abstract (frozen)

**Title:** Machine-Learning Surrogate Optimization for Time-Based Consolidation in Last-Mile Parcel Delivery

**Authors:** Lasse Bienzeisler\ :sup:`a`, Felix Petre\ :sup:`a`, Oskar Wage\ :sup:`b`, Bernhard Friedrich\ :sup:`a`

- :sup:`a` Institute of Transportation and Urban Engineering, TU Braunschweig
- :sup:`b` Institute of Cartography and Geoinformatics, Leibniz University Hannover

**Venue:** Transportation Research Procedia (EWGT 2026)

**Status:** **Submitted 2026-05-31, currently under review.** Not yet
accepted — please cite as a submitted manuscript (see BibTeX below).

## Abstract

Time-based consolidation (TBC) lets logistics service providers delay
eligible parcels for batched delivery, trading customer waiting time against
operational savings, but the cost-service balance and its spatial
distribution remain insufficiently quantified. We examine how the service
penalty and willingness-to-wait jointly drive cost, service, and fleet
requirements through machine-learning surrogate optimization that couples
Daganzo's continuum approximation with a LightGBM residual (2.95% MAPE),
applied to the Hanover Region (seven LSPs, 1.26 million weekly parcels).
Cost savings reach 22.8% at the cost-optimal extreme, with the Mo--Sa fleet
coefficient of variation dropping by up to 60% in the operationally
efficient service-penalty range of 0.25--0.75 EUR per parcel-day, depending
on the operator type. TBC pays most in rural regions with long stems, large
postal-code areas, and few delivered parcels per stop. Out-of-sample VROOM
re-routing confirms the surrogate is conservative, underestimating the
achieved savings by 1.3--2.1 pp.

## Keywords

Temporal consolidation; Parcel delivery; Last-mile logistics;
Vehicle routing problem; Delivery scheduling; Willingness to wait

## Citation (until acceptance)

```bibtex
@unpublished{bienzeisler2026tbc,
  title  = {Machine-Learning Surrogate Optimization for Time-Based
            Consolidation in Last-Mile Parcel Delivery},
  author = {Bienzeisler, Lasse and Petre, Felix and Wage, Oskar
            and Friedrich, Bernhard},
  year   = {2026},
  note   = {Manuscript submitted to Transportation Research Procedia
            (EWGT 2026); under review. Hanover Region case study.}
}
```

After the paper is accepted the entry will be upgraded to the proper
`@inproceedings` form with venue + DOI.
