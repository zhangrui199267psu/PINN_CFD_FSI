# PINN-CFD-FSI

Physics-Informed Neural Networks (PINNs) for Computational Fluid Dynamics and Fluid-Structure Interaction.

This repository evolves from a cylinder flow solver toward a full FSI benchmark with a flexible beam, following the Turek-Hron setup.

---

## Repository Structure

```
PINN_CFD_FSI/
├── papers/                          # Reference papers
│   ├── 1-s2.0-S2095034920300350-main.pdf   # Rao et al. 2020 (original PINN CFD)
│   └── TurekHron2006.pdf                   # Turek & Hron 2006 (FSI benchmark)
│
├── cylinder_flow/                   # Stage 1 — PINN for flow past a cylinder
│   ├── SteadyFlowCylinder_mixed.py  # Steady flow (mixed-form PINN, TF1)
│   ├── TransientFlowCylinder.py     # Transient flow (PINN, TF1)
│   ├── FluentSol.mat                # Reference solution from ANSYS Fluent
│   ├── pinn_cfd_0203.ipynb          # TF2 notebook — training & results
│   └── Results_pinn_cfd_0203.ipynb  # Results / post-processing notebook
│
└── beam_fsi/                        # Stage 2 — PINN for FSI with flexible beam
    └── pinn_cfd_beam_1_weight10.ipynb  # TF2 notebook — cylinder + attached beam
```

---

## Stage 1 — Flow Past a Cylinder

**Reference:** Rao, C., Sun, H., & Liu, Y. (2020). *Physics-informed deep learning for incompressible laminar flows.* Theoretical and Applied Mechanics Letters. [DOI](https://doi.org/10.1016/j.taml.2020.01.039) | [arXiv](https://arxiv.org/abs/2002.10558)

The original Python scripts (`SteadyFlowCylinder_mixed.py`, `TransientFlowCylinder.py`) implement a mixed-form PINN in TensorFlow 1.x. The notebooks (`pinn_cfd_0203.ipynb`, `Results_pinn_cfd_0203.ipynb`) are a TF2 re-implementation with:
- Latin Hypercube Sampling (LHS) for collocation points
- Navier-Stokes residuals as physics loss
- Comparison against the ANSYS Fluent reference (`FluentSol.mat`)

---

## Stage 2 — FSI with Flexible Beam (Turek-Hron)

**Reference:** Turek, S., & Hron, J. (2006). *Proposal for Numerical Benchmarking of Fluid-Structure Interaction between an Elastic Object and Laminar Incompressible Flow.* Fluid-Structure Interaction, Lecture Notes in Computational Science and Engineering.

**Notebook:** `beam_fsi/pinn_cfd_beam_1_weight10.ipynb`

Extends the cylinder PINN to include an elastic beam attached behind the cylinder (CSM/CFD/FSI benchmark). Key additions over Stage 1:
- Geometry includes both the cylinder and a rectangular beam region
- `DelObsPT` removes collocation points inside the cylinder **and** inside the beam
- Loss weight of 10 applied to boundary condition terms for improved convergence
- Domain and physics consistent with the Turek-Hron FSI2/FSI3 benchmark configurations

---

## Requirements

```
tensorflow >= 2.x
numpy
scipy
matplotlib
```

> The original `.py` scripts were developed for TensorFlow 1.10.0 (GPU). The notebooks have been updated to TensorFlow 2.x.

---

## Quick Start

```bash
# Cylinder flow (TF2 notebook)
jupyter notebook cylinder_flow/pinn_cfd_0203.ipynb

# Beam FSI (TF2 notebook)
jupyter notebook beam_fsi/pinn_cfd_beam_1_weight10.ipynb
```
