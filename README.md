# CATGCN: Cross-step Physics-aware Temporal Graph Neural Network for Biopharmaceutical Modeling

<p align="center">
  <img src="assets/framework.png" alt="CATGCN Framework" width="700"/>
</p>

---

CATGCN is a novel **Cross-step Physics-aware SpatioTemporal Graph Neural Network** designed for dynamic modeling in biopharmaceutical batch processes. It integrates physics knowledge derived from ODEs into neural modeling via an adaptive encoding module.

> 🧪 In this repo: you will find a complete framework for real-case analysis (Penicillin G & Erythromycin A), general-purpose temporal GNN physics encoders, and reusable tools for biomedical batch systems.

---

## 📁 Repository Structure

```bash
CATGCN/
│
├── CORREncoder_arch/         # Temporal correction module based on sequential feedback
├── Real_studys/              # Real-world case studies (Penicillin G, Erythromycin A)
├── STGDEncoder_tools/        # Tools for physics-based temporal graph construction
├── STGDEncoder_examples/     # Examples of building physical STG for arbitrary GNNs
└── assets/                   # Images for documentation (framework, metrics, etc.)
