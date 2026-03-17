# Aorta-Simulation-Preprocessing-Toolkit
This repository collects Python scripts to preprocess patient-specific aortic data starting from a **segmentation** and (optionally) **DICOM images**, producing geometry and meshes ready for **SimVascular** simulations. 

The workflow is organized in three main steps:

1. **Model generation** (from segmentation → cleaned inner surface + centerline + landing zones)
2. **Mesh generation** (from surface + centerline → SimVascular-compatible volumetric mesh, fluid and solid wall mesh)
3. **DICOMs to VTK** (from DICOM images → VTK grids + PC-MRA).  
   Reference implementation: https://github.com/saitta-s/flow4D

A `utils/` folder contains shared functions used across scripts, and `env.yml` provides a reproducible conda environment.

---
## Citation (required)

If you use this repository in academic work, please cite:


## Code overview

### `model_generation.py`

Starting with segmentation, this script generates a clean geometric model of the aorta (inner-lumen surface + centerline) and computes anatomical reference information for downstream use.

**What it does**
- Extracts the **inner aortic surface** from the segmentation.
- Generates a **triangulated surface mesh** (lumen).
- Applies **smoothing** to remove staircase artifacts and improve surface quality.
- Computes **landing zones** (regions of interest used for device planning and measurements).  
  Reference: https://doi.org/10.1007/s10278-021-00535-1
- Computes and saves the **centerline**, including all associated information.
- Saves outputs in standard formats (e.g., **VTK `.vtp`**) for the next pipeline step.

---

### `mesh_generation.py`

Generates a **SimVascular-compatible volumetric mesh** using the centerline and the inner surface produced in step 1.

**What it does**
- Loads the **centerline** and the **inner surface**.
- Builds a volumetric mesh compatible with the **SimVascular** solver workflow (**fluid + optional solid**).
- Supports **solid wall mesh** generation with:
  - **constant wall thickness**, or
  - **region-dependent thickness** (different thickness values assigned to different aortic regions).
- Exports meshes in formats usable by **SimVascular** and typical CFD workflows.

---

### `dicoms_to_vtk.py`

Converts DICOM images into **VTK grid** data and generates **PC-MRA** outputs compatible with both **VTK** and **3D Slicer**.

**What it does**
- Reads **DICOM series** and creates **VTK grids** based on the reference implementation:  
  https://github.com/saitta-s/flow4D
- Applies optional preprocessing steps when needed:
  1. **eddy current** correction
  2. **aliasing** correction
  3. **Gaussian filtering** (denoising)
- Generates **PC-MRA** and exports outputs compatible with:
  - **VTK pipelines**
  - **3D Slicer** (e.g., `.nii.gz`)

---

## Utilities

The `utils/` folder contains shared functions used throughout the pipeline (I/O helpers, geometry processing, centerline/landing-zone utilities, DICOM conversion helpers, etc.).

---

## Requirements / Installation

Create the environment from `env.yml`:

```bash
conda env create -f env.yml
conda activate <ENV_NAME>
