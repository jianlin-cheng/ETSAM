# ETSAM: Effectively Segmenting Membranes in cryo-Electron Tomograms

[![DOI:10.1101/2025.11.23.689996](http://img.shields.io/badge/DOI-10.1101/2025.11.23.689996-B31B1B.svg)](https://doi.org/10.1101/2025.11.23.689996)
[![Documentation](https://img.shields.io/badge/Documentation-jianlin--cheng.github.io%2FETSAM-0891B2.svg)](https://jianlin-cheng.github.io/ETSAM/)
[![Harvard Dataverse](https://img.shields.io/badge/Harvard%20Dataverse-10.7910%2FDVN%2FK4JKCW-A51C30.svg)](https://doi.org/10.7910/DVN/K4JKCW)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.17571925-1682D4.svg)](https://doi.org/10.5281/zenodo.17571925)

Cryogenic Electron Tomography (cryo-ET) is an emerging experimental technique to visualize cell structures and macromolecules in their native cellular environment. Accurate segmentation of cell structures in cryo-ET tomograms, such as cell membranes, is crucial to advance our understanding of cellular organization and function. However, several inherent limitations in cryo-ET tomograms, including the very low signal-to-noise ratio, missing wedge artifacts from limited tilt angles, and other noise artifacts, collectively hinder the reliable identification and delineation of these structures. In this study, we introduce ETSAM - a two-stage SAM2-based fine-tuned AI model that effectively segments cell membranes in cryo-ET tomograms. It is trained on a diverse dataset comprising 83 experimental tomograms from the CryoET Data Portal (CDP) database and 28 simulated tomograms generated using PolNet. ETSAM achieves state-of-the-art performance on an independent test set comprising 10 simulated tomograms and 10 experimental tomograms for which ground-truth annotations are available. It robustly segments cell membranes with high sensitivity and precision, achieving a more favorable precision–recall trade-off than other deep learning methods.

# ETSAM Pipeline 
![ETSAM Pipeline](<assets/etsam-framework.png>)

(A) shows the pipeline of the two-stage ETSAM. (B) shows the network architecture of the ETSAM block used in both stage 1 and stage 2.

# Requirements
## Hardware Requirements
<ins>ETSAM requires:</ins>
- a NVIDIA GPU with atleast 2GB VRAM, 4GB VRAM recommended. 
- atleast 24GB CPU RAM, recommended 32GB.

<ins>Tested on</ins>
- Intel® Core™ i7-14700K
- NVIDIA GeForce RTX 4070 with 12GB VRAM
- 32 GB DDR5 CPU RAM

_(AMD GPUs may work, but might require [AMD specific pytorch](https://pytorch.org/get-started/locally/) installation inside the etsam conda environment. Untested.)_

## Operating System
- Linux (Tested in Fedora 43 and Arch Linux)
- Windows (may work, but Untested)

# Documentation
Visit https://jianlin-cheng.github.io/ETSAM/ for installation and tutorial guide on how to use ETSAM. There are also detailed explanation on advanced usage and troubleshooting details available in the documentation.

# Cite
```
@article {Selvaraj2025.11.23.689996,
	author = {Selvaraj, Joel and Cheng, Jianlin},
	title = {ETSAM: Effectively Segmenting Cell Membranes in cryo-Electron Tomograms},
	elocation-id = {2025.11.23.689996},
	year = {2025},
	doi = {10.1101/2025.11.23.689996},
	publisher = {Cold Spring Harbor Laboratory},
	eprint = {https://www.biorxiv.org/content/early/2025/11/26/2025.11.23.689996.full.pdf},
	journal = {bioRxiv}
}
```
