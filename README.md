# Orbital Angular Momentum Holography using Neural Networks and Camera-in-the-Loop Optimization

This repository corresponds to the published paper:

**[Orbital Angular Momentum Holography using Neural Network and Camera-in-the-Loop](https://doi.org/10.1002/lpor.202500224)**  
*Laser & Photonics Reviews (2025): e00224.*

*Nima Asoudegi, Mo Mojahedi*  
Department of Electrical and Computer Engineering, University of Toronto  

## Article Description

This work presents methods for designing accurate phase-only Orbital Angular Momentum (OAM)-multiplexed holograms. Three design approaches are included:

1. **Conventional method** (using Gerchberg-Saxton algorithm)
2. **Optimization using Gradient Descent (GD) method**
3. **Deep learning approach (OAMnet)** using a U-Net neural network architecture

In conjuction with these methods, the project introduces a **Camera-in-the-Loop (CITL)** calibration technique to experimentally learn and correct imperfections (aberrations, vignetting, pixel crosstalk, and source profile variations) inherent in real-world holographic systems.

The GD method and OAMnet achieve upto 84% and 65% improvement, respectively, in accuracy compared to the conventional method in experimental measurements.

## Code description
This repository provides implementations of the three hologram design methods mentioned above, along with the necessary utilities for Camera-in-the-Loop calibration. The main components are as follows:
```plaintext
.
├── lib/
│   ├── oamlib/                     # Main library for OAM-multiplexed holography
│   ├── solvers.py                  # Implementations of the three hologram design methods mentioned above
|   ├── train_oamnet.py             # Training routine for OAMnet with Camera-in-the-Loop calibration 
│   ├── CITL.py                     # Camera-in-the-Loop experiment and training classes       
│   ├── device_manager_SDK4.py      # Device manager for Holoeye SLMs using API version v4.0.0 and Thorlabs camera SDK
|   └── utils/
|   │   ├── utils.py                # General utility functions
|   │   ├── slm.py                  # Hologram generation and SLM input data preparation utilities
|   │   └── holo.py                 # Light propagation and holography functions
└── CITL_experiment.ipynb          # Scripts for running Camera-in-the-Loop experiments and OAMnet training
```
## Hardwares & APIs
- **Thorlabs DKs** for camera integration and data acquisition.
  - Thorlabs CS505CU camera
- **Holoeye SLMs** code provided with both API versions v3.2.0 and v4.0.0.
  - Holoeye PLUTO-2.1 LCOS Spatial Light Modulator

## Citation

Please cite the following paper if you use this work:

```
@article{asoudegi2025orbital,
  title={Orbital Angular Momentum Holography Using Neural Network and Camera in the Loop},
  author={Asoudegi, Nima and Mojahedi, Mo},
  journal={Laser \& Photonics Reviews},
  pages={e00224},
  year={2025},
  publisher={Wiley Online Library}
}
```

**All rights reserved.** This repository is publicly available for transparency and referencing only. For further inquiries or permissions, contact the author at [n.asoudegi@gmail.com](mailto:n.asoudegi@gmail.com).

