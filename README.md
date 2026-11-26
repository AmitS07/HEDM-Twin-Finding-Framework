# HEDM-Twin-Detection-Framework

This framework could be used to find the deformation twin generated during in-situ HEDM (High-Energy X-ray Diffraction Microscopy) measurements.

## Pipeline Overview
* **Module-I:** Find the parent-twin pair from the HEDM dataset.
* **Module-II:** Examine the resolved shear stress (RSS) behavior between the parent and twin.
* **Module-III:** Find neighbor grains to the parent grain using Neper.
* **Module-IV:** Analyze the RSS behavior of neighbor grains.

## Repository Structure
```text
├── function_lib/      # Contains libraries related to the twin-finding-code related function
├── Folder_2/          # Conatins twin-finding function
├── Documentation file/      # A document file which guides step-wise-step for each module
└── README.md
```
