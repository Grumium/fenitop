"""
FEniTop: FEniCSx-based Topology Optimization

A simple FEniCSx implementation for 2D and 3D topology optimization
supporting parallel computing.

Reference:
    Jia, Y., Wang, C. & Zhang, X.S. FEniTop: a simple FEniCSx implementation
    for 2D and 3D topology optimization supporting parallel computing.
    Struct Multidisc Optim 67, 140 (2024).
    https://doi.org/10.1007/s00158-024-03818-7

Modules:
    - topopt: Main optimization orchestration
    - fem: Finite element analysis
    - parameterize: Density filter and Heaviside projection
    - sensitivity: Sensitivity analysis via automatic differentiation
    - optimize: OC and MMA optimizers
    - utility: Post-processing utilities (XDMF, STL export, visualization)
"""

from .topopt import topopt

__all__ = ['topopt']
__version__ = '1.0.0'
