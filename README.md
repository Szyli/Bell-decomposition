# Introduction
This repository contains my MSc thesis work titled [<b><i> Study and design of programmable Quantum Photonic Integrated Circuits (QPIC) implementing compactified linear optical unitaries </i></b>](https://github.com/Szyli/Msc-thesis) that I completed at Stockholm University, Sweden.

The work consists of the following.
  - Implementation of Unitary operation decomposition into Universal Multiport Interferometer circuit using [Bell's scheme](https://github.com/Szyli/Bell-decomposition/edit/main/README.md#bell-decomposition)
  - [Design and optimisation](https://github.com/Szyli/Bell-decomposition/edit/main/README.md#design--optimisation) of each building block of the implemented photonic circuit
  - [Programming of voltage control card](https://github.com/Szyli/Bell-decomposition/edit/main/README.md#programming-the-voltage-control-card)

The main language of the work is python with jupyter notebooks for testbenches. In addition, we also implemented some Ansys scripts that are readable but not recognised by GitHub. Also important to note that for the usage of these codes Ansys Lumerical subscription is required.

# Bell-decomposition
Implementation of B.A. Bell's decomposition of Linear Optical Unitaries that is a further compactifying of W.R. Clement's design for Universal Multiport Interferometers.
See the implementation of the latter [here](https://github.com/clementsw/interferometer).

Given a unitary matrix (``numpy array``) it prepares the order and value of the phase shifts in the circuit such that it will represent the operator. The ``matplotlib`` drawing of the circuit is available similarly to Clement's version.

# Design & optimisation
The building blocks of the integrated photonic circuit, considered in this work, for the implementation of unitary operations is the following.
  - Straight waveguide
  - Bent waveguide (*S-bend*)
  - 50:50 beamsplitter (*3dB coupler*)
    - Directional coupler (**DC**)
    - Multimode interference coupler (**MMI**)
      - Linear taper
  - Phase modulater (*thermo-optic*)

# Programming the voltage control card
asd
