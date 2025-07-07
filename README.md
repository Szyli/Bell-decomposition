> [!IMPORTANT]
> Please fork this repository to your own if you plan to work with it and leave this one unchanged.<br>
> In addition, please make sure to refer my work!

# Introduction
This repository contains my MSc thesis work titled [<b><i> Study and design of programmable Quantum Photonic Integrated Circuits (QPIC) implementing compactified linear optical unitaries </i></b>](https://github.com/Szyli/Msc-thesis) that I completed at Stockholm University, Sweden 2025.

The work consists of the following.
  - Implementation of Unitary operation decomposition into Universal Multiport Interferometer circuit using [Bell's scheme](https://github.com/Szyli/Bell-decomposition/edit/main/README.md#bell-decomposition)
  - [Design and optimisation](https://github.com/Szyli/Bell-decomposition/edit/main/README.md#design--optimisation) of each building block of the implemented photonic circuit
  - [Programming of voltage control card](https://github.com/Szyli/Bell-decomposition/edit/main/README.md#programming-the-voltage-control-card)

The main language of the work is python with jupyter notebooks for testbenches (see [``test_sMZI.ipynb``](https://github.com/Szyli/Bell-decomposition/blob/main/test_sMZI.ipynb)). In addition, I also implemented some Ansys scripts that are readable but not recognised by GitHub.
> [!WARNING]
> For the usage of these codes Ansys Lumerical subscription is required.

There are different still active branches.
  - The [**main**](https://github.com/Szyli/Bell-decomposition) which, as the name suggest, should contain the best version of the project.
  - The [**office**](https://github.com/Szyli/Bell-decomposition/tree/office) which corresponds to the developments carried out on the computer running the Ansys simulations
  - The [**office-card**](https://github.com/Szyli/Bell-decomposition/tree/office-card) which corresponds to the developments carried out on the computer connected to the voltage controller card
  - The [**Sophia**](https://github.com/Szyli/Bell-decomposition/tree/Sophia) branch is outdated (changes were implemented/merged to **main**) and corresponded to the developments carried out by a temporary colleague

> [!NOTE]
> The current state of the repository is not finalised and as such **office** is ahead of **main** regarding Ansys simulation related commits.

# Bell-decomposition
Implementation of B.A. Bell's decomposition of Linear Optical Unitaries that is a further compactifying of W.R. Clement's design for Universal Multiport Interferometers.
See the implementation of the latter [here](https://github.com/clementsw/interferometer). My implementation is under the [``sMZI.py``](https://github.com/Szyli/Bell-decomposition/blob/main/sMZI.py) file.

<b> B. A. Bell and I. A. Walmsley. <i>“Further compactifying linear optical unitaries”</i>. In: APL Photonics 6.7 (July 2021), DOI: [10.1063/5.0053421](https://doi.org/10.1063/5.0053421) </b>

<p align="center">
<figure>
  <img src="https://github.com/user-attachments/assets/c51848a4-1777-4559-b4ba-722c3bedf4a3" alt="Bell's circuit representing a 4x4 identity operation.", style="width:100%; height:80%"; border:0">
  <figcaption><p align="center"><i> Bell's circuit representing a 4x4 identity operation. </i></p></figcaption>
</figure>
</p>

Given a unitary matrix (``numpy array``) it prepares the order and value of the phase shifts in the circuit such that it will represent the operator. The ``matplotlib`` drawing of the circuit is available similarly to Clement's implementation. There is also an initialised functionality to build the full logical circuit defining and connecting its building blocks in Ansys INTERCONNECT. Once we obtain the optical properties of said elements we could simulate the performance of the circuit using test unitaries. Therefore, the next step was the design and optimisation of these building blocks.

# Design & optimisation
The building blocks of the integrated photonic circuit, considered in this work, for the implementation of unitary operations is the following.
  - Straight waveguide
  - Bent waveguide (*S-bend*)
  - 50:50 beamsplitter (*3dB coupler*)
    - Directional coupler (**DC**)
    - Multimode interference coupler (**MMI**)
      - Linear taper
  - Phase modulater (*thermo-optic*)

For the most part Ansys FDE and FDTD solvers suffised. They are used to get the mode solution of given cross section and to test EM propagation in 3D throughout the system respectively.
In the [``optics_simulator.py``](https://github.com/Szyli/Bell-decomposition/blob/main/optics_simulator.py) file I prepared function to generate the building blocks summarised above.

> [!TIP]
> I'd like to point out the way I coded the controll points of the Bézier curve that define the S-bends. Currently, it is automatic and hard coded given the length and the height of the bend.<br>A simple fix.<br>We can add an option to define exact coordinates of control points. Maybe a ``list`` of ``tuples`` where the length of the list is the number of control nodes with each entry being the coordinates following the order: ``(x,y)``.

For the thermo-optic phase modulation I also used the Ansys HEAT solver. The resulting heatmap of these simulation were exported to the FDE solver which, taking that into account, returned the mode solutions. The change in its effective index ($n$) gives the phase shift ($\Delta\phi$) following the relation

$$ \Delta\phi = 2\pi\frac{\Delta n}{\lambda}L $$

where $\lambda$ is the wavelength of the light and $L$ is the length of the heater wire. Heatspread in the direction of propagation was not considered but may be of small effect if the wire is placed tightly between the two S-bends. The outcome of these simulations is a relationship between the phase shift values and the electric power applied to reach them. By reverse engineering it, we can implement an algorithm that computes the necessary power (and voltage) to meet any desired phase shift values.

# Programming the voltage control card
The translation of phase shift values to voltages is carried out by the ``phase_controller.py`` file. In the end of ``test_sMZI.ipynb`` under the section **Voltage conversion** there is a short intro to its functionalities. In order to define output ports and test the voltage in real time you need the control card to be isntalled on your device!

<hr>

This is it for now. You may expect slight changes in the future but the repository is soon to lose support by me. It will most likely never be fully merged into a coherent **main** branch so please check the others, most importantly the **office** one.<br>

<p align="right"><i>2025.07.07.</i></p>
