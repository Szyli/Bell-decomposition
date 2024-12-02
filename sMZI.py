"""
    This module aims to implement the decomposition of unitary matrices
    using Symmetric Mach-Zehnder Interferometers based on Bell and Walmsley's further
    developed approach of Clement's scheme.
    
    William R. Clements et al.: An Optimal Design for Universal Multiport Interferometers
    B.A. Bell et al.: Further Compactifying Linear Optical Unitaries
    
    The following code was heavily influenced by Clement's interferometer from
    github: https://github.com/clementsw/interferometer
    
    @author: Csaba Szilard Racz //Szyli
"""

import numpy as np
import matplotlib.pyplot as plt

class Waveguide:
    """
    This class defines the straight waveguides (WG).
    Used on modes where neither phaseshifters (PS)
    nor symmetric Mach-Zehnder Interferometers (sMZI)
    are applied.
    
    ---
    
    Args:
        mode (int): the index of the mode on which the WG is located
    """
    def __init__(self, mode: int):
        self.mode = mode

    def __repr__(self):
        repr = "\n Straight waveguide on mode {}".format(self.mode)
        return repr

class Externalphaseshifter:
    """
    This class defines the external phaseshifters (PS)
    i.e. the ones that are not part of the
    symmetric Mach-Zehnder Interferometers.
    
    ---
    
    Args:
        mode (int): the index of the mode on which the PS is located
        phi (float): the angle of the PS in radians
    """
    def __init__(self, mode: int, phi: float):
        self.mode = mode
        self.phi = phi

    def __repr__(self):
        repr = "\n External phaseshift on mode {} with angle \t phi: {:.2f}".format(
            self.mode,
            self.phi)
        return repr

class Beamsplitter:
    """
    This class defines the tunable beam splitter cell
    i.e. a symmetric Mach-Zehnder Interferometer (sMZI)
    with two internal phase shifters (PS).
    
    ---
    
    The matrix describing the mode transformation is:
    
    >>``e^{i*summ}*sin(delta)``      ``e^{i*summ}cos(delta)``
    
    >>``e^{i*summ}*cos(delta)``     ``-e^{i*summ}sin(delta)``
    
    where
    - ``summ = (theta1 + theta2)/2``
    - ``delta = (theta1 - theta2)/2``
    
    with ``theta1`` and ``theta2`` being the internal phaseshift angles
    in radians.
    
    ---

    Args:
        mode1 (int): the index of the first mode ("upper")
        mode2 (int): the index of the second mode ("lower")
        theta1 (float): angle of internal phase shift on ``mode1``
        theta2 (float): angle of internal phase shift on ``mode2``
    """

    def __init__(self, mode1: int, mode2: int, theta1: float, theta2: float):
        self.mode1 = mode1
        self.mode2 = mode2
        self.theta1 = theta1
        self.theta2 = theta2

    def __repr__(self):
        repr = "\n Symmetric Mach-Zehnder Interferometer between modes {} and {}:\
                \n The angle on mode {}: {:.2f} \n The angle on mode {}: {:.2f}".format(
            self.mode1,
            self.mode2,
            self.mode1,
            self.theta1,
            self.mode2,
            self.theta2)
        return repr

class Interferometer:
    """
    This class defines the parameters of a programmable integrated
    photonic interferometer that evaluates the given Unitary operation.
    
    ---
    
    Args:
        num_of_modes (int): total number of modes (dimension of Unitary matrix)

    ---
    
    Notes
    ---
    In addition, the ``Interferometer`` class contains the following parameters:
    - ``input_phases``: An array containing the external phaseshifts (PS) applied
        at the input. These are the external PS:s applied at the beginning of each
        odd diagonals in the decomposition scheme.
    - ``output_phases``: An array containing the external phaseshifts (PS) applied
        at the output. These are the external PS:s applied at the beginning of each
        even diagonals in the decomposition scheme.
    - ``circuit``: A matrix (2D array) of the circuit where each entry defines the
        phaseshifts (PS) at the corresponding column `a` and row (mode) `b`. It excludes
        the ``input_phases`` and ``output_phases``.
    - ``mzi_list``: A dictionary where each key is the column number of the ``circuit`` matrix.
            The values are lists of tuples of two containing the mode indices
            of the internal phaseshift (PS) values ``theta1`` and ``theta2`` in radians.
    Representation:
    
    > ``column: [(row1,row2)]`` -> position of the internal phaseshifts in the ``circuit`` matrix.
    """

    def __init__(self, num_of_modes: int):
        self.num_of_modes = num_of_modes
        self.input_phases: np.ndarray[Externalphaseshifter] = np.array([], dtype=Externalphaseshifter)
        self.output_phases: np.ndarray[Externalphaseshifter] = np.array([], dtype=Externalphaseshifter)
        self.circuit: np.ndarray[float] = np.zeros(shape=(num_of_modes,num_of_modes), dtype=float)
        self.mzi_list: dict = {}
        
    def PS(self, phases: list[float] | np.ndarray[float]) -> np.ndarray[complex]:
        """
        Function that gives a matrix which describes phase shifts on the corresponding modes.
        
        ---
        
        Args:
            phases (list): phaseshifts (PS) to apply on the corresponding modes
        
        Returns
        ---
            A diagonal matrix with the corresponding phase shifts on the diagonal.
        """
        if isinstance(phases[0], Externalphaseshifter):
            exps: list[complex] = [np.exp(1j*p.phi) for p in phases]
        else:
            exps: list[complex] = [np.exp(1j*p) for p in phases]
        P_mat: np.ndarray[complex] = np.diag(exps)
        return P_mat

    def calculate_transformation(self, U: np.ndarray) -> None:
        """
        Function to tests if the given Unitary matrix can be restored using the
        decomposed elements.
        
        ---
        
        Args:
            U (array): initial/target unitary matrix
        
        ---
        
        Returns
        ---
            Prints message corresponding to the success of the recomposition.
            If the matrices equal (up to a certain tolerance) the decomposition works,
            otherwise, the difference is printed for further analysis.

        ---
                    
        Notes
        ---
            This function can also be used to get the unitary of a given circuit given that
            it has the respective geometry.
        """
        
        # preparing input/output external PS:s as "edge-vectors"
        In: list[float] = [0]*self.num_of_modes
        Out: list[float] = [0]*self.num_of_modes
        ps: Externalphaseshifter
        for ps in self.input_phases:
            In[ps.mode] = ps.phi
        for ps in self.output_phases:
            Out[ps.mode] = ps.phi
        
        # there are two different layers of BS, one starts at mode 0, the other at mode 1
        Beam_sp1: list[np.ndarray[complex]] = []
        i: int = 0
        while i < self.num_of_modes - 1:
            Beam_sp1.append(BS(self.num_of_modes, i, i+1))
            i = i + 2
        layerA: np.ndarray[complex] = matprod(Beam_sp1)
        
        Beam_sp2: list[np.ndarray[complex]] = []    
        i: int = 1
        while i < self.num_of_modes - 1:
            Beam_sp2.append(BS(self.num_of_modes, i, i+1))
            i = i + 2
        if not Beam_sp2:    # if it is empty
            layerB: np.ndarray[int] = np.eye(self.num_of_modes)
        else:
            layerB: np.ndarray[complex] = matprod(Beam_sp2)
        
        # we alternate between the two different BS layers and sandwich the phase shifts
        # since for an even number of modes we have a slightly different geometry, one has to treat the cases differently 
        U_list: list[np.ndarray[complex]] = []
        
        if self.num_of_modes%2 != 0:    # odd number of modes
            for k in range(self.num_of_modes):
                if k%2 == 0:
                    U_list.append(matprod([layerA, self.PS(self.circuit[:, self.num_of_modes-1-k]), layerA]))
                else:
                    U_list.append(matprod([layerB, self.PS(self.circuit[:, self.num_of_modes-1-k]), layerB]))
        else:   # even number of modes
            for k in range(self.num_of_modes):
                if k%2 == 0:
                    U_list.append(matprod([layerB, self.PS(self.circuit[:, self.num_of_modes-1-k]), layerB]))
                else:
                    U_list.append(matprod([layerA, self.PS(self.circuit[:, self.num_of_modes-1-k]), layerA])) 
        
        # multiply all BS layers and PS:S in the inner circuit (all described in the output matrix C)        
        U_mat: np.ndarray[complex] = matprod(U_list)
        
        # add the external PS to the input and output 
        U_imp: np.ndarray[complex] = matprod([self.PS(Out), U_mat, self.PS(In)])
        
        # compare with the desired unitary U, they should be equal up to a global phase, so we renormalize the test matrix
        test: np.ndarray[complex] = np.dot(U, U_imp.T.conj())
        test = test/test[0][0]
        
        if np.allclose(test, np.eye(self.num_of_modes, dtype=complex)):
            print("Re-composition finished.\tDecomposition was successful!")
        else:
            print("Matrices do not match!\nDifference:")
            print(test-np.eye(self.num_of_modes, dtype=complex))

    def waveguide_filler(self,
                source: np.ndarray[Externalphaseshifter]) -> list[Waveguide | Externalphaseshifter]:
        """
        Function to find modes on which there are no phaseshifts (PS) nor
        symmetric Mach-Zehnder Interferometers (sMZI) just waveguides (WG)
        and include them.
        
        ---
        
        Args:
            source (array): either ``input_phases`` or ``output_phases``
        
        ---
        
        Returns
        ---
            A list containing the input ``source`` supplemented with the missing waveguides.
        """
        
        modes_with_wg: list[bool | Externalphaseshifter | Waveguide] = [False]*self.num_of_modes
        elem: Externalphaseshifter
        for elem in source:
            modes_with_wg[elem.mode] = elem
        modes_with_wg = [Waveguide(idx) if elem==False else elem for idx, elem in enumerate(modes_with_wg)]

        return modes_with_wg

    def circuit_prep(self) -> np.ndarray[Beamsplitter | Externalphaseshifter | Waveguide]:
        """
        Function to finalize the circuit into one structure (array) called
        ``total_list``.
        
        ---
        
        Returns
        ---
            Array of circuit elements reading from first column, on the left of the circuit,
            to the right of it being the elements at the output.
        """
        
        total_list: np.ndarray[Beamsplitter | Externalphaseshifter | Waveguide] = np.array(self.waveguide_filler(self.input_phases))
        
        col: int
        row: tuple[int, int]
        elem: int
        for col in self.mzi_list:
            rows: np.ndarray[int] = np.arange(0, self.num_of_modes, 1)  # to follow which row was MZI and which a PS
            used_idx: tuple[int, int] = ()
            for row in self.mzi_list[col]:
                used_idx += row
                total_list = np.append(total_list, Beamsplitter(row[0],row[1], self.circuit[row[0],col], self.circuit[row[1],col]))
            # Update rows; Remaining are PS:s
            rows: np.ndarray = np.delete(rows, used_idx)
            for elem in rows:
                total_list = np.append(total_list, Externalphaseshifter(elem, self.circuit[elem, col]))
        
        total_list = np.append(total_list, self.waveguide_filler(self.output_phases))
        
        return total_list

    def curve(self, x1: float, x2: float, mode: bool, flip: bool = False,
            res: int = 10, scale: float = 0.5, c: float = 0) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Function to calculate coordinates of curved lines for the directional couplers of the circuit.
        
        ---
        
        Args:
            x1 (int): starting position in ``x``
            x2 (int): final position in ``x``
            mode (bool): True gives `cos`, False gives `sin`
            flip (bool): determines if curve needs to be mirrored around the x-axes
            res (int): resolution
            scale (float): relative height of curve to set return values corresponding
                to the environment
            c (float): shift in ``y`` direction

        ---
        
        Returns
        ---
            Two arrays ``(x, y)`` to plot.
        """
        x: np.ndarray[float] = np.linspace(0, np.pi/2, res)
        
        sign: int
        y: np.ndarray[float]
        if flip:
            sign = -1
        else:
            sign = 1
        if mode:
            y = np.cos(x) * sign * scale + c
        else:
            y = np.sin(x) * sign * scale + c
        
        x = np.linspace(x1, x2, res)
        
        return (x, y)
    
    def draw_curved_connectors(self, ord: bool, x_start: float, x_end: float, corr: float) -> None:
        """
        Function that draws the curved waveguids (directional couplers).
        
        ---
        
        Args:
            ord (bool): which pair of curved waveguides (WG) to draw
                
                > ``if True``: first (Coupling)
                
                > ``if False``: second (Splitter)
        
            x_start (float): position of first point in ``x``
            x_end (float): position of last point in ``x``
            corr (float): correction in ``y`` direction
        """
        
        c1: np.ndarray[float]
        c2: np.ndarray[float]
        if ord:
            c1, c2 = self.curve(x_start, x_end, mode=True, flip=False, c=corr)
            # line connecting UP to MID
            plt.plot(c1, c2, lw=1, color="blue")
            
            # line connecting BOT to MID
            c1, c2 = self.curve(x_start, x_end, mode=True, flip=True, c=corr)
            plt.plot(c1, c2, lw=1, color="blue")
        else:
            # line connecting MID to UP
            c1, c2 = self.curve(x_start, x_end, mode=False, flip=False, c=corr)
            plt.plot(c1, c2, lw=1, color="blue")
            
            # line connecting MID to BOT
            c1, c2 = self.curve(x_start, x_end, mode=False, flip=True, c=corr)
            plt.plot(c1, c2, lw=1, color="blue")
        
    def draw_external_ps(self, x: int, n: int, ext_ps: Externalphaseshifter,
                        used: bool, cl: str = "red",
                        internal: bool = False) -> bool:
        """
        Function to draw the external phasshifters (PS).
        
        ---
        
        Args:
            x (int): current position in ``x``
            n (int): total number of modes
            ext_ps (Externalphaseshifter): the external PS
            used (bool): whether the selected external PS has been used or not
            even (bool): whether we're on even or odd diagonal
            cl (str): color of text
            
        Returns
        ---
            A boolean representing the state of the current PS.
            
        > True means it has been used/drew.
        """
        
        if not used:
            used = True
            if not internal:
                # EXTERNAL PS on the edges of the circuit
                plt.plot((x+0.15, x+0.15), (n-ext_ps.mode-0.3, n-ext_ps.mode+0.2), lw=1, color="blue")
                circle = plt.Circle((x+0.15, n-ext_ps.mode), 0.1, fill=False)
                plt.gca().add_patch(circle)
                phase = "{:2f}".format(ext_ps.phi)
                if ext_ps.phi > 0:
                    plt.text(x+0.2, n-ext_ps.mode-0.3, phase[0:3], color=cl, fontsize=7)
                else:
                    plt.text(x+0.2, n-ext_ps.mode-0.3, phase[0:4], color=cl, fontsize=7)
            else:
                # EXTERNAL PS inside the circuit
                plt.plot((x+0.85, x+0.85), (n-ext_ps.mode-0.3, n-ext_ps.mode+0.2), lw=1, color="blue")
                circle = plt.Circle((x+0.85, n-ext_ps.mode), 0.1, fill=False)
                plt.gca().add_patch(circle)
                phase = "{:2f}".format(ext_ps.phi)
                if ext_ps.phi > 0:
                    plt.text(x+0.9, n-ext_ps.mode-0.3, phase[0:3], color=cl, fontsize=7)
                else:
                    plt.text(x+0.9, n-ext_ps.mode-0.3, phase[0:4], color=cl, fontsize=7)
    
        return used
    
    def draw(self, size: tuple[float, float] | None = None, show_plot: bool = True, save_fig: bool = False) -> None:  
        """
        Function to draw the circuit including the phaseshift values
        (in radians) on the PS:s.
        
        ---

        Args:
            size (tuple): ``x`` and ``y`` dimension of plot
            show_plot (bool): whether to show the generated plot
            save_fig (bool): whether to save the figure or not
        """
        
        mode_tracker: np.ndarray[float] = np.zeros(self.num_of_modes)
        sc: float = 0.5
        
        # calling the full source material (array of all elements)
        circuit = self.circuit_prep()

        if size == None:
            size = (self.num_of_modes*2 +4, self.num_of_modes*2)
            plt.figure(figsize = size)
        else:
            plt.figure(figsize = size)

        # initial/starting lines
        i: int
        for i in range(1, self.num_of_modes+1, 1):
            plt.plot((-1, 0), (i, i), lw=1, color="blue")

        used: bool = True
        
        for idx in range(len(circuit)):
            elem: Beamsplitter | Externalphaseshifter | Waveguide = circuit[idx]
            if isinstance(elem, Externalphaseshifter):
                used = False
                x = mode_tracker[elem.mode]
                
                # initial PS:s
                if (idx < self.num_of_modes) or (idx > (len(circuit)-self.num_of_modes-1)):
                    used = self.draw_external_ps(x, self.num_of_modes, elem, used)
                    plt.plot((x+0.25, x+0.85), (self.num_of_modes-elem.mode, self.num_of_modes-elem.mode), lw=1, color="blue")
                    # update
                    mode_tracker[elem.mode] = x+0.6
                    # for alignment in the odd modes cases 
                    if elem.mode == self.num_of_modes-2:
                        mode_tracker[-1] = x+0.6
                        plt.plot((x, x+0.6), (1, 1), lw=1, color="blue")
                else:
                    used = self.draw_external_ps(x, self.num_of_modes, elem, used, internal=True)
                    plt.plot((x, x+0.6), (self.num_of_modes-elem.mode, self.num_of_modes-elem.mode), lw=1, color="blue")
                    # update
                    mode_tracker[elem.mode] = x+0.6

            if isinstance(elem, Beamsplitter):
                x: float = np.max([mode_tracker[elem.mode1], mode_tracker[elem.mode2]])
                
                if used:
                    # no external PS to be placed; draw placeholders
                    plt.plot((x, x+0.3), (self.num_of_modes - elem.mode1, self.num_of_modes - elem.mode1), lw=1, color="blue")
                    plt.plot((x, x+0.3), (self.num_of_modes - elem.mode2, self.num_of_modes - elem.mode2), lw=1, color="blue")
                
                # creating curved lines
                self.draw_curved_connectors(ord=True, x_start=x+0.3, x_end=x+0.5,corr=self.num_of_modes-elem.mode1-sc)
                self.draw_curved_connectors(ord=False, x_start=x+0.5, x_end=x+0.7,corr=self.num_of_modes-elem.mode1-sc)
                
                # INTERNAL PS: theta1
                plt.plot((x+0.85, x+0.85), (self.num_of_modes+0.3-(elem.mode2 + elem.mode1)/2., self.num_of_modes+0.7-(elem.mode2 + elem.mode1)/2.), lw=1, color="blue")
                circle = plt.Circle((x+0.85, self.num_of_modes+0.5-(elem.mode2 + elem.mode1)/2.), 0.1, fill=False)
                plt.gca().add_patch(circle)
                inter_phase = "{:2f}".format(elem.theta1)
                plt.text(x+0.9, self.num_of_modes+0.7-(elem.mode2 + elem.mode1)/2, inter_phase[0:3], color="green", fontsize=7)
                
                # INTERNAL PS: theta2
                plt.plot((x+0.85, x+0.85), (self.num_of_modes-0.3-(elem.mode2 + elem.mode1)/2., self.num_of_modes-0.7-(elem.mode2 + elem.mode1)/2.), lw=1, color="blue")
                circle = plt.Circle((x+0.85, self.num_of_modes-0.5-(elem.mode2 + elem.mode1)/2.), 0.1, fill=False)
                plt.gca().add_patch(circle)
                inter_phase = "{:2f}".format(elem.theta2)
                plt.text(x+0.9, self.num_of_modes-0.7-(elem.mode2 + elem.mode1)/2, inter_phase[0:3], color="green", fontsize=7)
                
                # creating curved lines
                self.draw_curved_connectors(ord=True, x_start=x+1, x_end=x+1.2, corr=self.num_of_modes-elem.mode1-sc)
                self.draw_curved_connectors(ord=False, x_start=x+1.2, x_end=x+1.4, corr=self.num_of_modes-elem.mode1-sc)

                # connecting neighbouring cells
                plt.plot((x+1.4, x+2), (self.num_of_modes-elem.mode1, self.num_of_modes-elem.mode1), lw=1, color="blue")
                plt.plot((x+1.4, x+2), (self.num_of_modes-elem.mode2, self.num_of_modes-elem.mode2), lw=1, color="blue")
                
                # update and draw lines of modes otherwise left empty
                if x > mode_tracker[elem.mode1]:
                    plt.plot((mode_tracker[elem.mode1], x), (self.num_of_modes-elem.mode1, self.num_of_modes-elem.mode1), lw=1, color="blue")
                if x > mode_tracker[elem.mode2]:
                    plt.plot((mode_tracker[elem.mode2], x), (self.num_of_modes-elem.mode2, self.num_of_modes-elem.mode2), lw=1, color="blue")
                mode_tracker[elem.mode1] = x+2
                mode_tracker[elem.mode2] = x+2

        max_x: float = np.max(mode_tracker)
        i: int
        for i in range(self.num_of_modes):
            plt.plot((mode_tracker[i], max_x+1), (self.num_of_modes-i, self.num_of_modes-i), lw=1, color="blue")

        plt.text(max_x/3, self.num_of_modes+1, r"red: external phase shift ($\phi$)", color="red", fontsize=10)
        plt.text(max_x/3, self.num_of_modes+1-0.25, r"green: internal phase shifts ($\theta_1$,$\theta_2$)", color="green", fontsize=10)
        plt.text(-1, self.num_of_modes+0.2, "Light in", fontsize=10)
        plt.text(max_x+0.5, self.num_of_modes+0.2, "Light out", fontsize=10)
        plt.gca().axes.set_ylim([0.2, self.num_of_modes+1.2])
        plt.axis("off")
        
        if save_fig:
            plt.savefig("{}_{}-circuit.png".format(self.num_of_modes, self.num_of_modes))
        if show_plot:
            plt.show()

class AnsysCircuit:
    """
    This class is responsible for building the Integrated Optical
    Circuit based on the Unitary Decomposition in Ansys Interconnect.
    
    ---
    
    Args:
        num_of_modes (int): total number of modes/dimension of Unitary matrix
        list_of_elements (np.ndarray[Beamsplitter | Externalphaseshifter | Waveguide]): a list of all the elements in the circuit
    
    ---
    
    Notes
    ---
    In addition, the ``AnsysCircuit`` class contains the following parameter:
    - ``missing_elements``: A list of the compound elements that are not located
        in the Ansys custom elements library.
    """
    
    # include ANSYS
    import sys, os
    # Note: This path may very for users, therefore
    # Todo: Update to find path corresponding to limerical
    sys.path.append("D:\\lumerical\\api\\python")
    sys.path.append(os.path.abspath(""))

    def __init__(self, num_of_modes: int,
                list_of_elements: np.ndarray[Beamsplitter | Externalphaseshifter | Waveguide]):
        
        self.num_of_modes = num_of_modes
        self.elements = list_of_elements
        self.missing_elements: list = []

    def check_lib(self, interconnect) -> None:
        """
        Function to check if custom compounds are present in Ansys
        custom elements folder or not.

        ---

        Notes
        ---
        Nothing is returned but sets the property ``missing_elements``
        of class ``AnsysCircuit`` according to the following:
        
        - if all elements exist: return an empty list
        - if some exists: return a list of the nonexisting elements (to build later)
        - if the folder is empty or it does not exist: return a list of all the elements to be built
        """
        
        # get path of the custom elements
        ic = interconnect(hide=True)
        path: str = ic.customlibrary()
        path += "/Bell kits"    # name of the folder in which this code saves custom elements
        list_of_elem_names: list[str] = ["sMZI.ice", "Ext_PS.ice", "WG.ice"]
        missing_elements: list[str] = list_of_elem_names.copy()

        if self.os.path.exists(path):   # does the folder exist?
            dirlist: list[str] = self.os.listdir(path)
            if dirlist != 0:    # is it empty?
                if list_of_elem_names == dirlist:   # are all elements present?
                    missing_elements = []
                else:
                    missing_elements = [elem for elem in list_of_elem_names if elem not in dirlist]
        
        # set class property
        self.missing_elements = missing_elements

    def build_compounds(self) -> None:
        """
        Function to build missing compound elements in Ansys Interconnect.
        
        ---
        
        Acronyms of the components:
        - Straight Waveguide (WG)
        - External Phase Shifter (Ext_PS)
        - symmetric Mach-Zehnder Interferometer (sMZI)

        ---

        Evaluates Ansys script that builds the missing elements based on the
        ``missing_elements`` property of the class ``AnsysCircuit``.
        """

        script: str = "groupscope('::Root Element');\n"

        if "WG.ice" in self.missing_elements:   # create compound: Straight Waveguide
            print("Building WG")
            compound_name: str = 'WG'
            wave_guide_element: str = 'Straight Waveguide'
            x_spacing: int = 200

            script += "addelement('"+str(wave_guide_element)+"');\n"\
                    + "set('name','WGD');\n"\
                    + "set('x position', "+str(0)+");\n"\
                    + "set('y position', "+str(0)+");\n"\
                    + "selectall;\n"\
                    + "createcompound();\n"\
                    + "set('name',"+"'"+str(compound_name)+"');\n"\
                    + "set('prefix','"+str(compound_name)+"');\n"\
                    + "set('description','Straight WaveGuide');\n"\
                    + "addport('"+str(compound_name)+"','port_1', 'bidirectional', 'Optical Signal', 'left',0.5);\n"\
                    + "addport('"+str(compound_name)+"','port_2', 'bidirectional', 'Optical Signal', 'right',0.5);\n"\
                    + "groupscope('::Root Element::"+compound_name+"');\n"\
                    + "select('RELAY_1');\n"\
                    + "set('name','Port_1');\n"\
                    + "set('x position', "+str(-x_spacing)+");\n"\
                    + "set('y position', "+str(0)+");\n"\
                    + "connect('Port_1','port','WGD','port 1');\n"\
                    + "flipelement('Port_1');\n"\
                    + "select('RELAY_2');\n"\
                    + "set('name','Port_2');\n"\
                    + "set('x position', "+str(x_spacing)+");\n"\
                    + "set('y position', "+str(0)+");\n"\
                    + "connect('Port_2','port','WGD','port 2');\n"

            script += "groupscope('::Root Element');\n"\
                    + "select('"+str(compound_name)+"');\n"\
                    + "addtolibrary('Bell kits');\n"\
                    + "deleteall;\n"

        if "Ext_PS.ice" in self.missing_elements:   # create compound: External Phase Shifter
            print("Building Ext_PS")
            compound_name: str = 'Ext_PS'
            phase_shift_element: str = 'Optical Phase Shift'
            wave_guide_element: str = 'Straight Waveguide'
            x_spacing: int = 200

            script += "addelement('"+str(wave_guide_element)+"');\n"\
                    + "set('name','WGD_1');\n"\
                    + "set('x position', "+str(0)+");\n"\
                    + "set('y position', "+str(0)+");\n"\
                    + "addelement('"+str(phase_shift_element)+"');\n"\
                    + "set('name','Phi');\n"\
                    + "set('x position', "+str(x_spacing)+");\n"\
                    + "set('y position', "+str(0)+");\n"\
                    + "connect('WGD_1','port 2','Phi','port 1');\n"\
                    + "addelement('"+str(wave_guide_element)+"');\n"\
                    + "set('name','WGD_2');\n"\
                    + "set('x position', "+str(2*x_spacing)+");\n"\
                    + "set('y position', "+str(0)+");\n"\
                    + "connect('WGD_2','port 1','Phi','port 2');\n"\
                    + "selectall;\n"\
                    + "createcompound();\n"\
                    + "set('name',"+"'"+str(compound_name)+"');\n"\
                    + "set('prefix','"+str(compound_name)+"');\n"\
                    + "set('description','External Phase Shifter');\n"\
                    + "addport('"+str(compound_name)+"','port_1', 'bidirectional', 'Optical Signal', 'left',0.5);\n"\
                    + "addport('"+str(compound_name)+"','port_2', 'bidirectional', 'Optical Signal', 'right',0.5);\n"\
                    + "groupscope('::Root Element::"+compound_name+"');\n"\
                    + "select('RELAY_1');\n"\
                    + "set('name','Port_1');\n"\
                    + "set('x position', "+str(-x_spacing)+");\n"\
                    + "set('y position', "+str(0)+");\n"\
                    + "connect('Port_1','port','WGD_1','port 1');\n"\
                    + "flipelement('Port_1');\n"\
                    + "select('RELAY_2');\n"\
                    + "set('name','Port_2');\n"\
                    + "set('x position', "+str(3*x_spacing)+");\n"\
                    + "set('y position', "+str(0)+");\n"\
                    + "connect('Port_2','port','WGD_2','port 2');\n"

            script += "groupscope('::Root Element');\n"\
                    + "select('"+str(compound_name)+"');\n"\
                    + "addtolibrary('Bell kits');\n"\
                    + "deleteall;\n"

        if "sMZI.ice" in self.missing_elements: # create compound: symmetric Mach-Zehnder Interferometer
            print("Building sMZI")
            compound_name: str = 'sMZI'
            phase_shift_element: str = 'Optical Phase Shift'
            coupler_element: str = 'Waveguide Coupler'
            wave_guide_element: str = 'Straight Waveguide'
            
            x_spacing: int = 200
            y_spacing: int = 80

            script += "addelement('"+str(coupler_element)+"');\n"\
                    + "set('name','Coupler_1');\n"\
                    + "set('x position', "+str(0)+");\n"\
                    + "set('y position', "+str(y_spacing)+");\n"\
                    + "addelement('"+str(wave_guide_element)+"');\n"\
                    + "set('name','WGD_1');\n"\
                    + "set('x position', "+str(x_spacing)+");\n"\
                    + "set('y position', "+str(0)+");\n"\
                    + "addelement('"+str(phase_shift_element)+"');\n"\
                    + "set('name','Theta_1');\n"\
                    + "set('x position', "+str(2*x_spacing)+");\n"\
                    + "set('y position', "+str(0)+");\n"\
                    + "addelement('"+str(wave_guide_element)+"');\n"\
                    + "set('name','WGD_3');\n"\
                    + "set('x position', "+str(3*x_spacing)+");\n"\
                    + "set('y position', "+str(0)+");\n"\
                    + "connect('WGD_1','port 2','Theta_1','port 1');\n"\
                    + "connect('WGD_3','port 1','Theta_2','port 2');\n"\
                    + "addelement('"+str(wave_guide_element)+"');\n"\
                    + "set('name','WGD_2');\n"\
                    + "set('x position', "+str(x_spacing)+");\n"\
                    + "set('y position', "+str(2*y_spacing)+");\n"\
                    + "addelement('"+str(phase_shift_element)+"');\n"\
                    + "set('name','Theta_2');\n"\
                    + "set('x position', "+str(2*x_spacing)+");\n"\
                    + "set('y position', "+str(2*y_spacing)+");\n"\
                    + "addelement('"+str(wave_guide_element)+"');\n"\
                    + "set('name','WGD_4');\n"\
                    + "set('x position', "+str(3*x_spacing)+");\n"\
                    + "set('y position', "+str(2*y_spacing)+");\n"\
                    + "connect('WGD_2','port 2','Theta_2','port 1');\n"\
                    + "connect('WGD_4','port 1','Theta_2','port 2');\n"\
                    + "addelement('"+str(coupler_element)+"');\n"\
                    + "set('name','Coupler_2');\n"\
                    + "set('x position', "+str(2*x_spacing)+");\n"\
                    + "set('y position', "+str(y_spacing)+");\n"\
                    + "connect('Coupler_1','port 3','WGD_1','port 1');\n"\
                    + "connect('Coupler_1','port 4','WGD_2','port 1');\n"\
                    + "connect('Coupler_2','port 1','WGD_3','port 2');\n"\
                    + "connect('Coupler_2','port 2','WGD_4','port 2');\n"\
                    + "selectall;\n"\
                    + "createcompound();\n"\
                    + "set('name',"+"'"+str(compound_name)+"');\n"\
                    + "set('prefix','"+str(compound_name)+"');\n"\
                    + "set('description','Symmetric Mach-Zehnder Interferometer');\n"\
                    + "addport('"+str(compound_name)+"','port_1', 'bidirectional', 'Optical Signal', 'left',0.25);\n"\
                    + "addport('"+str(compound_name)+"','port_2', 'bidirectional', 'Optical Signal', 'left',0.75);\n"\
                    + "addport('"+str(compound_name)+"','port_3', 'bidirectional', 'Optical Signal', 'right',0.25);\n"\
                    + "addport('"+str(compound_name)+"','port_4', 'bidirectional', 'Optical Signal', 'right',0.75);\n"\
                    + "groupscope('::Root Element::"+compound_name+"');\n"\
                    + "select('RELAY_1');\n"\
                    + "set('name','Port_1');\n"\
                    + "set('x position', "+str(-x_spacing)+");\n"\
                    + "set('y position', "+str(0)+");\n"\
                    + "connect('Port_1','port','Coupler_1','port 1');\n"\
                    + "flipelement('Port_1');\n"\
                    + "select('RELAY_2');\n"\
                    + "set('name','Port_2');\n"\
                    + "set('x position', "+str(-x_spacing)+");\n"\
                    + "set('y position', "+str(2*y_spacing)+");\n"\
                    + "connect('Port_2','port','Coupler_1','port 2');\n"\
                    + "flipelement('Port_2');\n"\
                    + "select('RELAY_3');\n"\
                    + "set('name','Port_3');\n"\
                    + "set('x position', "+str(3*x_spacing)+");\n"\
                    + "set('y position', "+str(0)+");\n"\
                    + "connect('Port_3','port','Coupler_2','port 3');\n"\
                    + "select('RELAY_4');\n"\
                    + "set('name','Port_4');\n"\
                    + "set('x position', "+str(3*x_spacing)+");\n"\
                    + "set('y position', "+str(2*y_spacing)+");\n"\
                    + "connect('Port_4','port','Coupler_2','port 4');\n"

            script += "groupscope('::Root Element');\n"\
                    + "select('"+str(compound_name)+"');\n"\
                    + "addtolibrary('Bell kits');\n"\
                    + "deleteall;\n"

        # build elements
        ic = self.INTERCONNECT(hide=True)
        ic.eval(script)

    def build_WGD(self, ID: int, x: int, y: int, prior_elem: str = None, port: int = None) -> tuple[str, str]:
        """
        Function to build Straight Waveguides and connect to the prior
        element in the circuit with the defined port.
        
        ---

        Args:
            ID (int): serial number of current element
            prior_elem (str): name of the element to connect this with
            port (int): serial number of ``prior_elem``'s port to connect this element's port with
            x (int): ``x`` position of element
            y (int): ``y`` position of element

        Returns
        ---
            Ansys script to evaluate together with the name of this element.
            
        ---
        
        Notes
        ---
            Connection to ``prior_elem`` is to be thought of according to the following
            schematic where it is important to note that we connect the new elements
            to the previous ones on the left (located in the previous column of the
            circuit).
            
        > |``prior_elem``|--(``port``)--|``WGD``|
        """
        # might be unnecessary
        element_name: str = "WG"
        element_ID: str = "WG_" + str(ID)
        
        script: str = "addelement('"+element_name+"');\n"\
                    + "set('name','"+element_ID+"');\n"\
                    + "set('x position', "+str(x)+");\n"\
                    + "set('y position', "+str(y)+");\n"
        
        if port:
            # if port is not None
            # NOTE: port = 0 would also skip this statement but
            # we name ports by counting from 1
            port: str = "port_" + str(port)
            script += "connect('"+prior_elem+"','"+port+"','"+element_ID+"','port_1');"
        
        return script, element_ID

    def build_Ext_PS(self, ID: int, phi: float, x: int, y: int, prior_elem: str = None, port: int = None) -> tuple[str, str]:
        """
        Function to build an external phaseshifter (PS) with
        defined phase (in radian) and connect to the prior
        element in the circuit with the defined port.
        
        ---
        
        Args:
            ID (int): serial number of current element
            phi (float): value of phase shift (in radian)
            prior_elem (str): name of the element to connect this with
            port (int): serial number of ``prior_elem``'s port to connect this element's port with
            x (int): ``x`` position of element
            y (int): ``y`` position of element

        Returns
        ---
            Ansys script to evaluate together with the name of this element.
            
        ---
        
        Notes
        ---
            Connection to ``prior_elem`` is to be thought of according to the following
            schematic where it is important to note that we connect the new elements
            to the previous ones on the left (located in the previous column of the
            circuit).
            
        > |``prior_elem``|--(``port``)--|``Ext_PS``|
        """
        element_name: str = "Ext_PS"
        element_ID: str = "PS_" + str(ID)
        compound_name: str = "Phi"

        script: str = "addelement('"+element_name+"');\n"\
                    + "set('name','"+element_ID+"');\n"\
                    + "set('x position', "+str(x)+");\n"\
                    + "set('y position', "+str(y)+");\n"\
                    + "select('"+element_ID+"::"+compound_name+"');\n"\
                    + "set('phase shift',"+str(phi)+");\n"
        if port:
            # if port is not None
            # NOTE: port = 0 would also skip this statement but
            # we name ports by counting from 1
            port: str = "port_" + str(port)
            script += "connect('"+prior_elem+"','"+port+"','"+element_ID+"','port_1');"
        
        return script, element_ID

    def build_sMZI(self, ID: int, theta_1: float, theta_2: float, x: int, y: int,
                prior_elem_1: str = None, prior_elem_2: str = None,
                port_1: int = None, port_2: int = None) -> tuple[str, str]:
        """
        Function to build a Symmetric Mach-Zehnder Interferometer (sMZI) with
        defined phaseshifts (in radian) and connect to the prior
        elements in the circuit with the defined ports.
        
        ---

        Args:
            ID (int): serial number of this element
            theta_1 (float): value of phase shift on first(/upper) arm
            theta_2 (float): value of phase shift on second(/lower) arm
            prior_elem_1 (str): name of the element to connect the upper first(/upper) arm with
            prior_elem_2 (str): name of the element to connect the upper second(/lower) arm with
            port_1 (int): serial number of ``prior_elem_1``'s port to connect this element's
                first(/upper) port with
            port_2 (int): serial number of ``prior_elem_2``'s port to connect this element's
                second(/lower) port with
            x (int): ``x`` position of element
            y (int): ``y`` position of element

        Returns
        ---
            Ansys script to evaluate together with the name of this element.
            
        ---
        
        Notes
        ---
            Connection to ``prior_elem`` is to be thought of according to the following
            schematic where it is important to note that we connect the new elements
            to the previous ones on the left (located in the previous column of the
            circuit).
            
        > |``prior_elem_1``|--(``port_1``)--v

                                                >==|``sMZI``|

        > |``prior_elem_2``|--(``port_2``)--^
        """
        element_name: str = "sMZI"
        element_ID: str = "sMZI_" + str(ID)
        compound_name_1: str = "Theta_1"
        compound_name_2: str = "Theta_2"

        script: str = "addelement('"+element_name+"');\n"\
                    + "set('name','"+element_ID+"');\n"\
                    + "set('x position', "+str(x)+");\n"\
                    + "set('y position', "+str(y)+");\n"\
                    + "select('"+element_ID+"::"+compound_name_1+"');\n"\
                    + "set('phase shift',"+str(theta_1)+");\n"\
                    + "select('"+element_ID+"::"+compound_name_2+"');\n"\
                    + "set('phase shift',"+str(theta_2)+");\n"
        if port_1:
            # if port is not None
            # NOTE: port = 0 would also skip this statement but
            # we name ports by counting from 1
            port: str = "port_" + str(port_1)
            script += "connect('"+prior_elem_1+"','"+port+"','"+element_ID+"','port_1');"
        if port_2:
            port = "port_" + str(port_2)
            script += "connect('"+prior_elem_2+"','"+port+"','"+element_ID+"','port_2');"
        
        return script, element_ID


    def build_Circuit(self) -> None:
        """
        Function to build the integrated optical circuit, in Ansys Interconnect
        that implements the Unitary operation given by the user based on
        the scheme by B.A. Bell et al.

        ---

        Saves the circuit in ``.icp`` format for Ansys to open.

        """
        
        # include ANSYS
        from lumapi import INTERCONNECT
        
        # Pre-checks:
        self.check_lib(INTERCONNECT)
        if self.missing_elements:
            print("Building elements first:", self.missing_elements)
            self.build_compounds()
            print("Necessarry compounds have been built!")
        
        # structure: (name of element, port of element on its right, id of column)
        circuit_map: dict[tuple[None | int, None | int, None | int]] = dict.fromkeys(np.arange(0, self.num_of_modes, 1),(None, None, 0))
        input_connect: list[str] = []   # list of elements' name in the first column of the circuit
        eval_string: str = "groupscope('::Root Element');"
        wgd_id: int = 0
        ps_id: int = 0
        smzi_id: int = 0
        x_spacing: int = 200
        y_spacing: int = 160
        
        elem: Beamsplitter | Externalphaseshifter | Waveguide
        for elem in self.elements:
            if isinstance(elem, Waveguide):
                # if circuit_map[elem.mode]:
                script, name = self.build_WGD(wgd_id,
                                            circuit_map[elem.mode][2]*x_spacing, elem.mode*y_spacing,
                                            circuit_map[elem.mode][0], circuit_map[elem.mode][1])
                # update dictionary and input_connect
                if not circuit_map[elem.mode][0]:
                    input_connect.append(name)
                circuit_map[elem.mode] = (name, 2, circuit_map[elem.mode][2] + 1)
                eval_string += script
                wgd_id += 1
            elif isinstance(elem, Externalphaseshifter):
                # if circuit_map[elem.mode]:
                script, name = self.build_Ext_PS(ps_id, elem.phi,
                                                circuit_map[elem.mode][2]*x_spacing, elem.mode*y_spacing,
                                                circuit_map[elem.mode][0], circuit_map[elem.mode][1])
                # update dictionary and input_connect
                if not circuit_map[elem.mode][0]:
                    input_connect.append(name)
                circuit_map[elem.mode] = (name, 2, circuit_map[elem.mode][2] + 1)
                eval_string += script
                ps_id += 1
            elif isinstance(elem, Beamsplitter):
                # if circuit_map[elem.mode1] or circuit_map[elem.mode2]:
                # choose correct column (x position)
                if circuit_map[elem.mode1][2] > circuit_map[elem.mode2][2]:
                    x = circuit_map[elem.mode1][2]
                else:
                    x = circuit_map[elem.mode2][2]
                
                script, name =  self.build_sMZI(smzi_id, elem.theta1, elem.theta2,
                                        x*x_spacing, (elem.mode1 + elem.mode2)*0.5*y_spacing,
                                        circuit_map[elem.mode1][0], circuit_map[elem.mode2][0],
                                        circuit_map[elem.mode1][1], circuit_map[elem.mode2][1])
                # update dictionary; input_name is irrelevant since first column never
                # includes sMZI elements due to design
                circuit_map[elem.mode1] = (name, 3, x + 1)
                circuit_map[elem.mode2] = (name, 4, x + 1)
                eval_string += script
                smzi_id += 1
            else:
                print(f"Error! Element of type {type(elem)} is incorrect at index position: {ps_id+smzi_id}")
        
        # Finaly, compactify circuit
        circuit_name = f"U_{self.num_of_modes}x{self.num_of_modes}"
        input_name = "input"
        output_name = "output"

        eval_string += "selectall;\n"\
                    + "createcompound();\n"\
                    + "set('name',"+"'"+str(circuit_name)+"');\n"\
                    + "set('prefix','"+str(circuit_name)+"');\n"\
                    + "set('description','Integrated photonical representation of decomposed "+str(circuit_name)+" matrix based on scheme by Bell et.al.');\n"
        
        port_pos: float = 1/(self.num_of_modes+1)
        for id in range(1, self.num_of_modes+1, 1):
            eval_string += "addport('"+str(circuit_name)+"','"+input_name+"_"+str(id)+"', 'bidirectional', 'Optical Signal', 'left',"+str(port_pos*id)+");\n"\
                    + "addport('"+str(circuit_name)+"','"+output_name+"_"+str(id)+"', 'bidirectional', 'Optical Signal', 'right',"+str(port_pos*id)+");\n"
        
        eval_string += "groupscope('::Root Element::"+circuit_name+"');\n"
        for i in range(1,self.num_of_modes+1, 1):   # input RELAY connections
            rel_idx_in = 2*i-1
            rel_idx_out = i*2
            # input ports
            eval_string += "select('RELAY_"+str(rel_idx_in)+"');\n"\
                        + "set('name','input_"+str(i)+"');\n"\
                        + "set('x position', "+str(-x_spacing)+");\n"\
                        + "set('y position', "+str(y_spacing*(i-1))+");\n"\
                        + "connect('input_"+str(i)+"','port','"+str(input_connect[i-1])+"','port_1');\n"\
                        + "flipelement('input_"+str(i)+"');\n"
            # output ports
            eval_string += "select('RELAY_"+str(rel_idx_out)+"');\n"\
                        + "set('name','output_"+str(i)+"');\n"\
                        + "set('x position', "+str(circuit_map[i-1][2]*x_spacing)+");\n"\
                        + "set('y position', "+str(y_spacing*(i-1))+");\n"\
                        + "connect('output_"+str(i)+"','port','"+str(circuit_map[i-1][0])+"','port_2');\n"
        
        eval_string += "groupscope('::Root Element');\n"
        # evaluate complete circuit
        ic = self.INTERCONNECT(hide=True)
        ic.eval(eval_string)
        ic.save(f"{self.num_of_modes}x{self.num_of_modes}_test.icp")

        # return eval_string

# -----------------------------

def square_decomposition(U: np.ndarray[float | complex]) -> Interferometer:
    """
    This function implements the Unitary matrix decomposition based on
    the paper:
    
    **B.A. Bell et al.: Further Compactifying Linear Optical Unitaries**

    ---
    
    Introduction
    ---
        It includes (symmetric) Mach-Zehnder Interferometers (sMZI) with two phasehifters
        on both of their arms and phaseshifter outside of the sMZI:s. The matrix is decoded
        in a symmetric rectangular circuit.
    
    ---
    
    Returns
    ---
        An Interferometer instance that enables further work
        such as testing the success of the decomposition
        (``Interferometer.calculate_transformation``) and
        the creation of a drawing of the circuit
        (``Interferometer.draw``).
        The result of ``Interferometer.circuit_prep`` can
        be used in the class ``AnsysCircuit`` to prepare
        the same circuit for simulations.
    """
    m: int = U.shape[0]    # dimension of matrix = number of modes 'm'
    Itf: Interferometer = Interferometer(num_of_modes = m)
    V_mat: np.ndarray[float | complex] = np.conjugate(U)
    
    j: int
    for j in range(1, m): # odd diags: 1,3,5...
        if j%2 != 0:
            x: int = m-1
            y: int = j-1
            s: int = y+1 #place of the external phase shift P 
            # find external phaseshift that matches given elements' phases
            P_mat, phi = external_ps(m, s, V_mat[x, y], V_mat[x, y+1])
            V_mat = np.matmul(V_mat, P_mat)
            Itf.input_phases = np.append(Itf.input_phases, Externalphaseshifter(s, phi))
            
            k: int
            for k in range(1, j+1):
                modes: list[int] = [y, y+1]    # initial mode-pairs
                delta = custom_arctan(V_mat[x, y+1], V_mat[x, y])
                
                if k == j:  # redundant choice
                    summ: float = 0
                else:   # derivation shows
                    summ: float = np.angle(V_mat[x-1, y-1]) - np.angle(V_mat[x-1, y]*np.sin(delta) + V_mat[x-1, y+1]*np.cos(delta))
                
                M_mat = np.eye(m, dtype=np.complex128)
                M_mat[modes[0], modes[0]] =  np.sin(delta) * np.exp(1j*summ)
                M_mat[modes[1], modes[0]] =  np.cos(delta) * np.exp(1j*summ)
                M_mat[modes[0], modes[1]] =  np.cos(delta) * np.exp(1j*summ)
                M_mat[modes[1], modes[1]] = -np.sin(delta) * np.exp(1j*summ)
                V_mat = np.matmul(V_mat, M_mat)
                
                theta1, theta2 = internal_phases(delta, summ)
                
                #save the angles in the matrix circuit to easily read where to put which phase
                a, b = MZI_layer_coord(m, modes, j, k)
                Itf.circuit[a, b] = theta1
                Itf.circuit[a, b+1] = theta2
                
                if a not in Itf.mzi_list.keys():
                    Itf.mzi_list[a] = []
                Itf.mzi_list[a].append((b, b+1))
                
                # update coordinates
                x -= 1
                y -= 1
                
        else:   # even numbered diagonals (j = 2,4,6...)
            x: int = m-j
            y: int = 0
            s: int = x-1 #place of the external phase shift P
            
            P_mat, phi = external_ps(m, s, V_mat[x, y], V_mat[x-1, y])
            V_mat = np.matmul(P_mat, V_mat)
            Itf.output_phases = np.append(Itf.output_phases,Externalphaseshifter(s, phi))
            
            k: int
            for k in range(1, j+1):
                modes: list[int] = [x-1, x]     # initial mode-pairs
                delta = custom_arctan(-V_mat[x-1, y], V_mat[x, y])
                
                if k == j:
                    summ: float = 0
                else:
                    summ: float = np.angle(V_mat[x+1, y+1]) - np.angle(V_mat[x-1, y+1]*np.cos(delta) - V_mat[x, y+1]*np.sin(delta))
                
                M_mat = np.eye(m, dtype=np.complex128)
                M_mat[modes[0], modes[0]] =  np.sin(delta) * np.exp(1j*summ)
                M_mat[modes[1], modes[0]] =  np.cos(delta) * np.exp(1j*summ)
                M_mat[modes[0], modes[1]] =  np.cos(delta) * np.exp(1j*summ)
                M_mat[modes[1], modes[1]] = -np.sin(delta) * np.exp(1j*summ)
                V_mat = np.matmul(M_mat,V_mat)
                
                #calculate the actual phases theta1 and theta2
                theta1, theta2 = internal_phases(delta, summ)
                
                #save the angles in the matrix circuit to easily read where to put which phase
                a, b = MZI_layer_coord(m, modes, j, k)
                Itf.circuit[a, b] = theta1
                Itf.circuit[a, b+1] = theta2
                
                if a not in Itf.mzi_list.keys():
                    Itf.mzi_list[a] = []
                Itf.mzi_list[a].append((b, b+1))
                
                # update coordinates
                x += 1
                y += 1

    # Step 3 of the algorithm that implements the external phases in the middle of the circuit
    # in addiditon, we now want to move the external phases 'Q' to the residual positions.
    # this phase shift on the first mode ensures that we implement the actual unitary 'U',
    # without a global phase, however, for now we don't use it
    # V = np.dot(V,external_ps(m, 0, 0, V[0,0]))
    
    if m%2 != 0: # if the number of modes is odd
        for j in range(2, m+1):
            a: int = m - j
            P_mat, xi = external_ps(m, j-1, V_mat[0, 0], V_mat[j-1, j-1])
            V_mat = np.dot(V_mat, P_mat)
   
            if j%2 != 0: #if j is odd        
                for b in range(j-1, m):
                    Itf.circuit[a, b] = Itf.circuit[a, b] + xi
                for b in range(j, m):
                    Itf.circuit[a-1, b] = Itf.circuit[a-1, b] - xi
                
            else: #if j is even           
                for b in range(j):
                    Itf.circuit[a-1, b] = Itf.circuit[a-1, b] + xi
                for b in range(j-1):
                    Itf.circuit[a, b] = Itf.circuit[a, b] - xi
            
    else: #for even m
        for j in range(2, m+1):
            a: int = m - j
            P_mat, xi = external_ps(m, j-1, V_mat[0, 0], V_mat[j-1, j-1])
            V_mat = np.dot(V_mat, P_mat)
   
            if j%2 != 0: #if j is odd        
                for b in range(j):
                    Itf.circuit[a, b] = Itf.circuit[a, b] + xi
                for b in range(j-1):
                    Itf.circuit[a+1, b] = Itf.circuit[a+1, b] - xi
                    
            else: #if j is even           
                for b in range(j-1, m):
                    Itf.circuit[a+1, b] = Itf.circuit[a+1, b] + xi
                for b in range(j, m):
                    Itf.circuit[a, b] = Itf.circuit[a, b] - xi

    Itf.circuit = Itf.circuit.T
    Itf.mzi_list = dict(sorted(Itf.mzi_list.items()))
    
    return Itf

# -----------------------------

def random_unitary(n: int) -> np.ndarray[complex]:
    """
    Returns a random NxN unitary matrix.

    This code is inspired by the Matlab code written by Toby Cubitt:
    http://www.dr-qubit.org/matlab/randU.m

    ---

    Args:
        n (int): dimension of the NxN unitary matrix to generate

    ---

    Returns
    ---
        complex-valued 2D numpy array representing the interferometer
    """
    X_mat: np.ndarray[complex] = np.zeros([n, n], dtype=np.complex128)
    i: int
    j: int
    for i in range(n):
        for j in range(n):
            X_mat[i, j] = (np.random.normal() + 1j * np.random.normal()) / np.sqrt(2)

    q: np.ndarray[complex]
    r: np.ndarray[complex]
    q, r = np.linalg.qr(X_mat)
    r = np.diag(np.divide(np.diag(r), abs(np.diag(r))))
    U_mat = np.matmul(q, r)

    return U_mat

def external_ps(n: int, mode: int, v1: complex, v2: complex) -> tuple[np.ndarray[complex], float]:
    """
    Builds the representation of the external phaseshifter
    for the given diagonal.
    Purpose of this operator is to match the given elements'
    phases.
    
    ---
    
    Args:
        n (int): dimension of unitary matrix / number of modes
        mode (int): selected "diagonal" / index of mode
        V1 (complex): element of auxillary matrix
        V2 (complex): subsequent element of **v1**
    
    ---
    
    - for **even** diagonals ( j=2,4... ):
    >> ``v1 = V_mat[x, y]`` ,  ``v2 = V_mat[x-1, y]``
    
    - for **odd** diagonals ( j=1,3... ):
    >> ``v1 = V_mat[x, y]`` ,  ``v2 = V_mat[x, y+1]``
    
    ---
    
    Returns
    ---
        Diagonal matrix with phase-shift ``exp(i*phi)``
        at position ``[j, j]`` together with the angle ``phi`` itself.
    """
    phi = np.angle(v1) - np.angle(v2)
    P_mat = np.eye(n, dtype=np.complex128)
    P_mat[mode, mode] = np.exp(1j * phi)
    
    return P_mat, phi
    
def custom_arctan(v1: float | complex, v2: float | complex) -> float:
    """
    Computes the ``arctan`` of ``-v1/v2``.
    
    If ``v2=0`` returns ``pi/2``.
    
    ---
    
    Args:
        v1 (float/complex): subsequent element of **v2**
        v2 (float/complex): element of auxillary matrix
        
    ---
        
    - for **even** diagonals ( j=2,4... ):
    >> ``v1 = -V_mat[x-1, y]`` ,  ``v2 = V_mat[x, y]``
    
    - for **odd** diagonals ( j=1,3... ):
    >> ``v1 = V_mat[x, y+1]`` ,  ``v2 = V_mat[x, y]``
    
    ---
    
    Returns
    ---
        The angle of the arctan in radians.
    """
    if v2 != 0:
        ang: float | complex = np.arctan(-v1/v2)
        if isinstance(ang, np.complex128):
            return np.real(ang)
        else:
            return ang
    else:
        return np.pi/2
    
def internal_phases(delta: float, summ: float) -> tuple[float, float]:
    """
    Computes the internal phases ``theta1`` and ``theta2`` such that
    they reproduce ``delta`` and ``summ`` according to
    
    - ``summ = (theta1 + theta2)/2``
    - ``delta = (theta1 - theta2)/2``
    
    In addition we add a phase of ``-pi/2`` to each angle in order to
    ensure the proper description of the beamsplitter (BS).
    
    ---
    
    Args:
        delta (float): the angle computed to null elements
        summ (float): the phase computed to equalize phases
    
    ---
    
    
    Returns
    ---
        The internal phases ``theta1`` and ``theta2``.
    """
    theta1 = delta + summ - np.pi/2
    theta2 = summ - delta - np.pi/2
    
    return theta1, theta2

def MZI_layer_coord(m: int, modes: list[int], j: int, k: int) -> tuple[int, int]:
    """
    Gives new coordinates that define in which vertical layer (column)
    of the circuit the corresponding symmetric Mach-Zehnder Interferometer
    (sMZI) is placed.
    
    ---
    
    Args:
        m (int): the number of modes
        modes (list): the modes the sMZI is acting on
        j (int): the coordinate giving the diagonal of the circuit
        k (int): the counter within the diagonal ``j``
    
    Returns
    ---
        New coordinates ``a`` and ``b`` (starting at 0) that will help to place the MZI in the circuit
        and manage the shifting of the external PS:s, 'Q', from the middle of the circuit to the residual positions.
    
    a : the coordinate of the sMZI layer (column)
    
    b : the affected mode
    """
    
    if j%2 == 0 :
        a = m - k
    else:
        a = k - 1
    b = modes[0]
        
    return a, b

# Functions for re-composition:

def matprod(mat_list: list[np.ndarray] | np.ndarray[np.ndarray]) -> np.ndarray:
    """
    Computes the product of the matrices that are in the input list.
    
    ---
    
    Args:
        mat_list (list): a list of matrices that shall be mulitplied in the given order
    
    Returns
    ---
        A matrix that is the product of all matrices in ``mat_list``.
    """
    
    result: np.ndarray = np.eye(mat_list[0].shape[0])
    for i in mat_list:
        result = np.dot(result, i)
    return result

def BS(n: int, m1: int, m2: int) -> np.ndarray[complex]:
    """
    This function creates a matrix that descibes a 50/50 beam splitter
    (BS) acting on modes given modes.
    
    ---
    
    Args:
        n (int): the number of modes
        m1 (int): mode 1 the BS acts on (upper)
        m1 (int): mode 2 the BS acts on (lower)
        
    ---
    
    Returns
    ---
        An ``nxn`` matrix with embedded 50/50 BS on modes ``m1`` and ``m2``.
    """
    
    BS_mat: np.ndarray[complex] = np.eye(n, dtype=complex)
    BS_mat[m1, m1] = 1/(2**0.5)
    BS_mat[m1, m1+1] = 1j/(2**0.5)
    BS_mat[m2, m2-1] = 1j/(2**0.5)
    BS_mat[m2, m2] = 1/(2**0.5)
    
    return BS_mat

# TODO: main function to serve the user
# - Add reading from proper input
def function_to_rule_them_all(unitary_matrix: np.ndarray):
    """
    
    ---
    
    Args:
        unitary_matrix (array): Unitary matrix to decompose and represent
            in terms of an integrated photonic circuit
        
    ---
    
    """
    
    I = square_decomposition(unitary_matrix)
    I.calculate_transformation(unitary_matrix)
    I.draw(save_fig=False)
    
    AnsysCirc = AnsysCircuit(unitary_matrix.shape[0], I.circuit_prep())
    AnsysCirc.build_Circuit()
    # print(script)