"""
    This module aims to implement the decomposition of unitary matrices
    using Symmetric Mach-Zehnder Interferometers based on Bell's further
    developed approach of Clement's scheme.
    
    William R. Clements: An Optimal Design for Universal Multiport Interferometers
    B.A. Bell: Further Compactifying Linear Optical Unitaries
    
    The following code was heavily influenced by Clement's interferometer from
    github: https://github.com/clementsw/interferometer
    
    @author: Csaba Szilard Racz //Szyli
"""

import numpy as np
import matplotlib.pyplot as plt

class Externalphaseshifter:
    """
    This class defines the external phaseshifters (PS) applied
    before the first beamsplitters in each diagonals.
    May be used for the final layer of external PS:s
    positioned in the "middle" of the circuit.
    
    ---
    
    Args:
        mode (int): the index of the mode on which the PS is located
        phi (float): the angle of the PS
    """
    def __init__(self, mode: int, phi: float):
        self.mode = mode
        self.phi = phi

    def __repr__(self):
        repr = "\n External PS on mode {} with angle \t phi: {:.2f}".format(
            self.mode,
            self.phi,
            )
        return repr

class Beamsplitter:
    """
    This class defines a tunable beam splitter cell
    i.e. a sMZI with internal phase shifters.
    
    ---
    
    The matrix describing the mode transformation is:
    
    >>``e^{i*summ}*sin(delta)``      ``e^{i*summ}cos(delta)``
    
    >>``e^{i*summ}*cos(delta)``     ``-e^{i*summ}sin(delta)``
    
    where
    - ``summ = (theta1 + theta2)/2``
    - ``delta = (theta1 - theta2)/2``
    
    with ``theta1`` and ``theta2`` being the internal phase shifts.
    
    ---

    Args:
        mode1 (int): the index of the first mode (the first mode is mode 1)
        mode2 (int): the index of the second mode
        theta1 (float): angle of internal phase shift on mode1
        theta2 (float): angle of internal phase shift on mode2
    """
    

    def __init__(self, mode1: int, mode2: int, theta1: float, theta2: float):
        self.mode1 = mode1
        self.mode2 = mode2
        self.theta1 = theta1
        self.theta2 = theta2

    def __repr__(self):
        repr = "\n MZI between modes {} and {}: \n The angle on {}: {:.2f} \n The angle on {}: {:.2f}".format(
            self.mode1,
            self.mode2,
            self.mode1,
            self.theta1,
            self.mode2,
            self.theta2,
            )
        return repr

class Interferometer:
    """
    This class defines an interferometer.
    
    ---

    An interferometer contains parameters of the whole circuit; such as
    > - The number of modes (``num_of_modes``)
    > - The input phases (``input_phases``)
    > - The Output phases (``output_phases``)
    > - The circuit (``circuit``)
    > - The indices of internal phasesifters (``mzi_list``)
    
    ---
    
    Args:
        num_of_modes (int): total number of modes
        input_phases (array): the external PS P applied to the input
        output_phases (array): the external PS P applied to the output
        circuit (matrix): each entry will define the phase that needs to be put at the corresponding layer a and mode b
        mzi_list (dictionary): each key of it is the column number of the circuit matrix;
            the values are lists of tuples containing the mode indices of theta1 and theta2
            > column: [(row1,row2)] -> position of the internal phaseshifts in the circuit matrix
        
    """

    def __init__(self, num_of_modes: int):
        self.num_of_modes = num_of_modes
        self.input_phases: np.ndarray[Externalphaseshifter] = np.array([], dtype=Externalphaseshifter)
        self.output_phases: np.ndarray[Externalphaseshifter] = np.array([], dtype=Externalphaseshifter)
        self.circuit: np.ndarray[float] = np.zeros(shape=(num_of_modes,num_of_modes), dtype=float)
        self.mzi_list: dict = {}
        
    def PS(self, phases: list[float] | np.ndarray[float]) -> np.ndarray[complex]:
        """
        Gives a matrix that describes phase shifts on the corresponding modes
        
        ------
        
        Parameters
        ------
        phases : a list of phases that should be applied to the corresponding modes 
        
        Returns
        ------
        A diagonal matrix with the phase shifts on its diagonal
        """
        if isinstance(phases[0], Externalphaseshifter):
            exps: list[complex] = [np.exp(1j*p.phi) for p in phases]
        else:
            exps: list[complex] = [np.exp(1j*p) for p in phases]
        P: np.ndarray[complex] = np.diag(exps)
        return P

    def calculate_transformation(self, U: np.ndarray) -> None:
        """
        Tests if the calculated phases give back the target unitary
        
        ------
        
        Parameters
        ------
        U : target unitary matrix
        
        Returns
        ------
        Computes the matrix that is implemented using the phases given in In, Out and C and compares it
        to the target unitary U. If they are the same the decomposition works, 
        if they differ, the difference is printed for furhter error analysis.
        Note: this function can also be used to get the unitary, that is implemented by a given circuit,
        if it has the respective geometry
        """
        
        # preparing input/output external PS:s as "edge-vectors"
        In: list[float] = [0]*self.num_of_modes
        Out: list[float] = [0]*self.num_of_modes
        ps: Externalphaseshifter
        for ps in self.input_phases:
            In[ps.mode] = ps.phi
        for ps in self.output_phases:
            Out[ps.mode] = ps.phi
        
        #there are two different layers of BS, one starts at mode 0, the other at mode 1
        Beam_sp1: list[np.ndarray[complex]] = []
        i: int = 0
        while i < self.num_of_modes - 1:
            Beam_sp1.append(BS(self.num_of_modes, i, i+1))
            i = i + 2
        layerA: np.ndarray[complex] = matprod(Beam_sp1)
        
        Beam_sp2: list[np.ndarray[complex]] = []    
        i: int = 1
        while i < self.num_of_modes - 1:
            Beam_sp2.append(BS(self.num_of_modes,i,i+1))
            i = i + 2
        if not Beam_sp2:    # if it is empty
            layerB: np.ndarray[int] = np.eye(self.num_of_modes)
        else:
            layerB: np.ndarray[complex] = matprod(Beam_sp2)
        
        #we alternate between the two different BS layers and sandwich the phase shifts
        #since for an even number of modes we have a slightly different geometry, one has to treat the cases differently 
        U_list: list[np.ndarray[complex]] = []
        
        if self.num_of_modes%2 != 0:
            for k in range(self.num_of_modes):
                if k%2 == 0:
                    U_list.append(matprod([layerA,self.PS(self.circuit[:,self.num_of_modes-1-k]),layerA]))
                else:
                    U_list.append(matprod([layerB,self.PS(self.circuit[:,self.num_of_modes-1-k]),layerB]))
        else:
            for k in range(self.num_of_modes):
                if k%2 == 0:
                    U_list.append(matprod([layerB,self.PS(self.circuit[:,self.num_of_modes-1-k]),layerB]))
                else:
                    U_list.append(matprod([layerA,self.PS(self.circuit[:,self.num_of_modes-1-k]),layerA])) 
        
        #multiply all BS layers and PS in the inner circuit (all described in the output matrix C)        
        U_mat: np.ndarray[complex] = matprod(U_list)
        
        #add the external PS to the input and output 
        U_imp: np.ndarray[complex] = matprod([self.PS(Out),U_mat,self.PS(In)])
        
        #compare with the desired unitary U, they should be equal up to a global phase, so we renormalize the test matrix
        test: np.ndarray[complex] = np.dot(U,U_imp.T.conj())
        test = test/test[0][0]
        
        if np.allclose(test,np.eye(self.num_of_modes, dtype=complex)):
            print("Re-composition done.\tDecomposition was successful!")
        else:
            print("Matrices do not match!\nDifference:")
            print(test-np.eye(self.num_of_modes, dtype=complex))

    def circuit_prep(self) -> np.ndarray[Beamsplitter | Externalphaseshifter]:
        """
        Function to finalize the circuit into one structure called
        ``total_list``.
        """
        total_list: np.ndarray[Beamsplitter | Externalphaseshifter] = np.array([self.input_phases])
        
        col: int
        row: tuple[int, int]
        for col in self.mzi_list:
            rows: np.ndarray[int] = np.arange(0,self.num_of_modes,1) # to follow which row was MZI and which an external ps
            used_idx: tuple[int, int] = ()
            for row in self.mzi_list[col]:
                used_idx += row
                total_list = np.append(total_list, Beamsplitter(row[0],row[1], self.circuit[row[0],col], self.circuit[row[1],col]))
            rows: np.ndarray = np.delete(rows,used_idx) # remaining are external phaseshifts
            if rows.size:   # if the list is not empty
                for elem in rows:
                    total_list = np.append(total_list, Externalphaseshifter(elem, self.circuit[elem,col]))
        total_list = np.append(total_list, self.output_phases)
        
        return total_list

    def curve(self, x1: float, x2: float, mode: bool, flip: bool = False,
            res: int = 10, scale: float = 0.5, c: float = 0) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """
        
        Args
        ------
        mode : True gives `cos`; False gives `sin`
        flip : determines if curve needs to be mirrored around the x-axes
        res : resolution
        scale : relative height of curve
            to set return values corresponding to the environment
        c : shift in `y` direction
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
        
        return (x,y)
    
    def draw_curved_connectors(self, ord: bool, x_start: float, x_end: float, corr: float) -> None:
        """
        Function that puts the curved waveguids down (it's just a drawing).
        
        Args
        ------
        ord : which pair of curved wgs to draw
            ``if True``: first, ``if False``: second
        x_start : position of first point in `x`
        x_end : position of last point in `x`
        corr : correction in `y` direction
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
        
    def draw_external_ps(self, x: int, N: int, ext_ps: Externalphaseshifter,
                        used: bool, cl: str = "red",
                        internal: bool = False) -> bool:
        """
        Function to handle the external phasshifters on the drawing.
        
        Args
        ------
        x (int): current position in x
        N (int): total number of modes
        ext_ps (Externalphaseshifter): the external PS (class type)
        used (bool): whether the selected external PS has been used or not
        even (bool): whether we're on even or odd diagonal
        cl (str): color of text
        """
        
        if not used:
            used = True
            if not internal:
                # EXTERNAL PS on the edges of the circuit
                plt.plot((x+0.15, x+0.15), (N-ext_ps.mode-0.3, N-ext_ps.mode+0.2), lw=1, color="blue")
                circle = plt.Circle((x+0.15, N-ext_ps.mode), 0.1, fill=False)
                plt.gca().add_patch(circle)
                phase = "{:2f}".format(ext_ps.phi)
                if ext_ps.phi > 0:
                    plt.text(x+0.2, N-ext_ps.mode-0.3, phase[0:3], color=cl, fontsize=7)
                else:
                    plt.text(x+0.2, N-ext_ps.mode-0.3, phase[0:4], color=cl, fontsize=7)
            else:
                # EXTERNAL PS inside the circuit
                plt.plot((x+0.85, x+0.85), (N-ext_ps.mode-0.3, N-ext_ps.mode+0.2), lw=1, color="blue")
                circle = plt.Circle((x+0.85, N-ext_ps.mode), 0.1, fill=False)
                plt.gca().add_patch(circle)
                phase = "{:2f}".format(ext_ps.phi)
                if ext_ps.phi > 0:
                    plt.text(x+0.9, N-ext_ps.mode-0.3, phase[0:3], color=cl, fontsize=7)
                else:
                    plt.text(x+0.9, N-ext_ps.mode-0.3, phase[0:4], color=cl, fontsize=7)
    
        return used
    
    def draw(self, size: tuple[float, float] | None = None, show_plot: bool = True, save_fig: bool = False):  
        """
        Function to make a drawing of the interferometer.

        Args:
            size (tuple): x and y dimension of plot
            show_plot (bool): whether to show the generated plot
        """

        # TO BE INTERCHANGED FOR THE NEW CIRCUIT IMPLEMENTATION
        
        N: int = self.num_of_modes
        mode_tracker: np.ndarray[int] = np.zeros(N)
        sc: float = 0.5
        
        # calling the full source material
        circuit: np.ndarray[Beamsplitter | Externalphaseshifter] = self.circuit_prep()

        if size == None:
            size = (N*2+4, N*2)
            plt.figure(figsize = size)
        else:
            plt.figure(figsize = size)

        # initial/starting lines
        i: int
        for i in range(1, N+1, 1):
            plt.plot((-1, 0), (i, i), lw=1, color="blue")

        # external and MZIs without outputs
        used: bool = True
        
        for idx in range(len(circuit)):
            
            elem: Beamsplitter | Externalphaseshifter = circuit[idx]

            if isinstance(elem, Externalphaseshifter):
                used = False
                x: float = mode_tracker[elem.mode]
                
                # initial PS:s
                if (idx < len(self.input_phases)) or (idx > (len(circuit)-len(self.output_phases)-1)):
                    used = self.draw_external_ps(x, N, elem, used)
                    plt.plot((x+0.25, x+0.85), (N-elem.mode, N-elem.mode), lw=1, color="blue")
                    # update
                    mode_tracker[elem.mode] = x+0.6
                    # for alignment in the odd N cases 
                    if elem.mode == N-2:
                        mode_tracker[-1] = x+0.6
                        plt.plot((x, x+0.6), (1, 1), lw=1, color="blue")
                else:
                    used = self.draw_external_ps(x, N, elem, used, internal=True)
                    plt.plot((x, x+0.6), (N-elem.mode, N-elem.mode), lw=1, color="blue")
                    # update
                    mode_tracker[elem.mode] = x+0.6

            else:
                x: float = np.max([mode_tracker[elem.mode1], mode_tracker[elem.mode2]])
                
                if used:
                    # no external PS to be placed; placeholders
                    plt.plot((x, x+0.3), (N - elem.mode1, N - elem.mode1), lw=1, color="blue")
                    plt.plot((x, x+0.3), (N - elem.mode2, N - elem.mode2), lw=1, color="blue")
                
                # EXTERNAL PS
                # used = self.draw_external_ps(x, N, ext_ps, used, even)
                
                # creating curved lines
                self.draw_curved_connectors(ord=True, x_start=x+0.3, x_end=x+0.5,corr=N-elem.mode1-sc)
                self.draw_curved_connectors(ord=False, x_start=x+0.5, x_end=x+0.7,corr=N-elem.mode1-sc)
                
                # INTERNAL PS: theta1
                plt.plot((x+0.85, x+0.85), (N+0.3-(elem.mode2 + elem.mode1)/2., N+0.7-(elem.mode2 + elem.mode1)/2.), lw=1, color="blue")
                circle = plt.Circle((x+0.85, N+0.5-(elem.mode2 + elem.mode1)/2.), 0.1, fill=False)
                plt.gca().add_patch(circle)
                inter_phase = "{:2f}".format(elem.theta1)
                plt.text(x+0.9, N+0.7-(elem.mode2 + elem.mode1)/2, inter_phase[0:3], color="green", fontsize=7)
                
                # INTERNAL PS: theta2
                plt.plot((x+0.85, x+0.85), (N-0.3-(elem.mode2 + elem.mode1)/2., N-0.7-(elem.mode2 + elem.mode1)/2.), lw=1, color="blue")
                circle = plt.Circle((x+0.85, N-0.5-(elem.mode2 + elem.mode1)/2.), 0.1, fill=False)
                plt.gca().add_patch(circle)
                inter_phase = "{:2f}".format(elem.theta2)
                plt.text(x+0.9, N-0.7-(elem.mode2 + elem.mode1)/2, inter_phase[0:3], color="green", fontsize=7)
                
                # creating curved lines
                self.draw_curved_connectors(ord=True, x_start=x+1, x_end=x+1.2,corr=N-elem.mode1-sc)
                self.draw_curved_connectors(ord=False, x_start=x+1.2, x_end=x+1.4,corr=N-elem.mode1-sc)

                # connecting neighbouring cells
                plt.plot((x+1.4, x+2), (N-elem.mode1, N-elem.mode1), lw=1, color="blue")
                plt.plot((x+1.4, x+2), (N-elem.mode2, N-elem.mode2), lw=1, color="blue")
                
                # update
                if x > mode_tracker[elem.mode1]:
                    plt.plot((mode_tracker[elem.mode1], x), (N-elem.mode1, N-elem.mode1), lw=1, color="blue")
                if x > mode_tracker[elem.mode2]:
                    plt.plot((mode_tracker[elem.mode2], x), (N-elem.mode2, N-elem.mode2), lw=1, color="blue")
                mode_tracker[elem.mode1] = x+2
                mode_tracker[elem.mode2] = x+2

        max_x: float = np.max(mode_tracker)
        ii: int
        for ii in range(N):
            plt.plot((mode_tracker[ii], max_x+1), (N-ii, N-ii), lw=1, color="blue")

        plt.text(max_x/3, N+1, r"red: external phase shift ($\phi$)", color="red", fontsize=10)
        plt.text(max_x/3, N+1-0.25, r"green: internal phase shifts ($\theta_1$,$\theta_2$)", color="green", fontsize=10)
        # plt.text(max_x/3, N+1-0.5, r"brown: external phase shift ($\theta$)", color="brown", fontsize=10)
        plt.text(-1, N+0.2, "Light in", fontsize=10)
        plt.text(max_x+0.5, N+0.2, "Light out", fontsize=10)
        plt.gca().axes.set_ylim([0.2, N+1.2])
        plt.axis("off")
        
        if save_fig:
            plt.savefig("{}_{}-circuit.png".format(N,N))
        if show_plot:
            plt.show()

class AnsysCircuit:
    """
    This class is responsible for building the Integrated Optical
    Circuit using Ansys Interconnect based on the Unitary Decomposition.
    
    Args:
        num_of_modes (int): total number of modes/dimension of Unitary matrix
        list_of_elements (np.ndarray[Beamsplitter | Externalphaseshifter]): a list of all the elements in the circuit
    """
    
    def __init__(self, num_of_modes: int, list_of_elements: np.ndarray[Beamsplitter | Externalphaseshifter]):
        self.N = num_of_modes
        self.elements = list_of_elements
    
    def check_lib(self) -> list[str]:
        """
        Function to check if custom compounds are present or not.

        ---

        Notes:
        ---
        >>> if all elements exist: return an empty list
        >>> if some exists: return a list of the nonexisting elements (to build later)
        >>> if the folder is empty or it does not exist: return a list of all the elements to be built
        """
        # include ANSYS
        import sys, os
        sys.path.append("D:\\lumerical\\api\\python")
        sys.path.append(os.path.abspath(""))
        from lumapi import INTERCONNECT

        ic = INTERCONNECT(hide=True)
        path: str = ic.customlibrary()
        path += "/sMZI kits"
        list_of_compund_element_names: list[str] = ["sMZI.ice", "PS.ice", "WGD.ice"]
        if os.path.exists(path) and os.listdir(path) != 0:
            # print("I exists and I'm not empty!")
            if list_of_compund_element_names == os.listdir(path):
                # print("All compound elements exist!")
                return []
            else:
                missing_elements: list[str] = [elem for elem in list_of_compund_element_names if elem not in os.listdir(path)]
                return missing_elements
        else:
            return list_of_compund_element_names
        

    def build_compounds(self) -> str:
        """
        Function to build a compund elements.

        Returns:
            Ansys script to evaluate.
        """

        compound_name: str = 'sMZI'
        input_name: str = 'input'
        output_name: str = 'output'
        phase_shift_element: str = 'Optical Phase Shift'
        coupler_element: str = 'Waveguide Coupler'
        x_spacing: int = 200
        y_spacing: int = 80


        script: str = "groupscope('::Root Element');"
        script += "addelement('"+str(coupler_element)+"');\n"\
                + "set('name','Coupler_1');\n"\
                + "set('x position', "+str(0)+");\n"\
                + "set('y position', "+str(y_spacing)+");\n"\
                + "addelement('"+str(phase_shift_element)+"');\n"\
                + "set('name','Theta_1');\n"\
                + "set('x position', "+str(x_spacing)+");\n"\
                + "set('y position', "+str(0)+");\n"\
                + "addelement('"+str(phase_shift_element)+"');\n"\
                + "set('name','Theta_2');\n"\
                + "set('x position', "+str(x_spacing)+");\n"\
                + "set('y position', "+str(2*y_spacing)+");\n"\
                + "connect('Coupler_1','port 3','Theta_1','port 1');"\
                + "connect('Coupler_1','port 4','Theta_2','port 1');"\
                + "addelement('"+str(coupler_element)+"');\n"\
                + "set('name','Coupler_2');\n"\
                + "set('x position', "+str(2*x_spacing)+");\n"\
                + "set('y position', "+str(y_spacing)+");\n"\
                + "connect('Coupler_2','port 1','Theta_1','port 2');"\
                + "connect('Coupler_2','port 2','Theta_2','port 2');"\
                + "selectall;\n"\
                + "createcompound();\n"\
                + "set('name',"+"'"+str(compound_name)+"');\n"\
                + "set('prefix','"+str(compound_name)+"');\n"\
                + "set('description','Symmetric Mach-Zehnder Interferometer');\n"\
                + "addport('"+str(compound_name)+"','"+input_name+"_1'"+", 'bidirectional', 'Optical Signal', 'left',0.25);\n"\
                + "addport('"+str(compound_name)+"','"+input_name+"_2'"+", 'bidirectional', 'Optical Signal', 'left',0.75);\n"\
                + "addport('"+str(compound_name)+"','"+output_name+"_1'"+", 'bidirectional', 'Optical Signal', 'right',0.25);\n"\
                + "addport('"+str(compound_name)+"','"+output_name+"_2'"+", 'bidirectional', 'Optical Signal', 'right',0.75);\n"\
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
                + "connect('Port_2','port','Coupler_1','port 2');"\
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
                + "addtolibrary('sMZI kits');"

        return script

    def build_Ext_PS(self, ID: int, phi: float, x: int, y: int, prior_elem: str = None, port: int = None) -> tuple[str, str]:
        """
        Function to build an external PhaseShifter with
        defined phase (in rad) and connected to the prior
        element in the circuit with the defined port.
        
        Args:
            ID (int): serial number of this element
            phi (float): value of phase shift
            prior_elem (str): name of the element to connect this with
            port (int): serial number of ``prior_elem``:s port to connect this element's port with
            x (int): ``x`` position of element
            y (int): ``y`` position of element

        Returns:
            Ansys script to evaluate.
        """
        element_name: str = "Ext_PS"
        element_ID: str = "ps_" + str(ID)
        compound_name: str = "PS_1"

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
            # NOTE: in these cases (mainly input and output)
            # we need to also carefully redefine the waveguides
            # in these compund elements
            port: str = "port " + str(port)
            script += "connect('"+prior_elem+"','"+port+"','"+element_ID+"','port 1');"
        
        return script, element_ID

    def build_sMZI(self, ID: int, theta_1: float, theta_2: float, x: int, y: int,
                prior_elem_1: str = None, prior_elem_2: str = None,
                port_1: int = None, port_2: int = None) -> tuple[str, str]:
        """
        Function to build a Symmetric Mach-Zehnder Interferometer (sMZI) with
        defined phaseshifts (in rad) and connected to the prior
        elements in the circuit with the defined ports.

        Args:
            ID (int): serial number of this element
            theta_1 (float): value of phase shift on first(/upper) arm
            theta_2 (float): value of phase shift on second(/lower) arm
            prior_elem_1 (str): name of the element to connect the upper first(/upper) arm with
            prior_elem_2 (str): name of the element to connect the upper second(/lower) arm with
            port_1 (int): serial number of ``prior_elem_1``:s port to connect this element's
                first(/upper) port with
            port_2 (int): serial number of ``prior_elem_2``:s port to connect this element's
                second(/lower) port with
            x (int): ``x`` position of element
            y (int): ``y`` position of element

        Returns:
            Ansys script to evaluate.
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
            # NOTE: in these cases (mainly input and output)
            # we need to also carefully redefine the waveguides
            # in these compund elements
            port: str = "port " + str(port_1)
            script += "connect('"+prior_elem_1+"','"+port+"','"+element_ID+"','port 1');"
        if port_2:
            port = "port " + str(port_2)
            script += "connect('"+prior_elem_2+"','"+port+"','"+element_ID+"','port 2');"
        
        return script, element_ID

    def build_WGD(self, ID: int, x: int, y: int, prior_elem: str = None, port: int = None) -> tuple[str, str]:
        """
        Function to build Straight Waveguides.

        Args:
            ID (int): serial number of this element
            prior_elem (str): name of the element to connect this with
            port (int): serial number of ``prior_elem``:s port to connect this element's port with
            x (int): ``x`` position of element
            y (int): ``y`` position of element

        Returns:
            Ansys script to evaluate.
        """
        # might be unnecessary
        element_name: str = "WGD"
        element_ID: str = "wgd" + str(ID)
        
        script: str = "addelement('"+element_name+"');\n"\
                    + "set('name','"+element_ID+"');\n"\
                    + "set('x position', "+str(x)+");\n"\
                    + "set('y position', "+str(y)+");\n"
        
        if port:
            # if port is not None
            # NOTE: port = 0 would also skip this statement but
            # we name ports by counting from 1
            # NOTE: in these cases (mainly input and output)
            # we need to also carefully redefine the waveguides
            # in these compund elements
            port: str = "port " + str(port)
            script += "connect('"+prior_elem+"','"+port+"','"+element_ID+"','port 1');"
        
        return script, element_ID

    def build_Circuit(self):
        
        # include ANSYS
        import sys, os
        sys.path.append("D:\\lumerical\\api\\python")
        sys.path.append(os.path.abspath(""))
        from lumapi import INTERCONNECT

        elements = self.check_lib()

        if len(elements) < 3:
            print("Build elements first:")
            return elements
        
        # (name of element, port of element on its right, id of column)
        circuit_map: dict[tuple[None | int, None | int, None | int]] = dict.fromkeys(np.arange(0, self.N, 1),(None, None, 0))
        eval_string: str = "groupscope('::Root Element');"
        ps_id: int = 0
        smzi_id: int = 0
        x_spacing: int = 200
        y_spacing: int = 160
        
        elem: Beamsplitter | Externalphaseshifter
        for elem in self.elements:
            if isinstance(elem, Externalphaseshifter):
                if circuit_map[elem.mode]:  # if the first column of the circuit is set already
                    script, name = self.build_Ext_PS(ps_id, elem.phi,
                                                    circuit_map[elem.mode][2]*x_spacing, elem.mode*y_spacing,
                                                    circuit_map[elem.mode][0], circuit_map[elem.mode][1])
                    # update dictionary
                    circuit_map[elem.mode] = (name, 2, circuit_map[elem.mode][2] + 1)
                else:
                    # update dictionary
                    script, name = self.build_Ext_PS(ps_id, elem.phi, 0, elem.mode*y_spacing)
                    circuit_map[elem.mode] = (name, 2, 1)
                # extend evaluation string
                eval_string += script
                ps_id += 1
            elif isinstance(elem, Beamsplitter):
                if circuit_map[elem.mode1] or circuit_map[elem.mode2]:  # if the first column of the circuit  is set already
                    if circuit_map[elem.mode1][2] > circuit_map[elem.mode2][2]:
                        x = circuit_map[elem.mode1][2]
                    else:
                        x = circuit_map[elem.mode2][2]
                    script, name =  self.build_sMZI(smzi_id, elem.theta1, elem.theta2,
                                            x*x_spacing, (elem.mode1 + elem.mode2)*0.5*y_spacing,
                                            circuit_map[elem.mode1][0], circuit_map[elem.mode2][0],
                                            circuit_map[elem.mode1][1], circuit_map[elem.mode2][1])
                    # update dictionary
                    circuit_map[elem.mode1] = (name, 3, x + 1)
                    circuit_map[elem.mode2] = (name, 4, x + 1)
                else:
                    script, name =  self.build_sMZI(smzi_id, elem.theta1, elem.theta2,
                                                    0, (elem.mode1 + elem.mode2)*0.5*y_spacing)
                    circuit_map[elem.mode1] = (name, 3, 1)
                    circuit_map[elem.mode2] = (name, 4, 1)
                # extend evaluation string
                eval_string += script
                smzi_id += 1
            else:
                print(f"Error! Element type is incorrect at index position: {ps_id+smzi_id}")
            
        
        # evaluate complete circuit
        ic = INTERCONNECT(hide=True)
        ic.eval(eval_string)
        ic.save(f"{self.N}x{self.N}_test.icp")

        # return eval_string



def square_decomposition(U: np.ndarray[float | complex]) -> Interferometer:
    """
    Returns a rectangular mesh of beam splitters implementing matrix U.
    
    ---

    This code implements the decomposition algorithm in:
    

    Returns:
        an Interferometer instance
    """
    m: int = U.shape[0]    # dimension of matrix = number of modes 'm'
    I: Interferometer = Interferometer(num_of_modes = m)
    V: np.ndarray[float | complex] = np.conjugate(U)
    
    j: int
    for j in range(1, m): # odd diags: 1,3,5...
        if j%2 != 0:
            x: int = m-1
            y: int = j-1
            s: int = y+1 #place of the external phase shift P 
            # find external phaseshift that matches given elements' phases
            P, phi = external_ps(m, s, V[x,y], V[x,y+1])
            V = np.matmul(V,P)
            I.input_phases = np.append(I.input_phases, Externalphaseshifter(s, phi))
            
            k: int
            for k in range(1, j+1):
                modes: list[int] = [y, y+1]    # initial mode-pairs
                delta = custom_arctan(V[x,y+1], V[x,y])
                
                if k == j:  # redundant choice
                    summ: float = 0
                else:   # derivation shows
                    summ: float = np.angle(V[x-1,y-1]) - np.angle(V[x-1,y]*np.sin(delta) + V[x-1,y+1]*np.cos(delta))
                
                M = np.eye(m, dtype=np.complex128)
                M[modes[0],modes[0]] =  np.sin(delta) * np.exp(1j*summ)
                M[modes[1],modes[0]] =  np.cos(delta) * np.exp(1j*summ)
                M[modes[0],modes[1]] =  np.cos(delta) * np.exp(1j*summ)
                M[modes[1],modes[1]] = -np.sin(delta) * np.exp(1j*summ)
                V = np.matmul(V,M)
                
                theta1, theta2 = internal_phases(delta, summ)
                
                #save the angles in the matrix circuit to easily read where to put which phase
                a, b = MZI_layer_coord(m, modes, j, k)
                I.circuit[a,b] = theta1
                I.circuit[a,b+1] = theta2
                
                if a not in I.mzi_list.keys():
                    I.mzi_list[a] = []
                I.mzi_list[a].append((b, b+1))
                
                # update coordinates
                x -= 1
                y -= 1
                
        else:   # even numbered diagonals (j = 2,4,6...)
            x: int = m-j
            y: int = 0
            s: int = x-1 #place of the external phase shift P
            
            P, phi = external_ps(m, s, V[x,y], V[x-1,y])
            V = np.matmul(P, V)
            I.output_phases = np.append(I.output_phases,Externalphaseshifter(s, phi))
            
            k: int
            for k in range(1, j+1):
                modes: list[int] = [x-1, x]     # initial mode-pairs
                delta = custom_arctan(-V[x-1,y], V[x,y])
                
                if k == j:
                    summ: float = 0
                else:
                    summ: float = np.angle(V[x+1,y+1]) - np.angle(V[x-1,y+1]*np.cos(delta) - V[x,y+1]*np.sin(delta))
                
                M = np.eye(m, dtype=np.complex128)
                M[modes[0],modes[0]] =  np.sin(delta) * np.exp(1j*summ)
                M[modes[1],modes[0]] =  np.cos(delta) * np.exp(1j*summ)
                M[modes[0],modes[1]] =  np.cos(delta) * np.exp(1j*summ)
                M[modes[1],modes[1]] = -np.sin(delta) * np.exp(1j*summ)
                V = np.matmul(M,V)
                
                #calculate the actual phases theta1 and theta2
                theta1, theta2 = internal_phases(delta,summ)
                
                #save the angles in the matrix circuit to easily read where to put which phase
                a, b = MZI_layer_coord(m,modes,j,k)
                I.circuit[a,b] = theta1
                I.circuit[a,b+1] = theta2
                
                if a not in I.mzi_list.keys():
                    I.mzi_list[a] = []
                I.mzi_list[a].append((b,b+1))
                
                # update coordinates
                x += 1
                y += 1


    #add step 3 of the algorithm that implements the external phases in the middle of the cicuit
    #in addiditon we now want to move the external phases Q to the residual positions
    
    #this phase shift on the first mode ensures that we implement the actual unitary U, without global phase, 
    #however, for now we don't use it
    #V = np.dot(V,external_ps(m, 0, 0, V[0,0]))
    
    if m%2 != 0: # if the number of modes is odd
        for j in range(2, m+1):
            a: int = m - j
            P, xi = external_ps(m, j-1, V[0, 0], V[j-1, j-1])
            V = np.dot(V, P)
   
            if j%2 != 0: #if j is odd        
                for b in range(j-1,m):
                    I.circuit[a, b] = I.circuit[a, b] + xi
                for b in range(j,m):
                    I.circuit[a-1, b] = I.circuit[a-1, b] - xi
                
            else: #if j is even           
                for b in range(j):
                    I.circuit[a-1, b] = I.circuit[a-1, b] + xi
                for b in range(j-1):
                    I.circuit[a, b] = I.circuit[a, b] - xi
            
    else: #for even m
        for j in range(2, m+1):
            a: int = m - j
            P, xi = external_ps(m, j-1, V[0, 0], V[j-1, j-1])
            V = np.dot(V, P)
   
            if j%2 != 0: #if j is odd        
                for b in range(j):
                    I.circuit[a, b] = I.circuit[a, b] + xi
                for b in range(j-1):
                    I.circuit[a+1, b] = I.circuit[a+1, b] - xi
                    
            else: #if j is even           
                for b in range(j-1, m):
                    I.circuit[a+1, b] = I.circuit[a+1, b] + xi
                for b in range(j, m):
                    I.circuit[a, b] = I.circuit[a, b] - xi

    I.circuit = I.circuit.T
    I.mzi_list = dict(sorted(I.mzi_list.items()))
    
    return I


def random_unitary(N: int) -> np.ndarray[complex]:
    """
    Returns a random NxN unitary matrix

    This code is inspired by Matlab code written by Toby Cubitt:
    http://www.dr-qubit.org/matlab/randU.m

    Args:
        N (int): dimension of the NxN unitary matrix to generate

    Returns:
        complex-valued 2D numpy array representing the interferometer
    """
    X: np.ndarray[complex] = np.zeros([N, N], dtype=np.complex128)
    ii: int
    jj: int
    for ii in range(N):
        for jj in range(N):
            X[ii, jj] = (np.random.normal() + 1j * np.random.normal()) / np.sqrt(2)

    q: np.ndarray[complex]
    r: np.ndarray[complex]
    q, r = np.linalg.qr(X)
    r = np.diag(np.divide(np.diag(r), abs(np.diag(r))))
    U = np.matmul(q, r)

    return U

def external_ps(N: int, mode: int, V1: complex, V2: complex) -> tuple[np.ndarray[complex], float]:
    """
    Builds the external phase-shifter for the given diagonal.
    Purpose of this operator is to match the given elements'
    phases.
    
    Parameters
    ------
    N : dimension of unitary matrix / number of modes
    mode : selected "diagonal" / index of mode
    V1 : element of auxillary matrix
    V2 : subsequent element of **V1**
    
    ---
    
    - for **even** diagonals ( j=2,4... ):
    >> ``V1 = V[x,y]`` ,  ``V2 = V[x-1,y]``
    
    - for **odd** diagonals ( j=1,3... ):
    >> ``V1 = V[x,y]`` ,  ``V2 = V[x,y+1]``
    
    Returns
    ------
    Diagonal matrix with phase-shift ``exp(i*phi)``
    at position ``[j,j]`` and the angle phi itself.
    """
    phi = np.angle(V1) - np.angle(V2)
    P = np.eye(N, dtype=np.complex128)
    P[mode, mode] = np.exp(1j * phi)
    
    return P, phi
    
def custom_arctan(V1: float | complex, V2: float | complex) -> float:
    """
    Computes the ``arctan`` of ``-V1/V2``.
    
    ---
    If ``V2=0`` returns ``pi/2``.
    
    """
    if V2 != 0:
        ang = np.arctan(-V1/V2)
        if isinstance(ang, np.complex128):
            return np.real(ang)
        else:
            return ang
    else:
        return np.pi/2
    
def internal_phases(delta: float, summ:float) -> tuple[float, float]:
    """
    Computes the internal phases theta1 and theta2 
    
    ------
    
    Parameters
    ------
    delta : the angle computed to null the elements
    summ : the phase computed to equalize the phases
    
    Returns
    ------
    The interal phases 'theta1' and 'theta2', that reproduce 'delta' and 'summ' according to
    summ = (theta1+theta2)/2
    detla = (theta1-theta2)/2
    In addition we add a phase -pi/2 to each angle to ensure the proper description of the BS
    """
    theta1 = delta + summ - np.pi/2
    theta2 = summ - delta - np.pi/2
    
    return theta1, theta2

def MZI_layer_coord(m: int, modes: list, j: int, k: int) -> tuple[int, int]:
    """
    Gives new coordinates that define in which vertical layer of the circuit 
    the corresponding MZI is placed
    
    ------
    
    Parameters
    ------
    m : the number of modes
    modes : the modes the MZI is acting on
    j : the coordinate giving the diagonal of the circuit
    k : the counter within the diagonal j
    
    Returns
    ------
    New coordinates a and b (starting at 0) that will help to place the MZI in the circiut 
    and manage the shifting of the external PS Q from the middle of the circuit to the residual positions
    
    a : the coordinate of the MZI layer
    
    b : the affected mode
    """
    
    if j%2 == 0 :
        a = m - k
    else:
        a = k - 1
    b = modes[0]
        
    return a, b

# Functions for re-composition:

def matprod(mat_list: list | np.ndarray) -> np.ndarray:
    """
    Computes the product of the matrices in the list
    
    ------
    
    Parameters
    ------
    mat_list : a list of matrices that shall be mulitplied in the given order
    
    Returns
    ------
    A matrix that is the product of all matrices in mat_list
    """
    
    result: np.ndarray = np.eye(mat_list[0].shape[0])
    for i in mat_list:
        result = np.dot(result, i)
    return result

def BS(m: int, m1: int, m2: int) -> np.ndarray[complex]:
    """
    Gives a matrix that descibes a 50/50 beam splitter acting on modes m1 and m1 embedded in a mxm matrix
    
    ------
    
    Parameters
    ------
    m : the number of modes
    m1 : mode 1 the BS acts on
    m1 : mode 2 the BS acts on
    Returns
    ------
    An mxm matrix with embedded 50/50 BS on modes m1 and m2
    """
    B: np.ndarray[complex] = np.eye(m, dtype=complex)
    B[m1,m1] = 1/(2**0.5)
    B[m1,m1+1] = 1j/(2**0.5)
    B[m2,m2-1] = 1j/(2**0.5)
    B[m2,m2] = 1/(2**0.5)
    
    return B

# TODO: main function to serve the user
# read from input
def function_to_rule_them_all(unitary_matrix: np.ndarray):
    
    I = square_decomposition(unitary_matrix)
    I.calculate_transformation(unitary_matrix)
    I.draw(save_fig=False)
    
    AnsysCirc = AnsysCircuit(unitary_matrix.shape[0], I.circuit_prep())
    script = AnsysCirc.build_Circuit()
    print(script)