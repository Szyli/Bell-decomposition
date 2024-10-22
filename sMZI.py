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
    def __init__(self, mode, phi):
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
    

    def __init__(self, mode1, mode2, theta1, theta2):
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

    def __init__(self, num_of_modes):
        self.num_of_modes = num_of_modes
        self.input_phases = np.array([], dtype=Externalphaseshifter)
        self.output_phases = np.array([], dtype=Externalphaseshifter)
        self.circuit = np.zeros(shape=(num_of_modes,num_of_modes), dtype=complex)
        self.mzi_list = {}
        

    def calculate_transformation(self) -> np.ndarray:
        """
        Calculate unitary matrix describing the transformation implemented by the interferometer.
        Used to verify the implementation.
    
        Returns:
            complex-valued 2D numpy array representing the interferometer
        """
        N = self.num_of_modes
        U = np.eye(N, dtype=np.complex_)

        for BS in self.BS_list:
            T = np.eye(N, dtype=np.complex_)
            T[BS.mode1 - 1, BS.mode1 - 1] = np.exp(1j * BS.phi) * np.cos(BS.theta)
            T[BS.mode1 - 1, BS.mode2 - 1] = -np.sin(BS.theta)
            T[BS.mode2 - 1, BS.mode1 - 1] = np.exp(1j * BS.phi) * np.sin(BS.theta)
            T[BS.mode2 - 1, BS.mode2 - 1] = np.cos(BS.theta)
            U = np.matmul(T,U)

        while np.size(self.output_phases) < N:  # Autofill for users who don't want to bother with output phases
            self.output_phases.append(0)

        D = np.diag(np.exp([1j * phase for phase in self.output_phases]))
        U = np.matmul(D,U)
        return U

    def circuit_prep(self):
        """
        Function to finalize the circuit into one structure called
        ``total_list``.
        """
        total_list = np.array([self.input_phases])
        for col in self.mzi_list:
            rows = np.arange(0,self.num_of_modes,1) # to follow which row was MZI and which an external ps
            used_idx = ()
            for row in self.mzi_list[col]:
                used_idx += row
                total_list = np.append(total_list, Beamsplitter(row[0],row[1],self.circuit[row[0],col],self.circuit[row[1],col]))
            rows = np.delete(rows,used_idx)
            # if the list is not empty
            if rows.size:
                for elem in rows:
                    total_list = np.append(total_list, Externalphaseshifter(elem, self.circuit[elem,col]))
        total_list = np.append(total_list, self.output_phases)
        
        return total_list

    def curve(self, x1: float, x2: float,
            mode: bool, flip: bool = False,
            res: int = 10, scale: float = 0.5,
            c: float = 0):
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
        x = np.linspace(0, np.pi/2, res)
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
    
    def draw_curved_connectors(self, ord, x_start, x_end, corr):
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
                        internal: bool = False):
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
    
    def draw(self, size = None, show_plot=True, save_fig = False):  
        """
        Function to make a drawing of the interferometer.

        Args:
            size (tuple): x and y dimension of plot
            show_plot (bool): whether to show the generated plot
        """

        # TO BE INTERCHANGED FOR THE NEW CIRCUIT IMPLEMENTATION
        
        N = self.num_of_modes
        mode_tracker = np.zeros(N)
        sc = 0.5
        
        # calling the full source material
        circuit = self.circuit_prep()

        if size == None:
            size = (N*2+4,N*2)
            plt.figure(figsize=size)
        else:
            plt.figure(figsize=size)

        # initial/starting lines
        for i in range(1,N+1,1):
            plt.plot((-1, 0), (i, i), lw=1, color="blue")

        # external and MZIs without outputs
        used = True
        
        for idx in range(len(circuit)):
            
            elem = circuit[idx]

            if isinstance(elem, Externalphaseshifter):
                used = False
                x = mode_tracker[elem.mode]
                
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
                x = np.max([mode_tracker[elem.mode1], mode_tracker[elem.mode2]])
                
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

        max_x = np.max(mode_tracker)
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
            
        


def square_decomposition(U):
    """
    Returns a rectangular mesh of beam splitters implementing matrix U.
    
    ---

    This code implements the decomposition algorithm in:
    

    Returns:
        an Interferometer instance
    """
    m = U.shape[0]    # dimension of matrix = number of modes 'm'
    I = Interferometer(num_of_modes=m)
    V = np.conjugate(U)
    
    for j in range(1, m): # odd diags: 1,3,5...
        if j%2 != 0:
            x = m-1
            y = j-1
            s = y+1 #place of the external phase shift P 
            # find external phaseshift that matches given elements' phases
            P, phi = external_ps(m, s, V[x,y], V[x,y+1])

            V = np.matmul(V,P)
            
            I.input_phases = np.append(I.input_phases,Externalphaseshifter(s, phi))
            
            for k in range(1, j+1):
                modes = [y, y+1]    # initial mode-pairs
                
                delta = custom_arctan(V[x,y+1], V[x,y])
                
                if k == j:  # redundant choice
                    summ = 0
                else:   # derivation shows
                    summ = np.angle(V[x-1,y-1]) - np.angle(V[x-1,y]*np.sin(delta) + V[x-1,y+1]*np.cos(delta))
                
                M = np.eye(m, dtype=np.complex_)
                M[modes[0],modes[0]] =  np.sin(delta) * np.exp(1j*summ)
                M[modes[1],modes[0]] =  np.cos(delta) * np.exp(1j*summ)
                M[modes[0],modes[1]] =  np.cos(delta) * np.exp(1j*summ)
                M[modes[1],modes[1]] = -np.sin(delta) * np.exp(1j*summ)
                V = np.matmul(V,M)
                
                theta1, theta2 = internal_phases(delta,summ)
                
                #save the angles in the matrix circuit to easily read where to put which phase
                a, b = MZI_layer_coord(m,modes,j,k)
                I.circuit[a,b] = theta1
                I.circuit[a,b+1] = theta2
                
                if a not in I.mzi_list.keys():
                    I.mzi_list[a] = []
                I.mzi_list[a].append((b,b+1))
                
                # update coordinates
                x -= 1
                y -= 1
                
        else:   # even numbered diagonals (j = 2,4,6...)
            x = m-j
            y = 0
            s = x-1 #place of the external phase shift P
            
            P, phi = external_ps(m, s, V[x,y], V[x-1,y])

            V = np.matmul(P,V)
            
            I.output_phases = np.append(I.output_phases,Externalphaseshifter(s, phi))
          
            for k in range(1, j+1):
                modes = [x-1, x]     # initial mode-pairs
                
                delta = custom_arctan(-V[x-1,y], V[x,y])
                if k == j:
                    summ = 0
                else:
                    summ = np.angle(V[x+1,y+1]) - np.angle(V[x-1,y+1]*np.cos(delta) - V[x,y+1]*np.sin(delta))
                
                M = np.eye(m, dtype=np.complex_)
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
    
    if m%2 != 0: #if the number of modes is odd
        for j in range(2,m+1):
            a = m - j
            P , xi = external_ps(m, j-1, V[0,0], V[j-1,j-1])
            V = np.dot(V,P)
   
            if j%2 != 0: #if j is odd        
                for b in range(j-1,m):
                    I.circuit[a,b] = I.circuit[a,b] + xi
                for b in range(j,m):
                    I.circuit[a-1,b] = I.circuit[a-1,b] - xi
                
            else: #if j is even           
                for b in range(j):
                    I.circuit[a-1,b] = I.circuit[a-1,b] + xi
                for b in range(j-1):
                    I.circuit[a,b] = I.circuit[a,b] - xi
            
    else: #for even m
        for j in range(2,m+1):
            a = m - j
            P , xi = external_ps(m, j-1, V[0,0], V[j-1,j-1])
            V = np.dot(V,P)
   
            if j%2 != 0: #if j is odd        
                for b in range(j):
                    I.circuit[a,b] = I.circuit[a,b] + xi
                for b in range(j-1):
                    I.circuit[a+1,b] = I.circuit[a+1,b] - xi
                    
            else: #if j is even           
                for b in range(j-1,m):
                    I.circuit[a+1,b] = I.circuit[a+1,b] + xi
                for b in range(j,m):
                    I.circuit[a,b] = I.circuit[a,b] - xi

    I.circuit = I.circuit.T
    I.mzi_list = dict(sorted(I.mzi_list.items()))
    
    return I


def random_unitary(N: int) -> np.ndarray:
    """
    Returns a random NxN unitary matrix

    This code is inspired by Matlab code written by Toby Cubitt:
    http://www.dr-qubit.org/matlab/randU.m

    Args:
        N (int): dimension of the NxN unitary matrix to generate

    Returns:
        complex-valued 2D numpy array representing the interferometer
    """
    X = np.zeros([N, N], dtype=np.complex_)
    for ii in range(N):
        for jj in range(N):
            X[ii, jj] = (np.random.normal() + 1j * np.random.normal()) / np.sqrt(2)

    q, r = np.linalg.qr(X)
    r = np.diag(np.divide(np.diag(r), abs(np.diag(r))))
    U = np.matmul(q, r)

    return U

def external_ps(N: int, mode: int, V1: np.complex_, V2: np.complex_) -> tuple[np.ndarray, float]:
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
    P = np.eye(N, dtype=np.complex_)
    P[mode,mode] = np.exp(1j * phi)
    
    return P, phi

def phase_match(V1: np.complex_, V2: np.complex_):
    """
    Phase matcher function. Used to find ``summ``.
    
    ---
    
    Parameters
    ------
    V1 : element of auxillary matrix at ``[x+1,y+1]`` or ``[x-1,y-1]``
    V2 : element of auxillary matrix at ``[x,y+1]`` or ``[x-1,y]``
    
    Returns
    ------
    The angle for ``summ``
    """
    phi1 = np.angle(V1)
    phi2 = np.angle(V2)
    
    print('angle1: ', phi1, '\nangle2: ', phi2)
    
def custom_arctan(V1, V2):
    """
    Computes the ``arctan`` of ``-V1/V2``.
    
    ---
    If ``V2=0`` returns ``pi/2``.
    
    """
    if V2 != 0:
        return np.arctan(-V1/V2)
    else:
        return np.pi/2
    
def internal_phases(delta,summ):
    """
    Computes the internal phases theta1 and theta2 
    
    ------
    
    Parameters
    ------
    delta: the angle computed to null the elements
    summ: the phase computed to equalize the phases
    
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

def MZI_layer_coord(m,modes,j,k):
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

