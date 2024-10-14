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
        repr = "\n MZI between modes {} and {}: \n Theta angle on {}: {:.2f} \n the angle on {}: {:.2f}".format(
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

    An interferometer contains an ordered list of variable beam splitters,
    represented here by ``BS_list``. For ``BS`` in ``BS_list``, ``BS[0]``
    and ``BS[1]`` correspond to the labels of the two modes being interfered (which start at 1).
    
    This transformation is parametrized by ``BS[2]`` (theta1) and by ``BS[3]`` (theta2)
    which determines the beam splitter reflectivity.
    The interferometer also contains a list of input phases in ``initial_phases``
    that is the only external phaseshifter that is not located at the output.
    Output phases are described by ``output_phases``.
    
    ---
    
    Args:
        BS_list (list): of beamsplitters; elements are of ``Beamsplitter`` class type
        initial_phases (list): of phaseshifts at the beginning of the diagonals
        output_phases (list): of phaseshifts put at the output of the circuit to complete it
    """

    def __init__(self):
        self.BS_list = []
        self.initial_phases = []
        self.output_phases = []

    def add_BS(self, BS):
        """Adds a beam splitter at the output of the current interferometer

        Args:
            BS (Beamsplitter): a Beamsplitter instance
        """
        self.BS_list.append(BS)

    def add_phase(self, mode, phase):    
        """Use this to manually add a phase shift to a selected mode at the output of the interferometer
        
        Args:
            mode (int): the mode index. The first mode is mode 1
            phase (float): the real-valued phase to add
        """
        while mode > np.size(self.output_phases):
            self.output_phases.append(0)
        self.output_phases[mode-1] = phase

    def count_modes(self) -> int:
        """
        Calculate number of modes involved in the transformation. 
        
        ---
        
        This is required for the functions ``calculate_transformation`` and ``draw``.
        
        ---

        Returns:
            the number of modes in the transformation
        """
        highest_index = max([max([BS.mode1, BS.mode2]) for BS in self.BS_list])
        return highest_index

    def calculate_transformation(self) -> np.ndarray:
        """
        Calculate unitary matrix describing the transformation implemented by the interferometer.
        Used to verify the implementation.
    
        Returns:
            complex-valued 2D numpy array representing the interferometer
        """
        N = int(self.count_modes())
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

    def draw(self, show_plot=True):  
        """Use matplotlib to make a drawing of the interferometer

        Args:
            show_plot (bool): whether to show the generated plot
        """

        import matplotlib.pyplot as plt
        plt.figure()
        N = self.count_modes()
        mode_tracker = np.zeros(N)

        for ii in range(N):
            plt.plot((-1, 0), (ii, ii), lw=1, color="blue")

        for BS in self.BS_list:
            x = np.max([mode_tracker[BS.mode1 - 1], mode_tracker[BS.mode2 - 1]])
            plt.plot((x+0.3, x+1), (N - BS.mode1, N - BS.mode2), lw=1, color="blue")
            plt.plot((x, x+0.3), (N - BS.mode1, N - BS.mode1), lw=1, color="blue")
            plt.plot((x, x+0.3), (N - BS.mode2, N - BS.mode2), lw=1, color="blue")
            plt.plot((x+0.3, x+1), (N - BS.mode2, N - BS.mode1), lw=1, color="blue")
            plt.plot((x+0.4, x+0.9), (N - (BS.mode2 + BS.mode1)/2, N - (BS.mode2 + BS.mode1)/2), lw=1, color="blue")
            reflectivity = "{:2f}".format(np.cos(BS.theta)**2)
            plt.text(x+0.9, N + 0.05 - (BS.mode2 + BS.mode1)/2, reflectivity[0:3], color="green", fontsize=7)

            plt.plot((x+0.15, x+0.15), (N+0.3-(BS.mode2 + BS.mode1)/2., N+0.7-(BS.mode2 + BS.mode1)/2.), lw=1, color="blue")
            circle = plt.Circle((x+0.15, N+0.5-(BS.mode2 + BS.mode1)/2.), 0.1, fill=False)
            plt.gca().add_patch(circle)
            phase = "{:2f}".format(BS.phi)
            if BS.phi > 0:
                plt.text(x+0.2, N+0.7-(BS.mode2 + BS.mode1)/2., phase[0:3], color="red", fontsize=7)
            else:
                plt.text(x+0.2, N+0.7-(BS.mode2 + BS.mode1)/2., phase[0:4], color="red", fontsize=7)
            if x > mode_tracker[BS.mode1-1]:
                plt.plot((mode_tracker[BS.mode1-1], x), (N-BS.mode1, N-BS.mode1), lw=1, color="blue")
            if x > mode_tracker[BS.mode2-1]:
                plt.plot((mode_tracker[BS.mode2-1], x), (N-BS.mode2, N-BS.mode2), lw=1, color="blue")
            mode_tracker[BS.mode1-1] = x+1
            mode_tracker[BS.mode2-1] = x+1

        max_x = np.max(mode_tracker)
        for ii in range(N):
            plt.plot((mode_tracker[ii], max_x+1), (N-ii-1, N-ii-1), lw=1, color="blue")
            while np.size(self.output_phases) < N:  # Autofill for users who don't want to bother with output phases
                self.output_phases.append(0)
            if self.output_phases[ii] != 0:
                plt.plot((max_x+0.5, max_x+0.5), (N-ii-1.2, N-ii-0.8), lw=1, color="blue")
                circle = plt.Circle((max_x+0.5, N-ii-1), 0.1, fill=False)
                plt.gca().add_patch(circle)
                phase = str(self.output_phases[ii])
                if BS.phi > 0:
                    plt.text(max_x+0.6, N-ii-0.8, phase[0:3], color="red", fontsize=7)
                else:
                    plt.text(max_x+0.6, N-ii-0.8, phase[0:4], color="red", fontsize=7)


        plt.text(max_x/2, -0.7, "green: BS reflectivity", color="green", fontsize=10)
        plt.text(max_x/2, -1.4, "red: phase shift", color="red", fontsize=10)
        plt.text(-1, N-0.3, "Light in", fontsize=10)
        plt.text(max_x+0.5, N-0.3, "Light out", fontsize=10)
        plt.gca().axes.set_ylim([-1.8, N+0.2])
        plt.axis("off")
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
    I = Interferometer()
    m = U.shape[0]    # dimension of matrix = number of modes 'm'
    V = np.conjugate(U.T)
    even = []

    for j in range(1, m):
        # odd diags: 1,3,5...
        if j%2 != 0: # ii%2
            x = m-1
            y = j-1
            s = y+1 #place of the external phase shift P 
            # find external phaseshift that matches given elements' phases
            P = external_ps(m, s, V[x,y], V[x,y+1])
            V = np.matmul(V,P)
            
            for k in range(1, j+1):
                modes = [y, y+1]    # initial mode-pairs
                
                delta = custom_arctan(V[x,y+1], V[x,y])
                
                if k == j:
                    # redundant choice (?)
                    summ = 0
                else:
                    # derivation shows
                    summ = np.angle(V[x-1,y-1]) - np.angle(V[x-1,y]*np.sin(delta) + V[x-1,y+1]*np.cos(delta))
                
                M = np.eye(m, dtype=np.complex_)
                M[modes[0],modes[0]] =  np.sin(delta) * np.exp(1j*summ)
                M[modes[1],modes[0]] =  np.cos(delta) * np.exp(1j*summ)
                M[modes[0],modes[1]] =  np.cos(delta) * np.exp(1j*summ)
                M[modes[1],modes[1]] = -np.sin(delta) * np.exp(1j*summ)
                V = np.matmul(V,M)
                
                theta1, theta2 = internal_phases(delta,summ)
                
                I.BS_list.append(Beamsplitter(modes[0], modes[1], theta1, theta2))
                # print("j,k: {:.2f},{:.2f}\nnulled: {:.2f}".format(j,k,V[x,y]))
                
                # update coordinates
                x -= 1
                y -= 1
                
        # even numbered diagonals (j = 2,4,6...)
        else:
            x = m-j
            y = 0
            s = x-1 #place of the external phase shift P
            
            P = external_ps(m, s, V[x,y], V[x-1,y])
            V = np.matmul(P,V)
        
            for k in range(1, j+1): # jj
                modes = [x-1, x]     # initial mode-pairs
                
                delta = custom_arctan(-V[x-1,y], V[x,y])
                if k == j:
                    summ = 0
                else:
                    # derivation shows
                    summ = np.angle(V[x+1,y+1]) - np.angle(V[x-1,y+1]*np.cos(delta) - V[x,y+1]*np.sin(delta))
                
                M = np.eye(m, dtype=np.complex_)
                M[modes[0],modes[0]] =  np.sin(delta) * np.exp(1j*summ)
                M[modes[1],modes[0]] =  np.cos(delta) * np.exp(1j*summ)
                M[modes[0],modes[1]] =  np.cos(delta) * np.exp(1j*summ)
                M[modes[1],modes[1]] = -np.sin(delta) * np.exp(1j*summ)
                V = np.matmul(M,V)
                
                #calculate the actual phases theta1 and theta2
                theta1, theta2 = internal_phases(delta,summ)
                
                even.append(Beamsplitter(modes[0], modes[1], theta1, theta2))
                
                # print("j,k: {:.2f},{:.2f}\nnulled: {:.2f}".format(j,k,V[x,y]))
                
                # update coordinates
                x += 1
                y += 1

    #add step 3 of the algorithm that implements the external phases in the middle of the cicuit
    for j in range(2,m+1):
        #xi = np.angle(V[0][0])-np.angle(V[j-1][j-1])
        V = np.dot(V,external_ps(m, j-1, V[0,0], V[j-1,j-1]))
        
    #add the even MZIs to the BS list:
    for BS in np.flip(even, 0):
        I.BS_list.append(BS)
    
    # for BS in np.flip(left_T, 0):
    #     modes = [int(BS.mode1), int(BS.mode2)]
    #     invT = np.eye(N, dtype=np.complex_)
    #     invT[modes[0]-1, modes[0]-1] = np.exp(-1j * BS.phi) * np.cos(BS.theta)
    #     invT[modes[0]-1, modes[1]-1] = np.exp(-1j * BS.phi) * np.sin(BS.theta)
    #     invT[modes[1]-1, modes[0]-1] = -np.sin(BS.theta)
    #     invT[modes[1]-1, modes[1]-1] = np.cos(BS.theta)
    #     U = np.matmul(invT, U)
    #     theta = custom_arctan(U[modes[1]-1, modes[0]-1], U[modes[1]-1, modes[1]-1])
    #     phi   =  custom_angle(U[modes[1]-1, modes[0]-1], U[modes[1]-1, modes[1]-1])
    #     invT[modes[0]-1, modes[0]-1] = np.exp(-1j * phi) * np.cos(theta)
    #     invT[modes[0]-1, modes[1]-1] = np.exp(-1j * phi) * np.sin(theta)
    #     invT[modes[1]-1, modes[0]-1] = -np.sin(theta)
    #     invT[modes[1]-1, modes[1]-1] = np.cos(theta)
    #     U = np.matmul(U, invT) 
    #     I.BS_list.append(Beamsplitter(modes[0], modes[1], theta, phi))
    # # output (external) phases
    # phases = np.diag(U)
    # I.output_phases = [np.angle(i) for i in phases]
    #return I
    return V


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

def external_ps(N: int, j: int, V1: np.complex_, V2: np.complex_) -> np.ndarray:
    """
    Builds the external phase-shifter for the given diagonal.
    Purpose of this operator is to match the given elements'
    phases.
    
    Parameters
    ------
    N : dimension of unitary matrix / number of modes
    j : selected "diagonal"
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
    at position ``[j,j]``.
    """
    phi = np.angle(V1) - np.angle(V2)
    P = np.eye(N, dtype=np.complex_)
    P[j,j] = np.exp(1j * phi)
    
    return P

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

def custom_angle(x1, x2):
    if x2 != 0:
        return np.angle(x1/x2)
    else:
        return 0
    
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
    detla = (theta1+theta2)/2
    In addition we add a phase -pi/2 to each angle to ensure the proper description of the BS
    """
    theta1 = delta+summ - np.pi/2
    theta2 = summ-delta - np.pi/2
    
    return theta1, theta2

