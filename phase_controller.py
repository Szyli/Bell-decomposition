import numpy as np
from scipy.optimize import curve_fit

class PhaseControl:
    """
    Class to prepare phase-voltage relation for the circuit control.
    
    ---
    
    Attributes:
        path (str): the path of the dataset describing the relation between
            heating power and phase shift
            
    Methods
    ---
    
    ``data_reader()``:
        Reads the data file; expected to be txt of two columns
    ``fit()``:
        Fits a linear function on the given data and returns its params together with the fitting's errors
    """
    def data_reader(self, path: str) -> np.ndarray[np.ndarray, np.ndarray]:
        """
        This function reads the txt file at the defined path
        into a numpy array. It is expected to have two columns
        for the power and the corresponding phase shift value.

        ---

        Args:
            path (str): the path of the saved data (format: `.txt`)

        Returns:
            Numpy array containing the power and corresponding phase shift.
        """
        data_set = np.loadtxt(path)
        return data_set
        
    def linear(self, x: float |np.ndarray, a: float) -> float | np.ndarray:
        """
        This function returns the value of defined linear function by
        parameters at `x` where in this contaxt it's the heating power
        and the parameters are the following physical values.
        
        ---
        
        Args:
            x (float | array): the argument of the function; The electric power in Watts
            a (float): the steepness of the linear function
            b (float): displacement in `y` of the linear function
        Returns:
            The solution of the linear function at `x`;
            The phase shift value at given power.
        """
        return a*x

    def fit(self, data: np.ndarray) -> tuple[list, list]:
        """
        This function executes a linear curve fit on the simulation
        data of phase shift versus heating power.

        ---

        Args:
            data (array): Numpy array containing the power and corresponding phase shift.
        """
        popt, pcov = curve_fit(self.linear, data[:,0], data[:,1])
        perr = np.sqrt(np.diag(pcov))

        return popt, perr
    
    def __init__(self, path: str):
        self.params, self.errors = self.fit(self.data_reader(path))

# -----------SUPPORT FUNCTION-----------

def currcalc(v: float | np.ndarray, r: float, curr_lim: float) -> float | np.ndarray:
    """
    This function computes the current present at given voltage
    and resistivity.

    ---

    Args:
        v (float | array): the voltage in [`V`]
        r (float): the resistance in [`ohm`]
        curr_lim (float): the maximum current supported in [`mA`]
    Returns:
        The current in the wire given the parameters in [`mA`].
    """
    curr = (v/r) * 1e-3
    if isinstance(curr, float) and curr > curr_lim:
        print("The necessary current oversteps the given limit!\t{:.4f} [mA]".format(curr))
    if isinstance(curr, np.ndarray) and (curr > curr_lim).any():
        print("The necessary currents overstep the given limit!")
    return curr

def voltcalc(p: float |np.ndarray, r: float,
            v_lim: float) -> float | np.ndarray:
    """
    This function computes the voltage for given power
    and resistivity.

    ---

    Args:
        p (float | array): power in [`W`]
        r (float): resistance in [`ohm`]
        v_lim (float): voltage limit in [`V`]

    Returns:
        The voltage in [`V`].
    """
    v = np.sqrt(p*r)
    if isinstance(v, float) and v > v_lim:
        print("The necessary voltage oversteps the given limit!\t{:.4f} [V]".format(v))
    if isinstance(v, np.ndarray) and (v > v_lim).any():
        print("The necessary currents overstep the given limit!")
    return v

def powcalc(phase: np.ndarray | float, a:float, b:float=0)->np.ndarray |float:
    """
    Function to compute power from given phase.
    It's simply the rearranged form of the fitted linear function.

    ---

    Args:
        phase (array | float): desired phase shift to achieve
        a (float): steepness of linear relation between power and phase shift
        b (float): displacement of linear relation between power and phase shift

    Returns:
        The power in [`W`] necessary to achieve the given phase shift.
    """
    return (phase-b)/a

def rescalc(rho: float, l: float, a: float) -> float:
    """
    This function computes the resistance of the heater wire.

    ---

    Args:
        rho (float): the resistivity of the wire in [`ohm * m`]
        l (float): the length of the wire in [`um`]
        a (float): the cross section of the wire in [`um^2`]

    Returns:
        The resistance of the wire in [`ohm`].
    """
    r = rho * (l/a) * 1e6
    return r

# -----------MAIN FUNCTION-----------

def phase_to_voltage(path: str, ps: float | np.ndarray,
                    wire_resistivity: float,
                    wire_length: float,
                    wire_cross_section: float,
                    voltage_limit: float = 10,
                    current_limit: float = 20) -> float | np.ndarray:
    """
    This function returns the appropriate voltage for the given
    phase shift(s). For that it first computes the necessary
    power in watts which is simply the rearranged form of the fitted
    linear function.
    
    ---
    
    Args:
        path (str): the path of the dataset describing the relation between
            power and phase shift (from simulation)
        ps (float | array): the desired phase shift(s) in [`rad`]
        wire_resistivity (float): the resistivity (rho) of the wire in [`ohm * m`]
        wire_length (float): the wire's length in [`um`]
        wire_cross_section (float): the wire's cross section area in [`um^2`]
        voltage_limit (float): the highest voltage in [`volts`] supported
        current_limit (float): the highest current in [`mA`] supported
    Returns:
        The voltage corresponding to the given phase shift value
        based on the simulation data.
    """
    # prepare settings
    sett = PhaseControl(path=path)
    
    if(any(sett.errors > 1e-2)):
        print("The error on the fitted function parameter:\n" \
                "a = {:.4f}".format(*sett.errors))
    
    resistance = rescalc(wire_resistivity, wire_length, wire_cross_section)

    power = powcalc(ps, *sett.params)
    voltage = voltcalc(power, resistance, voltage_limit)
    current = currcalc(voltage, resistance, current_limit)

    # if voltage==0 or current==0:
    #     print("The voltage or current exceeds the given limits!")
    # else:
    # print("Voltage:\t{:.4f} [V]\nCurrent:\t{:.4f} [mA]\nPower:\t\t{:.4f} [W]".format(
    #     voltage, current, power))
    print(power)

    return voltage

def voltage_map(v: float | np.ndarray, positions: list[tuple] | tuple):
    """
    This function merges the positions of the phase shifters from the
    sMZI.py file with the voltages converted from phases into a dictionary.
    
    ---
    
    Args:
        v (float | array): the voltages in [`V`]
        positions (list[tuple] | tuple): the positions of the phase shifters
        
    Note:
        The ordering of the positions given here must agree with the
        dictionary returned by `class Interferometer` when called
        `phase_map` from the sMZI.py file that was used for the
        `phase_to_voltage` function.
    
    Returns:
        Dictionary with the positions and corresponding voltages.
    """
    volt_map = {}
    idx=0
    for pos in positions:
        volt_map[pos] = v[idx]
        idx += 1
    return volt_map