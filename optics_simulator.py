
import numpy as np
# include ANSYS
import sys, os
# Note: This path may very for users, therefore
# Todo: Update to find path corresponding to lumerical
sys.path.append("D:\\lumerical\\api\\python")
sys.path.append(os.path.abspath(""))

from lumapi import MODE

class StraightWGD:
    """
    This class defines a straight waveguide (wgd).

    **All parameters of length to be added in microns!**

    ---

    Args:
        base_width (float): the width of the base (bottom of the waveguide)
        base_height (flaot): the height of the wgd
        base_angle (flaot): the angle between the base and the sides
        material (str): material of the waveguide
        poles (str): `x` and `y` coordinates of the starting and the ending
              point of the wgd

     ---

    Functions
    ---

    - ``builder``: returns the script for ansys MODE to evaluate the
                    defined straight waveguide.
    """
    def __init__(self, base_width: float,
                base_height: float, base_angle: float,
                material: str, poles: str):

        self.width = base_width * 1e-6
        self.height = base_height * 1e-6
        self.angle = base_angle
        self.material = material
        self.pole = poles + " * 1e-6"

    def builder(self) -> str:
        """
          Function to prepare script for ansys MODE to evaluate the
        defined straight waveguide.
        """
        script = "addwaveguide;\n"\
                +"set(\"base width\", "+str(self.width)+");\n"\
                +"set(\"base height\", "+str(self.height)+");\n"\
                +"set(\"base angle\", "+str(self.angle)+");\n"\
                +"set(\"poles\", "+self.pole+");\n"\
                +"set(\"material\", \""+self.material+"\");\n"\
                +"set(\"override mesh order from material database\",1);\n"\
                +"set(\"mesh order\",1);\n"
        return script

class BentWGD:
    """
    This class defines a bent waveguide (wgd).

    **All parameters of length to be added in microns!**

    ---

    Args:
        base_width (float): the width of the base (bottom of the waveguide)
        base_height (flaot): the height of the wgd
        base_angle (flaot): the angle between the base and the sides
        material (str): material of the waveguide
        poles (str): `x` and `y` coordinates of the starting and the ending
              point of the wgd

     ---

    Functions
    ---

    - ``builder``: returns the script for ansys MODE to evaluate the
                    defined bent waveguide.
    """

    def __init__(self, base_width: float,
                base_height: float, base_angle: float,
                material: str, poles: str):

        self.width = base_width * 1e-6
        self.height = base_height * 1e-6
        self.angle = base_angle
        self.material = material
        self.pole = poles + " * 1e-6"

    def builder(self) -> str:

        script = "addwaveguide;\n"\
                +"set(\"base width\", "+str(self.width)+");\n"\
                +"set(\"base height\", "+str(self.height)+");\n"\
                +"set(\"base angle\", "+str(self.angle)+");\n"\
                +"set(\"poles\", "+self.pole+");\n"\
                +"set(\"material\", \""+self.material+"\");\n"\
                +"set(\"override mesh order from material database\",1);\n"\
                +"set(\"mesh order\",1);\n"
        return script


def coupler(parent: str, mode1: str, mode2: str, z_pos: float,
            base_height: float, base_width: float, base_angle: float,
            material: str, y_span: float, gap: float,
            Lc: float, Ls: float, Lw: float) -> str:
    """
    This function makes the script for directional couplers in Ansys MODE.

    **All parameters of length to be added in microns!**

    ---

    Args:
        parent (str): the name of the structure group in which to
                        move the coupler object
        mode1 (str): name of the upper mode in the coupler
        mode2 (str): name of the lower mode in the coupler
        z_pos (float): position of the coupler object in the `z` direction
        base_height (flaot): the height of the wgd
        base_width (float): the width of the base (bottom of the waveguide)
        base_angle (flaot): the angle between the base and the sides
        material (str): material of the waveguide
        y_span (float): center-to-center distance between the two input ports
        gap (float): gap between two coupling sections
        Lc (float): length of the central straight waveguide sections for coupling
        Ls (float): x span of the four individual s-bend sections
        Lw (float): length of the four individual straight waveguide sections
            at the input and output ports
    """

    z_pos = z_pos*1e-6

    script = "addstructuregroup;\n"\
            +"set('name', '"+parent+"');\n"\
    # mode 1
    script += "addstructuregroup;\n"\
            +"set('name', '"+mode1+"');\n"\
            +"addtogroup('"+parent+"');\n"\
            + StraightWGD(base_width, base_height,
                        base_angle, material,
                        "[0,"+str(y_span/2)+";"\
                        +str(Lw)+","+str(y_span/2)+"]").builder()\
            +"addtogroup('"+parent+"::"+mode1+"');\n"\
            + BentWGD(base_width, base_height,
                    base_angle, material,
                    "["+str(Lw)+","+str(y_span/2)+";"\
                    +str(Lw+Ls/2)+","+str(y_span/2)+";"\
                    +str(Lw+Ls/2)+","+str((gap+base_width)/2)+";"\
                    +str(Lw+Ls)+","+str((gap+base_width)/2)+"]").builder()\
            +"addtogroup('"+parent+"::"+mode1+"');\n"\
            + StraightWGD(base_width, base_height,
                        base_angle, material,
                        "["+str(Lw+Ls)+","+str((gap+base_width)/2)+";"\
                        +str(Lw+Ls+Lc)+","+str((gap+base_width)/2)+"]").builder()\
            +"addtogroup('"+parent+"::"+mode1+"');\n"\
            + BentWGD(base_width, base_height,
                    base_angle, material,
                    "["+str(Lw+Ls+Lc)+","+str((gap+base_width)/2)+";"\
                    +str(Lw+3*Ls/2+Lc)+","+str((gap+base_width)/2)+";"\
                    +str(Lw+3*Ls/2+Lc)+","+str(y_span/2)+";"\
                    +str(Lw+2*Ls+Lc)+","+str(y_span/2)+"]").builder()\
            +"addtogroup('"+parent+"::"+mode1+"');\n"\
            + StraightWGD(base_width, base_height,
                        base_angle, material,
                        "["+str(Lw+2*Ls+Lc)+","+str(y_span/2)+";"\
                        +str(2*Lw+2*Ls+Lc)+","+str(y_span/2)+"]").builder()\
            +"addtogroup('"+parent+"::"+mode1+"');\n"\
    # mode 2
    script += "addstructuregroup;\n"\
            +"set('name', '"+mode2+"');\n"\
            +"addtogroup('"+parent+"');\n"\
            + StraightWGD(base_width, base_height,
                        base_angle, material,
                        "[0,"+str(-y_span/2)+";"\
                        +str(Lw)+","+str(-y_span/2)+"]").builder()\
            +"addtogroup('"+parent+"::"+mode2+"');\n"\
            + BentWGD(base_width, base_height,
                    base_angle, material,
                    "["+str(Lw)+","+str(-y_span/2)+";"\
                    +str(Lw+Ls/2)+","+str(-y_span/2)+";"\
                    +str(Lw+Ls/2)+","+str(-(gap+base_width)/2)+";"\
                    +str(Lw+Ls)+","+str(-(gap+base_width)/2)+"]").builder()\
            +"addtogroup('"+parent+"::"+mode2+"');\n"\
            + StraightWGD(base_width, base_height,
                        base_angle, material,
                        "["+str(Lw+Ls)+","+str(-(gap+base_width)/2)+";"\
                        +str(Lw+Ls+Lc)+","+str(-(gap+base_width)/2)+"]").builder()\
            +"addtogroup('"+parent+"::"+mode2+"');\n"\
            + BentWGD(base_width, base_height,
                    base_angle, material,
                    "["+str(Lw+Ls+Lc)+","+str(-(gap+base_width)/2)+";"\
                    +str(Lw+3*Ls/2+Lc)+","+str(-(gap+base_width)/2)+";"\
                    +str(Lw+3*Ls/2+Lc)+","+str(-y_span/2)+";"\
                    +str(Lw+2*Ls+Lc)+","+str(-y_span/2)+"]").builder()\
            +"addtogroup('"+parent+"::"+mode2+"');\n"\
            + StraightWGD(base_width, base_height,
                        base_angle, material,
                        "["+str(Lw+2*Ls+Lc)+","+str(-y_span/2)+";"\
                        +str(2*Lw+2*Ls+Lc)+","+str(-y_span/2)+"]").builder()\
            +"addtogroup('"+parent+"::"+mode2+"');\n"


    script += "select('"+parent+"');\n"\
            +"set(\"z\", "+str(z_pos)+");\n"

    return script

def insulator(param: dict, dist: float, material: str) -> str:
    """
    This function prepares a script for Ansys MODE to build
    a rectangle around the waveguide of given parameters.
    
    **All parameters of length to be added in microns!**
    
    ---
    
    Args:
        param (dict): this dictionary contains parameters of the
            given object i.e. a directional coupler such as
            - `base height` (*float*): the height of the wgd
            - `y span` (*float*): center-to-center distance between the two input ports
            - `Lc` (*float*): length of the central straight waveguide sections for coupling
            - `Ls` (*float*): x span of the four individual s-bend sections
            - `Lw` (*float*): length of the four individual straight waveguide sections
                at the input and output ports
        dist (float): distance between the two layers
        material (str): the material of the insulator
        
    """

    script = "addrect;\n"\
            +"set(\"x min\",0);\n"\
            +"set(\"x max\","+str(2*param["Lw"]+2*param["Ls"]+param["Lc"])+"*1e-6);\n"\
            +"set(\"y min\","+str(-param["y span"])+"*1e-6);\n"\
            +"set(\"y max\","+str(param["y span"])+"*1e-6);\n"\
            +"set(\"z min\","+str(-2*param["base height"]-dist)+"*1e-6);\n"\
            +"set(\"z max\","+str(2*param["base height"]+dist)+"*1e-6);\n"\
            +"set(\"material\", \""+material+"\");\n"\
            +"set('name', 'insulator');\n"

    return script

def eme_simulator(y_span: float, z_span: float,
                Lc: float, Ls: float, Lw: float,
                y_pos: float, z_pos: float) -> str:
    """
    This function writes script for Ansys MODE to prepare
    an EME solver for given parameters of directional coupler.
    It includes four EME ports at the inputs and outputs.
    
    ---
    
    Args:
        y_span (float): distance in `y` direction that the simulation area covers
        z_span (float): distance in `z` direction that the simulation area covers
        Lc (float): length of the central straight waveguide sections for coupling
        Ls (float): x span of the four individual s-bend sections
        Lw (float): length of the four individual straight waveguide sections
                at the input and output ports
        y_pos (float): position of ports in `y` direction
            (it is a symmetric system therefore, give positive number)
        z_pos (float): position of ports in `z` direction
            (it is a symmetric system therefore, give positive number)
    """

    cell_spans = "["+str(Lw)+";"+str(Ls)+";"+str(Lc)+";"\
                +str(Ls)+";"+str(Lw)+"]"

    # subcells: 0 = none,  1 = CVCS
    script = "addeme;\n"
    script += "set(\"x min\", 0);\n"\
            +"set(\"y min\", "+str(-y_span)+"*1e-6);\n"\
            +"set(\"y max\", "+str(y_span)+"*1e-6);\n"\
            +"set(\"z min\", "+str(-z_span)+"*1e-6);\n"\
            +"set(\"z max\", "+str(z_span)+"*1e-6);\n"\
            +"set(\"number of cell groups\",5);\n"\
            +"set(\"group spans\","+cell_spans+"*1e-6);\n"\
            +"set(\"cells\",[1; 10; 1; 10; 1]);\n"\
            +"set(\"subcell method\",[0; 1; 0; 1; 0]);\n"

    # ports
    script += "addemeport;\n"\
            +"addemeport;\n"\
        +"select(\"EME::Ports::port_1\");\n"\
            +"set(\"port location\",\"left\");\n"\
            +"set(\"use full simulation span\",1);\n"\
            +"set(\"y\","+str(y_pos)+"*1e-6);\n"\
            +"set(\"y span\",5.5e-6);\n"\
            +"set(\"z\","+str(z_pos)+"*1e-6);\n"\
            +"set(\"z span\",7e-6);\n"\
            +"set(\"mode selection\",\"fundamental mode\");\n"\
        +"select(\"EME::Ports::port_2\");\n"\
            +"set(\"port location\",\"left\");\n"\
            +"set(\"use full simulation span\",1);\n"\
            +"set(\"y\","+str(-y_pos)+"*1e-6);\n"\
            +"set(\"y span\",5.5e-6);\n"\
            +"set(\"z\","+str(z_pos)+"*1e-6);\n"\
            +"set(\"z span\",7e-6);\n"\
            +"set(\"mode selection\",\"fundamental mode\");\n"\
        +"select(\"EME::Ports::port_3\");\n"\
            +"set(\"port location\",\"right\");\n"\
            +"set(\"use full simulation span\",1);\n"\
            +"set(\"y\","+str(y_pos)+"*1e-6);\n"\
            +"set(\"y span\",5.5e-6);\n"\
            +"set(\"z\","+str(z_pos)+"*1e-6);\n"\
            +"set(\"z span\",7e-6);\n"\
            +"set(\"mode selection\",\"fundamental mode\");\n"\
        +"select(\"EME::Ports::port_4\");\n"\
            +"set(\"port location\",\"right\");\n"\
            +"set(\"use full simulation span\",1);\n"\
            +"set(\"y\","+str(-y_pos)+"*1e-6);\n"\
            +"set(\"y span\",5.5e-6);\n"\
            +"set(\"z\","+str(z_pos)+"*1e-6);\n"\
            +"set(\"z span\",7e-6);\n"\
            +"set(\"mode selection\",\"fundamental mode\");\n"

    script += "addemeprofile;\n"\
            +"set(\"x min\", 0);\n"\
            +"set(\"y min\", "+str(-y_span)+"*1e-6);\n"\
            +"set(\"y max\", "+str(y_span)+"*1e-6);\n"\
            +"addemeindex;\n"\
            +"set(\"x min\", 0);\n"\
            +"set(\"y min\", "+str(-y_span)+"*1e-6);\n"\
            +"set(\"y max\", "+str(y_span)+"*1e-6);\n"

    return script


# main builder functions


def dir_coupler_builder(param_1: dict, param_2: dict,
                        dist: float, material: str,
                        addEMEsolver: bool= True) -> None:
    """
    Prepares script to build directional coupler in the triplex scheme
    in Ansys MODE.
    
    **All parameters of length to be added in microns!**

    ---

    Args:
        param_1 (dict): Dictionary of "lower layer" waveguide's parameters
        param_2 (dict): Dictionary of "upper layer" waveguide's parameters
        dist (float): distance between the two layers
        material (str): material of the covering insulator
        addEMEsolver (bool): whether to add or not the EME solver to the
            scope

    ---

    Note
    ---
    parameter keys:
    > - base height (*float*): the height of the wgd
    > - base width (*float*): the width of the base (bottom of the waveguide)
    > - base angle (*float*): the angle between the base and the sides
    > - material (*str*): material of the waveguide
    > - y span (*float*): center-to-center distance between the two input ports
    > - gap (*float*): gap between two coupling sections
    > - Lc (*float*): length of the central straight waveguide sections for coupling
    > - Ls (*float*): x span of the four individual s-bend sections
    > - Lw (*float*): length of the four individual straight waveguide sections
                at the input and output ports
    """
    
    parent1 = "lower layer"
    parent2 = "upper layer"
    mode1 = "mode_1"
    mode2 = "mode_2"

    dist = dist + param_1["base height"]/2 + param_2["base height"]/2

    script: str = "groupscope('::model');\n"

    script += coupler(parent2, mode1, mode2, dist, *param_2.values())
    script += coupler(parent1, mode1, mode2, 0, *param_1.values())
    script += insulator(param_2, dist, material)


    if addEMEsolver:
        script += eme_simulator(1.5*param_2["y span"]/2, 2*param_2["base height"]+dist,
                            param_2["Lc"], param_2["Ls"], param_2["Lw"], param_2["y span"]/2,
                            dist)



    mode = MODE(hide = True)
    mode.eval(script)
    mode.save(f"simulations.lms")


def straight_wgd_builder(param_1, param_2, dist, material):

    dist = dist + param_1["base height"]/2 + param_2["base height"]/2


    script: str = "groupscope('::model');\n"

    script += StraightWGD(param_1["base width"], param_1["base height"],
                        param_1["base angle"], param_1["material"],
                        "[0,"+str(param_1["y span"]/2)+";"\
                        +str(param_1["Lw"])+","+str(param_1["y span"]/2)+"]").builder()\
            +StraightWGD(param_2["base width"], param_2["base height"],
                        param_2["base angle"], param_2["material"],
                        "[0,"+str(param_2["y span"]/2)+";"\
                        +str(param_2["Lw"])+","+str(param_2["y span"]/2)+"]").builder()\
            +"set(\"z\", "+str(dist)+"*1e-6);\n"\
            +insulator(param_2, dist, material)

    mode = MODE(hide = True)
    mode.eval(script)
    mode.save(f"straight.lms")

