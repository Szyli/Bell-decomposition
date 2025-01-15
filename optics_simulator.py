
import numpy as np
# include ANSYS
import sys, os
# Note: This path may vary for users, therefore
# Todo: Update to find path corresponding to lumerical
sys.path.append("D:\\lumerical\\api\\python")
sys.path.append(os.path.abspath(""))

def mode_select(mode: str, vis: bool) -> None:
    if mode == "MODE":
        from lumapi import MODE
        solver = MODE(hide = vis)
        return solver
    if mode == "FDTD":
        from lumapi import FDTD
        solver = FDTD(hide = vis)
        return solver
    else:
        print("invalid Ansys lumapi given!")
    print("{mode} selected.")

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

    def builder(self, name: str) -> str:
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
                +"set(\"mesh order\",1);\n"\
                +"set('name','"+name+"');\n"
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

    def builder(self, name: str) -> str:

        script = "addwaveguide;\n"\
                +"set(\"base width\", "+str(self.width)+");\n"\
                +"set(\"base height\", "+str(self.height)+");\n"\
                +"set(\"base angle\", "+str(self.angle)+");\n"\
                +"set(\"poles\", "+self.pole+");\n"\
                +"set(\"material\", \""+self.material+"\");\n"\
                +"set(\"override mesh order from material database\",1);\n"\
                +"set(\"mesh order\",1);\n"\
                +"set('name','"+name+"');\n"
        return script

def single_straight(parent: str, mode: str, z_pos: float,
                    x_pos: float, y_pos: float,
                    base_height: float, base_width: float, base_angle: float,
                    material: str, Lw: float) -> str:
    """
    Function to build single waveguide of triplex scheme in Ansys.

    **All parameters of length to be added in microns!**

    ---

    Args:
        parent (str): the name of the structure group in which this one
                        will be located
        mode (str): name of this mode
        z_pos (float): position of the object in the `z` direction
        x_pos (float): starting position of object in `x` direction
        y_pos (float): starting position of object in `y` direction
        base_height (flaot): the height of the wgd
        base_width (float): the width of the base (bottom of the waveguide)
        base_angle (flaot): the angle between the base and the sides
        material (str): material of the waveguide
        Lw (float): length of the straight waveguide
    """
    
    # given mode
    script = "addstructuregroup;\n"\
            +"set('name', '"+parent+"');\n"\
            +"addtogroup('"+mode+"');\n"\
            + StraightWGD(base_width, base_height,
                        base_angle, material,
                        "["+str(x_pos)+","+str(y_pos)+";"\
                        +str(x_pos+Lw)+","+str(y_pos)+"]").builder(name="straight wg "+parent+" "+mode)\
            +"addtogroup('"+mode+"::"+parent+"');\n"
    
    # positioning
    script += "select('"+mode+"::"+parent+"');\n"\
            +"set(\"z\", "+str(z_pos)+");\n"

    return script

def double_straight():
    ...

def coupler(parent: str, mode1: str, mode2: str, z_pos: float,
            x_pos: float, y_pos1: float,
            base_height: float, base_width: float, base_angle: float,
            material: str, y_span: float, gap: float,
            Lc: float, Ls: float) -> str:
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
        x_pos (float): starting position of object in `x` direction
        y_pos1 (float): starting position of upper mode in `y` direction
        base_height (flaot): the height of the wgd
        base_width (float): the width of the base (bottom of the waveguide)
        base_angle (flaot): the angle between the base and the sides
        material (str): material of the waveguide
        y_span (float): center-to-center distance between the two input ports
        gap (float): gap between two coupling sections
        Lc (float): length of the central straight waveguide sections for coupling
        Ls (float): x span of the four individual s-bend sections
    """

    # mode 1
    script = "addstructuregroup;\n"\
            +"set('name', '"+parent+" 1');\n"\
            +"addtogroup('"+mode1+"');\n"\
            + BentWGD(base_width, base_height,
                base_angle, material,
                "["+str(x_pos)+","+str(y_pos1+y_span)+";"\
                +str(x_pos+Ls/2)+","+str(y_pos1+y_span)+";"\
                +str(x_pos+Ls/2)+","+str(y_pos1+(gap+base_width)/2)+";"\
                +str(x_pos+Ls)+","+str(y_pos1+(gap+base_width)/2)+"]").builder(name="S-bend in mode_1 "+parent)\
            +"addtogroup('"+mode1+"::"+parent+" 1');\n"\
            + StraightWGD(base_width, base_height,
                        base_angle, material,
                        "["+str(x_pos+Ls)+","+str(y_pos1+(gap+base_width)/2)+";"\
                        +str(x_pos+Ls+Lc)+","+str(y_pos1+(gap+base_width)/2)+"]").builder(name="straight coupler mode_1 "+parent)\
            +"addtogroup('"+mode1+"::"+parent+" 1');\n"\
            + BentWGD(base_width, base_height,
                    base_angle, material,
                    "["+str(x_pos+Ls+Lc)+","+str(y_pos1+(gap+base_width)/2)+";"\
                    +str(x_pos+3*Ls/2+Lc)+","+str(y_pos1+(gap+base_width)/2)+";"\
                    +str(x_pos+3*Ls/2+Lc)+","+str(y_pos1+y_span)+";"\
                    +str(x_pos+2*Ls+Lc)+","+str(y_pos1+y_span)+"]").builder(name="S-bend out mode_1 "+parent)\
            +"addtogroup('"+mode1+"::"+parent+" 1');\n"
    # mode 2
    script += "addstructuregroup;\n"\
            +"set('name', '"+parent+" 2');\n"\
            +"addtogroup('"+mode2+"');\n"\
            + BentWGD(base_width, base_height,
                    base_angle, material,
                    "["+str(x_pos)+","+str(-y_span+y_pos1)+";"\
                    +str(x_pos+Ls/2)+","+str(-y_span+y_pos1)+";"\
                    +str(x_pos+Ls/2)+","+str(-(gap+base_width)/2 +y_pos1)+";"\
                    +str(x_pos+Ls)+","+str(-(gap+base_width)/2 +y_pos1)+"]").builder(name="S-bend in mode_2 "+parent)\
            +"addtogroup('"+mode2+"::"+parent+" 2');\n"\
            + StraightWGD(base_width, base_height,
                        base_angle, material,
                        "["+str(x_pos+Ls)+","+str(-(gap+base_width)/2 +y_pos1)+";"\
                        +str(x_pos+Ls+Lc)+","+str(-(gap+base_width)/2 +y_pos1)+"]").builder(name="straight coupler mode_2 "+parent)\
            +"addtogroup('"+mode2+"::"+parent+" 2');\n"\
            + BentWGD(base_width, base_height,
                    base_angle, material,
                    "["+str(x_pos+Ls+Lc)+","+str(-(gap+base_width)/2 +y_pos1)+";"\
                    +str(x_pos+3*Ls/2+Lc)+","+str(-(gap+base_width)/2 +y_pos1)+";"\
                    +str(x_pos+3*Ls/2+Lc)+","+str(-y_span +y_pos1)+";"\
                    +str(x_pos+2*Ls+Lc)+","+str(-y_span +y_pos1)+"]").builder(name="S-bend out mode_2 "+parent)\
            +"addtogroup('"+mode2+"::"+parent+" 2');\n"

    # positioning
    script += "select('"+mode1+"::"+parent+" 1');\n"\
            +"set(\"z\", "+str(z_pos)+");\n"\
            +"select('"+mode2+"::"+parent+" 2');\n"\
            +"set(\"z\", "+str(z_pos)+");\n"

    return script

def insulator(x_span: float, y_span: float,
                material: str,
                z_min: float, z_max: float, idx: int) -> str:
    """
    This function prepares a script for Ansys to build
    a rectangle around the waveguide of given parameters.
    
    **All parameters of length to be added in microns!**
    
    ---
    
    Args:
        x_span (float): the total distance in `x` the rectangle spans
        y_span (float): the total distance in `y` the rectangle spans
        z_min (float): the bottom position of the rectangle in `z`
        z_max (float): the top position of the rectangle in `z`
        material (str): the material of the insulator
    """

    script = "addrect;\n"\
            +"set(\"x min\",0);\n"\
            +"set(\"x max\","+str(x_span)+"*1e-6);\n"\
            +"set(\"y min\","+str(-y_span/2)+"*1e-6);\n"\
            +"set(\"y max\","+str(y_span/2)+"*1e-6);\n"\
            +"set(\"z min\","+str(z_min)+"*1e-6);\n"\
            +"set(\"z max\","+str(z_max)+"*1e-6);\n"\
            +"set(\"material\", \""+material+"\");\n"\
            +"set('name', 'insulator_"+str(idx)+"');\n"

    return script


"""
# def eme_simulator(y_span: float, z_span: float,
#                 Lc: float, Ls: float, Lw: float,
#                 y_pos: float, z_pos: float) -> str:
#     ""
#     This function writes script for Ansys MODE to prepare
#     an EME solver for given parameters of directional coupler.
#     It includes four EME ports at the inputs and outputs.
    
#     ---
    
#     Args:
#         y_span (float): distance in `y` direction that the simulation area covers
#         z_span (float): distance in `z` direction that the simulation area covers
#         Lc (float): length of the central straight waveguide sections for coupling
#         Ls (float): x span of the four individual s-bend sections
#         Lw (float): length of the four individual straight waveguide sections
#                 at the input and output ports
#         y_pos (float): position of ports in `y` direction
#             (it is a symmetric system therefore, give positive number)
#         z_pos (float): position of ports in `z` direction
#             (it is a symmetric system therefore, give positive number)
#     ""

#     cell_spans = "["+str(Ls)+";"+str(Lc)+";"+str(Ls)+"]"

#     total_length = 2*Ls + Lc
#     total_width = y_span*2

#     # subcells: 0 = none,  1 = CVCS
#     script = "addeme;\n"
#     script += "set(\"x min\", 0.01*1e-6);\n"\
#             +"set(\"y min\", "+str(-y_span)+"*1e-6);\n"\
#             +"set(\"y max\", "+str(y_span)+"*1e-6);\n"\
#             +"set(\"z min\", "+str(-z_span)+"*1e-6);\n"\
#             +"set(\"z max\", "+str(z_span)+"*1e-6);\n"\
#             +"set(\"number of cell groups\",3);\n"\
#             +"set(\"group spans\","+cell_spans+"*1e-6);\n"\
#             +"set(\"cells\",[20; 10; 20]);\n"\
#             +"set(\"subcell method\",[1; 1; 1]);\n"\
#             +"set(\"display cells\",1);\n"\
#             +"set(\"mesh cells y\","+str(total_width)+");\n"\
#             +"set(\"mesh cells z\","+str(z_span*10)+");\n"\

#     # ports
#     script += "addemeport;\n"\
#             +"addemeport;\n"\
#         +"select(\"EME::Ports::port_1\");\n"\
#             +"set(\"port location\",\"left\");\n"\
#             +"set(\"use full simulation span\",0);\n"\
#             +"set(\"y\","+str(y_pos)+"*1e-6);\n"\
#             +"set(\"y span\",5.5e-6);\n"\
#             +"set(\"z\","+str(z_pos)+"*0.4e-6);\n"\
#             +"set(\"z span\",7e-6);\n"\
#             +"set(\"mode selection\",\"fundamental mode\");\n"\
#         +"select(\"EME::Ports::port_2\");\n"\
#             +"set(\"port location\",\"left\");\n"\
#             +"set(\"use full simulation span\",0);\n"\
#             +"set(\"y\","+str(-y_pos)+"*1e-6);\n"\
#             +"set(\"y span\",5.5e-6);\n"\
#             +"set(\"z\","+str(z_pos)+"*1e-6);\n"\
#             +"set(\"z span\",7e-6);\n"\
#             +"set(\"mode selection\",\"fundamental mode\");\n"\
#         +"select(\"EME::Ports::port_3\");\n"\
#             +"set(\"port location\",\"right\");\n"\
#             +"set(\"use full simulation span\",0);\n"\
#             +"set(\"y\","+str(y_pos)+"*1e-6);\n"\
#             +"set(\"y span\",5.5e-6);\n"\
#             +"set(\"z\","+str(z_pos)+"*1e-6);\n"\
#             +"set(\"z span\",7e-6);\n"\
#             +"set(\"mode selection\",\"fundamental mode\");\n"\
#         +"select(\"EME::Ports::port_4\");\n"\
#             +"set(\"port location\",\"right\");\n"\
#             +"set(\"use full simulation span\",0);\n"\
#             +"set(\"y\","+str(-y_pos)+"*1e-6);\n"\
#             +"set(\"y span\",5.5e-6);\n"\
#             +"set(\"z\","+str(z_pos)+"*1e-6);\n"\
#             +"set(\"z span\",7e-6);\n"\
#             +"set(\"mode selection\",\"fundamental mode\");\n"

#     # mesh
#     script += "addmesh;\n"\
#             +""



#     script += "addemeprofile;\n"\
#             +"set(\"monitor type\",\"2D Z-normal\");\n"\
#             +"set(\"x min\", 0);\n"\
#             +"set(\"x max\", "+str(total_length)+"*1e-6);\n"\
#             +"set(\"y min\", "+str(-y_span)+"*1e-6);\n"\
#             +"set(\"y max\", "+str(y_span)+"*1e-6);\n"\
#             +"set(\"x resolution\",2000);\n"\
#             +"addemeindex;\n"\
#             +"set(\"monitor type\",\"2D Z-normal\");\n"\
#             +"set(\"x min\", 0);\n"\
#             +"set(\"x max\", "+str(total_length)+"*1e-6);\n"\
#             +"set(\"y min\", "+str(-y_span)+"*1e-6);\n"\
#             +"set(\"y max\", "+str(y_span)+"*1e-6);\n"

#     return script
"""


# layout extraction

def layout(num_of_cells: int, materials: list,
           path: str) -> str:
    """
    This function extracts the layout of the built geometry.
    Use it after geometry is built and while it is open.
    
    ---
    
    
    Args:
        num_of_cells (int): total number of layers in the geometry
        materials (list): list of the materials of all layers in the
                        correct order
        path (str): the path of the Ansys script that manages the automated
                    layout extraction
        
    Returns:
        String of instructions for Ansys to evaluate.
    """
    
    # loads automation code
    script: str = "feval("+str(path)+" + \"/GDSexport_core.lsfx\");\n"\
                + "feval("+str(path)+" + \"/GDS_auto_export.lsf\");\n"

    script += "layer_def = cell("+str(num_of_cells)+");\n"

    for layer in range(num_of_cells):
        script += "layer = struct;\n"\
                + "layer.z =0.0e-6;\n"\
                + "layer.material =  \""+str(materials[layer])+"\";\n"\
                + "layer.layer = "+str(layer)+";\n"\
                + "layer_def{"+str(layer)+"} = layer_"+str(layer)+";\n"
                # .layer property: optional? , "accept integer, or string"

    # file name definition
    script += "gds_filename = filebasename(currentfilename) + \".gds\";\n"

    # using automated function to extract layout
    script += "GDS_AUTOEXPORT_MAIN({\"layer_def\": layer_def, \"gds_filename\": gds_filename, \"verbose\": false });\n"
    # verbose can be changed into parameter

    return script

# main builder functions

def single_straight_wgd_builder(lower_wg: dict, upper_wg: dict,
                        dist: float, insulation: dict,
                        insul_height_bot: float, insul_height_top: float,
                        name: str, version: str,
                        materials: list,
                        layout_extract: bool = False,
                        path: str = "C:/Users/Admin/Desktop/Interferometer/gds_auto_export_package/gds_auto_export_package",
                        hideAnsys: bool = True):

    # distance in `z` between two layers of Si3N4
    dist_gap = dist + lower_wg["base height"]/2 + upper_wg["base height"]/2
    dist_gap = dist_gap*1e-6
    num_of_cells = 5    # 6 with Si

    lower_parent = "lower layer"
    upper_parent = "upper layer"
    mode = "mode x"

    script: str = "groupscope('::model');\n"
    script += "addstructuregroup;\n"\
            +"set('name', '"+mode+"');\n"

    # upper wg
    script += single_straight(upper_parent,mode,dist_gap,0,0,*upper_wg.values())
    # lower wg
    script += single_straight(lower_parent,mode,0,0,0,*lower_wg.values())
    # insulations
    
    # list of insulation thicknesses
    layer_thicks = [insul_height_bot,
                    lower_wg["base height"] + dist + upper_wg["base height"],
                    insul_height_top]
    z_min = -(insul_height_bot + lower_wg["base height"]/2)
    for layer in range(num_of_cells-2):
        z_max = z_min + layer_thicks[layer]
    
        script += insulator(*insulation.values(), z_min, z_max, layer)
    
        z_min = z_max

    if layout_extract:
        script += layout(num_of_cells, materials, path)
        
    mode = mode_select(version, hideAnsys)
    
    mode.eval(script)
    mode.save(f"{name}")

def double_straight_wgd_builder(lower_wg: dict, upper_wg: dict,
                        dist: float, insulation: dict,
                        name: str, hideAnsys: bool = True):

    dist = dist + lower_wg["base height"]/2 + upper_wg["base height"]/2
    dist = dist*1e-6

    lower_parent = "lower layer"
    upper_parent = "upper layer"
    mode = "mode x"


    script: str = "groupscope('::model');\n"
    script += "addstructuregroup;\n"\
            +"set('name', '"+mode+"');\n"

    # upper wg
    script += single_straight(upper_parent,mode,dist,0,0,*upper_wg.values())
    # lower wg
    script += single_straight(lower_parent,mode,0,0,0,*lower_wg.values())
    

    script += insulator(*insulation.values())

    mode = FDTD(hide = hideAnsys)
    mode.eval(script)
    mode.save(f"{name}")

def dir_coupler_builder(lower_wg: dict, upper_wg: dict,
                        dist: float, insulation: dict,
                        name: str, version: str,
                        layout_extract: bool = False,
                        hideAnsys: bool = True) -> None:
    """
    Prepares script to build directional coupler in the triplex scheme
    in Ansys.
    
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
    
    lower_parent = "lower layer"
    upper_parent = "upper layer"
    mode1 = "mode_1"
    mode2 = "mode_2"

    dist = dist + lower_wg["base height"]/2 + upper_wg["base height"]/2
    dist = dist*1e-6

    script: str = "groupscope('::model');\n"
    script += "addstructuregroup;\n"\
            +"set('name', '"+mode1+"');\n"\
            +"addstructuregroup;\n"\
            +"set('name', '"+mode2+"');\n"
    
    # upper wg
    script += coupler(upper_parent, mode1, mode2, dist, 0, 0, *upper_wg.values())
    # lower wg
    script += coupler(lower_parent, mode1, mode2, 0, 0, 0, *lower_wg.values())
    script += insulator(*insulation.values())

    if layout_extract:
        script += layout(num_of_cells, materials, path)

    mode = mode_select(version, hideAnsys)

    mode.eval(script)
    mode.save(f"{name}")


def circuit(lower_wg: dict, upper_wg: dict, insulation: dict, dist: float,
            name: str, hideAnsys: bool = True) -> None:
    """
    This function puts together the final circuit.
    """
    
    dist = dist + lower_wg["base height"]/2 + upper_wg["base height"]/2
    dist = dist*1e-6
    
    
    
    ...