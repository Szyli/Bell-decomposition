
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

class GeometryHolder:
    """
    
    """

    def __init__(self, bot_height: float, top_height: float,
                top_width: float, base_angle: float, dist: float,
                bend_height: float, bend_length: float,
                gap: float, Lc: float,
                wg_material: str, insul_material: str,
                straight: bool = False, wg_length: float = 0):
        
        # parameters for single straight wg cell
        if straight:
            self.upper_wg: dict = {
                    "base height" : top_height,
                    "base width" : bot_width_calc(base_angle,
                                                top_width, top_height),
                    "base angle" : base_angle,
                    "material" : wg_material,
                    "Lw" : wg_length,
                    }
            
            self.lower_wg: dict = {
                    "base height" : bot_height,
                    "base width" : bot_width_calc(base_angle,
                                                bot_width_calc(base_angle,
                                                    self.upper_wg["base width"], dist),
                                                bot_height),
                    "base angle" : base_angle,
                    "material" : wg_material,
                    "Lw" : wg_length
                    }

            self.insulation: dict = {
                    "x span" : wg_length,
                    "y span" : bend_height*4 + gap,
                    "material" : insul_material
                    }
        # parameters for coupler cell
        elif straight == False:
            self.upper_wg: dict = {
                    "base height" : top_height,
                    "base width" : bot_width_calc(base_angle,
                                                top_width, top_height),
                    "base angle" : base_angle,
                    "material" : wg_material,
                    "y span" : bend_height,
                    "Ls" : bend_length,
                    "gap" : gap,
                    "Lc" : Lc
                    }
            
            self.lower_wg: dict = {
                    "base height" : bot_height,
                    "base width" : bot_width_calc(base_angle,
                                                bot_width_calc(base_angle,
                                                    self.upper_wg["base width"], dist),
                                                bot_height),
                    "base angle" : base_angle,
                    "material" : wg_material,
                    "y span" : bend_height,
                    "Ls" : bend_length,
                    "gap" : gap,
                    "Lc" : Lc
                    }
            self.insulation: dict = {
                    "x span" : bend_length*2 + Lc,
                    "y span" : bend_height*4 + gap,
                    "material" : insul_material
                    }


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
            material: str, y_span: float, Ls: float,
            gap: float, Lc: float, ) -> str:
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
                z_mid: float, z_span: float, idx: int,
                parent: str, groupname: str) -> str:
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

    # if idx == 0:
    #     groupname = "bot insulation"
    # elif idx == 1:
    #     groupname = "mid insulation"
    # elif idx == 2:
    #     groupname = "top insulation"
    # else:
    #     print("Wrong index in insulation builder!")
    #     return

    script = "addstructuregroup;\n"\
            +"set('name', '"+groupname+"');\n"\
            +"addtogroup('"+parent+"');\n"\
            +"set(\"z\", "+str(z_mid)+");\n"\
            +"addrect;\n"\
            +"addtogroup('"+parent+"::"+groupname+"');\n"\
            +"set(\"x min\",0);\n"\
            +"set(\"x max\","+str(x_span)+"*1e-6);\n"\
            +"set(\"y min\","+str(-y_span/2)+"*1e-6);\n"\
            +"set(\"y max\","+str(y_span/2)+"*1e-6);\n"\
            +"set(\"z\", 0);\n"\
            +"set(\"z span\","+str(z_span)+");\n"\
            +"set(\"material\", \""+material+"\");\n"\
            +"set('name', 'insulator_"+str(idx)+"');\n"

    return script

def bot_width_calc(angle: float, width: float, height: float) -> float:
    """
    Function to calculate base width of waveguide.

    ---

    Args:
        angle (float): base angle
        width (float): top width
        height (float): height of the wg
    """

    angle = angle * np.pi / 180
    w_part = height / np.tan(angle)
        
    return float(width + 2*w_part)


# layout extraction

def layout(num_of_cells: int, struct: dict,
           path: str, name: str) -> str:
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
    script: str = "feval('"+path+"/GDSexport_core.lsfx');\n"\
                + "feval('"+path+"/GDS_auto_export.lsf');\n"

    script += "layer_def = cell("+str(num_of_cells)+");\n"

    idx = 1
    for z_pos in struct.keys():
        # layer_name = str(struct[z_pos].split(" ")[0])
        script += "layer_"+str(idx)+" = struct;\n"\
                + "layer_"+str(idx)+".z = "+str(z_pos)+";\n"\
                + "layer_"+str(idx)+".material =  \""+str(struct[z_pos])+"\";\n"\
                + "layer_"+str(idx)+".layer = "+str(idx)+";\n"\
                + "layer_def{"+str(idx)+"} = layer_"+str(idx)+";\n"
            # .layer property: optional? , "accept integer, or string" => string doesn't work
        idx += 1

    # file name definition
    script += "gds_filename = '"+name+".gds';\n"

    # using automated function to extract layout
    script += "GDS_AUTOEXPORT_MAIN({\"layer_def\": layer_def, \"gds_filename\": gds_filename, \"verbose\": false });\n"
    # verbose can be changed into parameter

    return script

# main builder functions

def single_straight_wgd_builder(lower_wg: dict, upper_wg: dict,
                        dist: float, insulation: dict,
                        insul_height_bot: float, insul_height_top: float,
                        mode_idx: int,
                        name: str, version: str,
                        layout_extract: bool = False,
                        path: str = "C:/Users/Admin/Desktop/Interferometer/gds_auto_export_package/gds_auto_export_package",
                        hideAnsys: bool = True):

    # distance in `z` between two layers of Si3N4
    dist_gap = dist + lower_wg["base height"]/2 + upper_wg["base height"]/2
    dist_gap = dist_gap*1e-6
    num_of_cells = 5    # 6 with Si base but that doesn't work when too thick
    insul_height_bot = insul_height_bot*1e-6
    insul_height_top = insul_height_top*1e-6
    dist = dist*1e-6

    lower_parent = "lower layer"
    upper_parent = "upper layer"
    cover_parent = "covering"
    mode = "mode_{}".format(mode_idx)

    script: str = "groupscope('::model');\n"
    script += "addstructuregroup;\n"\
            +"set('name', '"+mode+"');\n"

    # upper wg
    script += single_straight(upper_parent,mode,dist_gap,0,0,*upper_wg.values())
    # lower wg
    script += single_straight(lower_parent,mode,0,0,0,*lower_wg.values())
    
    structure: dict = {
        0 : lower_wg["material"],
        dist_gap : upper_wg["material"]
    }
    
    # list of insulation and base thicknesses
    layer_thicks = [insul_height_bot,
                    lower_wg["base height"]*1e-6 + dist + upper_wg["base height"]*1e-6,
                    insul_height_top]
    
    groupnames = ["bot insulation",
                "mid insulation",
                "top insulation"]
    
    z_min = -(insul_height_bot + 0.5*lower_wg["base height"]*1e-6)

    script += "addstructuregroup;\n"\
            +"set('name', '"+cover_parent+"');\n"
    
    for layer in range(num_of_cells-2): # -2 for the waveguides
        z_max = z_min + layer_thicks[layer]
        z_mid = z_min+(z_max-z_min)/2
    
        script += insulator(*insulation.values(), z_mid, layer_thicks[layer], layer,
                        cover_parent, groupnames[layer])
        
        structure[z_mid] = insulation["material"]

        z_min = z_max

    if layout_extract:
        script += layout(num_of_cells, structure, path, name)
        
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
                        insul_height_bot: float, insul_height_top: float,
                        top_mode_idx: int, bot_mode_idx: int,
                        name: str, version: str,
                        layout_extract: bool = False,
                        path: str = "C:/Users/Admin/Desktop/Interferometer/gds_auto_export_package/gds_auto_export_package",
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
    
    # distance in `z` between two layers of Si3N4
    dist_gap = dist + lower_wg["base height"]/2 + upper_wg["base height"]/2
    dist_gap = dist_gap*1e-6
    num_of_cells = 5    # 6 with Si base but that doesn't work when too thick
    insul_height_bot = insul_height_bot*1e-6
    insul_height_top = insul_height_top*1e-6
    dist = dist*1e-6
    # base_height = base_height*1e-6
    
    lower_parent = "lower layer"
    upper_parent = "upper layer"
    cover_parent = "covering"
    mode1 = "mode_{}".format(top_mode_idx)
    mode2 = "mode_{}".format(bot_mode_idx)

    script: str = "groupscope('::model');\n"

    script += "addstructuregroup;\n"\
            +"set('name', '"+mode1+"');\n"\
            +"addstructuregroup;\n"\
            +"set('name', '"+mode2+"');\n"

    # upper wg
    script += coupler(upper_parent, mode1, mode2, dist_gap, 0, 0, *upper_wg.values())
    # lower wg
    script += coupler(lower_parent, mode1, mode2, 0, 0, 0, *lower_wg.values())
    
    structure: dict = {
        0 : lower_wg["material"],
        dist_gap : upper_wg["material"]
    }

    # list of insulation and base thicknesses
    layer_thicks = [insul_height_bot,
                    lower_wg["base height"]*1e-6 + dist + upper_wg["base height"]*1e-6,
                    insul_height_top]
    
    groupnames = ["bot insulation",
                "mid insulation",
                "top insulation"]
    
    z_min = -(insul_height_bot + 0.5*lower_wg["base height"]*1e-6)

    script += "addstructuregroup;\n"\
            +"set('name', '"+cover_parent+"');\n"

    for layer in range(num_of_cells-2): # -2 for the waveguides
        z_max = z_min + layer_thicks[layer]
        z_mid = z_min+(z_max-z_min)/2
    
        script += insulator(*insulation.values(), z_mid, layer_thicks[layer], layer,
                        cover_parent, groupnames[layer])
        
        structure[z_mid] = insulation["material"]

        z_min = z_max

    if layout_extract:
        script += layout(num_of_cells, structure, path, name)

    mode = mode_select(version, hideAnsys)

    mode.eval(script)
    mode.save(f"{name}")


def circuit() -> None:
    """
    This function puts together the final circuit.
    """
    
    
    
    
    
    ...