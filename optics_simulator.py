
import numpy as np
import sys, os
# Note: This path may vary for users, therefore
# Todo: Update to find path corresponding to lumerical
sys.path.append("D:\\lumerical\\api\\python")
sys.path.append(os.path.abspath(""))

def mode_select(mode: str, vis: bool) -> None:
    """
    Use this function to select the Ansys Lumerical solver to use.

    ---

    Args:
        mode (str): The name of the solver. Currently supported:
            - MODE
            - FDTD
        vis (bool): `True` if you want Ansys hidden during the run and `False` otherwise
    Note
    ---
        `DEVICE` is not a supported solver since it handles objects' materials differently.
        A working practice is to build the system in `MODE` or `FDTD` and copy the objects
        from the object tree.
    Returns:
        Lumapi solver object.
    """
    if mode == "MODE":
        from lumapi import MODE
        solver = MODE(hide = vis)
        print("{} was selected.".format(mode))
        return solver
    if mode == "FDTD":
        from lumapi import FDTD
        solver = FDTD(hide = vis)
        print("{} was selected.".format(mode))
        return solver
    else:
        print("Invalid Ansys solver mode given!")
        print("{} was selected.".format(mode))

class StraightWGD:
    """
    This class defines a straight waveguide (wgd).

    **All parameters of length to be added in microns!**

    ---

    Attributes:
        base_width (float): The width of the base (bottom of the waveguide)
        base_height (flaot): The height of the wgd
        base_angle (flaot): The angle between the base and the sides
        material (str): Material of the waveguide
        poles (str): `x` and `y` coordinates of the starting and the ending
              point of the wgd
    Methods
    ---

    ``builder(name)``:
        Prepares the script for selected ansys solver to evaluate a straight waveguide
    """
    def __init__(self, base_width: float,
                base_height: float, base_angle: float,
                material: str, poles: str) -> None:

        self.width = base_width * 1e-6
        self.height = base_height * 1e-6
        self.angle = base_angle
        self.material = material
        self.pole = poles + " * 1e-6"

    def builder(self, name: str) -> str:
        """
        Returns the script building defined straight waveguide.

        ---

        Args:
            name (str): the name of the waveguide object
        Returns:
            Script to evaluate defined straight waveguide
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

class MMI:
    """
    This class defines an Multi Mode Interferometer (MMI) coupler.

    **All parameters of length to be added in microns!**

    ---

    Args:

     ---

    Functions
    ---

    - ``builder``: returns the script for ansys to evaluate the
                    defined straight waveguide.
    """
    def __init__(self, mmi_width: float,
                mmi_height: float, mmi_length: float,
                base_angle: float, material: str,
                x_pos: float, z_pos: float, y_pos: float=0):

        self.width = mmi_width * 1e-6
        self.height = mmi_height * 1e-6
        self.angle = base_angle
        # self.length = mmi_length * 1e-6
        self.pole = "["+str(x_pos* 1e-6)+","+str(y_pos* 1e-6)+";"\
                        +str((x_pos + mmi_length) * 1e-6)+","+str(y_pos* 1e-6)+"]"
        self.material = material
        # self.x_pos = x_pos * 1e-6
        self.z_pos = z_pos

    def builder(self, name: str) -> str:
        """
          Function to prepare script for ansys to evaluate the
        defined MMI.
        """
        
        # script = "addrect;\n"\
        #         +"set(\"y min\", "+str(-self.width/2)+");\n"\
        #         +"set(\"y max\", "+str(self.width/2)+");\n"\
        #         +"set(\"z\", "+str(self.z_pos)+");\n"\
        #         +"set(\"z span\", "+str(self.height)+");\n"\
        #         +"set(\"x min\", "+str(self.x_pos)+");\n"\
        #         +"set(\"x max\", "+str(self.x_pos + self.length)+");\n"\
        #         +"set(\"material\", \""+self.material+"\");\n"\
        #         +"set(\"override mesh order from material database\",1);\n"\
        #         +"set(\"mesh order\",1);\n"\
        #         +"set('name','"+name+"');\n"
        script = "addwaveguide;\n"\
                +"set(\"base width\", "+str(self.width)+");\n"\
                +"set(\"base height\", "+str(self.height)+");\n"\
                +"set(\"base angle\", "+str(self.angle)+");\n"\
                +"set(\"z\", "+str(self.z_pos)+");\n"\
                +"set(\"poles\", "+self.pole+");\n"\
                +"set(\"material\", \""+self.material+"\");\n"\
                +"set(\"override mesh order from material database\",1);\n"\
                +"set(\"mesh order\",1);\n"\
                +"set('name','"+name+"');\n"
        
        return script

class LinearTaper:
    """
    This class defines a linear taper.

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
    def __init__(self, taper_height: float,
                taper_width_l: float, taper_width_r: float,
                taper_length: float, base_angle: float,
                material: str, x_pos: float, y_pos: float, z_pos: float,
                mesh_order: int = 1, hfrac_ref: float = 0):

        self.thickness = taper_height * 1e-6
        self.width_l = taper_width_l * 1e-6
        self.width_r = taper_width_r * 1e-6
        self.angle_side = base_angle
        self.len = taper_length * 1e-6
        self.material = material
        self.mesh_order = mesh_order
        self.x = x_pos * 1e-6
        self.y = y_pos * 1e-6
        self.z = z_pos * 1e-6
        self.hfrac_ref = hfrac_ref
        self.code = '\
            #####################################################################\n\
            # Linear taper\n\
            # This object makes a linear taper with angled sidewalls.\n\
            #\n\
            # Input properties\n\
            # thickness:         waveguide height\n\
            # base angle:        sidewall angle of the waveguide\n\
            # width_l & width_r: the left and right width of the taper\n\
            # len:               length of the taper\n\
            # hfrac_ref:         the reference height at where the widths are specified, 0 =< hfrac_ref =< 1.  This is relative to the height of the trapeziod (waveguide cross section)\n\
            #		The most common cases:\n\
            #		1   : top width\n\
            #		0.5 : middle width\n\
            #		0   : bottom width\n\
            #\n\
            # Tags: integrated optics waveguide linear taper\n\
            #\n\
            # Copyright 2018 Lumerical Solutions Inc\n\
            #####################################################################\n\
            \n\
            delta_w = 2*thickness*tan((90-angle_side)*pi/180);\n\
            \n\
            ?"width_l = " + num2str(width_l);\n\
            ?"width_r = " + num2str(width_r) + endl;\n\
            \n\
            width_top_l = width_l - (1-hfrac_ref)*delta_w;\n\
            width_top_r = width_r - (1-hfrac_ref)*delta_w;\n\
            ?"width_top_l = " + num2str(width_top_l);\n\
            ?"width_top_r = " + num2str(width_top_r) + endl;\n\
            \n\
            width_bot_l = width_l + hfrac_ref*delta_w;\n\
            width_bot_r = width_r + hfrac_ref*delta_w;\n\
            ?"width_bot_l = " + num2str(width_bot_l);\n\
            ?"width_bot_r = " + num2str(width_bot_r) + endl;\n\
            \n\
            # Error checking:\n\
            \n\
            if ((hfrac_ref>1) or (hfrac_ref<0)){\n\
                ?"Error: hfrac_ref must be between 0 and 1.";\n\
                break;\n\
            }\n\
            \n\
            if ((width_top_l<0) or (width_top_r<0) or (width_bot_l<0) or (width_bot_r<0)){\n\
                ?"Error: width and angle values are not correct.";\n\
                break;\n\
            }\n\
            \n\
            zmin = -thickness/2;\n\
            zmax = thickness/2;\n\
            \n\
            xmin = -len/2;\n\
            xmax = len/2;\n\
            \n\
            ymin_bot_l = -width_bot_l/2;\n\
            ymax_bot_l = width_bot_l/2;\n\
            \n\
            ymin_bot_r = -width_bot_r/2;\n\
            ymax_bot_r = width_bot_r/2;\n\
            \n\
            ymin_top_l = -width_top_l/2;\n\
            ymax_top_l = width_top_l/2;\n\
            \n\
            ymin_top_r = -width_top_r/2;\n\
            ymax_top_r = width_top_r/2;\n\
            \n\
            vtx=    [xmin,ymin_bot_l,zmin;    #1\n\
                    xmax,ymin_bot_r,zmin;     #2\n\
                    xmax,ymax_bot_r,zmin;     #3\n\
                    xmin,ymax_bot_l,zmin;     #4\n\
                    xmin,ymin_top_l,zmax;     #5\n\
                    xmax,ymin_top_r,zmax;     #6\n\
                    xmax,ymax_top_r,zmax;     #7\n\
                    xmin,ymax_top_l,zmax];    #8\n\
            \n\
            a = cell(6);\n\
            for(i = 1:6){\n\
                a{i} = cell(1);\n\
            }\n\
            \n\
            #facets:\n\
            \n\
            a{1}{1} = [1,4,3,2]; #bottom facet\n\
            a{2}{1} = [1,2,6,5];\n\
            a{3}{1} = [2,3,7,6];\n\
            a{4}{1} = [3,4,8,7];\n\
            a{5}{1} = [1,5,8,4];\n\
            a{6}{1} = [5,6,7,8]; #top facet\n\
            \n\
            addplanarsolid(vtx,a);\n\
            \n\
            if (material=="<Object defined dielectric>"){\n\
                setnamed("solid", "index",1.4);\n\
            }\n\
            else{\n\
                setnamed("solid", "material",material);\n\
                set("override mesh order from material database",1);\n\
                setnamed("solid", "mesh order", mesh_order);\n\
            }'

    def builder(self, name: str) -> str:
        """
          Function to prepare script for ansys MODE to evaluate the
        defined straight waveguide.
        """
        
        script = "addstructuregroup;\n"\
                +"adduserprop(\"thickness\", 2, "+str(self.thickness)+");\n"\
                +"adduserprop(\"width_l\", 2, "+str(self.width_l)+");\n"\
                +"adduserprop(\"width_r\", 2, "+str(self.width_r)+");\n"\
                +"adduserprop(\"angle_side\", 0, "+str(self.angle_side)+");\n"\
                +"adduserprop(\"len\", 2, "+str(self.len)+");\n"\
                +"adduserprop(\"material\", 5, '"+self.material+"');\n"\
                +"adduserprop(\"mesh_order\", 0, "+str(self.mesh_order)+");\n"\
                +"adduserprop(\"hfrac_ref\", 0, "+str(self.hfrac_ref)+");\n"\
                +"set('construction group', 1);\n"\
                +"set('script', '"+self.code+"');\n"\
                +"set(\"x\", "+str(self.x)+");\n"\
                +"set(\"y\", "+str(self.y)+");\n"\
                +"set(\"z\", "+str(self.z)+");\n"\
                +"set('name','"+name+"');\n"
        return script

class GeometryHolder:
    """
    This class is meant to prepare all the proper parameters by the user input
    in order to build desired objects.

    **All parameters of length to be added in microns!**

    ---

    Attributes:
        bot_height (float):
        top_height (float):
        top_width (float):
        base_angle (float):
        dist (float):
        bend_height (float):
        bend_length (float):
        gap (float):
        Lc (float):
        wg_material (str):
        insul_material (str):
        multimode (bool):
        mmi_length (float):
        mmi_taper_length (float):
        taper_width (float):
        straight (bool):
        wg_length (float):
    Methods
    ---

    ``bot_width_calc(angle, width, height)``:
        Computes the bottom width of waveguide from the top width, height and side angle.
    """
    
    def bot_width_calc(self, angle: float, width: float, height: float) -> float:
        """
        Function to calculate base width of given waveguide.

        ---

        Args:
            angle (float): base angle
            width (float): top width
            height (float): height of the wg
        Returns:
            The base width of the waveguide.
        """

        angle = angle * np.pi / 180
        w_part = height / np.tan(angle)
        
        return float(width + 2*w_part)

    def mmi_top_width_calc(self, gap: float, taper_width: float) -> float:
        """
        Function to calculate the top width of the MMI region based on
        gap and taper width.

        Args:
            gap (float): the distance between the neighbouring tapers' wider ends
            taper_width (float): the width of the taper's wider end
        Returns:
            The top width of the MMI region.
        """

        # hard coded part to estimate the effective width over the MMI's width
        # it is largest for TE modes where sgm=0
        lmbd = 1.55     # wavelength in use (lambda)
        nc = 1.444      # effective index of cladding
        nr = 1.9961     # effective index of waveguide
        sgm = 0         # (sigma); for `TE = 0` for `TM = 1`

        ext = (lmbd/np.pi)*(nc/nr)**(2*sgm) * (nr**2 - nc**2)**(-0.5)
        We = 3*(gap + taper_width)
        # We = W_mmi + ext = 3(gap+width)

        return We - ext

    def mmi_length_calc(self, gap: float, taper_width: float) -> float:
        """
        Function to compute the MMI region's length.

        Args:
            gap (float): the distance between the neighbouring tapers' wider ends
            taper_width (float): the width of the taper's wider end
        Returns:
            The length of the MMI region.
        """
        n0 = 1.53344        # effective idnex of fundamental mode
        lmbd = 1.55         # wavelength in use (lambda)
        We = 3*(gap + taper_width)

        return (4*n0*We**2) / (6*lmbd)

    def __init__(self, bot_height: float, top_height: float,
                top_width: float, base_angle: float, dist: float,
                bend_height: float, bend_length: float,
                gap: float, Lc: float,
                wg_material: str, insul_material: str,
                multimode: bool = False,
                mmi_taper_length: float = 0, taper_width: float = 0,
                straight: bool = False, wg_length: float = 0):
        
        # parameters for single straight wg cell
        if straight:
            self.upper_wg: dict = {
                    "base height" : top_height,
                    "base width" : self.bot_width_calc(base_angle,
                                                top_width, top_height),
                    "base angle" : base_angle,
                    "material" : wg_material,
                    "Lw" : wg_length,
                    }
            
            self.lower_wg: dict = {
                    "base height" : bot_height,
                    "base width" : self.bot_width_calc(base_angle,
                                                self.bot_width_calc(base_angle,
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
        # parameters for directional coupler cell
        elif (straight == False) and (multimode == False):
            self.upper_wg: dict = {
                    "base height" : top_height,
                    "base width" : self.bot_width_calc(base_angle,
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
                    "base width" : self.bot_width_calc(base_angle,
                                                self.bot_width_calc(base_angle,
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
        # parameters for MMI coupler cell
        elif (straight == False) and (multimode == True):
            self.upper_wg: dict = {
                    "base height" : top_height,
                    "base width" : self.bot_width_calc(base_angle,
                                                top_width, top_height),
                    "taper width": self.bot_width_calc(base_angle,
                                                taper_width, top_height),
                    "taper length": mmi_taper_length,
                    "mmi base width": self.bot_width_calc(base_angle,
                                                self.mmi_top_width_calc(gap,taper_width),
                                                top_height),
                    "mmi length": self.mmi_length_calc(gap, taper_width),
                    "base angle" : base_angle,
                    "material" : wg_material,
                    "y span" : bend_height,
                    "Ls" : bend_length,
                    "gap" : gap,
                    "Lc" : Lc
                    }
            
            self.lower_wg: dict = {
                    "base height" : bot_height,
                    "base width" : self.bot_width_calc(base_angle,
                                                self.bot_width_calc(base_angle,
                                                    self.upper_wg["base width"], dist),
                                                bot_height),
                    "taper width": self.bot_width_calc(base_angle,
                                                self.bot_width_calc(base_angle,
                                                    self.upper_wg["taper width"], dist),
                                                top_height),
                    "taper length": mmi_taper_length,
                    "mmi base width": self.bot_width_calc(base_angle,
                                                self.bot_width_calc(base_angle,
                                                    self.upper_wg["mmi base width"], dist),
                                                bot_height),
                    "mmi length": self.upper_wg["mmi length"],
                    "base angle" : base_angle,
                    "material" : wg_material,
                    "y span" : bend_height,
                    "Ls" : bend_length,
                    "gap" : gap,
                    "Lc" : Lc
                    }

            self.insulation: dict = {
                    "x span" : bend_length*2 + 2*Lc +\
                                2* mmi_taper_length + self.upper_wg["mmi length"],
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

def mmi(parent: str, mode1: str, mode2: str, z_pos: float,
            x_pos: float, y_pos: float, y_pos_adj: float,
            base_height: float, base_width: float,
            taper_width: float, taper_length: float,
            mmi_width: float, mmi_length: float,
            base_angle: float,
            material: str, y_span: float, Ls: float,
            gap: float, Lc: float) -> str:
    """
    This function makes the script for MMI couplers in Ansys.

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
        Lc (float): length of the connector straight waveguide between bend and taper
        Ls (float): x span of the four individual s-bend sections
    """

    # mode 1
    script = "addstructuregroup;\n"\
            +"set('name', '"+parent+" 1');\n"\
            +"addtogroup('"+mode1+"');\n"\
            + BentWGD(base_width, base_height,
                base_angle, material,
                "["+str(x_pos)+","+str(y_pos+y_span+y_pos_adj)+";"\
                +str(x_pos+Ls/2)+","+str(y_pos+y_span+y_pos_adj)+";"\
                +str(x_pos+Ls/2)+","+str(y_pos+y_pos_adj)+";"\
                +str(x_pos+Ls)+","+str(y_pos+y_pos_adj)+"]").builder(name="S-bend in " + mode1 +" "+ parent)\
            +"addtogroup('"+mode1+"::"+parent+" 1');\n"\
            + StraightWGD(base_width, base_height,
                        base_angle, material,
                        "["+str(x_pos+Ls)+","+str(y_pos+y_pos_adj)+";"\
                        +str(x_pos+Ls+Lc)+","+str(y_pos+y_pos_adj)+"]").builder(name="straight connector in " + mode1 +" "+parent)\
            +"addtogroup('"+mode1+"::"+parent+" 1');\n"\
            + LinearTaper(base_height,base_width,taper_width,taper_length,
                    base_angle,material,
                    x_pos+Ls+Lc+taper_length/2,
                    y_pos+y_pos_adj,
                    z_pos).builder(name="taper in " + mode1 +" "+parent)\
            +"addtogroup('"+mode1+"::"+parent+" 1');\n"\
            + LinearTaper(base_height,taper_width,base_width,taper_length,
                    base_angle,material,
                    x_pos+Ls+Lc+1.5*taper_length+mmi_length,
                    y_pos+y_pos_adj,
                    z_pos).builder(name="taper out " + mode1 +" "+parent)\
            +"addtogroup('"+mode1+"::"+parent+" 1');\n"\
            + StraightWGD(base_width, base_height,
                        base_angle, material,
                        "["+str(x_pos+Ls+Lc+taper_length*2+mmi_length)+","+str(y_pos+y_pos_adj)+";"\
                        +str(x_pos+Ls+2*Lc+taper_length*2+mmi_length)+","+str(y_pos+y_pos_adj)+"]").builder(name="straight connector out " + mode1 +" "+parent)\
            +"addtogroup('"+mode1+"::"+parent+" 1');\n"\
            + BentWGD(base_width, base_height,
                    base_angle, material,
                    "["+str(x_pos+Ls+2*Lc+taper_length*2+mmi_length)+","+str(y_pos+y_pos_adj)+";"\
                    +str(x_pos+1.5*Ls+2*Lc+taper_length*2+mmi_length)+","+str(y_pos+y_pos_adj)+";"\
                    +str(x_pos+1.5*Ls+2*Lc+taper_length*2+mmi_length)+","+str(y_pos+y_pos_adj+y_span)+";"\
                    +str(x_pos+2*Ls+2*Lc+taper_length*2+mmi_length)+","+str(y_pos+y_pos_adj+y_span)+"]").builder(name="S-bend out " + mode1 +" "+parent)\
            +"addtogroup('"+mode1+"::"+parent+" 1');\n"
    # mode 2
    script += "addstructuregroup;\n"\
            +"set('name', '"+parent+" 2');\n"\
            +"addtogroup('"+mode2+"');\n"\
            + BentWGD(base_width, base_height,
                    base_angle, material,
                    "["+str(x_pos)+","+str(-y_span-y_pos_adj+y_pos)+";"\
                    +str(x_pos+Ls/2)+","+str(-y_span-y_pos_adj+y_pos)+";"\
                    +str(x_pos+Ls/2)+","+str(-y_pos_adj +y_pos)+";"\
                    +str(x_pos+Ls)+","+str(-y_pos_adj +y_pos)+"]").builder(name="S-bend in " + mode2 +" "+parent)\
            +"addtogroup('"+mode2+"::"+parent+" 2');\n"\
            + StraightWGD(base_width, base_height,
                        base_angle, material,
                        "["+str(x_pos+Ls)+","+str(-y_pos_adj +y_pos)+";"\
                        +str(x_pos+Ls+Lc)+","+str(-y_pos_adj +y_pos)+"]").builder(name="straight connector in " + mode2 +" "+parent)\
            +"addtogroup('"+mode2+"::"+parent+" 2');\n"\
            + LinearTaper(base_height,base_width,taper_width,taper_length,
                    base_angle,material,
                    x_pos+Ls+Lc+taper_length/2,
                    y_pos-y_pos_adj,
                    z_pos).builder(name="taper in " + mode2 +" "+parent)\
            +"addtogroup('"+mode2+"::"+parent+" 2');\n"\
            + LinearTaper(base_height,taper_width,base_width,taper_length,
                    base_angle,material,
                    x_pos+Ls+Lc+1.5*taper_length+mmi_length,
                    y_pos-y_pos_adj,
                    z_pos).builder(name="taper out " + mode2 +" "+parent)\
            +"addtogroup('"+mode2+"::"+parent+" 2');\n"\
            + StraightWGD(base_width, base_height,
                        base_angle, material,
                        "["+str(x_pos+Ls+Lc+taper_length*2+mmi_length)+","+str(y_pos-y_pos_adj)+";"\
                        +str(x_pos+Ls+2*Lc+taper_length*2+mmi_length)+","+str(y_pos-y_pos_adj)+"]").builder(name="straight connector out " + mode2 +" "+parent)\
            +"addtogroup('"+mode2+"::"+parent+" 2');\n"\
            + BentWGD(base_width, base_height,
                    base_angle, material,
                    "["+str(x_pos+Ls+2*Lc+taper_length*2+mmi_length)+","+str(y_pos-y_pos_adj)+";"\
                    +str(x_pos+1.5*Ls+2*Lc+taper_length*2+mmi_length)+","+str(y_pos-y_pos_adj)+";"\
                    +str(x_pos+1.5*Ls+2*Lc+taper_length*2+mmi_length)+","+str(y_pos-y_span-y_pos_adj)+";"\
                    +str(x_pos+2*Ls+2*Lc+taper_length*2+mmi_length)+","+str(y_pos-y_span-y_pos_adj)+"]").builder(name="S-bend out " + mode2 +" "+parent)\
            +"addtogroup('"+mode2+"::"+parent+" 2');\n"

    script += MMI(mmi_width, base_height, mmi_length,
                base_angle, material,
                x_pos+Ls+Lc+taper_length, z_pos).builder(name="MMI "+parent)

    # positioning
    script += "select('"+mode1+"::"+parent+" 1');\n"\
            +"set(\"z\", "+str(z_pos)+");\n"\
            +"select('"+mode2+"::"+parent+" 2');\n"\
            +"set(\"z\", "+str(z_pos)+");\n"

    return script

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

def mmi_coupler_builder(lower_wg: dict, upper_wg: dict,
                        dist: float,
                        insulation: dict,
                        insul_height_bot: float, insul_height_top: float,
                        top_mode_idx: int, bot_mode_idx: int,
                        name: str, version: str,
                        layout_extract: bool = False,
                        path: str = "C:/Users/Admin/Desktop/Interferometer/gds_auto_export_package/gds_auto_export_package",
                        hideAnsys: bool = True) -> None:
    """
        Function that builds MMI coupler in Ansys.
    """
    # distance in `z` between two layers of Si3N4
    dist_gap = dist + lower_wg["base height"]/2 + upper_wg["base height"]/2
    dist_gap = dist_gap*1e-6
    num_of_cells = 5    # 6 with Si base but that doesn't work when too thick
    insul_height_bot = insul_height_bot*1e-6
    insul_height_top = insul_height_top*1e-6
    dist = dist*1e-6

    y_adj = (upper_wg["gap"]+lower_wg["taper width"])/2
    
    lower_parent = "lower layer"
    upper_parent = "upper layer"
    cover_parent = "cladding"
    mode1 = "mode_{}".format(top_mode_idx)
    mode2 = "mode_{}".format(bot_mode_idx)

    script: str = "groupscope('::model');\n"

    script += "addstructuregroup;\n"\
            +"set('name', '"+mode1+"');\n"\
            +"addstructuregroup;\n"\
            +"set('name', '"+mode2+"');\n"
    
    # upper layer
    script += mmi(upper_parent, mode1, mode2, dist_gap, 0, 0, y_adj,
                *upper_wg.values())
    # lower layer
    script += mmi(lower_parent, mode1, mode2, 0, 0, 0, y_adj,
                *lower_wg.values())
    
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

    # return script

    mode.eval(script)
    mode.save(f"{name}")


def circuit() -> None:
    """
    This function puts together the final circuit.
    """
    
    
    
    
    
    ...

