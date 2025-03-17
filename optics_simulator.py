
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

# --------------------------------------------
# basic geometries
# --------------------------------------------

class StraightWGD:
    """
    This class defines a straight waveguide.

    **All parameters of length to be added in microns!**

    ---

    Attributes:
        base_width (float): the width of the base (bottom of the waveguide)
        base_height (flaot): the height of the wgd
        base_angle (flaot): the angle between the base and the sides
        material (str): material of the waveguide
        poles (str): `x` and `y` coordinates of poles defining the
                Bézier-curve for the waveguide.
    Note
    ---
        For straight waveguides we need two pair of pole points.
        It is a string of list with semicolumns separating the poles
        and columns separating the coordinates.
        
    Example:
        
    >>``poles = "[0,10; 10,10]"``
    Methods
    ---

    ``builder(name)``:
        Prepares the script for selected ansys solver to evaluate a straight waveguide
    """
    def __init__(self, base_width: float,
                base_height: float, base_angle: float,
                material: str, poles: str) -> None:

        self.width = base_width
        self.height = base_height
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
                +"set(\"base width\", "+str(self.width)+"* 1e-6);\n"\
                +"set(\"base height\", "+str(self.height)+"* 1e-6);\n"\
                +"set(\"base angle\", "+str(self.angle)+");\n"\
                +"set(\"poles\", "+self.pole+");\n"\
                +"set(\"material\", \""+self.material+"\");\n"\
                +"set(\"override mesh order from material database\",1);\n"\
                +"set(\"mesh order\",1);\n"\
                +"set('name','"+name+"');\n"
        return script

class BentWGD:
    """
    This class defines a bent waveguide.

    **All parameters of length to be added in microns!**

    ---

    Attributes:
        base_width (float): the width of the base (bottom of the waveguide)
        base_height (flaot): the height of the wgd
        base_angle (flaot): the angle between the base and the sides
        material (str): material of the waveguide
        poles (str): `x` and `y` coordinates of poles defining the
                Bézier-curve that is this bent waveguide.
    Note
    ---
        For S-bends we need four pair of pole points. It is a string
        of list with semicolumns separating the poles and columns
        separating the coordinates.
        
    Example:
        
    >>``poles = "[0,10; 5,10; 5,0; 10,0]"``
        
    Methods
    ---

    ``builder(name)``:
        Prepares the script for selected ansys solver to evaluate a bent waveguide
    """

    def __init__(self, base_width: float,
                base_height: float, base_angle: float,
                material: str, poles: str):

        self.width = base_width
        self.height = base_height
        self.angle = base_angle
        self.material = material
        self.pole = poles + " * 1e-6"

    def builder(self, name: str) -> str:
        """
        Returns the script building defined bent waveguide.

        ---

        Args:
            name (str): the name of the bent waveguide object
        Returns:
            Script to evaluate defined straight waveguide
        """
        script = "addwaveguide;\n"\
                +"set(\"base width\", "+str(self.width)+"* 1e-6);\n"\
                +"set(\"base height\", "+str(self.height)+"* 1e-6);\n"\
                +"set(\"base angle\", "+str(self.angle)+");\n"\
                +"set(\"poles\", "+self.pole+");\n"\
                +"set(\"material\", \""+self.material+"\");\n"\
                +"set(\"override mesh order from material database\",1);\n"\
                +"set(\"mesh order\",1);\n"\
                +"set('name','"+name+"');\n"
        return script

class MMI:
    """
    This class defines a Multi Mode Interferometer (MMI) region.

    **All parameters of length to be added in microns!**

    ---

    Args:
        mmi_width (float): the base (bottom) width of the MMI region
        mmi_height (float): the height of the MMI region
        base_angle (float): the angle between the base and the sides
        material (str): the material of the MMI region
        x_pos (float): the starting position in `x`
        y_pos (float): the starting position in `y` that
                    is also the center of the MMI region in `y`
    Methods
    ---

    ``builder(name)``:
        Prepares the script for selected ansys solver to evaluate an MMI region
    """
    def __init__(self, mmi_width: float,
                mmi_height: float, mmi_length: float,
                base_angle: float, material: str,
                x_pos: float, y_pos: float=0):

        self.width = mmi_width
        self.height = mmi_height
        self.angle = base_angle
        self.pole = "["+str(x_pos)+"* 1e-6,"+str(y_pos)+"* 1e-6;"\
                        +str(x_pos + mmi_length)+"* 1e-6"+str(y_pos)+"* 1e-6]"
        self.material = material

    def builder(self, name: str) -> str:
        """
        Assambles the script to build MMI region.
        
        ---

        Args:
            name (str): the name of the MMI object
        Returns:
            Script to evaluate defined MMI region
        """
        script = "addwaveguide;\n"\
                +"set(\"base width\", "+str(self.width)+"* 1e-6);\n"\
                +"set(\"base height\", "+str(self.height)+"* 1e-6);\n"\
                +"set(\"base angle\", "+str(self.angle)+");\n"\
                +"set(\"z\", 0);\n"\
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
        taper_height (float): the height of the taper waveguide
        taper_width_l (float): the width of the taper waveguide on the left
        taper_width_r (float): the width of the taper waveguide on the right
        taper_length (float): the length of the taper        
        base_angle (float): the angle between the base and the sides
        material (str): material of the taper waveguide
        x_pos (float): the starting position in `x`
        y_pos (float): the starting position in `y` that
                    is also the center of the MMI region in `y`
        mesh_order (int): by default `1`. Determines the rank in the system
        hfrac_ref (float): by default `0`. the reference height at where the widths are specified,
                0 =< hfrac_ref =< 1. 
                This is relative to the height of the trapeziod (waveguide cross section)\n\

    Methods
    ---

    ``builder(name)``:
        Prepares the script for selected ansys solver to evaluate a linear straight taper.
    """
    
    def __init__(self, taper_height: float,
                taper_width_l: float, taper_width_r: float,
                taper_length: float, base_angle: float,
                material: str, x_pos: float, y_pos: float,
                mesh_order: int = 1, hfrac_ref: float = 0):

        self.thickness = taper_height
        self.width_l = taper_width_l
        self.width_r = taper_width_r
        self.angle_side = base_angle
        self.len = taper_length
        self.material = material
        self.mesh_order = mesh_order
        self.x = x_pos
        self.y = y_pos
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


    def builder(self, name: str, layout: bool = False) -> str:
        """
          Function to prepare script for ansys MODE to evaluate the
        defined straight waveguide.
        """

        if not layout:
            script = "addstructuregroup;\n"\
                +"adduserprop(\"thickness\", 2, "+str(self.thickness)+"* 1e-6);\n"\
                +"adduserprop(\"width_l\", 2, "+str(self.width_l)+"* 1e-6);\n"\
                +"adduserprop(\"width_r\", 2, "+str(self.width_r)+"* 1e-6);\n"\
                +"adduserprop(\"angle_side\", 0, "+str(self.angle_side)+");\n"\
                +"adduserprop(\"len\", 2, "+str(self.len)+"* 1e-6);\n"\
                +"adduserprop(\"material\", 5, '"+self.material+"');\n"\
                +"adduserprop(\"mesh_order\", 0, "+str(self.mesh_order)+");\n"\
                +"adduserprop(\"hfrac_ref\", 0, "+str(self.hfrac_ref)+");\n"\
                +"set('construction group', 1);\n"\
                +"set('script', '"+self.code+"');\n"\
                +"set(\"x\", "+str(self.x)+"* 1e-6);\n"\
                +"set(\"y\", "+str(self.y)+"* 1e-6);\n"\
                +"set(\"z\", 0);\n"\
                +"set('name','"+name+"');\n"
        else:
            script = "thickness = "+str(self.thickness)+"* 1e-6;\n"\
                    +"angle_side = "+str(self.angle_side)+";\n"\
                    +"width_l = "+str(self.width_l)+"* 1e-6;\n"\
                    +"width_r = "+str(self.width_r)+"* 1e-6;\n"\
                    +"len = "+str(self.len)+"* 1e-6;\n"\
                    +"material = '"+self.material+"';\n"\
                    +"mesh_order = "+str(self.mesh_order)+";"\
                    +"hfrac_ref = "+str(self.hfrac_ref)+";\n"\
                    +"delta_w = 2*thickness*tan((90-angle_side)*pi/180);\n"\
                    +"width_top_l = width_l - (1-hfrac_ref)*delta_w;\n"\
                    +"width_top_r = width_r - (1-hfrac_ref)*delta_w;\n"\
                    +"width_bot_l = width_l + hfrac_ref*delta_w;\n"\
                    +"width_bot_r = width_r + hfrac_ref*delta_w;\n"\
                    +"zmin = -thickness/2;\n"\
                    +"zmax = thickness/2;\n"\
                    +"xmin = -len/2;\n"\
                    +"xmax = len/2;\n"\
                    +"ymin_bot_l = -width_bot_l/2;\n"\
                    +"ymax_bot_l = width_bot_l/2;\n"\
                    +"ymin_bot_r = -width_bot_r/2;\n"\
                    +"ymax_bot_r = width_bot_r/2;\n"\
                    +"ymin_top_l = -width_top_l/2;\n"\
                    +"ymax_top_l = width_top_l/2;\n"\
                    +"ymin_top_r = -width_top_r/2;\n"\
                    +"ymax_top_r = width_top_r/2;\n"\
                    +"vtx=    [xmin,ymin_bot_l,zmin;"\
                    +        "xmax,ymin_bot_r,zmin;"\
                    +        "xmax,ymax_bot_r,zmin;"\
                    +        "xmin,ymax_bot_l,zmin;"\
                    +        "xmin,ymin_top_l,zmax;"\
                    +        "xmax,ymin_top_r,zmax;"\
                    +        "xmax,ymax_top_r,zmax;"\
                    +        "xmin,ymax_top_l,zmax];\n"\
                    +"a = cell(6);\n"\
                    +"for(i = 1:6){"\
                    +"a{i} = cell(1);"\
                    +"}"\
                    +"a{1}{1} = [1,4,3,2]; #bottom facet\n"\
                    +"a{2}{1} = [1,2,6,5];\n"\
                    +"a{3}{1} = [2,3,7,6];\n"\
                    +"a{4}{1} = [3,4,8,7];\n"\
                    +"a{5}{1} = [1,5,8,4];\n"\
                    +"a{6}{1} = [5,6,7,8]; #top facet\n"\
                    +"addplanarsolid(vtx,a);\n"\
                    +"setnamed('solid', 'material', material);\n"\
                    +"set('override mesh order from material database', 1);\n"\
                    +"setnamed('solid', 'mesh order', mesh_order);\n"\
                    +"set(\"x\", "+str(self.x)+"* 1e-6);\n"\
                    +"set(\"y\", "+str(self.y)+"* 1e-6);\n"\
                    +"set(\"z\", 0);\n"\
                    +"set('name','"+name+"');\n"\
                    +"clear;\n"

        return script

class GeometryHolder:
    """
    This class is meant to prepare all the proper parameters by the user input
    in order to build desired objects.
    
    By "bottom waveguide" we mean the thinner lower layer in an
    assymetric double strips triplex scheme. Then, by "upper waveguide"
    we mean the waveguide core.
    
    The class has three modes which return parameters accordingly:
        - Single Straight Waveguide
        - Directional Coupler
        - Multi Mode Interference Coupler

    **All parameters of length to be added in microns!**

    ---

    Attributes:
        bot_height (float): height of the bottom waveguide layer
        top_height (float): height of the upper waveguide layer
        top_width (float): width of the upper layer
        base_angle (float): the angle between the base and the sides
        dist (float): the "triplex distance" that is the height of the
                    cladding between the two stripes in the triplex scheme
        bend_height (float): the height of the S-bend in `y` direction
        bend_length (float): the length of the S-bend in `x` direction (aka. propagation direction)
        gap (float): the smallest distance between two neighbouring waveguides in `y` direction
        L (float): the length of the straight waveguide or the coupling or connector region's
        wg_material (str): the material of the waveguides
        insul_material (str): the material of the cladding
        mode (int): choose what object's geometry to prepare
            - Single Straight Waveguide : `0`
            - Directional Coupler : `1`
            - Multi Mode Interference Coupler : `2`
        mmi_taper_length (float): the length of the linear taper (0 by default)
        taper_width (float): width of the wider end of linear taper (0 by default)
    Methods
    ---

    ``bot_width_calc(angle, width, height)``:
        Computes the bottom width of waveguides from the top width, height and side angle.
    ``mmi_top_width_calc(gap, taper_width)``:
        Computes the top width of the MMI region designed for two fold image creation at the output based on the gap and the taper width.
    ``mmi_length_calc(gap, taper_width)``:
        Computes the length of the MMI region designed for two fold image creation at the output based on the gap and the taper width (aka. the MMI's width).
    """
    
    def bot_width_calc(self, angle: float, width: float, height: float) -> float:
        """
        Function to calculate base width of given waveguide.

        ---

        Args:
            base_angle (float): the angle between the base and the sides
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
        # The math: We = 3(gap+width) = W_mmi + ext

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
                gap: float, L: float,
                wg_material: str, insul_material: str,
                mode: int = 0,
                mmi_taper_length: float = 0, taper_width: float = 0):
        
        # parameters for single straight wg cell
        if mode==0:
            self.upper_wg: dict = {
                    "base height" : top_height,
                    "base width" : self.bot_width_calc(base_angle,
                                                top_width, top_height),
                    "base angle" : base_angle,
                    "material" : wg_material,
                    "Lw" : L,
                    }
            
            self.lower_wg: dict = {
                    "base height" : bot_height,
                    "base width" : self.bot_width_calc(base_angle,
                                                self.bot_width_calc(base_angle,
                                                    self.upper_wg["base width"], dist),
                                                bot_height),
                    "base angle" : base_angle,
                    "material" : wg_material,
                    "Lw" : L
                    }

            self.insulation: dict = {
                    "x span" : L,
                    "y span" : bend_height*4 + gap,
                    "material" : insul_material
                    }
        # parameters for directional coupler cell
        elif mode==1:
            self.upper_wg: dict = {
                    "base height" : top_height,
                    "base width" : self.bot_width_calc(base_angle,
                                                top_width, top_height),
                    "base angle" : base_angle,
                    "material" : wg_material,
                    "y span" : bend_height,
                    "Ls" : bend_length,
                    "gap" : gap,
                    "Lc" : L
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
                    "Lc" : L
                    }
            
            self.insulation: dict = {
                    "x span" : bend_length*2 + L,
                    "y span" : bend_height*4 + gap,
                    "material" : insul_material
                    }
        # parameters for MMI coupler cell
        elif mode==2:
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
                    "gap" : (gap+taper_width)/2,
                    "Lc" : L
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
                    "gap" : (gap+taper_width)/2,
                    "Lc" : L
                    }

            self.insulation: dict = {
                    "x span" : bend_length*2 + 2*L +\
                                2* mmi_taper_length + self.upper_wg["mmi length"],
                    "y span" : bend_height*4 + gap,
                    "material" : insul_material
                    }

# --------------------------------------------
# building blocks
# --------------------------------------------

def single_straight(parent: str, mode: str,
                    z_pos: float, x_pos: float, y_pos: float,
                    base_height: float, base_width: float, base_angle: float,
                    material: str, Lw: float) -> str:
    """
    Function to build single waveguide of triplex scheme in Ansys.

    **All parameters of length to be added in microns!**

    ---

    Args:
        parent (str): the name of the structure group in which this one
                        will be added
        mode (str): name of this waveguide mode
        z_pos (float): position of the object in the `z` direction
        x_pos (float): starting position of object in `x` direction
        y_pos (float): starting position of object in `y` direction
        base_height (flaot): the height of the wgd
        base_width (float): the width of the base (bottom of the waveguide)
        base_angle (flaot): the angle between the base and the sides
        material (str): material of the waveguide
        Lw (float): length of the straight waveguide
    Returns:
        The script to evaluate defined straight waveguide by selected Ansys solver.
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
    a rectangle around the given chosen structure of given parameters.
    
    **All parameters of length to be added in microns!**
    
    ---
    
    Args:
        x_span (float): the total distance in `x` that the rectangle spans
        y_span (float): the total distance in `y` that the rectangle spans
        material (str): the material of the cladding
        z_mid (float): the center position of the rectangle in `z`
        z_span (float): the span of the rectangle in `z`
        parent (str): the name of the structure group in which this one
                        will be located
        groupname (str): the name of this structure group
    Returns:
        The script to evaluate defined cladding by selected Ansys solver.
    """
    script = "addstructuregroup;\n"\
            +"set('name', '"+groupname+"');\n"\
            +"addtogroup('"+parent+"');\n"\
            +"set(\"z\", "+str(z_mid)+"*1e-6);\n"\
            +"addrect;\n"\
            +"addtogroup('"+parent+"::"+groupname+"');\n"\
            +"set(\"x min\",0);\n"\
            +"set(\"x max\","+str(x_span)+"*1e-6);\n"\
            +"set(\"y min\","+str(-y_span/2)+"*1e-6);\n"\
            +"set(\"y max\","+str(y_span/2)+"*1e-6);\n"\
            +"set(\"z\", 0);\n"\
            +"set(\"z span\","+str(z_span)+"*1e-6);\n"\
            +"set(\"material\", \""+material+"\");\n"\
            +"set('name', 'cladding_"+str(idx)+"');\n"

    return script

def coupler(parent: str, mode1: str, mode2: str,
            z_pos: float, x_pos: float, y_pos: float,
            base_height: float, base_width: float, base_angle: float,
            material: str, y_span: float, Ls: float,
            gap: float, Lc: float, ) -> str:
    """
    This function makes the script for directional couplers in selected Ansys solver.

    **All parameters of length to be added in microns!**

    ---

    Args:
        parent (str): the name of the structure group in which to
                        move the coupler object
        mode1 (str): name of the upper mode in the coupler
        mode2 (str): name of the lower mode in the coupler
        z_pos (float): position of the coupler object in the `z` direction
        x_pos (float): starting position of object in `x` direction
        y_pos (float): starting position of upper mode in `y` direction
        y_pos_adj (float): the adjustment of the wgds center in `y` direction
        base_height (flaot): the height of the wgd
        base_width (float): the width of the base (bottom of the waveguide)
        base_angle (flaot): the angle between the base and the sides
        material (str): material of the waveguide
        y_span (float): center-to-center distance between the two inputs
        Ls (float): x span of the individual s-bend sections
        gap (float): gap between two coupling sections (center of waveguides are aligned in `z`)
        Lc (float): length of the central straight waveguide sections for coupling
    Returns:
        The script to evaluate defined directional coupler by selected Ansys solver.
    """

    # mode 1
    script = "addstructuregroup;\n"\
            +"set('name', '"+parent+" 1');\n"\
            +"addtogroup('"+mode1+"');\n"\
            + BentWGD(base_width, base_height,
                base_angle, material,
                "["+str(x_pos)+","+str(y_pos+y_span+gap)+";"\
                +str(x_pos+Ls/2)+","+str(y_pos+y_span+gap)+";"\
                +str(x_pos+Ls/2)+","+str(y_pos+gap)+";"\
                +str(x_pos+Ls)+","+str(y_pos+gap)+"]").builder(name="S-bend in mode_1 "+parent)\
            +"addtogroup('"+mode1+"::"+parent+" 1');\n"\
            + StraightWGD(base_width, base_height,
                        base_angle, material,
                        "["+str(x_pos+Ls)+","+str(y_pos+gap)+";"\
                        +str(x_pos+Ls+Lc)+","+str(y_pos+gap)+"]").builder(name="straight coupler mode_1 "+parent)\
            +"addtogroup('"+mode1+"::"+parent+" 1');\n"\
            + BentWGD(base_width, base_height,
                    base_angle, material,
                    "["+str(x_pos+Ls+Lc)+","+str(y_pos+gap)+";"\
                    +str(x_pos+3*Ls/2+Lc)+","+str(y_pos+gap)+";"\
                    +str(x_pos+3*Ls/2+Lc)+","+str(y_pos+y_span+gap)+";"\
                    +str(x_pos+2*Ls+Lc)+","+str(y_pos+y_span+gap)+"]").builder(name="S-bend out mode_1 "+parent)\
            +"addtogroup('"+mode1+"::"+parent+" 1');\n"
    # mode 2
    script += "addstructuregroup;\n"\
            +"set('name', '"+parent+" 2');\n"\
            +"addtogroup('"+mode2+"');\n"\
            + BentWGD(base_width, base_height,
                    base_angle, material,
                    "["+str(x_pos)+","+str(-y_span-gap+y_pos)+";"\
                    +str(x_pos+Ls/2)+","+str(-y_span-gap+y_pos)+";"\
                    +str(x_pos+Ls/2)+","+str(-gap+y_pos)+";"\
                    +str(x_pos+Ls)+","+str(-gap+y_pos)+"]").builder(name="S-bend in mode_2 "+parent)\
            +"addtogroup('"+mode2+"::"+parent+" 2');\n"\
            + StraightWGD(base_width, base_height,
                        base_angle, material,
                        "["+str(x_pos+Ls)+","+str(-gap+y_pos)+";"\
                        +str(x_pos+Ls+Lc)+","+str(-gap+y_pos)+"]").builder(name="straight coupler mode_2 "+parent)\
            +"addtogroup('"+mode2+"::"+parent+" 2');\n"\
            + BentWGD(base_width, base_height,
                    base_angle, material,
                    "["+str(x_pos+Ls+Lc)+","+str(y_pos-gap)+";"\
                    +str(x_pos+3*Ls/2+Lc)+","+str(y_pos-gap)+";"\
                    +str(x_pos+3*Ls/2+Lc)+","+str(y_pos-y_span-gap)+";"\
                    +str(x_pos+2*Ls+Lc)+","+str(y_pos-y_span-gap)+"]").builder(name="S-bend out mode_2 "+parent)\
            +"addtogroup('"+mode2+"::"+parent+" 2');\n"

    # positioning
    script += "select('"+mode1+"::"+parent+" 1');\n"\
            +"set(\"z\", "+str(z_pos)+");\n"\
            +"select('"+mode2+"::"+parent+" 2');\n"\
            +"set(\"z\", "+str(z_pos)+");\n"

    return script

def mmi(parent: str, mode1: str, mode2: str, z_pos: float,
            x_pos: float, y_pos: float,
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
        y_pos (float): starting position of upper mode in `y` direction
        base_height (flaot): the height of the wgd
        base_width (float): the width of the base (bottom of the waveguide)
        base_angle (flaot): the angle between the base and the sides
        material (str): material of the waveguide
        y_span (float): center-to-center distance between the two input ports
        gap (float): gap between two coupling sections (with aligned wgd centers in `y` direction)
        Lc (float): length of the connector straight waveguide between bend and taper
        Ls (float): x span of the four individual s-bend sections
    Returns:
        The script to evaluate an MMI coupler design.
    """

    # mode 1
    script = "addstructuregroup;\n"\
            +"set('name', '"+parent+" 1');\n"\
            +"addtogroup('"+mode1+"');\n"\
            + BentWGD(base_width, base_height,
                base_angle, material,
                "["+str(x_pos)+","+str(y_pos+y_span+gap)+";"\
                +str(x_pos+Ls/2)+","+str(y_pos+y_span+gap)+";"\
                +str(x_pos+Ls/2)+","+str(y_pos+gap)+";"\
                +str(x_pos+Ls)+","+str(y_pos+gap)+"]").builder(name="S-bend in " + mode1 +" "+ parent)\
            +"addtogroup('"+mode1+"::"+parent+" 1');\n"\
            + StraightWGD(base_width, base_height,
                        base_angle, material,
                        "["+str(x_pos+Ls)+","+str(y_pos+gap)+";"\
                        +str(x_pos+Ls+Lc)+","+str(y_pos+gap)+"]").builder(name="straight connector in " + mode1 +" "+parent)\
            +"addtogroup('"+mode1+"::"+parent+" 1');\n"\
            + LinearTaper(base_height,base_width,taper_width,taper_length,
                    base_angle,material,
                    x_pos+Ls+Lc+taper_length/2,
                    y_pos+gap).builder(name="taper in " + mode1 +" "+parent)\
            +"addtogroup('"+mode1+"::"+parent+" 1');\n"\
            + LinearTaper(base_height,taper_width,base_width,taper_length,
                    base_angle,material,
                    x_pos+Ls+Lc+1.5*taper_length+mmi_length,
                    y_pos+gap).builder(name="taper out " + mode1 +" "+parent)\
            +"addtogroup('"+mode1+"::"+parent+" 1');\n"\
            + StraightWGD(base_width, base_height,
                        base_angle, material,
                        "["+str(x_pos+Ls+Lc+taper_length*2+mmi_length)+","+str(y_pos+gap)+";"\
                        +str(x_pos+Ls+2*Lc+taper_length*2+mmi_length)+","+str(y_pos+gap)+"]").builder(name="straight connector out " + mode1 +" "+parent)\
            +"addtogroup('"+mode1+"::"+parent+" 1');\n"\
            + BentWGD(base_width, base_height,
                    base_angle, material,
                    "["+str(x_pos+Ls+2*Lc+taper_length*2+mmi_length)+","+str(y_pos+gap)+";"\
                    +str(x_pos+1.5*Ls+2*Lc+taper_length*2+mmi_length)+","+str(y_pos+gap)+";"\
                    +str(x_pos+1.5*Ls+2*Lc+taper_length*2+mmi_length)+","+str(y_pos+gap+y_span)+";"\
                    +str(x_pos+2*Ls+2*Lc+taper_length*2+mmi_length)+","+str(y_pos+gap+y_span)+"]").builder(name="S-bend out " + mode1 +" "+parent)\
            +"addtogroup('"+mode1+"::"+parent+" 1');\n"
    # mode 2
    script += "addstructuregroup;\n"\
            +"set('name', '"+parent+" 2');\n"\
            +"addtogroup('"+mode2+"');\n"\
            + BentWGD(base_width, base_height,
                    base_angle, material,
                    "["+str(x_pos)+","+str(-y_span-gap+y_pos)+";"\
                    +str(x_pos+Ls/2)+","+str(-y_span-gap+y_pos)+";"\
                    +str(x_pos+Ls/2)+","+str(-gap+y_pos)+";"\
                    +str(x_pos+Ls)+","+str(-gap+y_pos)+"]").builder(name="S-bend in " + mode2 +" "+parent)\
            +"addtogroup('"+mode2+"::"+parent+" 2');\n"\
            + StraightWGD(base_width, base_height,
                        base_angle, material,
                        "["+str(x_pos+Ls)+","+str(-gap+y_pos)+";"\
                        +str(x_pos+Ls+Lc)+","+str(-gap+y_pos)+"]").builder(name="straight connector in " + mode2 +" "+parent)\
            +"addtogroup('"+mode2+"::"+parent+" 2');\n"\
            + LinearTaper(base_height,base_width,taper_width,taper_length,
                    base_angle,material,
                    x_pos+Ls+Lc+taper_length/2,
                    y_pos-gap).builder(name="taper in " + mode2 +" "+parent)\
            +"addtogroup('"+mode2+"::"+parent+" 2');\n"\
            + LinearTaper(base_height,taper_width,base_width,taper_length,
                    base_angle,material,
                    x_pos+Ls+Lc+1.5*taper_length+mmi_length,
                    y_pos-gap).builder(name="taper out " + mode2 +" "+parent)\
            +"addtogroup('"+mode2+"::"+parent+" 2');\n"\
            + StraightWGD(base_width, base_height,
                        base_angle, material,
                        "["+str(x_pos+Ls+Lc+taper_length*2+mmi_length)+","+str(y_pos-gap)+";"\
                        +str(x_pos+Ls+2*Lc+taper_length*2+mmi_length)+","+str(y_pos-gap)+"]").builder(name="straight connector out " + mode2 +" "+parent)\
            +"addtogroup('"+mode2+"::"+parent+" 2');\n"\
            + BentWGD(base_width, base_height,
                    base_angle, material,
                    "["+str(x_pos+Ls+2*Lc+taper_length*2+mmi_length)+","+str(y_pos-gap)+";"\
                    +str(x_pos+1.5*Ls+2*Lc+taper_length*2+mmi_length)+","+str(y_pos-gap)+";"\
                    +str(x_pos+1.5*Ls+2*Lc+taper_length*2+mmi_length)+","+str(y_pos-y_span-gap)+";"\
                    +str(x_pos+2*Ls+2*Lc+taper_length*2+mmi_length)+","+str(y_pos-y_span-gap)+"]").builder(name="S-bend out " + mode2 +" "+parent)\
            +"addtogroup('"+mode2+"::"+parent+" 2');\n"

    script += "addstructuregroup;\n"\
            + "set('name', 'MMI "+parent+"');\n"\
            + MMI(mmi_width, base_height, mmi_length,
                base_angle, material,
                x_pos+Ls+Lc+taper_length).builder(name="MMI "+parent)\
            + "addtogroup('MMI "+parent+"');\n"\

    # positioning
    script += "select('"+mode1+"::"+parent+" 1');\n"\
            +"set(\"z\", "+str(z_pos)+"* 1e-6);\n"\
            +"select('"+mode2+"::"+parent+" 2');\n"\
            +"set(\"z\", "+str(z_pos)+"* 1e-6);\n"\
            +"select('MMI "+parent+"');\n"\
            +"set(\"z\", "+str(z_pos)+"* 1e-6);\n"

    return script

# --------------------------------------------
# layout extraction
# --------------------------------------------

def layout(bot_wg_height: float, top_wg_height: float,
            dist: float, wg_material: str,
            path: str, name: str) -> str:
    """
    This function prepares the data and extracts the layout
    of built objects.
    Use it after geometry is built and while it is open (inside the file).
    
    **All parameters of length to be added in microns!**
    
    ---
    
    
    Args:
        bot_wg_height (float): the height of the lower waveguide
        top_wg_height (float): the height of the upper waveguide
        dist (float): the distance between the two waveguide layer
        wg_material (str): the material of the waveguide
        path (str): the path of the Ansys script that manages the automated
                    layout extraction
        name (str): the name of the GDS file to be generated
    Returns:
        String of instructions for Ansys to evaluate layout extraction.
    """
    
    # prepare structure
    struct: dict = {
        bot_wg_height/2 : wg_material,
        dist + bot_wg_height/2 + top_wg_height - 1e-4 : wg_material    #adjustment for unknown reaason but it works at this level
    }
    
    
    # loads automation code written by Ansys group.
    # For more see: https://optics.ansys.com/hc/en-us/articles/1500006203341-GDSII-Export-Automation
    script: str = "feval('"+path+"/GDSexport_core.lsfx');\n"\
                + "feval('"+path+"/GDS_auto_export.lsf');\n"

    script += "layer_def = cell("+str(len(struct))+");\n"

    idx = 1
    for z_pos in struct.keys():
        # layer_name = str(struct[z_pos].split(" ")[0])
        script += "layer_"+str(idx)+" = struct;\n"\
                + "layer_"+str(idx)+".z = "+str(z_pos)+"* 1e-6;\n"\
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

# --------------------------------------------
# main functions to call for building simulations
# --------------------------------------------

def single_straight_wgd_builder(lower_wg: dict, upper_wg: dict,
                        dist: float, insulation: dict,
                        insul_height_bot: float, insul_height_top: float,
                        mode_idx: int,
                        name: str, version: str,
                        layout_extract: bool = False,
                        hideAnsys: bool = True,
                        path: str = "C:/Users/Admin/Desktop/Interferometer/gds_auto_export_package/gds_auto_export_package") -> None:
    """
    Prepares script to build a single waveguide of the triplex scheme
    in Ansys.
    
    **All parameters of length to be added in microns!**

    ---

    Args:
        lower_wg (dict): dictionary of "lower layer" waveguide's parameters
        upper_wg (dict): dictionary of "upper layer" waveguide's parameters
        dist (float): distance between the two layers in `z`
        insulation (dict): dictionary of cladding's parameters
        insul_height_bot (float): the height of the cladding on top of the upper waveguide
        insul_height_top (float): the height of the cladding below the lower waveguide
        mode_idx (int): the index of the waveguide mode
        name (str): the name of the Ansys file to which it is created
        version (str): the Ansys solver used in which to build the system
            
            Available options:
            - MODE
            - FDTD
        layout_extract (bool): whether to extract the system's layout or not
        hideAnsys (bool): whether to open Ansys or not during evaluation
        path (str): the path for the Ansys layout extraction automation code base
    Note
    ---
    Parameter keys of the waveguide dictionaries (upper/lower) are strings and their values are:
        **base height** : *float*
        the height of the wgd
        **base width** : *float*
        the width of the base (bottom of the waveguide)
        **base angle** : *float*
        the angle between the base and the sides
        **material** : *str*
        material of the waveguide
        **Lw** : *float*
        the length of the straight waveguide
    
    Parameter keys of the cladding (/insulation) dictionary are strings and their values are:
        **x span** : *float*
        the total span of the cladding in `x` dimension
        **y span** : *float*
        the total span of the cladding in `y` dimension
        **material** : *str*
        material of the cladding

    Returns:
        Build desired design and, if chosen, extracts the layout of it into a GDS file.
    """

    # distance in `z` between two layers of Si3N4
    dist_gap = dist + lower_wg["base height"]/2 + upper_wg["base height"]/2
    insul_height_bot = insul_height_bot
    insul_height_top = insul_height_top

    lower_parent = "lower layer"
    upper_parent = "upper layer"
    cover_parent = "cladding"
    mode = "mode_{}".format(mode_idx)

    script: str = "groupscope('::model');\n"
    script += "addstructuregroup;\n"\
            +"set('name', '"+mode+"');\n"

    # upper wg
    script += single_straight(upper_parent, mode, dist_gap, 0, 0,
                            *upper_wg.values())
    # lower wg
    script += single_straight(lower_parent, mode, 0, 0, 0,
                            *lower_wg.values())
    
    # list of insulation and base thicknesses
    layer_thicks = [insul_height_bot,
                    lower_wg["base height"] + dist + upper_wg["base height"]*1e-6,
                    insul_height_top]
    
    groupnames = ["bot clad",
                  "mid clad",
                  "top clad"]
    
    z_min = -(insul_height_bot + 0.5*lower_wg["base height"])

    script += "addstructuregroup;\n"\
            +"set('name', '"+cover_parent+"');\n"
    
    # cladding
    for layer in range(len(layer_thicks)):
        z_max = z_min + layer_thicks[layer]
        z_mid = z_min+(z_max-z_min)/2
    
        script += insulator(*insulation.values(), z_mid, layer_thicks[layer], layer,
                        cover_parent, groupnames[layer])
        z_min = z_max

    if layout_extract:
        script += layout(lower_wg["base height"], upper_wg["base height"],
                        dist, upper_wg["material"], path, name)
        
    mode = mode_select(version, hideAnsys)
    mode.eval(script)
    mode.save(f"{name}")
    
    return 0

def dir_coupler_builder(lower_wg: dict, upper_wg: dict,
                        dist: float, insulation: dict,
                        insul_height_bot: float, insul_height_top: float,
                        top_mode_idx: int, bot_mode_idx: int,
                        name: str, version: str,
                        layout_extract: bool = False,
                        hideAnsys: bool = True,
                        path: str = "C:/Users/Admin/Desktop/Interferometer/gds_auto_export_package/gds_auto_export_package") -> None:
    """
    Prepares script to build directional coupler in the triplex scheme
    in Ansys.
    
    **All parameters of length to be added in microns!**

    ---

    Args:
        lower_wg (dict): dictionary of "lower layer" waveguide's parameters
        upper_wg (dict): dictionary of "upper layer" waveguide's parameters
        dist (float): distance between the two layers in `z`
        insulation (dict): dictionary of cladding's parameters
        insul_height_bot (float): the height of the cladding on top of the upper waveguide
        insul_height_top (float): the height of the cladding below the lower waveguide
        top_mode_idx (int): the index of the upper waveguide mode (in `y` dimension)
        bot_mode_idx (int): the index of the lower waveguide mode (in `y` dimension)
        name (str): the name of the Ansys file to which it is created
        version (str): the Ansys solver used in which to build the system
            
            Available options:
            - MODE
            - FDTD
        layout_extract (bool): whether to extract the system's layout or not
        hideAnsys (bool): whether to open Ansys or not during evaluation
        path (str): the path for the Ansys layout extraction automation code base
    Note
    ---
    Parameter keys of the waveguide dictionaries (upper/lower) are strings and their values are:
        **base height** : *float*
        the height of the wgd
        **base width** : *float*
        the width of the base (bottom of the waveguide)
        **base angle** : *float*
        the angle between the base and the sides
        **material** : *str*
        material of the waveguide
        **y span** : *float*
        the height of the S-bend (in `y` dimension)
        **Ls** : *float*
        the length of the S-bend (in `x` dimension)
        **gap** : *float*
        gap between two coupling sections (with aligned top and bot waveguide centers)
        **Lc** : *float*
        the straight waveguide's length that connects the bend with the tapers
    
    Parameter keys of the cladding (/insulation) dictionary are strings and their values are:
        **x span** : *float*
        the total span of the cladding in `x` dimension
        **y span** : *float*
        the total span of the cladding in `y` dimension
        **material** : *str*
        material of the cladding

    Returns:
        Builds desired design and, if chosen, extracts the layout of it into a GDS file.
    """
    
    # distance in `z` between two layers of Si3N4
    dist_gap = dist + lower_wg["base height"]/2 + upper_wg["base height"]/2
    insul_height_bot = insul_height_bot
    insul_height_top = insul_height_top
    
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

    # upper wg
    script += coupler(upper_parent, mode1, mode2, dist_gap, 0, 0,
                    *upper_wg.values())
    # lower wg
    script += coupler(lower_parent, mode1, mode2, 0, 0, 0,
                    *lower_wg.values())

    # list of insulation and base thicknesses
    layer_thicks = [insul_height_bot,
                    lower_wg["base height"] + dist + upper_wg["base height"],
                    insul_height_top]
    
    groupnames = ["bot clad",
                  "mid clad",
                  "top clad"]
    
    z_min = -(insul_height_bot + 0.5*lower_wg["base height"])

    script += "addstructuregroup;\n"\
            +"set('name', '"+cover_parent+"');\n"
    
    # cladding
    for layer in range(len(layer_thicks)):
        z_max = z_min + layer_thicks[layer]
        z_mid = z_min+(z_max-z_min)/2
    
        script += insulator(*insulation.values(), z_mid, layer_thicks[layer], layer,
                        cover_parent, groupnames[layer])
        z_min = z_max

    if layout_extract:
        script += layout(lower_wg["base height"], upper_wg["base height"],
                        dist, upper_wg["material"], path, name)

    mode = mode_select(version, hideAnsys)
    mode.eval(script)
    mode.save(f"{name}")
    
    return 0

def mmi_coupler_builder(lower_wg: dict, upper_wg: dict,
                        dist: float, insulation: dict,
                        insul_height_bot: float, insul_height_top: float,
                        top_mode_idx: int, bot_mode_idx: int,
                        name: str, version: str,
                        layout_extract: bool = False,
                        hideAnsys: bool = True,
                        path: str = "C:/Users/Admin/Desktop/Interferometer/gds_auto_export_package/gds_auto_export_package") -> None:
    """
    Prepares script to build a single waveguide of the triplex scheme
    in Ansys.
    
    **All parameters of length to be added in microns!**

    ---

    Args:
        lower_wg (dict): dictionary of "lower layer" waveguide's parameters
        upper_wg (dict): dictionary of "upper layer" waveguide's parameters
        dist (float): distance between the two layers in `z`
        insulation (dict): dictionary of cladding's parameters
        insul_height_bot (float): the height of the cladding on top of the upper waveguide
        insul_height_top (float): the height of the cladding below the lower waveguide
        top_mode_idx (int): the index of the upper waveguide mode (in `y` dimension)
        bot_mode_idx (int): the index of the lower waveguide mode (in `y` dimension)
        name (str): the name of the Ansys file to which it is created
        version (str): the Ansys solver used in which to build the system
            
            Available options:
            - MODE
            - FDTD
        layout_extract (bool): whether to extract the system's layout or not
        hideAnsys (bool): whether to open Ansys or not during evaluation
        path (str): the path for the Ansys layout extraction automation code base
    Note
    ---
    Parameter keys of the waveguide dictionaries (upper/lower) are strings and their values are:
        **base height** : *float*
        the height of the wgd
        **base width** : *float*
        the width of the base (bottom of the waveguide)
        **taper width** : *float*
        the width of the base (bottom of the waveguide)
        **taper length** : *float*
        the length of the taper
        **mmi base width** : *float*
        the width of the MMI region's base (bottom of the waveguide)
        **mmi length** : *float*
        the length of the MMI region
        **base angle** : *float*
        the angle between the base and the sides
        **material** : *str*
        material of the waveguide
        **y span** : *float*
        the height of the S-bend (in `y` dimension)
        **Ls** : *float*
        the length of the S-bend (in `x` dimension)
        **gap** : *float*
        gap between two coupling sections (with aligned top and bot waveguide centers)
        **Lc** : *float*
        the straight waveguide's length that connects the bend with the tapers
    
    Parameter keys of the cladding (/insulation) dictionary are strings and their values are:
        **x span** : *float*
        the total span of the cladding in `x` dimension
        **y span** : *float*
        the total span of the cladding in `y` dimension
        **material** : *str*
        material of the cladding
    
    Returns:
        Builds desired design and, if chosen, extracts the layout of it into a GDS file.
    """
    # distance in `z` between two layers of Si3N4
    dist_gap = dist + lower_wg["base height"]/2 + upper_wg["base height"]/2
    insul_height_bot = insul_height_bot
    insul_height_top = insul_height_top
    
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
    script += mmi(upper_parent, mode1, mode2, dist_gap, 0, 0,
                *upper_wg.values())
    # lower layer
    script += mmi(lower_parent, mode1, mode2, 0, 0, 0,
                *lower_wg.values())
    
    # list of cladding and base thicknesses
    layer_thicks = [insul_height_bot,
                    lower_wg["base height"] + dist + upper_wg["base height"],
                    insul_height_top]
    
    groupnames = ["bot clad",
                  "mid clad",
                  "top clad"]
    
    z_min = -(insul_height_bot + 0.5*lower_wg["base height"])
    
    script += "addstructuregroup;\n"\
            +"set('name', '"+cover_parent+"');\n"

    # cladding
    for layer in range(len(layer_thicks)):
        z_max = z_min + layer_thicks[layer]
        z_mid = z_min+(z_max-z_min)/2
    
        script += insulator(*insulation.values(), z_mid, layer_thicks[layer], layer,
                        cover_parent, groupnames[layer])
        z_min = z_max

    if layout_extract:
        script += layout(lower_wg["base height"], upper_wg["base height"],
                        dist, upper_wg["material"], path, name)

    mode = mode_select(version, hideAnsys)
    mode.eval(script)
    mode.save(f"{name}")
    
    return 0

# --------------------------------------------
# complete circuit extracter (?)
# --------------------------------------------

def circuit() -> None:
    """
    This function puts together the final circuit.
    """
    
    
    
    
    
    ...

