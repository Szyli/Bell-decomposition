
import numpy as np
# include ANSYS
import sys, os
# Note: This path may very for users, therefore
# Todo: Update to find path corresponding to limerical
sys.path.append("D:\\lumerical\\api\\python")
sys.path.append(os.path.abspath(""))

from lumapi import MODE

class StraightWGD:
    def __init__(self, base_width,
                base_height, base_angle,
                material, poles: str):
        
        self.width = base_width * 1e-6
        self.height = base_height * 1e-6
        self.angle = base_angle
        self.material = material
        self.pole = poles + " * 1e-6"
        
    def builder(self):
        
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
    def __init__(self, base_width,
                base_height, base_angle,
                material, poles: str):
        
        self.width = base_width * 1e-6
        self.height = base_height * 1e-6
        self.angle = base_angle
        self.material = material
        self.pole = poles + " * 1e-6"
        
    def builder(self):
        
        script = "addwaveguide;\n"\
                +"set(\"base width\", "+str(self.width)+");\n"\
                +"set(\"base height\", "+str(self.height)+");\n"\
                +"set(\"base angle\", "+str(self.angle)+");\n"\
                +"set(\"poles\", "+self.pole+");\n"\
                +"set(\"material\", \""+self.material+"\");\n"\
                +"set(\"override mesh order from material database\",1);\n"\
                +"set(\"mesh order\",1);\n"
        return script
        

def coupler(parent, mode1, mode2, z_pos,
            base_height, base_width, base_angle,
            material, y_span, gap,
            Lc, Ls, Lw):
    
    z_pos = z_pos*1e-6

    script = "addstructuregroup;\n"\
            +"set('name', '"+parent+"');\n"\

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

def insulator(param: dict, dist, material):
    
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

def eme_simulator(y_span, z_span, Lc, Ls, Lw, y_pos, z_pos):


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


def dir_coupler_builder(param_1: dict, param_2: dict,
                        dist: float, material: str, addEMEsolver: bool= True) -> None:
    """
    Builds directional coupler in triplex scheme.
    
    ---

    Args:
        param_1 (dict): Dictionary of "lower layer" waveguide's parameters
        param_2 (dict): Dictionary of "upper layer" waveguide's parameters
        dist (float): distance between the two layers (um)
    
    ---

    Note
    ---
    parameter keys:
    > - base height
    > - base width
    > - base angle
    > - material
    > - y span
    > - gap
    > - Lc
    > - Ls
    > - Lw
    """
    # parameters
    # Lc: length of the central straight waveguide sections for coupling
    # Lw: length of the four individual straight waveguide sections 
    #     at the input and output ports
    # Ls: x span of the four individual s-bend sections
    # gap: gap between two coupling sections
    # y_span: center-to-center distance between the two input ports
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

