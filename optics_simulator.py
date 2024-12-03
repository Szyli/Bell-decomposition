
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
            +"set(\"y min\","+str(-param["y span"]/2 - param["base width"]/2)+"*1e-6);\n"\
            +"set(\"y max\","+str(param["y span"]/2 + param["base width"]/2)+"*1e-6);\n"\
            +"set(\"z min\","+str(-1)+"*1e-6);"\
            +"set(\"z max\","+str(dist+1)+"*1e-6);"\
            +"set(\"material\", \""+material+"\");\n"\
            +"set('name', 'insulator');\n"
    
    return script


def dir_coupler_builder(param_1: dict, param_2: dict,
                        dist: float, material: str) -> None:
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
    script += insulator(param_1, dist, material)
    
    


    mode = MODE(hide = True)
    mode.eval(script)
    mode.save(f"couplers.lms")







