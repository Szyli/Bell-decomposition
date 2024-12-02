
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
                +"set(\"material\", \""+self.material+"\");\n"
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
                +"set(\"material\", \""+self.material+"\");\n"
        return script
        

def coupler(parent, mode1, mode2,
            base_width, base_height, base_angle,
            material, y_span, gap,
            Ls, Lw, Lc, y_pos):
    
    y_pos = y_pos*1e-6

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
            +"set(\"z\", "+str(y_pos)+");\n"

    return script


def builder():
    # parameters
    base_height = 0.175
    base_width = 1.149
    base_angle = 82
    material = "Si3N4 (Silicon Nitride) - Luke"
    y_span = 8
    gap = 0.4
    Lc = 3
    Ls = 5
    Lw = 3
    # Lc: length of the central straight waveguide sections for coupling
    # Lw: length of the four individual straight waveguide sections 
    #     at the input and output ports
    # Ls: x span of the four individual s-bend sections
    # gap: gap between two coupling sections
    # y_span: center-to-center distance between the two input ports
    parent1 = "upper layer"
    parent2 = "lower layer"
    mode1 = "mode_1"
    mode2 = "mode_2"

    script: str = "groupscope('::model');\n"

    script += coupler(parent1,mode1,mode2,
                    base_width,base_height,base_angle,
                    material,y_span,gap,Ls,Lw,Lc,0)
    
    script += coupler(parent2,mode1,mode2,
                    base_width,base_height,base_angle,
                    material,y_span,gap,Ls,Lw,Lc,0.21385)

    
        




    mode = MODE(hide = True)
    mode.eval(script)
    mode.save(f"couplers.lms")







