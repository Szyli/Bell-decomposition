"""
    This file is responsible to handle the execution of the
    whole project.
    
    @author: Csaba Szilard Racz //Szyli
"""

from customtkinter import *
# from CTkPDFViewer import *
# import sMZI as itf



class DisplayFrame(CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.label = CTkLabel(self, text="Output will appear here", font=("Ariel", 18))
        self.label.pack(pady=20)

    def update_label(self, new_text):
        """Update the label in this frame with new_text."""
        self.label.configure(text=new_text)


class TitleFrame(CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        
        CTkLabel(self,text="This application executes the "\
                +"decomposition of given unitary matrix of size NxN based "\
                +"on the scheme by B.A.Bell et.all.",
                font=("Ariel", 22),corner_radius=5,fg_color="#0058B5",
                height=40).grid(row=0, column=0,
                                padx=0, pady=(10,0),
                                sticky="nwe", columnspan=2)

class OptionsFrame(CTkFrame):
    def __init__(self, master, display_frame):
        super().__init__(master)

        intro_button = CTkButton(self,text="Introduction",font=("Ariel",20),
                                fg_color="#0058B5", command=self.button_callback)
        intro_button.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        # Todo: what happens when we press
        intro_button = CTkButton(self,text="Decomposition and circuit diagram",
                                font=("Ariel",20),fg_color="#0058B5")
        intro_button.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="w")
        # Todo: what happens when we press
        intro_button = CTkButton(self,text="Simulations",font=("Ariel",20),
                                fg_color="#0058B5")
        intro_button.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="w")
        # Todo: what happens when we press
        
    def button_callback(self):
        """Callback to print a message on the adjacent frame."""
        self.display_frame.update_label("Hello, World!")
        
        
class PDFViewer():
    ...
    # root = CTk()
    # root.geometry("700x600")
    # pdf_frame = CTkPDFViewer(root, file="my_file.pdf")
    # pdf_frame.pack(fill="both", expand=True, padx=10, pady=10)
    # root.mainloop()

class App(CTk):
    def __init__(self):
        super().__init__()

        width = self.winfo_screenwidth()
        height = self.winfo_screenheight()

        x_coord = -10
        y_coord = 0
        # print(width,height,x_coord,y_coord)
        self.geometry("{}x{}+{}+{}".format(width, height, x_coord, y_coord))
        self.attributes("-topmost", True)  # sets above every other window

        set_appearance_mode("system")
        set_default_color_theme("blue")

        self.title("Programmable Integrated Quantum Photonic Circuit")
        
        self.grid_columnconfigure((0,1), weight=1)
        self.grid_rowconfigure((0), weight=0)
        self.grid_rowconfigure((1), weight=1)
        
        self.title_frame = TitleFrame(self)
        self.title_frame.grid(row=0, column=0, padx=10, pady=(10, 0),
                                sticky="N", columnspan=2)

        self.options_frame = OptionsFrame(self, None)
        self.options_frame.grid(row=1, column=0, padx=10, pady=(10, 0),
                                sticky="nsw")
        
        

        self.display_frame = DisplayFrame(self)
        self.display_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        # Link the frames
        self.options_frame.display_frame = self.display_frame




# creating application window
root = App()

root.mainloop()
