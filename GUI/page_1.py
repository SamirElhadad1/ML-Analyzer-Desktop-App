import customtkinter as ctk
from tkinter import messagebox

class page_1(ctk.CTkFrame):
    def __init__(self, master, switch_frame):
        super().__init__(master, fg_color="#f0f4f8")
        self.switch_frame = switch_frame
        self.create_widgets()

    def create_widgets(self):
        # Main container
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(expand=True, fill="both")
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)

        # Center frame
        center_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        center_frame.grid(row=0, column=0, sticky="nsew")

        # Title
        self.title_label = ctk.CTkLabel(
            center_frame,
            text="ML Analyzer",
            font=ctk.CTkFont(family="Roboto", size=36, weight="bold"),
            text_color="#2C3E50"
        )
        self.title_label.pack(pady=(100, 10))

        # Subtitle
        self.subtitle_label = ctk.CTkLabel(
            center_frame,
            text="Predict, Classify & Cluster Any Dataset with Ease",
            font=ctk.CTkFont(family="Roboto", size=16),
            text_color="#4A5568"
        )
        self.subtitle_label.pack(pady=(0, 20))

        # Start Button
        self.start_button = ctk.CTkButton(
            center_frame,
            text="Start Analysis",
            command=self.start_analysis,
            width=200,
            height=50,
            font=ctk.CTkFont(family="Roboto", size=16, weight="bold"),
            fg_color="#28A745",
            hover_color="#218838"
        )
        self.start_button.pack(pady=20)

        # Status Bar
        self.status_var = ctk.StringVar(value="Welcome to ML Analyzer")
        self.status_bar = ctk.CTkLabel(
            self,
            textvariable=self.status_var,
            font=ctk.CTkFont(family="Roboto", size=12),
            anchor="w",
            fg_color="#e0e6ed",
            corner_radius=0
        )
        self.status_bar.pack(side="bottom", fill="x", padx=10, pady=5)

    def start_analysis(self):
        self.status_var.set("Navigating to Upload Page...")
        self.start_button.configure(state="disabled")
        self.after(300, self._navigate_to_page_2)

    def _navigate_to_page_2(self):
        try:
            from GUI.page_2 import page_2
            self.switch_frame(self.master, page_2, switch_frame=self.switch_frame)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to navigate: {str(e)}")
            self.start_button.configure(state="normal")
            self.status_var.set("Navigation failed")