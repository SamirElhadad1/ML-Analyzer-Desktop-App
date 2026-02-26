import customtkinter as ctk
from tkinter import messagebox
from GUI.page_1 import page_1
from GUI.page_2 import page_2
from GUI.page_3 import page_3

def switch_frame(master, new_frame_class, **kwargs):
    for widget in master.winfo_children():
        widget.destroy()
    try:
        if isinstance(new_frame_class, str):
            module_path, class_name = new_frame_class.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            frame_class = getattr(module, class_name)
        else:
            frame_class = new_frame_class
        frame = frame_class(master, **kwargs)
        frame.pack(fill='both', expand=True)
    except (TypeError, ImportError, AttributeError) as e:
        messagebox.showerror("Error", f"Failed to switch frame: {str(e)}")

def main():
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    
    root = ctk.CTk()
    root.title("ML Analyzer: Predict, Classify & Cluster")
    root.geometry("1000x700")
    root.minsize(800, 600)

    try:
        root.iconbitmap("assets/icon.ico")
    except:
        pass

    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - 1000) // 2
    y = (screen_height - 700) // 2
    root.geometry(f"1000x700+{x}+{y}")

    switch_frame(root, page_1, switch_frame=switch_frame)
    
    try:
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()