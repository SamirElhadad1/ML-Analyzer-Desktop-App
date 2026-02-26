import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from GUI.page_1 import page_1
from Ml.predict import make_prediction, PredictionError
from Ml.preprocess import preprocess_data, PreprocessingError
from tkinter import ttk

class page_2(ctk.CTkFrame):
    def __init__(self, master, switch_frame):
        super().__init__(master, fg_color="#f5f5f5")
        self.switch_frame = switch_frame
        self.df = None
        self.processed_df = None
        self.preprocessing_artifacts = None
        self.task = ctk.StringVar()
        self.algorithm = ctk.StringVar()
        self.target_column = ctk.StringVar()
        self.create_widgets()

    def create_widgets(self):
        main_container = ctk.CTkFrame(self, fg_color="#ffffff", corner_radius=10)
        main_container.pack(expand=True, fill="both", padx=15, pady=10)

        # Title
        ctk.CTkLabel(
            main_container,
            text="Configure Dataset & Model",
            font=ctk.CTkFont(family="Roboto", size=18, weight="bold")
        ).pack(pady=10)

        # Upload Section
        upload_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        upload_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(
            upload_frame,
            text="Upload CSV",
            command=self.load_csv,
            width=150,
            font=ctk.CTkFont(family="Roboto", size=14),
            fg_color="#007BFF",
            hover_color="#0056b3"
        ).pack(side="left", padx=5)

        # Columns and Config
        config_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        config_frame.pack(fill="x", padx=10, pady=5)

        # Columns Display
        columns_frame = ctk.CTkFrame(config_frame, fg_color="#ffffff", corner_radius=8)
        columns_frame.pack(side="left", fill="y", padx=5)
        ctk.CTkLabel(
            columns_frame,
            text="Columns",
            font=ctk.CTkFont(family="Roboto", size=14, weight="bold")
        ).pack(padx=10, pady=5)
        self.columns_listbox = ctk.CTkTextbox(
            columns_frame,
            height=100,
            width=300,
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.columns_listbox.pack(padx=5, pady=5)

        # Configuration Options
        options_frame = ctk.CTkFrame(config_frame, fg_color="#ffffff", corner_radius=8)
        options_frame.pack(side="left", fill="both", expand=True, padx=5)

        # Target Column
        target_frame = ctk.CTkFrame(options_frame, fg_color="transparent")
        target_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(
            target_frame,
            text="Target Column",
            font=ctk.CTkFont(family="Roboto", size=14, weight="bold")
        ).pack(anchor="w", padx=10)
        self.target_combo = ctk.CTkComboBox(
            target_frame,
            variable=self.target_column,
            state="readonly",
            width=300,
            font=ctk.CTkFont(family="Roboto", size=12)
        )
        self.target_combo.pack(anchor="w", padx=15, pady=2)

        # Task Selection
        task_frame = ctk.CTkFrame(options_frame, fg_color="transparent")
        task_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(
            task_frame,
            text="Task Type",
            font=ctk.CTkFont(family="Roboto", size=14, weight="bold")
        ).pack(anchor="w", padx=10)
        for task, label in [
            ("classification", "Classification"),
            ("regression", "Regression"),
            ("clustering", "Clustering")
        ]:
            ctk.CTkRadioButton(
                task_frame,
                text=label,
                variable=self.task,
                value=task,
                command=self.update_algorithms,
                font=ctk.CTkFont(family="Roboto", size=12)
            ).pack(side="left", padx=15)

        # Algorithm Selection
        algo_frame = ctk.CTkFrame(options_frame, fg_color="transparent")
        algo_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(
            algo_frame,
            text="Algorithm",
            font=ctk.CTkFont(family="Roboto", size=14, weight="bold")
        ).pack(anchor="w", padx=10)
        self.algo_combo = ctk.CTkComboBox(
            algo_frame,
            variable=self.algorithm,
            state="readonly",
            width=300,
            font=ctk.CTkFont(family="Roboto", size=12)
        )
        self.algo_combo.pack(anchor="w", padx=15, pady=2)

        # Data Preview
        preview_frame = ctk.CTkFrame(main_container, fg_color="#ffffff", corner_radius=8)
        preview_frame.pack(fill="both", expand=True, padx=10, pady=5)
        ctk.CTkLabel(
            preview_frame,
            text="Data Preview",
            font=ctk.CTkFont(family="Roboto", size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        self.preview_tree = ttk.Treeview(preview_frame, show="headings", height=6)
        self.preview_tree.pack(side="left", fill="both", expand=True, padx=5)
        scrollbar_y = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_tree.yview)
        scrollbar_x = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.preview_tree.xview)
        self.preview_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        scrollbar_y.pack(side="right", fill="y")
        scrollbar_x.pack(side="bottom", fill="x")

        # Navigation Buttons
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(side="bottom", fill="x", padx=20, pady=10)
        ctk.CTkButton(
            button_frame,
            text="Reset",
            command=self.reset,
            width=120,
            font=ctk.CTkFont(family="Roboto", size=14),
            fg_color="#FF6D00",
            hover_color="#e65c00"
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            button_frame,
            text="Back",
            command=lambda: self.switch_frame(self.master, page_1, switch_frame=self.switch_frame),
            width=120,
            font=ctk.CTkFont(family="Roboto", size=14),
            fg_color="#6C757D",
            hover_color="#5A6268"
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            button_frame,
            text="Continue",
            command=self.goto_train,
            width=120,
            font=ctk.CTkFont(family="Roboto", size=14),
            fg_color="#28A745",
            hover_color="#218838"
        ).pack(side="right", padx=5)

        # Status Bar
        self.status_var = ctk.StringVar(value="Welcome to ML Analyzer")
        self.status_bar = ctk.CTkLabel(
            self,
            textvariable=self.status_var,
            font=ctk.CTkFont(family="Roboto", size=12),
            anchor="w",
            fg_color="#e0e6ed",
            corner_radius=6,
            height=30,
            width=400
        )
        self.status_bar.pack(pady=(10, 30))


    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                if self.df.empty:
                    raise ValueError("CSV file is empty")
                self.columns_listbox.delete("1.0", "end")
                columns = list(self.df.columns)
                self.columns_listbox.insert("end", "\n".join(columns))
                self.target_combo.configure(values=columns)
                self.target_column.set(columns[-1] if columns else "")
                self.update_preview()
                self.set_status("Dataset loaded successfully", "green")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
                self.set_status("Error loading dataset", "red")

    def validate_dataset(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please upload a CSV file first")
            self.set_status("No dataset loaded", "red")
            return
        try:
            self.processed_df = self.preprocess_data()
            if self.processed_df is not None:
                messagebox.showinfo("Success", "Dataset is valid and ready for processing")
                self.set_status("Dataset validated", "green")
            else:
                raise PreprocessingError("Preprocessing failed")
        except Exception as e:
            messagebox.showerror("Error", f"Dataset validation failed: {str(e)}")
            self.set_status("Validation failed", "red")

    def update_preview(self):
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
        if self.df is not None:
            df = self.df.head(10)
            self.preview_tree["columns"] = list(df.columns)
            for col in df.columns:
                self.preview_tree.heading(col, text=col)
                self.preview_tree.column(col, width=100, anchor="center")
            for i, row in df.iterrows():
                self.preview_tree.insert("", "end", values=list(row), tags=("even" if i % 2 == 0 else "odd"))
            self.preview_tree.tag_configure("even", background="#f0f0f0")
            self.preview_tree.tag_configure("odd", background="#ffffff")

    def update_algorithms(self):
        task = self.task.get()
        if task == "classification":
            self.algo_combo.configure(values=["KNN", "SVM", "Decision Tree"])
        elif task == "regression":
            self.algo_combo.configure(values=["Linear Regression"])
        elif task == "clustering":
            self.algo_combo.configure(values=["KMeans"])
        else:
            self.algo_combo.configure(values=[])
        self.algorithm.set("")

    def preprocess_data(self):
        if self.df is None:
            return None
        try:
            target_col = self.target_column.get() if self.task.get() != "clustering" else None
            self.processed_df, self.preprocessing_artifacts = preprocess_data(
                df=self.df,
                target_column=self.target_column.get(),
                handle_missing='mean',
                encoding_method='label',
                scale_numeric=True
            )
            return self.processed_df
        except PreprocessingError as e:
            messagebox.showerror("Error", str(e))
            return None

    def reset(self):
        self.df = None
        self.processed_df = None
        self.preprocessing_artifacts = None
        self.task.set("")
        self.algorithm.set("")
        self.target_column.set("")
        self.columns_listbox.delete("1.0", "end")
        self.target_combo.configure(values=[])
        self.algo_combo.configure(values=[])
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
        self.set_status("Configuration reset", "blue")

    def goto_train(self):
        if not self.task.get():
            messagebox.showerror("Error", "Please select a task type")
            self.set_status("Missing task type", "red")
            return
        if not self.algorithm.get():
            messagebox.showerror("Error", "Please select an algorithm")
            self.set_status("Missing algorithm", "red")
            return
        if self.task.get() != "clustering" and not self.target_column.get():
            messagebox.showerror("Error", "Please select a target column")
            self.set_status("Missing target column", "red")
            return
        if self.df is None:
            messagebox.showerror("Error", "Please upload a CSV file")
            self.set_status("Missing dataset", "red")
            return
        self.processed_df = self.preprocess_data()
        if self.processed_df is None:
            self.set_status("Preprocessing failed", "red")
            return
        from GUI.page_3 import page_3
        self.switch_frame(
            self.master,
            page_3,
            switch_frame=self.switch_frame,
            df=self.processed_df,
            task=self.task.get(),
            algorithm=self.algorithm.get(),
            target_column=self.target_column.get(),
            preprocessing_artifacts=self.preprocessing_artifacts
        )

    def set_status(self, message, color):
        self.status_var.set(message)
        self.status_bar.configure(text_color=color)