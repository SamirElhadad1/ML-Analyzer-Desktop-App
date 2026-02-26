import customtkinter as ctk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from GUI.page_1 import page_1
from GUI.page_2 import page_2
from Ml.predict import make_prediction, PredictionError

class page_3(ctk.CTkFrame):
    def __init__(self, master, df, task, algorithm, target_column, preprocessing_artifacts, switch_frame):
        super().__init__(master, fg_color="#f0f4f8")
        self.df = df
        self.task = task
        self.algorithm = algorithm
        self.target_column = target_column
        self.preprocessing_artifacts = preprocessing_artifacts
        self.switch_frame = switch_frame
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.feature_columns = [col for col in df.columns if col != target_column]
        self.create_widgets()
        self.train_model()

    def create_widgets(self):
        self.train_button = ctk.CTkButton(self, text="Train", command=self.train_model)
        self.train_button.pack(pady=10)
        self.metrics_text = ctk.CTkTextbox(self, height=100, width=400)
        self.metrics_text.pack(pady=10)
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.plot_canvas = None
        # ... you can add more widgets here as needed ...

    def train_model(self):
        self.train_button.configure(state="disabled")
        self.set_status("Training model...", "blue")
        try:
            if self.task == "clustering":
                X = self.df
                if self.algorithm == "KMeans":
                    self.model = KMeans(n_clusters=3, random_state=42)
                self.model.fit(X)
                self.y_pred = self.model.labels_
            else:
                X = self.df.drop(columns=[self.target_column])
                y = self.df[self.target_column]
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                if self.task == "classification":
                    if self.algorithm == "KNN":
                        self.model = KNeighborsClassifier()
                    elif self.algorithm == "SVM":
                        self.model = SVC()
                    elif self.algorithm == "Decision Tree":
                        self.model = DecisionTreeClassifier()
                else:
                    self.model = LinearRegression()
                self.model.fit(self.X_train, self.y_train)
                self.y_pred = self.model.predict(self.X_test)
            self.update_evaluation_metrics()
            self.create_performance_plot()
            self.set_status("Training completed", "green")
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.set_status("Training failed", "red")
        finally:
            self.train_button.configure(state="normal")

    def update_evaluation_metrics(self):
        self.metrics_text.delete("1.0", "end")
        try:
            if self.task == "classification":
                acc = accuracy_score(self.y_test, self.y_pred)
                prec = precision_score(self.y_test, self.y_pred, average='macro', zero_division=0)
                rec = recall_score(self.y_test, self.y_pred, average='macro', zero_division=0)
                f1 = f1_score(self.y_test, self.y_pred, average='macro', zero_division=0)
                self.metrics_text.insert("end", f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}\n")
            elif self.task == "regression":
                mae = mean_absolute_error(self.y_test, self.y_pred)
                rmse = mean_squared_error(self.y_test, self.y_pred, squared=False)
                r2 = r2_score(self.y_test, self.y_pred)
                self.metrics_text.insert("end", f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR² Score: {r2:.4f}\n")
            else:
                self.metrics_text.insert("end", "Clustering completed.\nCluster assignments generated.\n")
        except Exception as e:
            messagebox.showerror("Metrics Error", f"Failed to update metrics: {str(e)}")

    def create_performance_plot(self):
        if self.plot_canvas:
            self.plot_canvas.get_tk_widget().destroy()
        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            if self.task == "classification":
                cm = confusion_matrix(self.y_test, self.y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title("Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
            elif self.task == "regression":
                ax.scatter(self.y_test, self.y_pred, alpha=0.5, color="#28A745")
                ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Predicted vs Actual Values")
            elif self.task == "clustering" and self.df.shape[1] >= 2:
                ax.scatter(self.df.iloc[:, 0], self.df.iloc[:, 1], c=self.y_pred, cmap='viridis')
                ax.set_title("KMeans Clustering")
                ax.set_xlabel(self.df.columns[0])
                ax.set_ylabel(self.df.columns[1])
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to create plot: {str(e)}")
        self.plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    def set_status(self, message, color):
        # Dummy status method to avoid attribute errors
        pass

    # [Other methods remain unchanged for brevity and since they are already functional and clean]
    # make_prediction, save_results, clear_results, go_back, start_new, set_status
