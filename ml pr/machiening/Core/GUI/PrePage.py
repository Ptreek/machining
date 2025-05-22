import customtkinter as ctk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Optional

class PrePage:
    def __init__(self, parent, df, app=None):
        self.parent = parent
        self.app = app
        # Use the app's DataFrame as the source of truth
        self.df = self.app.df if self.app and isinstance(self.app.df, pd.DataFrame) and not self.app.df.empty else pd.DataFrame()
        self.processed_df = self.df.copy()
        self.chart_frames = []
        self.setup_ui()
        if not self.df.empty:
            self.populate_tabs()
        else:
            self.show_error("No valid dataset provided. Please upload a dataset in the Dataset tab.")

    def setup_ui(self):
        self.main_frame = ctk.CTkFrame(self.parent, fg_color="#f5f7fa")
        self.main_frame.pack(fill="both", expand=True, padx=15, pady=15)

        sidebar_frame = ctk.CTkScrollableFrame(
            self.main_frame, fg_color="#ffffff", width=250, corner_radius=10,
            border_width=1, border_color="#e2e8f0"
        )
        sidebar_frame.pack(side="left", fill="y", padx=(0, 10), pady=10)

        content_frame = ctk.CTkFrame(self.main_frame, fg_color="#ffffff", corner_radius=10, border_width=1, border_color="#e2e8f0")
        content_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(content_frame, text="Data Cleaning & Preprocessing", font=("Arial", 20, "bold"), text_color="#2d3748").pack(pady=(15, 5))
        ctk.CTkLabel(content_frame, text="Clean and preprocess the dataset", font=("Arial", 14), text_color="#718096").pack(pady=(0, 20))

        self.tab_view = ctk.CTkTabview(content_frame)
        self.tab_view.pack(fill="both", expand=True, padx=10, pady=10)

        self.tabs = [
            "Remove Duplicates", "Remove Null Rows", "Remove Outliers",
            "Normalization", "Encoding"
        ]
        for tab_name in self.tabs:
            tab = self.tab_view.add(tab_name)
            frame = ctk.CTkFrame(tab, fg_color="#f0f4f8", corner_radius=10)
            frame.pack(fill="both", expand=True, padx=10, pady=10)
            self.chart_frames.append(frame)
            ctk.CTkButton(
                sidebar_frame, text=tab_name, font=("Arial", 14),
                fg_color="#4a90e2", hover_color="#357abd", text_color="white",
                command=lambda t=tab_name: self.tab_view.set(t)
            ).pack(fill="x", pady=5, padx=20)

        button_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        button_frame.pack(fill="x", pady=10)
        ctk.CTkButton(
            button_frame, text="Apply Preprocessing", font=("Arial", 14),
            fg_color="#4a90e2", hover_color="#357abd", text_color="white",
            command=self.apply_preprocessing
        )
        ctk.CTkButton(
            button_frame, text="Reset Data", font=("Arial", 14),
            fg_color="#e2e8f0", hover_color="#d1d9e6", text_color="#2d3748",
            command=self.reset_data
        ).pack(side="left", padx=10)

    def show_error(self, message):
        for frame in self.chart_frames:
            error_label = ctk.CTkLabel(frame, text=message, text_color="red", font=("Arial", 14))
            error_label.pack(expand=True)

    def populate_tabs(self):
        methods = [
            self.show_remove_duplicates, self.show_remove_null_rows,
            self.show_remove_outliers, self.show_normalization,
            self.show_encoding
        ]
        for frame, method in zip(self.chart_frames, methods):
            for widget in frame.winfo_children():
                widget.destroy()
            method(frame)

    def update_ui(self, df):
        """Update the UI with the dataset from the main app."""
        self.df = df if isinstance(df, pd.DataFrame) and not df.empty else pd.DataFrame()
        self.processed_df = self.df.copy()
        for frame in self.chart_frames:
            for widget in frame.winfo_children():
                widget.destroy()
        if not self.df.empty:
            self.populate_tabs()
        else:
            self.show_error("No valid dataset provided. Please upload a dataset in the Dataset tab.")

    def apply_preprocessing(self):
        try:
            if not self.app:
                raise ValueError("Main application reference not provided.")
            
            # Update the main app's DataFrame with the processed data
            self.app.df = self.processed_df.copy()
            
            # Update the Dataset page
            if self.app.dataset_page:
                self.app.dataset_page.update_ui(self.app.df, self.app.dataset_name)
            
            # Update the Visualization page by reinitializing it if it exists
            if hasattr(self.app, 'visualization_page') and self.app.visualization_page:
                self.app.clear_main_frame()
                self.app.visualization_page = Visualise(self.app.main_frame, self.app.df, self.app.report_path)
            
            # Update the ML Model page by reinitializing it if it exists
            if hasattr(self.app, 'model_page') and self.app.model_page:
                self.app.clear_main_frame()
                self.app.model_page = MLModelPage(self.app.main_frame, self.app.df)
                self.app.model_page.setup_ui()
            
            # Refresh the current preprocessing page
            self.update_ui(self.app.df)
            
            messagebox.showinfo("Success", "Cleaning and preprocessing applied and dataset updated across all tabs.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply preprocessing: {str(e)}")

    def reset_data(self):
        self.processed_df = self.df.copy()
        self.populate_tabs()
        messagebox.showinfo("Success", "Dataset reset to original state.")

    def show_remove_duplicates(self, frame):
        try:
            duplicates = self.df.duplicated().sum()
            text = f"Number of Duplicate Rows: {duplicates}\n\nPreview of Duplicates:\n{self.df[self.df.duplicated()].head()}"
            text_box = ctk.CTkTextbox(frame, width=600, height=300, font=("Courier", 14))
            text_box.insert("0.0", text)
            text_box.configure(state="disabled")
            text_box.pack(expand=True, pady=10)
            ctk.CTkButton(
                frame, text="Remove Duplicates", font=("Arial", 14),
                fg_color="#4a90e2", hover_color="#357abd", text_color="white",
                command=self.apply_remove_duplicates
            ).pack(pady=10)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def apply_remove_duplicates(self):
        try:
            initial_rows = len(self.processed_df)
            self.processed_df = self.processed_df.drop_duplicates()
            removed_rows = initial_rows - len(self.processed_df)
            messagebox.showinfo("Success", f"Removed {removed_rows} duplicate rows successfully.")
            self.populate_tabs()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove duplicates: {str(e)}")

    def show_remove_null_rows(self, frame):
        try:
            null_rows = self.df.isnull().any(axis=1).sum()
            text = f"Number of Rows with Null Values: {null_rows}\n\nPreview of Rows with Nulls:\n{self.df[self.df.isnull().any(axis=1)].head()}"
            text_box = ctk.CTkTextbox(frame, width=600, height=300, font=("Courier", 14))
            text_box.insert("0.0", text)
            text_box.configure(state="disabled")
            text_box.pack(expand=True, pady=10)
            ctk.CTkButton(
                frame, text="fill Null Rows", font=("Arial", 14),
                fg_color="#4a90e2", hover_color="#357abd", text_color="white",
                command=self.apply_remove_null_rows
            ).pack(pady=10)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def apply_remove_null_rows(self):
        try:
            initial_rows = len(self.processed_df)
            removed_rows = sum(self.processed_df.isnull().any(axis=1))
            self.processed_df = self.processed_df.fillna(0)
            messagebox.showinfo("Success", f"fill {removed_rows} null values successfully.")
            self.populate_tabs()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fill null rows: {str(e)}")

    def show_remove_outliers(self, frame):
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns available for outlier removal.")
            temp_df = self.df.copy()
            outlier_counts = {}
            for col in numeric_cols:
                Q1 = temp_df[col].quantile(0.25)
                Q3 = temp_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = temp_df[(temp_df[col] < lower_bound) | (temp_df[col] > upper_bound)][col]
                outlier_counts[col] = len(outliers)
            text = f"Outliers Detected (IQR Method):\n\n{outlier_counts}"
            text_box = ctk.CTkTextbox(frame, width=600, height=300, font=("Courier", 14))
            text_box.insert("0.0", text)
            text_box.configure(state="disabled")
            text_box.pack(expand=True, pady=10)
            ctk.CTkButton(
                frame, text="Remove Outliers (IQR)", font=("Arial", 14),
                fg_color="#4a90e2", hover_color="#357abd", text_color="white",
                command=self.apply_remove_outliers
            ).pack(pady=10)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def apply_remove_outliers(self):
        try:
            numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns available for outlier removal.")
            initial_rows = len(self.processed_df)
            for col in numeric_cols:
                Q1 = self.processed_df[col].quantile(0.25)
                Q3 = self.processed_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.processed_df = self.processed_df[(self.processed_df[col] >= lower_bound) & (self.processed_df[col] <= upper_bound)]
            removed_rows = initial_rows - len(self.processed_df)
            messagebox.showinfo("Success", f"Removed {removed_rows} rows with outliers.")
            self.populate_tabs()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove outliers: {str(e)}")

    def show_normalization(self, frame):
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns available for normalization.")
            temp_df = self.df.copy()
            scaler = StandardScaler()
            temp_df[numeric_cols] = scaler.fit_transform(temp_df[numeric_cols])
            text = "Normalization (Standard Scaling) applied.\n\nPreview:\n" + str(temp_df[numeric_cols].head())
            text_box = ctk.CTkTextbox(frame, width=600, height=300, font=("Courier", 14))
            text_box.insert("0.0", text)
            text_box.configure(state="disabled")
            text_box.pack(expand=True, pady=10)
            ctk.CTkButton(
                frame, text="Apply Normalization", font=("Arial", 14),
                fg_color="#4a90e2", hover_color="#357abd", text_color="white",
                command=self.apply_normalization
            ).pack(pady=10)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def apply_normalization(self):
        try:
            numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                raise ValueError("No numeric columns available for normalization.")
            scaler = StandardScaler()
            self.processed_df[numeric_cols] = scaler.fit_transform(self.processed_df[numeric_cols])
            messagebox.showinfo("Success", "Normalization applied successfully.")
            self.populate_tabs()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply normalization: {str(e)}")

    def show_encoding(self, frame):
        try:
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
            if categorical_cols.empty:
                raise ValueError("No categorical columns available for encoding.")
            temp_df = self.df.copy()
            le = LabelEncoder()
            for col in categorical_cols:
                temp_df[col] = le.fit_transform(temp_df[col].astype(str))
            text = "Label Encoding applied to categorical columns.\n\nPreview:\n" + str(temp_df[categorical_cols].head())
            text_box = ctk.CTkTextbox(frame, width=600, height=300, font=("Courier", 14))
            text_box.insert("0.0", text)
            text_box.configure(state="disabled")
            text_box.pack(expand=True, pady=10)
            ctk.CTkButton(
                frame, text="Apply Label Encoding", font=("Arial", 14),
                fg_color="#4a90e2", hover_color="#357abd", text_color="white",
                command=self.apply_encoding
            ).pack(pady=10)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"Error: {str(e)}", text_color="red", font=("Arial", 14)).pack(expand=True)

    def apply_encoding(self):
        try:
            categorical_cols = self.processed_df.select_dtypes(include=['object', 'category']).columns
            if categorical_cols.empty:
                raise ValueError("No categorical columns available for encoding.")
            le = LabelEncoder()
            for col in categorical_cols:
                self.processed_df[col] = le.fit_transform(self.processed_df[col].astype(str))
            messagebox.showinfo("Success", "Label encoding applied successfully.")
            self.populate_tabs()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply encoding: {str(e)}")
