import customtkinter as ctk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, silhouette_score
import threading
import tkinter as tk
from typing import Optional, List, Tuple, Callable

class MLModelPage(ctk.CTkFrame):
    def __init__(self, master, df: Optional[pd.DataFrame] = None):
        super().__init__(master, fg_color="#f5f7fa")
        self.pack(expand=True, fill="both")
        self.df: pd.DataFrame = df if isinstance(df, pd.DataFrame) and not df.empty else pd.DataFrame()
        self.target_column: Optional[str] = None
        self.current_textbox: Optional[ctk.CTkTextbox] = None
        self.loading_label: Optional[ctk.CTkLabel] = None
        self.warning_label: Optional[ctk.CTkLabel] = None
        self.model_thread: Optional[threading.Thread] = None
        self.model_buttons: dict = {}

        # Initialize columns and validate DataFrame
        self.columns: List[str] = list(self.df.columns) if not self.df.empty else []
        if self.df.empty:
            self.show_error("No valid dataset provided. Please upload a dataset in the Dataset tab.")

        # Define models by category: (name, training function, requires numeric target for supervised)
        self.supervised_models: List[Tuple[str, Callable, bool]] = [
            ("Linear Regression", self.train_linear_regression, True),
            ("SVM", self.train_svm, False),
            ("Logistic Regression", self.train_logistic_regression, False),
            ("Random Forest", self.train_random_forest, False),
            ("Decision Tree", self.train_decision_tree, False),
            ("KNN", self.train_knn, False),
            ("Naive Bayes", self.train_naive_bayes, False)
        ]
        self.unsupervised_models: List[Tuple[str, Callable]] = [
            ("K-Means", self.train_kmeans),
            ("DBSCAN", self.train_dbscan)
        ]

        self.setup_ui()

    def setup_ui(self) -> None:
        """Set up the main UI components: button frame (left) and results frame (right)."""
        # Left Frame: Model selection buttons
        self.button_frame = ctk.CTkFrame(self, fg_color="#ffffff", width=300, corner_radius=10, border_width=1, border_color="#e2e8f0")
        self.button_frame.pack(side="left", fill="y", padx=(10, 5), pady=10)
        self.button_frame.pack_propagate(False)

        # Right Frame: Model results
        self.results_frame = ctk.CTkFrame(self, fg_color="#ffffff", corner_radius=10, border_width=1, border_color="#e2e8f0")
        self.results_frame.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)

        # Button frame: Supervised and Unsupervised buttons
        ctk.CTkLabel(self.button_frame, text="Model Selection", font=("Arial", 18, "bold"), text_color="#2d3748").pack(pady=(15, 25))
        
        self.supervised_button = ctk.CTkButton(
            self.button_frame, text="Supervised", font=("Arial", 16, "bold"),
            fg_color="#e2e8f0", hover_color="#d1d9e6", text_color="#2d3748",
            command=self.show_supervised_models, state="normal" if self.columns else "disabled"
        )
        self.supervised_button.pack(fill="x", padx=20, pady=8)

        self.unsupervised_button = ctk.CTkButton(
            self.button_frame, text="Unsupervised", font=("Arial", 16, "bold"),
            fg_color="#e2e8f0", hover_color="#d1d9e6", text_color="#2d3748",
            command=self.show_unsupervised_models, state="normal" if self.columns else "disabled"
        )
        self.unsupervised_button.pack(fill="x", padx=20, pady=8)

        # Results frame: Title and target selection
        ctk.CTkLabel(self.results_frame, text="Model Results", font=("Arial", 20, "bold"), text_color="#2d3748").pack(pady=(15, 5))
        ctk.CTkLabel(self.results_frame, text="Select a target column and model to view results", font=("Arial", 14), text_color="#718096").pack(pady=(0, 20))

        ctk.CTkLabel(self.results_frame, text="Select Target Column:", font=("Arial", 16), text_color="#2d3748").pack(pady=(10, 0))
        self.target_dropdown = ctk.CTkComboBox(
            self.results_frame, values=self.columns, command=self.on_target_selected,
            state="disabled" if not self.columns else "normal",
            font=("Arial", 14), text_color="#2d3748", dropdown_font=("Arial", 14)
        )
        self.target_dropdown.pack(pady=(0, 10))

        # Parameter frame (empty for now)
        self.param_frame = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        self.param_frame.pack(fill="x", padx=20, pady=10)

    def show_error(self, message: str) -> None:
        """Display an error message in the results frame."""
        self.clear_results_frame()
        self.current_textbox = ctk.CTkTextbox(self.results_frame, width=700, height=300, font=("Courier", 14), wrap="none")
        self.current_textbox.insert("0.0", f"Error: {message}")
        self.current_textbox.configure(state="disabled")
        self.current_textbox.pack(expand=True, pady=20, padx=20)

    def show_warning(self, message: str) -> None:
        """Display a warning message in the results frame."""
        if self.warning_label:
            self.warning_label.destroy()
        self.warning_label = ctk.CTkLabel(
            self.results_frame, text=f"Warning: {message}", 
            font=("Arial", 14, "bold"), text_color="#e67e22"
        )
        self.warning_label.pack(pady=(5, 0))

    def clear_warning(self) -> None:
        """Remove the warning label if it exists."""
        if self.warning_label:
            self.warning_label.destroy()
            self.warning_label = None

    def clear_sub_buttons(self) -> None:
        """Remove all model-specific buttons from the button frame."""
        for btn in self.model_buttons.values():
            btn.destroy()
        self.model_buttons.clear()

    def show_supervised_models(self) -> None:
        """Display buttons for supervised models, enabling/disabling based on target column type."""
        self.clear_sub_buttons()
        self.supervised_button.configure(fg_color="#4a90e2", text_color="white")
        self.unsupervised_button.configure(fg_color="#e2e8f0", text_color="#2d3748")

        for model_name, _, requires_numeric_target in self.supervised_models:
            state = "disabled"
            if self.target_column and not self.df.empty:
                target_series = self.df[self.target_column]
                is_numeric = pd.api.types.is_numeric_dtype(target_series)
                is_valid = is_numeric if requires_numeric_target else True
                state = "normal" if is_valid else "disabled"
            btn = ctk.CTkButton(
                self.button_frame, text=f"Run {model_name}", font=("Arial", 14),
                fg_color="#6aa8f0", hover_color="#8bbcff", text_color="white",
                command=lambda name=model_name: self.run_model(name),
                state=state
            )
            btn.pack(fill="x", padx=40, pady=4)
            self.model_buttons[model_name] = btn

    def show_unsupervised_models(self) -> None:
        """Display buttons for unsupervised models."""
        self.clear_sub_buttons()
        self.unsupervised_button.configure(fg_color="#4a90e2", text_color="white")
        self.supervised_button.configure(fg_color="#e2e8f0", text_color="#2d3748")

        for model_name, _ in self.unsupervised_models:
            btn = ctk.CTkButton(
                self.button_frame, text=f"Run {model_name}", font=("Arial", 14),
                fg_color="#6aa8f0", hover_color="#8bbcff", text_color="white",
                command=lambda name=model_name: self.run_model(name),
                state="normal" if not self.df.empty else "disabled"
            )
            btn.pack(fill="x", padx=40, pady=4)
            self.model_buttons[model_name] = btn

    def on_target_selected(self, choice: str) -> None:
        """Handle target column selection, check for null values, and refresh supervised model buttons."""
        self.target_column = choice
        self.clear_results_frame()
        self.clear_warning()
        
        # Check for null values in the target column
        if not self.df.empty and self.target_column:
            if self.df[self.target_column].isnull().any():
                self.show_warning(f"Target column '{self.target_column}' contains null values. They will be handled during preprocessing.")
        
        self.show_supervised_models()

    def clear_results_frame(self) -> None:
        """Clear the results frame of any existing widgets except warnings."""
        if self.current_textbox:
            self.current_textbox.destroy()
            self.current_textbox = None
        if self.loading_label:
            self.loading_label.destroy()
            self.loading_label = None
        for widget in self.param_frame.winfo_children():
            widget.destroy()

    def show_loading(self) -> None:
        """Display a loading message during model training."""
        self.clear_results_frame()
        self.loading_label = ctk.CTkLabel(self.results_frame, text="Training model, please wait...", font=("Arial", 14, "italic"), text_color="#718096")
        self.loading_label.pack(pady=20)
        self.results_frame.update()

    def run_model(self, model_name: str) -> None:
        """Start a new thread to run the selected model."""
        if self.model_thread and self.model_thread.is_alive():
            self.show_error("A model is already running. Please wait for it to complete.")
            return

        if self.df.empty:
            self.show_error("No dataset loaded.")
            return

        self.show_loading()
        self.model_thread = threading.Thread(target=self._run_model_thread, args=(model_name,))
        self.model_thread.start()

    def _run_model_thread(self, model_name: str) -> None:
        """Run the model in a separate thread and update UI with results."""
        try:
            all_models = self.supervised_models + self.unsupervised_models
            for name, train_func, *args in all_models:
                if name == model_name:
                    result = train_func()
                    break
            else:
                result = f"Error: Model '{model_name}' not found."

            def update_ui():
                self.clear_results_frame()
                self.clear_warning()
                self.current_textbox = ctk.CTkTextbox(self.results_frame, width=700, height=300, font=("Courier", 14), wrap="none")
                self.current_textbox.insert("0.0", result)
                self.current_textbox.configure(state="disabled")
                self.current_textbox.pack(expand=True, pady=20, padx=20)
                # Re-check for nulls in the target column
                if self.target_column and not self.df.empty and self.df[self.target_column].isnull().any():
                    self.show_warning(f"Target column '{self.target_column}' contains null values. They were handled during preprocessing.")
                self.model_thread = None

            self.after(0, update_ui)

        except Exception as e:
            def update_ui():
                self.clear_results_frame()
                self.clear_warning()
                self.current_textbox = ctk.CTkTextbox(self.results_frame, width=700, height=300, font=("Courier", 14), wrap="none")
                self.current_textbox.insert("0.0", f"Error: {str(e)}")
                self.current_textbox.configure(state="disabled")
                self.current_textbox.pack(expand=True, pady=20, padx=20)
                self.model_thread = None

            self.after(0, update_ui)

    def preprocess_data(self, for_clustering: bool = False) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for supervised or unsupervised learning."""
        if self.df.empty:
            raise ValueError("No dataset loaded.")

        df = self.df.copy()
        if for_clustering:
            X = df.select_dtypes(include=[np.number])
            if X.empty:
                raise ValueError("No numeric columns available for clustering.")
            preprocessor = Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])
            return preprocessor.fit_transform(X)
        else:
            if not self.target_column:
                raise ValueError("Target column not selected.")
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]

            numeric_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns

            # Limit categories for high-cardinality columns
            max_categories = 50
            cat_encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore", max_categories=max_categories)
            preprocessor = ColumnTransformer([
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler())
                ]), numeric_cols),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", cat_encoder)
                ]), categorical_cols)
            ])

            X = preprocessor.fit_transform(X)

            # Encode target if categorical for classification
            if y.dtype == "object" or y.dtype.name == "category":
                le = LabelEncoder()
                y = le.fit_transform(y)
            elif not pd.api.types.is_numeric_dtype(y):
                raise ValueError("Target column must be numeric or categorical.")
            return X, y

    def train_linear_regression(self) -> str:
        """Train and evaluate Linear Regression model."""
        X, y = self.preprocess_data()
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Linear Regression requires a numeric target column.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        return f"""Linear Regression Results for Column '{self.target_column}':
Mean Squared Error (Train): {mse_train:.4f}
Mean Squared Error (Test): {mse_test:.4f}
R² Score (Train): {r2_train:.4f}
R² Score (Test): {r2_test:.4f}"""

    def train_svm(self) -> str:
        """Train and evaluate SVM model."""
        X, y = self.preprocess_data()
        if pd.api.types.is_numeric_dtype(y) and not np.all(y.astype(int) == y):
            raise ValueError("SVM requires a categorical or integer target for classification.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = SVC(kernel="rbf", random_state=42)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        accuracy_train = accuracy_score(y_train, y_train_pred)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        report = classification_report(y_test, y_test_pred, zero_division=0)

        return f"""SVM Results for Column '{self.target_column}':
Accuracy (Train): {accuracy_train:.4f}
Accuracy (Test): {accuracy_test:.4f}

Classification Report (Test):
{report}"""

    def train_logistic_regression(self) -> str:
        """Train and evaluate Logistic Regression model."""
        X, y = self.preprocess_data()
        if pd.api.types.is_numeric_dtype(y) and not np.all(y.astype(int) == y):
            raise ValueError("Logistic Regression requires a categorical or integer target.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        accuracy_train = accuracy_score(y_train, y_train_pred)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        report = classification_report(y_test, y_test_pred, zero_division=0)

        return f"""Logistic Regression Results for Column '{self.target_column}':
Accuracy (Train): {accuracy_train:.4f}
Accuracy (Test): {accuracy_test:.4f}

Classification Report (Test):
{report}"""

    def train_random_forest(self) -> str:
        """Train and evaluate Random Forest model."""
        X, y = self.preprocess_data()
        if pd.api.types.is_numeric_dtype(y) and not np.all(y.astype(int) == y):
            raise ValueError("Random Forest requires a categorical or integer target.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        accuracy_train = accuracy_score(y_train, y_train_pred)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        report = classification_report(y_test, y_test_pred, zero_division=0)

        return f"""Random Forest Results for Column '{self.target_column}':
Accuracy (Train): {accuracy_train:.4f}
Accuracy (Test): {accuracy_test:.4f}

Classification Report (Test):
{report}"""

    def train_decision_tree(self) -> str:
        """Train and evaluate Decision Tree model."""
        X, y = self.preprocess_data()
        if pd.api.types.is_numeric_dtype(y) and not np.all(y.astype(int) == y):
            raise ValueError("Decision Tree requires a categorical or integer target.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        accuracy_train = accuracy_score(y_train, y_train_pred)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        report = classification_report(y_test, y_test_pred, zero_division=0)

        return f"""Decision Tree Results for Column '{self.target_column}':
Accuracy (Train): {accuracy_train:.4f}
Accuracy (Test): {accuracy_test:.4f}

Classification Report (Test):
{report}"""

    def train_knn(self) -> str:
        """Train and evaluate KNN model."""
        X, y = self.preprocess_data()
        if pd.api.types.is_numeric_dtype(y) and not np.all(y.astype(int) == y):
            raise ValueError("KNN requires a categorical or integer target.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        accuracy_train = accuracy_score(y_train, y_train_pred)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        report = classification_report(y_test, y_test_pred, zero_division=0)

        return f"""KNN Results for Column '{self.target_column}':
Accuracy (Train): {accuracy_train:.4f}
Accuracy (Test): {accuracy_test:.4f}

Classification Report (Test):
{report}"""

    def train_naive_bayes(self) -> str:
        """Train and evaluate Naive Bayes model."""
        X, y = self.preprocess_data()
        if pd.api.types.is_numeric_dtype(y) and not np.all(y.astype(int) == y):
            raise ValueError("Naive Bayes requires a categorical or integer target.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GaussianNB()
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        accuracy_train = accuracy_score(y_train, y_train_pred)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        report = classification_report(y_test, y_test_pred, zero_division=0)

        return f"""Naive Bayes Results for Column '{self.target_column}':
Accuracy (Train): {accuracy_train:.4f}
Accuracy (Test): {accuracy_test:.4f}

Classification Report (Test):
{report}"""

    def train_kmeans(self) -> str:
        """Train and evaluate K-Means clustering model."""
        X = self.preprocess_data(for_clustering=True)
        n_clusters = min(3, X.shape[0])  # Ensure n_clusters <= number of samples

        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)

        silhouette = silhouette_score(X, labels) if len(set(labels)) > 1 else "N/A (single cluster or insufficient samples)"

        return f"""K-Means Clustering Results:
Number of Clusters: {n_clusters}
Silhouette Score: {silhouette if isinstance(silhouette, str) else f"{silhouette:.4f}"}"""

    def train_dbscan(self) -> str:
        """Train and evaluate DBSCAN clustering model."""
        X = self.preprocess_data(for_clustering=True)
        eps = 0.5  # Default value
        min_samples = min(5, X.shape[0])  # Ensure min_samples <= number of samples

        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        silhouette = silhouette_score(X, labels) if n_clusters > 1 else "N/A (insufficient clusters or samples)"

        return f"""DBSCAN Clustering Results:
Number of Clusters: {n_clusters} (excluding noise)
Noise Points: {n_noise}
eps: {eps}
Min Samples: {min_samples}
Silhouette Score: {silhouette if isinstance(silhouette, str) else f"{silhouette:.4f}"}"""