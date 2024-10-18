import tkinter as tk
from tkinter import ttk
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

class ClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Classification App")

        self.data_path = r'C:\Users\aggel\OneDrive\Υπολογιστής\Cardiac_Project\output\features_combined.xlsx'

        self.best_params = {
            'SVM': {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'},
            'MLP': {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': (100, 50), 'solver': 'adam'},
            'Random Forest': {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}
        }

        self.model_scores = {model: {} for model in ['SVM', 'MLP', 'Naive Bayes', 'Random Forest']}

        self.load_data()
        self.create_widgets()

    def load_data(self):
        start_time = time.time()
        self.df_features = pd.read_excel(self.data_path)

        self.X = self.df_features.drop(['category', 'NormalOrAbnormal', 'filename'], axis=1)
        self.y = self.df_features['category']

        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        self.X_scaled = self.X_scaled[~np.isnan(self.y)]
        self.y = self.y[~np.isnan(self.y)]

        smote = SMOTE(random_state=42)
        self.X_resampled, self.y_resampled = smote.fit_resample(self.X_scaled, self.y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_resampled, self.y_resampled, test_size=0.2, random_state=42, stratify=self.y_resampled
        )

        end_time = time.time()
        print(f"Data loaded. Total time: {end_time - start_time:.2f} seconds")

    def create_widgets(self):
        self.model_label = ttk.Label(self.root, text="Επιλέξτε Μοντέλο:")
        self.model_label.pack()

        self.model_var = tk.StringVar()
        self.model_var.set("SVM")

        self.model_options = ['SVM', 'MLP', 'Naive Bayes', 'Random Forest']
        self.model_menu = ttk.OptionMenu(self.root, self.model_var, self.model_options[0], *self.model_options)
        self.model_menu.pack()

        self.run_button = ttk.Button(self.root, text="Εκτέλεση", command=self.run_model)
        self.run_button.pack()

        self.plot_roc_button = ttk.Button(self.root, text="ROC Διαγράμμα", command=self.plot_roc_curve)
        self.plot_roc_button.pack()

        self.plot_confusion_matrix_button = ttk.Button(self.root, text="Confusion Matrix", command=self.plot_confusion_matrix)
        self.plot_confusion_matrix_button.pack()

        self.plot_precision_recall_button = ttk.Button(self.root, text="Precision-Recall Διαγράμμα", command=self.plot_precision_recall_curve)
        self.plot_precision_recall_button.pack()

        self.plot_learning_curve_button = ttk.Button(self.root, text="Learning Curve", command=self.plot_learning_curve)
        self.plot_learning_curve_button.pack()

        self.show_params_button = ttk.Button(self.root, text="Show Best Params", command=self.show_best_params)
        self.show_params_button.pack()

        self.clear_button = ttk.Button(self.root, text="Clear", command=self.clear_text)
        self.clear_button.pack()

        self.results_text = tk.Text(self.root, height=20, width=100)
        self.results_text.pack()

    def run_model(self):
        model_name = self.model_var.get()
        print(f"Running model: {model_name}")

        start_time = time.time()
        
        if model_name == "SVM":
            model = SVC(C=self.best_params['SVM']['C'], gamma=self.best_params['SVM']['gamma'], kernel=self.best_params['SVM']['kernel'], probability=True)
        elif model_name == "MLP":
            model = MLPClassifier(activation=self.best_params['MLP']['activation'], alpha=self.best_params['MLP']['alpha'], hidden_layer_sizes=self.best_params['MLP']['hidden_layer_sizes'], solver=self.best_params['MLP']['solver'], max_iter=10000)
        elif model_name == "Naive Bayes":
            model = GaussianNB()
        elif model_name == "Random Forest":
            model = RandomForestClassifier(max_depth=self.best_params['Random Forest']['max_depth'], min_samples_split=self.best_params['Random Forest']['min_samples_split'], n_estimators=self.best_params['Random Forest']['n_estimators'])

        unique_classes = np.unique(self.y_train)
        if len(unique_classes) > 1:
            model.fit(self.X_train, self.y_train)
            self.y_pred = model.predict(self.X_test)
            self.y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            self.y_pred_threshold = (self.y_pred_proba >= 0.5).astype(int)

            end_time = time.time()
            print(f"Model {model_name} training and prediction completed in {end_time - start_time:.2f} seconds")

            accuracy = accuracy_score(self.y_test, self.y_pred_threshold)
            f1 = f1_score(self.y_test, self.y_pred_threshold)
            precision = precision_score(self.y_test, self.y_pred_threshold)
            recall = recall_score(self.y_test, self.y_pred_threshold)
            tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred_threshold).ravel()
            specificity = tn / (tn + fp)

            self.model_scores[model_name] = {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'y_pred_proba': self.y_pred_proba,
                'y_pred_threshold': self.y_pred_threshold,  # Store the threshold predictions
                'model': model
            }

            self.results_text.insert(tk.END, f"Model: {model_name}\n")
            self.results_text.insert(tk.END, f"accuracy: {accuracy:.2f}\n")
            self.results_text.insert(tk.END, f"f1_score: {f1:.2f}\n")
            self.results_text.insert(tk.END, f"precision: {precision:.2f}\n")
            self.results_text.insert(tk.END, f"recall: {recall:.2f}\n")
            self.results_text.insert(tk.END, f"specificity: {specificity:.2f}\n")

            self.results_text.insert(tk.END, "Αποτελέσματα ανα ασθενή:\n")
            for index, (true, pred) in enumerate(zip(self.y_test, self.y_pred_threshold), 1):
                self.results_text.insert(tk.END, f"Patient Index: {index}, Εμφάνισε: {pred}, Ενώ στην πραγματικότητα ήταν: {true}\n")
        else:
            self.results_text.insert(tk.END, f"Model {model_name} cannot be trained because the training set has only one class.\n")

    def plot_roc_curve(self):
        if not self.model_scores:
            self.results_text.insert(tk.END, "Πρέπει πρώτα να εκτελεστούν τα μοντέλα.\n")
            return
        
        plt.figure(figsize=(10, 6))
        for model_name in self.model_options:
            if model_name in self.model_scores:
                fpr, tpr, _ = roc_curve(self.y_test, self.model_scores[model_name]['y_pred_proba'])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{model_name} (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()

    def plot_confusion_matrix(self):
        if not self.model_scores:
            self.results_text.insert(tk.END, "Πρέπει πρώτα να εκτελεστούν τα μοντέλα.\n")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for i, model_name in enumerate(self.model_options):
            if model_name in self.model_scores:
                cm = confusion_matrix(self.y_test, self.model_scores[model_name]['y_pred_threshold'])
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(ax=axes[i // 2, i % 2], values_format='d', cmap='Blues')
                axes[i // 2, i % 2].set_title(f'Confusion Matrix - {model_name}')
                axes[i // 2, i % 2].set_xticks(range(len(set(self.y_test))))
                axes[i // 2, i % 2].set_yticks(range(len(set(self.y_test))))
                axes[i // 2, i % 2].set_xticklabels(['0', '1'])
                axes[i // 2, i % 2].set_yticklabels(['0', '1'])
                axes[i // 2, i % 2].set_xlabel('Predicted label')
                axes[i // 2, i % 2].set_ylabel('True label')
        plt.tight_layout()
        plt.show()

    def plot_precision_recall_curve(self):
        if not self.model_scores:
            self.results_text.insert(tk.END, "Πρέπει πρώτα να εκτελεστούν τα μοντέλα.\n")
            return
        
        plt.figure(figsize=(10, 6))
        for model_name in self.model_options:
            if model_name in self.model_scores:
                precision, recall, _ = precision_recall_curve(self.y_test, self.model_scores[model_name]['y_pred_proba'])
                pr_auc = auc(recall, precision)
                plt.plot(recall, precision, lw=2, label=f'{model_name} (area = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.show()

    def plot_learning_curve(self):
        if not self.model_scores:
            self.results_text.insert(tk.END, "Πρέπει πρώτα να εκτελεστούν τα μοντέλα.\n")
            return
        
        plt.figure(figsize=(10, 6))
        for model_name in self.model_options:
            if model_name in self.model_scores:
                model = self.model_scores[model_name]['model']
                try:
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    train_sizes = np.linspace(0.2, 1.0, 5)
                    train_sizes, train_scores, test_scores = learning_curve(model, self.X_resampled, self.y_resampled, cv=skf, train_sizes=train_sizes, n_jobs=-1, error_score='raise')
                    train_scores_mean = np.mean(train_scores, axis=1)
                    test_scores_mean = np.mean(test_scores, axis=1)
                    plt.plot(train_sizes, train_scores_mean, 'o-', label=f'{model_name} Train')
                    plt.plot(train_sizes, test_scores_mean, 'o-', label=f'{model_name} Test')
                except Exception as e:
                    self.results_text.insert(tk.END, f"Error plotting learning curve for {model_name}: {e}\n")
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend(loc="best")
        plt.show()

    def show_best_params(self):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Best parameters for each model:\n")
        for model, params in self.best_params.items():
            self.results_text.insert(tk.END, f"{model}: {params}\n")

    def clear_text(self):
        self.results_text.delete(1.0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ClassificationApp(root)
    root.mainloop()
