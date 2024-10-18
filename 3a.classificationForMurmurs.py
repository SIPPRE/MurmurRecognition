import tkinter as tk
from tkinter import ttk
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
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

        # Διαχωρισμός των χαρακτηριστικών (features) και της ετικέτας (label)
        self.df_features = self.df_features.dropna(subset=['NormalOrAbnormal'])
        self.X = self.df_features.drop(['category', 'filename', 'NormalOrAbnormal'], axis=1)
        self.y = self.df_features['NormalOrAbnormal']

        # Κανονικοποίηση των χαρακτηριστικών
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        # Χρήση του SMOTE για αντιμετώπιση του unbalanced dataset
        smote = SMOTE(random_state=42)
        self.X_resampled, self.y_resampled = smote.fit_resample(self.X_scaled, self.y)

        # Διαχωρισμός δεδομένων σε training και test set (80% training, 20% test)
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

        self.plot_button = ttk.Button(self.root, text="Δημιουργία Διαγραμμάτων", command=self.plot_results)
        self.plot_button.pack()

        self.full_stats_button = ttk.Button(self.root, text="Full Stats", command=self.plot_full_stats)
        self.full_stats_button.pack()

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
            'specificity': specificity
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

    def plot_results(self):
        if not self.model_scores:
            self.results_text.insert(tk.END, "Πρέπει πρώτα να εκτελεστούν τα μοντέλα.\n")
            return
        
        true_counts = self.y_test.value_counts()
        pred_counts = pd.Series(self.y_pred_threshold).value_counts()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].pie(true_counts, labels=true_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Κατανομή Κατηγοριών στο Test Set (πραγματικά δεδομένα)')
        axes[1].pie(pred_counts, labels=pred_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Κατανομή Προβλέψεων στο Test Set')

        plt.show()

    def plot_full_stats(self):
        if not all(model in self.model_scores for model in self.model_options):
            self.results_text.insert(tk.END, "Πρέπει να εκτελεστούν όλα τα μοντέλα πρώτα.\n")
            return
        
        accuracies = [self.model_scores[model]['accuracy'] for model in self.model_options]
        f1_scores = [self.model_scores[model]['f1'] for model in self.model_options]
        precisions = [self.model_scores[model]['precision'] for model in self.model_options]
        recalls = [self.model_scores[model]['recall'] for model in self.model_options]
        specificities = [self.model_scores[model]['specificity'] for model in self.model_options]

        bar_width = 0.1
        r1 = range(len(accuracies))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        r4 = [x + bar_width for x in r3]
        r5 = [x + bar_width for x in r4]

        fig, ax = plt.subplots(figsize=(14, 8))

        ax.bar(r1, accuracies, color='blue', width=bar_width, edgecolor='grey', label='Accuracy')
        ax.bar(r2, f1_scores, color='green', width=bar_width, edgecolor='grey', label='F1 Score')
        ax.bar(r3, precisions, color='red', width=bar_width, edgecolor='grey', label='Precision')
        ax.bar(r4, recalls, color='cyan', width=bar_width, edgecolor='grey', label='Recall')
        ax.bar(r5, specificities, color='purple', width=bar_width, edgecolor='grey', label='Specificity')

        ax.set_xlabel('Models', fontweight='bold')
        ax.set_xticks([r + 2 * bar_width for r in range(len(accuracies))])
        ax.set_xticklabels(self.model_options)
        ax.set_title('Comparison of Model Performance Metrics')
        ax.set_ylabel('Scores')

        for i in range(len(accuracies)):
            ax.text(r1[i], accuracies[i] + 0.01, f"{accuracies[i]:.2f}", ha='center', va='bottom')
            ax.text(r2[i], f1_scores[i] + 0.01, f"{f1_scores[i]:.2f}", ha='center', va='bottom')
            ax.text(r3[i], precisions[i] + 0.01, f"{precisions[i]:.2f}", ha='center', va='bottom')
            ax.text(r4[i], recalls[i] + 0.01, f"{recalls[i]:.2f}", ha='center', va='bottom')
            ax.text(r5[i], specificities[i] + 0.01, f"{specificities[i]:.2f}", ha='center', va='bottom')

        ax.legend()
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

