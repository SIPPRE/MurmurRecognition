import tkinter as tk
from tkinter import ttk
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
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
        self.X = self.df_features.drop(['category', 'filename', 'NormalOrAbnormal'], axis=1)
        self.y = self.df_features['category']

        # Αφαίρεση γραμμών που περιέχουν NaN στην ετικέτα
        self.X = self.X[self.y.notna()]
        self.y = self.y[self.y.notna()]

        if len(self.y.unique()) <= 1:
            raise ValueError("The target 'y' needs to have more than 1 class. Got 1 class instead.")

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

        self.full_stats_button = ttk.Button(self.root, text="Πλήρης Στατιστικά", command=self.plot_full_stats)
        self.full_stats_button.pack()

        self.roc_curve_button = ttk.Button(self.root, text="ROC Curve", command=self.plot_roc_curve)
        self.roc_curve_button.pack()

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

        fig, ax = plt.subplots(figsize=(14, 8))
        bar_width = 0.15
        index = range(len(self.model_options))

        bars1 = ax.bar([i - 2*bar_width for i in index], accuracies, bar_width, label='Accuracy')
        bars2 = ax.bar([i - bar_width for i in index], f1_scores, bar_width, label='F1 Score')
        bars3 = ax.bar(index, precisions, bar_width, label='Precision')
        bars4 = ax.bar([i + bar_width for i in index], recalls, bar_width, label='Recall')
        bars5 = ax.bar([i + 2*bar_width for i in index], specificities, bar_width, label='Specificity')

        ax.set_xlabel('Models')
        ax.set_ylabel('Scores')
        ax.set_title('Comparison of Model Performance Metrics')
        ax.set_xticks(index)
        ax.set_xticklabels(self.model_options)
        ax.legend()

        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)
        add_labels(bars4)
        add_labels(bars5)

        plt.show()

    def plot_roc_curve(self):
        if not self.model_scores:
            self.results_text.insert(tk.END, "Πρέπει πρώτα να εκτελεστούν τα μοντέλα.\n")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        for model_name in self.model_options:
            if model_name == "SVM":
                model = SVC(C=self.best_params['SVM']['C'], gamma=self.best_params['SVM']['gamma'], kernel=self.best_params['SVM']['kernel'], probability=True)
            elif model_name == "MLP":
                model = MLPClassifier(activation=self.best_params['MLP']['activation'], alpha=self.best_params['MLP']['alpha'], hidden_layer_sizes=self.best_params['MLP']['hidden_layer_sizes'], solver=self.best_params['MLP']['solver'], max_iter=10000)
            elif model_name == "Naive Bayes":
                model = GaussianNB()
            elif model_name == "Random Forest":
                model = RandomForestClassifier(max_depth=self.best_params['Random Forest']['max_depth'], min_samples_split=self.best_params['Random Forest']['min_samples_split'], n_estimators=self.best_params['Random Forest']['n_estimators'])

            model.fit(self.X_train, self.y_train)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'{model_name} (area = {roc_auc:.2f})')

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.legend(loc="lower right")
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

