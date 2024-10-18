import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Φόρτωση των δεδομένων
data_path = r'C:\Users\aggel\OneDrive\Υπολογιστής\Cardiac_Project\data\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data.csv'
df = pd.read_csv(data_path)

# Φιλτράρισμα των ασθενών με φύσημα
df_with_murmur = df[df['Murmur'] == 'Present']

# Δημιουργία γραφημάτων
def create_figures():
    figures = []

    # Δημιουργία των πρώτων 5 διαγραμμάτων σε μία εικόνα
    fig_set_1, axes1 = plt.subplots(2, 3, figsize=(20, 12))
    axes1 = axes1.flatten()

    # 1. Pie Chart για την Κατανομή των Κατηγοριών Φυσημάτων (όλοι οι ασθενείς)
    df['Murmur'].value_counts().plot.pie(ax=axes1[0], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral'])
    axes1[0].set_title('Κατανομή των Κατηγοριών Φυσημάτων', fontsize=14)
    axes1[0].set_ylabel('')

    # 2. Pie Chart για την Κατανομή Φύλου (όλοι οι ασθενείς)
    df['Sex'].value_counts().plot.pie(ax=axes1[1], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen'])
    axes1[1].set_title('Κατανομή Φύλου', fontsize=14)
    axes1[1].set_ylabel('')

    # 3. Pie Chart για την Ηλικία (όλοι οι ασθενείς)
    df['Age'].value_counts().plot.pie(ax=axes1[2], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral', 'gold'])
    axes1[2].set_title('Κατανομή Ηλικίας', fontsize=14)
    axes1[2].set_ylabel('')
    fig_set_1.text(0.5, 0.01, 'Neonate: birth to 27 days old\nInfant: 28 days old to 1 year old\nChild: 1 to 11 years old\nAdolescent: 12 to 18 years old\nYoung Adult: 19 to 21 years old', 
                  ha='center', fontsize=10, bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})

    # 4. Bar plot για την πιο ακουστή θέση φυσημάτων (μόνο οι ασθενείς με φύσημα)
    most_audible_location = df_with_murmur['Most audible location'].value_counts(normalize=True) * 100
    most_audible_location.plot(kind='bar', color='mediumseagreen', ax=axes1[3])
    axes1[3].set_title('Πιο Ακουστή Θέση Φυσημάτων (ασθενείς με φύσημα)', fontsize=14)
    axes1[3].set_xlabel('Πιο Ακουστή Θέση')
    axes1[3].set_ylabel('Ποσοστό (%)')
    axes1[3].annotate('AV: Αορτική Βαλβίδα\nPV: Πνευμονική Βαλβίδα\nTV: Τριγλώχινα Βαλβίδα\nMV: Μιτροειδής Βαλβίδα', 
                     xy=(0.5, -0.25), xytext=(0.5, -0.25), fontsize=12, verticalalignment='center', bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', ec='black', lw=1), xycoords='axes fraction')

    # 5. Pie Chart για το Systolic murmur timing (μόνο οι ασθενείς με φύσημα)
    df_with_murmur['Systolic murmur timing'].value_counts().plot.pie(ax=axes1[4], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral'])
    axes1[4].set_title('Κατανομή Systolic Murmur Timing', fontsize=14)
    axes1[4].set_ylabel('')

    # Αφαίρεση του κενής υποπλοκής
    fig_set_1.delaxes(axes1[5])

    figures.append(fig_set_1)

    # Δημιουργία των επόμενων 6 διαγραμμάτων σε άλλη εικόνα
    fig_set_2, axes2 = plt.subplots(3, 2, figsize=(20, 16))
    axes2 = axes2.flatten()

    # 6. Pie Chart για το Systolic murmur shape (μόνο οι ασθενείς με φύσημα)
    df_with_murmur['Systolic murmur shape'].value_counts().plot.pie(ax=axes2[0], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral'])
    axes2[0].set_title('Κατανομή Systolic Murmur Shape', fontsize=14)
    axes2[0].set_ylabel('')

    # 7. Pie Chart για το Systolic murmur pitch (μόνο οι ασθενείς με φύσημα)
    df_with_murmur['Systolic murmur pitch'].value_counts().plot.pie(ax=axes2[1], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral'])
    axes2[1].set_title('Κατανομή Systolic Murmur Pitch', fontsize=14)
    axes2[1].set_ylabel('')

    # 8. Pie Chart για το Systolic murmur quality (μόνο οι ασθενείς με φύσημα)
    df_with_murmur['Systolic murmur quality'].value_counts().plot.pie(ax=axes2[2], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral'])
    axes2[2].set_title('Κατανομή Systolic Murmur Quality', fontsize=14)
    axes2[2].set_ylabel('')

    # 9. Pie Chart για το Outcome (μόνο οι ασθενείς με φύσημα)
    df_with_murmur['Outcome'].value_counts().plot.pie(ax=axes2[3], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral'])
    axes2[3].set_title('Κατανομή Outcome', fontsize=14)
    axes2[3].set_ylabel('')

    # 10. Pie chart για ασθενείς με προβλήματα στη συστολή και στη διαστολή
    def classify_murmur_timing(row):
        if pd.notna(row['Systolic murmur timing']) and pd.notna(row['Diastolic murmur timing']):
            return 'Systolic and Diastolic'
        elif pd.notna(row['Systolic murmur timing']):
            return 'Systolic only'
        elif pd.notna(row['Diastolic murmur timing']):
            return 'Diastolic only'
        else:
            return 'None'

    df_with_murmur['Murmur classification'] = df_with_murmur.apply(classify_murmur_timing, axis=1)
    murmur_classification_counts = df_with_murmur['Murmur classification'].value_counts()
    murmur_classification_counts.plot.pie(ax=axes2[4], autopct='%1.1f%%', startangle=0, colors=['lightblue', 'lightgreen', 'lightcoral', 'lightgray'], labeldistance=1.1)
    axes2[4].set_title('Κατανομή ασθενών με προβλήματα στη Συστολή και Διάστολη (ασθενείς με φύσημα)', fontsize=14)
    axes2[4].set_ylabel('')

    # 11. Bar plot για τα ποσοστά των θέσεων των φυσημάτων (μόνο οι ασθενείς με φύσημα)
    murmur_locations = df_with_murmur['Murmur locations'].value_counts(normalize=True) * 100
    murmur_locations.plot(kind='bar', color='cornflowerblue', ax=axes2[5])
    axes2[5].set_title('Ποσοστά Θέσεων Φυσημάτων (ασθενείς με φύσημα)', fontsize=14)
    axes2[5].set_xlabel('Θέση Φυσημάτων')
    axes2[5].set_ylabel('Ποσοστό (%)')
    axes2[5].annotate('AV: Αορτική Βαλβίδα\nPV: Πνευμονική Βαλβίδα\nTV: Τριγλώχινα Βαλβίδα\nMV: Μιτροειδής Βαλβίδα', 
                     xy=(1.05, 0.8), xytext=(1.05, 0.8), fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', ec='black', lw=1), xycoords='axes fraction')

    figures.append(fig_set_2)

    return figures

class FullscreenApp:
    def __init__(self, figures):
        self.root = tk.Tk()
        self.root.state('zoomed')  # Maximized window state
        self.figures = figures
        self.current_figure_set = 0

        self.root.bind("<Right>", self.next_figure_set)
        self.root.bind("<Left>", self.previous_figure_set)
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        self.show_figure_set()

        save_button = tk.Button(self.root, text="Save Figures", command=self.save_figures)
        save_button.pack()

        self.root.mainloop()

    def show_figure_set(self):
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Button):
                continue
            widget.destroy()
        canvas = FigureCanvasTkAgg(self.figures[self.current_figure_set], master=self.root)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def next_figure_set(self, event):
        self.current_figure_set = (self.current_figure_set + 1) % len(self.figures)
        self.show_figure_set()

    def previous_figure_set(self, event):
        self.current_figure_set = (self.current_figure_set - 1) % len(self.figures)
        self.show_figure_set()

    def save_figures(self):
        file_path_1 = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")], title="Save Figure Set 1")
        if file_path_1:
            self.figures[0].savefig(file_path_1, bbox_inches='tight')
        file_path_2 = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")], title="Save Figure Set 2")
        if file_path_2:
            self.figures[1].savefig(file_path_2, bbox_inches='tight')

# Δημιουργία των γραφημάτων
figures = create_figures()

# Έναρξη της εφαρμογής παραθύρου σε πλήρη οθόνη
app = FullscreenApp(figures) 