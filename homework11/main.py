import os
import sys

# FIX TCL/TK PATHS 
python_home = r"D:\Programmes\Python" 

tcl_dir = os.path.join(python_home, "tcl")

if not os.path.exists(tcl_dir):
    tcl_dir = os.path.join(python_home, "Lib", "tcl")

if os.path.exists(tcl_dir):
    os.environ["TCL_LIBRARY"] = os.path.join(tcl_dir, "tcl8.6")
    os.environ["TK_LIBRARY"] = os.path.join(tcl_dir, "tk8.6")
    print(f"Manual Tcl/Tk path set to: {tcl_dir}")
else:
    print(f"Warning: Tcl folder not found at {tcl_dir}")

import numpy as np
import re
import math
from collections import Counter
import tkinter as tk
from tkinter import filedialog, messagebox

ALPHABET = "abcdefghijklmnopqrstuvwxyz "

def clean_text(text: str) -> str:
    """
    Cleans the text:
    1. Converts to lowercase.
    2. Replaces newlines with spaces.
    3. Removes any character that is not in ALPHABET.
    4. Normalizes multiple spaces into one.
    """
    text = text.lower()
    text = text.replace("\n", " ")
    
    escaped_alphabet = re.escape(ALPHABET)
    
    # Remove everything NOT in the alphabet
    text = re.sub(f'[^{escaped_alphabet}]+', '', text)
    
    # Collapse multiple spaces and trim edges
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def bigram_matrix(text, alphabet=ALPHABET):
    """
    Builds a transition probability matrix
    from the training text.
    """
    text = clean_text(text)
    counts = Counter()
    
    # Count pairs (bigrams)
    for i in range(len(text) - 1):
        a, b = text[i], text[i + 1]
        if a in alphabet and b in alphabet:
            counts[(a, b)] += 1
            
    n = len(alphabet)
    M = np.zeros((n, n))
    
    # Map characters to indices ('a' -> 0, 'b' -> 1)
    char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
    
    for (a, b), c in counts.items():
        i, j = char_to_idx[a], char_to_idx[b]
        M[i, j] = c
        
    # Add smoothing to avoid zero probabilities (log(0) = error)
    M += 1e-6 
    
    # Normalize rows so they sum to 1 (probabilities)
    row_sums = M.sum(axis=1, keepdims=True)
    M = M / row_sums
    return M, char_to_idx

def calculate_adequacy(text, M, char_to_idx, alphabet=ALPHABET):
    """
    Calculates the 'score' of a text.
    Higher score (closer to 0 (-2.5)) = More Adequate.
    Lower score (-6.0) = Less Adequate.
    """
    clean = clean_text(text)
    
    if len(clean) < 2:
        return -99999.0

    log_prob_sum = 0
    pairs_count = 0
    
    for i in range(len(clean) - 1):
        char1 = clean[i]
        char2 = clean[i+1]
        
        if char1 in char_to_idx and char2 in char_to_idx:
            idx1 = char_to_idx[char1]
            idx2 = char_to_idx[char2]
            
            prob = M[idx1, idx2]
            log_prob_sum += math.log(prob)
            pairs_count += 1
            
    if pairs_count == 0:
        return -99999.0

    return log_prob_sum / pairs_count

class TextAdequacyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Adequacy Check System (Lab 11)")
        self.root.geometry("650x450")

        self.model_matrix = None
        self.char_map = None
        self.threshold = -4.5 # Default threshold
        
        # TRAINING 
        self.frame_train = tk.LabelFrame(root, text="1. Model Training", padx=10, pady=10)
        self.frame_train.pack(fill="x", padx=10, pady=5)
        
        self.btn_load_train = tk.Button(self.frame_train, text="Load Training Text (.txt)", command=self.load_training_file)
        self.btn_load_train.pack(side="left")
        
        self.lbl_status = tk.Label(self.frame_train, text="Model not trained", fg="red")
        self.lbl_status.pack(side="left", padx=10)

        # TESTING 
        self.frame_test = tk.LabelFrame(root, text="2. Text Verification", padx=10, pady=10)
        self.frame_test.pack(fill="both", expand=True, padx=10, pady=5)
        
        tk.Label(self.frame_test, text="Enter text to check:").pack(anchor="w")
        
        self.txt_input = tk.Text(self.frame_test, height=5)
        self.txt_input.pack(fill="x", pady=5)
        
        self.btn_check = tk.Button(self.frame_test, text="Check Adequacy", command=self.check_adequacy)
        self.btn_check.pack(pady=5)
        
        self.lbl_result = tk.Label(self.frame_test, text="Result: -", font=("Arial", 12, "bold"))
        self.lbl_result.pack(pady=10)

    def load_training_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                if len(text) < 100:
                    messagebox.showwarning("Warning", "Text file is too short for training.")
                    return

                # Train the model
                self.model_matrix, self.char_map = bigram_matrix(text)

                self.lbl_status.config(text=f"Trained on {len(text)} chars", fg="green")
                
                #  AUTO-CALCULATE THRESHOLD
                sample_score = calculate_adequacy(text, self.model_matrix, self.char_map)
                self.threshold = sample_score - 1.5 
                
                print(f"Training Sample Score: {sample_score:.4f}")
                print(f"New Threshold set to: {self.threshold:.4f}")
                
                messagebox.showinfo("Success", f"Model trained!\nThreshold set to: {self.threshold:.2f}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not read file: {e}")

    def check_adequacy(self):
        if self.model_matrix is None:
            messagebox.showwarning("Warning", "Please load a training file first!")
            return
            
        text = self.txt_input.get("1.0", tk.END).strip()
        if not text:
            return

        score = calculate_adequacy(text, self.model_matrix, self.char_map)

        if score == -99999.0:
             verdict = "INVALID INPUT"
             color = "orange"
        elif score > self.threshold:
            verdict = "ADEQUATE TEXT"
            color = "green"
        else:
            verdict = "RANDOM"
            color = "red"
        
        self.lbl_result.config(text=f"Score: {score:.2f}\nVerdict: {verdict}", fg=color)

if __name__ == "__main__":
    root = tk.Tk()
    app = TextAdequacyApp(root)
    root.mainloop()