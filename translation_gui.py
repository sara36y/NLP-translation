import tkinter as tk
from tkinter import ttk, scrolledtext
from NLP import BilingualTranslationPipeline
import threading
from datetime import datetime

class TranslationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Arabic-English Translation System")
        self.root.geometry("800x600")
        
        # Initialize translation pipelines
        self.ar_en_pipeline = BilingualTranslationPipeline('ar-en')
        self.en_ar_pipeline = BilingualTranslationPipeline('en-ar')


        # Load models for both pipelines
        self.ar_en_pipeline.load_model()
        self.en_ar_pipeline.load_model()

        # Set up the main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create and set up widgets
        self.setup_widgets()
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(3, weight=1)
        
    def setup_widgets(self):
        # Direction selection
        self.direction_var = tk.StringVar(value="ar-en")
        direction_frame = ttk.Frame(self.main_frame)
        direction_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        ttk.Radiobutton(direction_frame, text="Arabic → English", 
                       variable=self.direction_var, value="ar-en",
                       command=self.update_direction).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(direction_frame, text="English → Arabic", 
                       variable=self.direction_var, value="en-ar",
                       command=self.update_direction).pack(side=tk.LEFT, padx=5)
        
        # Input text
        ttk.Label(self.main_frame, text="Input Text:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.input_text = scrolledtext.ScrolledText(self.main_frame, wrap=tk.WORD, height=10)
        self.input_text.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Translate button
        self.translate_button = ttk.Button(self.main_frame, text="Translate", command=self.translate)
        self.translate_button.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Output text
        ttk.Label(self.main_frame, text="Translation:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.output_text = scrolledtext.ScrolledText(self.main_frame, wrap=tk.WORD, height=10)
        self.output_text.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Clear button
        self.clear_button = ttk.Button(self.main_frame, text="Clear", command=self.clear_text)
        self.clear_button.grid(row=7, column=0, columnspan=2, pady=5)
        
    def update_direction(self):
        """Update the translation direction"""
        direction = self.direction_var.get()
        self.status_var.set(f"Translation direction set to: {'Arabic → English' if direction == 'ar-en' else 'English → Arabic'}")
        
    def translate(self):
        """Handle translation in a separate thread"""
        def translation_thread():
            try:
                self.translate_button.state(['disabled'])
                self.status_var.set("Translating...")
                
                # Get input text
                input_text = self.input_text.get("1.0", tk.END).strip()
                if not input_text:
                    self.status_var.set("Please enter text to translate")
                    return
                
                # Select appropriate pipeline
                pipeline = self.ar_en_pipeline if self.direction_var.get() == "ar-en" else self.en_ar_pipeline
                
                # Perform translation
                start_time = datetime.now()
                result = pipeline.translate_text(input_text)
                elapsed = (datetime.now() - start_time).total_seconds()
                
                # Update output
                self.output_text.delete("1.0", tk.END)
                self.output_text.insert("1.0", result['translated'])
                
                # Update status
                self.status_var.set(f"Translation completed in {elapsed:.2f} seconds")
                
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
            finally:
                self.translate_button.state(['!disabled'])
        
        # Start translation in a separate thread
        threading.Thread(target=translation_thread, daemon=True).start()
        
    def clear_text(self):
        """Clear both input and output text areas"""
        self.input_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)
        self.status_var.set("Text areas cleared")

def main():
    root = tk.Tk()
    app = TranslationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 