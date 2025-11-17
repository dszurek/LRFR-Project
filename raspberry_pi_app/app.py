"""Main GUI Application for Raspberry Pi 5 LRFR System.

Full-featured facial recognition application with:
- Gallery management
- Webcam capture
- Real-time identification
- Performance metrics display
"""

import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import gc
from typing import Optional, List, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from pipeline import LRFRPipeline
from gallery_manager import GalleryManager
from webcam_capture import WebcamCapture
from face_detector import FaceDetector


class LRFRApp:
    """Main application GUI for LRFR on Raspberry Pi 5."""
    
    def __init__(self, root: tk.Tk):
        """Initialize application.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title(config.WINDOW_TITLE)
        self.root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        
        # Application state
        self.pipeline: Optional[LRFRPipeline] = None
        self.gallery: Optional[GalleryManager] = None
        self.webcam: Optional[WebcamCapture] = None
        self.current_vlr_size = config.DEFAULT_VLR_SIZE
        self.verification_mode = tk.StringVar(value="1:N")  # "1:1" or "1:N"
        self.selected_person = tk.StringVar(value="")
        
        # Processing state
        self.is_processing = False
        self.last_result: Optional[Dict] = None
        
        # Setup GUI
        self._create_widgets()
        
        # Initialize components in background
        self.root.after(100, self._initialize_components)
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # ===== Left Panel: Controls =====
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        row = 0
        
        # Resolution selection
        ttk.Label(control_frame, text="VLR Resolution:").grid(row=row, column=0, sticky=tk.W, pady=5)
        row += 1
        
        self.resolution_var = tk.IntVar(value=config.DEFAULT_VLR_SIZE)
        for vlr_size in config.VLR_SIZES:
            ttk.Radiobutton(
                control_frame,
                text=f"{vlr_size}×{vlr_size}",
                variable=self.resolution_var,
                value=vlr_size,
                command=self._on_resolution_change
            ).grid(row=row, column=0, sticky=tk.W)
            row += 1
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # Verification mode
        ttk.Label(control_frame, text="Mode:").grid(row=row, column=0, sticky=tk.W, pady=5)
        row += 1
        
        ttk.Radiobutton(
            control_frame,
            text="1:1 Verification",
            variable=self.verification_mode,
            value="1:1",
            command=self._on_mode_change
        ).grid(row=row, column=0, sticky=tk.W)
        row += 1
        
        ttk.Radiobutton(
            control_frame,
            text="1:N Identification",
            variable=self.verification_mode,
            value="1:N",
            command=self._on_mode_change
        ).grid(row=row, column=0, sticky=tk.W)
        row += 1
        
        # Person selection (for 1:1 mode)
        self.person_select_label = ttk.Label(control_frame, text="Select Person:")
        self.person_select_combo = ttk.Combobox(
            control_frame,
            textvariable=self.selected_person,
            state="readonly",
            width=15
        )
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # Action buttons
        ttk.Button(
            control_frame,
            text="📷 Capture from Webcam",
            command=self._on_capture_clicked
        ).grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        ttk.Button(
            control_frame,
            text="📁 Load from File",
            command=self._on_load_file_clicked
        ).grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        ttk.Button(
            control_frame,
            text="👥 Manage Gallery",
            command=self._on_manage_gallery_clicked
        ).grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready", foreground="green")
        self.status_label.grid(row=row, column=0, sticky=tk.W, pady=(20, 0))
        row += 1
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        # ===== Top Right: Image Display =====
        image_frame = ttk.LabelFrame(main_frame, text="Images", padding="10")
        image_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Three image panels: Input, Upscaled, Matched
        image_subframe = ttk.Frame(image_frame)
        image_subframe.pack(fill=tk.BOTH, expand=True)
        
        for i, title in enumerate(["Input (VLR)", "Upscaled (DSR)", "Matched Identity"]):
            panel = ttk.Frame(image_subframe)
            panel.grid(row=0, column=i, padx=5, pady=5)
            
            ttk.Label(panel, text=title, font=('TkDefaultFont', 10, 'bold')).pack()
            
            label = ttk.Label(panel, text="No image", relief=tk.SUNKEN, width=25, height=15)
            label.pack(padx=5, pady=5)
            
            if i == 0:
                self.input_image_label = label
            elif i == 1:
                self.upscaled_image_label = label
            else:
                self.matched_image_label = label
        
        # ===== Middle Right: Results =====
        results_frame = ttk.LabelFrame(main_frame, text="Top-5 Predictions", padding="10")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Results text area
        self.results_text = tk.Text(results_frame, height=8, width=60, wrap=tk.WORD)
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure text tags
        self.results_text.tag_configure("header", font=('TkDefaultFont', 11, 'bold'))
        self.results_text.tag_configure("match", foreground="green", font=('TkDefaultFont', 10, 'bold'))
        self.results_text.tag_configure("nomatch", foreground="red")
        self.results_text.tag_configure("timing", foreground="blue", font=('TkDefaultFont', 9, 'italic'))
        
        # ===== Bottom Right: Performance Metrics =====
        metrics_frame = ttk.LabelFrame(main_frame, text="Performance Metrics", padding="10")
        metrics_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        self.metrics_text = tk.Text(metrics_frame, height=6, width=60, wrap=tk.WORD)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        self.metrics_text.tag_configure("metric", font=('TkDefaultFont', 9))
    
    def _initialize_components(self):
        """Initialize pipeline and gallery in background."""
        self._set_status("Initializing...", "blue")
        self.progress.start()
        
        def init_thread():
            try:
                # Initialize gallery
                self.gallery = GalleryManager()
                
                # Initialize pipeline with default resolution
                self.pipeline = LRFRPipeline(vlr_size=self.current_vlr_size)
                
                # Compute embeddings if gallery exists
                if self.gallery.size() > 0:
                    self.gallery.compute_all_embeddings(self.pipeline)
                
                # Update UI
                self.root.after(0, self._on_initialization_complete)
                
            except Exception as e:
                self.root.after(0, lambda: self._on_initialization_error(str(e)))
        
        threading.Thread(target=init_thread, daemon=True).start()
    
    def _on_initialization_complete(self):
        """Called when initialization completes."""
        self.progress.stop()
        self._set_status(f"Ready (Gallery: {self.gallery.size()} people)", "green")
        self._update_person_list()
    
    def _on_initialization_error(self, error: str):
        """Called when initialization fails."""
        self.progress.stop()
        self._set_status(f"Error: {error}", "red")
        messagebox.showerror("Initialization Error", f"Failed to initialize:\n{error}")
    
    def _set_status(self, message: str, color: str = "black"):
        """Update status label."""
        self.status_label.config(text=message, foreground=color)
        self.root.update_idletasks()
    
    def _update_person_list(self):
        """Update person selection dropdown."""
        if self.gallery is None:
            return
        
        names = self.gallery.get_all_names()
        self.person_select_combo['values'] = names
        
        if names and not self.selected_person.get():
            self.selected_person.set(names[0])
    
    def _on_resolution_change(self):
        """Handle resolution change."""
        new_size = self.resolution_var.get()
        
        if new_size == self.current_vlr_size:
            return
        
        # Confirm change (requires reloading models)
        if not messagebox.askyesno(
            "Change Resolution",
            f"Changing resolution to {new_size}×{new_size} requires reloading models.\nContinue?"
        ):
            self.resolution_var.set(self.current_vlr_size)
            return
        
        self.current_vlr_size = new_size
        self._reload_pipeline()
    
    def _reload_pipeline(self):
        """Reload pipeline with new resolution."""
        self._set_status("Reloading models...", "blue")
        self.progress.start()
        
        def reload_thread():
            try:
                # Clean up old pipeline first
                old_pipeline = self.pipeline
                self.pipeline = None
                del old_pipeline
                gc.collect()
                
                # Load new pipeline
                self.pipeline = LRFRPipeline(vlr_size=self.current_vlr_size)
                
                # Recompute embeddings
                if self.gallery.size() > 0:
                    self.gallery.compute_all_embeddings(self.pipeline)
                
                self.root.after(0, lambda: self._on_reload_complete())
                
            except Exception as e:
                self.root.after(0, lambda: self._on_reload_error(str(e)))
        
        threading.Thread(target=reload_thread, daemon=True).start()
    
    def _on_reload_complete(self):
        """Called when pipeline reload completes."""
        self.progress.stop()
        self._set_status(f"Ready ({self.current_vlr_size}×{self.current_vlr_size})", "green")
    
    def _on_reload_error(self, error: str):
        """Called when reload fails."""
        self.progress.stop()
        self._set_status(f"Error: {error}", "red")
        messagebox.showerror("Reload Error", f"Failed to reload models:\n{error}")
    
    def _on_mode_change(self):
        """Handle verification mode change."""
        mode = self.verification_mode.get()
        
        if mode == "1:1":
            # Show person selection
            row = self.person_select_label.grid_info()['row'] if self.person_select_label.grid_info() else 7
            self.person_select_label.grid(row=row, column=0, sticky=tk.W, pady=5)
            self.person_select_combo.grid(row=row+1, column=0, sticky=(tk.W, tk.E), pady=5)
        else:
            # Hide person selection
            self.person_select_label.grid_remove()
            self.person_select_combo.grid_remove()
    
    def _on_capture_clicked(self):
        """Handle webcam capture button click."""
        if self.is_processing:
            messagebox.showwarning("Busy", "Processing in progress, please wait...")
            return
        
        if self.gallery.size() == 0:
            messagebox.showwarning("No Gallery", "Please add people to gallery first.")
            return
        
        # Open webcam and capture
        self._capture_from_webcam()
    
    def _capture_from_webcam(self):
        """Capture face from webcam."""
        self._set_status("Opening webcam...", "blue")
        
        try:
            webcam = WebcamCapture()
            if not webcam.open():
                raise RuntimeError("Failed to open webcam")
            
            self._set_status("Position yourself and press SPACE to capture", "blue")
            
            # Capture face
            face = webcam.capture_face(output_size=config.HR_SIZE)
            webcam.close()
            
            if face is None:
                self._set_status("Capture cancelled", "orange")
                return
            
            # Process captured face
            self._process_face(face)
            
        except Exception as e:
            self._set_status(f"Webcam error: {str(e)}", "red")
            messagebox.showerror("Webcam Error", f"Failed to capture from webcam:\n{str(e)}")
    
    def _on_load_file_clicked(self):
        """Handle load from file button click."""
        if self.is_processing:
            messagebox.showwarning("Busy", "Processing in progress, please wait...")
            return
        
        if self.gallery.size() == 0:
            messagebox.showwarning("No Gallery", "Please add people to gallery first.")
            return
        
        # Open file dialog
        filename = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not filename:
            return
        
        # Load and process image
        try:
            image = cv2.imread(filename)
            if image is None:
                raise ValueError("Failed to load image")
            
            # Detect and crop face
            detector = FaceDetector()
            face = detector.detect_and_crop(image, output_size=config.HR_SIZE)
            
            if face is None:
                messagebox.showwarning("No Face", "No face detected in image. Please select a clear face image.")
                return
            
            self._process_face(face)
            
        except Exception as e:
            messagebox.showerror("File Error", f"Failed to load image:\n{str(e)}")
    
    def _process_face(self, face_image: np.ndarray):
        """Process face through pipeline and display results."""
        if self.pipeline is None or self.gallery is None:
            messagebox.showerror("Not Ready", "System not initialized yet.")
            return
        
        self.is_processing = True
        self._set_status("Processing...", "blue")
        self.progress.start()
        
        def process_thread():
            try:
                # Run pipeline
                result = self.pipeline.process_image(face_image, return_intermediate=True)
                
                # Get gallery embeddings
                gallery_embeddings, gallery_names = self.gallery.get_gallery_embeddings()
                
                if not gallery_embeddings:
                    raise ValueError("No embeddings in gallery")
                
                # Perform identification
                mode = self.verification_mode.get()
                
                if mode == "1:1":
                    # Verify against selected person
                    selected = self.selected_person.get()
                    if not selected:
                        raise ValueError("No person selected for 1:1 verification")
                    
                    person_idx = gallery_names.index(selected)
                    is_match, similarity = self.pipeline.verify_1_to_1(
                        result["embedding"],
                        gallery_embeddings[person_idx]
                    )
                    
                    result["mode"] = "1:1"
                    result["target_person"] = selected
                    result["is_match"] = is_match
                    result["similarity"] = similarity
                    result["predictions"] = [(selected, similarity, is_match)]
                    
                else:  # 1:N
                    predictions = self.pipeline.identify_1_to_n(
                        result["embedding"],
                        gallery_embeddings,
                        gallery_names
                    )
                    
                    result["mode"] = "1:N"
                    result["predictions"] = predictions
                
                # Update UI
                self.root.after(0, lambda: self._on_processing_complete(result))
                
            except Exception as e:
                self.root.after(0, lambda: self._on_processing_error(str(e)))
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def _on_processing_complete(self, result: Dict):
        """Called when processing completes."""
        self.progress.stop()
        self.is_processing = False
        self._set_status("Processing complete", "green")
        
        # Store result
        self.last_result = result
        
        # Update displays
        self._update_image_displays(result)
        self._update_results_display(result)
        self._update_metrics_display(result)
        
        # Clean up memory after display update
        gc.collect()
    
    def _on_processing_error(self, error: str):
        """Called when processing fails."""
        self.progress.stop()
        self.is_processing = False
        self._set_status(f"Error: {error}", "red")
        messagebox.showerror("Processing Error", f"Failed to process image:\n{error}")
    
    def _update_image_displays(self, result: Dict):
        """Update image display panels."""
        # Input (VLR)
        if "vlr_image" in result:
            self._display_image(result["vlr_image"], self.input_image_label)
        
        # Upscaled (DSR output)
        if "sr_image" in result:
            self._display_image(result["sr_image"], self.upscaled_image_label)
        
        # Matched identity (rank-1 prediction)
        if "predictions" in result and len(result["predictions"]) > 0:
            top_match_name = result["predictions"][0][0]
            thumbnail = self.gallery.get_person_thumbnail(top_match_name)
            if thumbnail is not None:
                self._display_image(thumbnail, self.matched_image_label)
    
    def _display_image(self, cv_image: np.ndarray, label: ttk.Label):
        """Convert OpenCV image to Tkinter and display."""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Resize to fit display
        rgb = cv2.resize(rgb, config.RESULT_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb)
        
        # Convert to Tkinter PhotoImage
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Update label
        label.configure(image=tk_image, text="")
        label.image = tk_image  # Keep reference to prevent garbage collection
        
        # Clean up intermediate objects
        del pil_image
        del rgb
    
    def _update_results_display(self, result: Dict):
        """Update results text area with predictions."""
        self.results_text.delete(1.0, tk.END)
        
        mode = result.get("mode", "1:N")
        predictions = result.get("predictions", [])
        
        if mode == "1:1":
            # 1:1 Verification
            target = result.get("target_person", "Unknown")
            is_match = result.get("is_match", False)
            similarity = result.get("similarity", 0.0)
            
            self.results_text.insert(tk.END, "1:1 Verification Result\n", "header")
            self.results_text.insert(tk.END, f"\nTarget: {target}\n")
            
            if is_match:
                self.results_text.insert(tk.END, f"✓ MATCH (Confidence: {similarity:.2%})\n", "match")
            else:
                self.results_text.insert(tk.END, f"✗ NO MATCH (Similarity: {similarity:.2%})\n", "nomatch")
            
            threshold = config.VERIFICATION_THRESHOLD_1_1
            self.results_text.insert(tk.END, f"\nThreshold: {threshold:.2%}\n", "timing")
            
        else:
            # 1:N Identification
            self.results_text.insert(tk.END, "1:N Identification Results\n", "header")
            self.results_text.insert(tk.END, f"\nTop-{len(predictions)} Matches:\n\n")
            
            for rank, (name, similarity, is_valid) in enumerate(predictions, 1):
                prefix = "✓" if is_valid else "✗"
                tag = "match" if is_valid else "nomatch"
                
                self.results_text.insert(tk.END, f"{rank}. {prefix} {name}\n", tag)
                self.results_text.insert(tk.END, f"   Confidence: {similarity:.2%}\n\n")
            
            threshold = config.IDENTIFICATION_THRESHOLD_1_N
            self.results_text.insert(tk.END, f"\nMinimum threshold: {threshold:.2%}\n", "timing")
    
    def _update_metrics_display(self, result: Dict):
        """Update performance metrics display."""
        self.metrics_text.delete(1.0, tk.END)
        
        timings = result.get("timings", {})
        total_time = result.get("total_time", 0)
        
        self.metrics_text.insert(tk.END, "Processing Time Breakdown:\n\n", "metric")
        
        # Per-stage timings
        for stage, time_ms in timings.items():
            self.metrics_text.insert(tk.END, f"  {stage.capitalize()}: {time_ms:.1f} ms\n", "metric")
        
        self.metrics_text.insert(tk.END, f"\nTotal Time: {total_time:.1f} ms\n", "metric")
        self.metrics_text.insert(tk.END, f"Throughput: {1000/total_time:.1f} FPS\n\n", "metric")
        
        # Model info
        self.metrics_text.insert(tk.END, f"Resolution: {self.current_vlr_size}×{self.current_vlr_size} → 112×112\n", "metric")
        self.metrics_text.insert(tk.END, f"Gallery Size: {self.gallery.size()} people\n", "metric")
    
    def _on_manage_gallery_clicked(self):
        """Open gallery management dialog."""
        GalleryDialog(self.root, self.gallery, self.pipeline, self._update_person_list)


class GalleryDialog:
    """Dialog for managing gallery."""
    
    def __init__(self, parent, gallery: GalleryManager, pipeline: LRFRPipeline, callback):
        """Initialize gallery management dialog."""
        self.gallery = gallery
        self.pipeline = pipeline
        self.callback = callback
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Gallery Management")
        self.dialog.geometry("600x400")
        
        self._create_widgets()
        self._refresh_list()
    
    def _create_widgets(self):
        """Create dialog widgets."""
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Gallery list
        list_frame = ttk.LabelFrame(main_frame, text="Gallery", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.gallery_listbox = tk.Listbox(list_frame, height=10)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.gallery_listbox.yview)
        self.gallery_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.gallery_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Add Person", command=self._on_add_person).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Remove Selected", command=self._on_remove_person).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear All", command=self._on_clear_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=self.dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        # Status
        self.status_label = ttk.Label(main_frame, text=f"{self.gallery.size()}/{config.MAX_GALLERY_SIZE} people")
        self.status_label.pack(pady=(10, 0))
    
    def _refresh_list(self):
        """Refresh gallery list."""
        self.gallery_listbox.delete(0, tk.END)
        
        for name in self.gallery.get_all_names():
            person = self.gallery.get_person(name)
            num_images = len(person.image_paths) if person else 0
            self.gallery_listbox.insert(tk.END, f"{name} ({num_images} images)")
        
        self.status_label.config(text=f"{self.gallery.size()}/{config.MAX_GALLERY_SIZE} people")
    
    def _on_add_person(self):
        """Add new person to gallery."""
        if self.gallery.is_full():
            messagebox.showwarning("Gallery Full", f"Gallery is full ({config.MAX_GALLERY_SIZE} people max)")
            return
        
        AddPersonDialog(self.dialog, self.gallery, self.pipeline, self._refresh_list)
    
    def _on_remove_person(self):
        """Remove selected person."""
        selection = self.gallery_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a person to remove")
            return
        
        name = self.gallery.get_all_names()[selection[0]]
        
        if messagebox.askyesno("Confirm", f"Remove '{name}' from gallery?"):
            self.gallery.remove_person(name)
            self._refresh_list()
            self.callback()  # Update main window
    
    def _on_clear_all(self):
        """Clear entire gallery."""
        if self.gallery.size() == 0:
            return
        
        if messagebox.askyesno("Confirm", f"Remove all {self.gallery.size()} people from gallery?"):
            self.gallery.clear()
            self._refresh_list()
            self.callback()


class AddPersonDialog:
    """Dialog for adding a person to gallery."""
    
    def __init__(self, parent, gallery: GalleryManager, pipeline: LRFRPipeline, callback):
        """Initialize add person dialog."""
        self.gallery = gallery
        self.pipeline = pipeline
        self.callback = callback
        self.images = []
        
        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Add Person")
        self.dialog.geometry("500x400")
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Name entry
        ttk.Label(main_frame, text="Name:").pack(anchor=tk.W)
        self.name_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.name_var, width=30).pack(fill=tk.X, pady=(0, 10))
        
        # Image collection
        images_frame = ttk.LabelFrame(main_frame, text="Images", padding="10")
        images_frame.pack(fill=tk.BOTH, expand=True)
        
        button_frame = ttk.Frame(images_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Capture from Webcam", command=self._capture_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load from File", command=self._load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear All", command=self._clear_images).pack(side=tk.LEFT, padx=5)
        
        self.images_label = ttk.Label(images_frame, text=f"0/{config.MAX_IMAGES_PER_PERSON} images")
        self.images_label.pack(pady=10)
        
        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(action_frame, text="Add to Gallery", command=self._on_add).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _capture_image(self):
        """Capture image from webcam."""
        if len(self.images) >= config.MAX_IMAGES_PER_PERSON:
            messagebox.showwarning("Limit Reached", f"Maximum {config.MAX_IMAGES_PER_PERSON} images per person")
            return
        
        try:
            webcam = WebcamCapture()
            if not webcam.open():
                raise RuntimeError("Failed to open webcam")
            
            face = webcam.capture_face(output_size=config.HR_SIZE)
            webcam.close()
            
            if face is not None:
                self.images.append(face)
                self._update_images_label()
        
        except Exception as e:
            messagebox.showerror("Webcam Error", f"Failed to capture:\n{str(e)}")
    
    def _load_image(self):
        """Load image from file."""
        if len(self.images) >= config.MAX_IMAGES_PER_PERSON:
            messagebox.showwarning("Limit Reached", f"Maximum {config.MAX_IMAGES_PER_PERSON} images per person")
            return
        
        filename = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            image = cv2.imread(filename)
            if image is None:
                raise ValueError("Failed to load image")
            
            self.images.append(image)
            self._update_images_label()
        
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load image:\n{str(e)}")
    
    def _clear_images(self):
        """Clear all images."""
        self.images.clear()
        self._update_images_label()
    
    def _update_images_label(self):
        """Update images count label."""
        self.images_label.config(text=f"{len(self.images)}/{config.MAX_IMAGES_PER_PERSON} images")
    
    def _on_add(self):
        """Add person to gallery."""
        name = self.name_var.get().strip()
        
        if not name:
            messagebox.showwarning("No Name", "Please enter a name")
            return
        
        if len(self.images) < config.MIN_IMAGES_PER_PERSON:
            messagebox.showwarning(
                "Not Enough Images",
                f"Please provide at least {config.MIN_IMAGES_PER_PERSON} images"
            )
            return
        
        # Add to gallery
        success = self.gallery.add_person(name, self.images, pipeline=self.pipeline, overwrite=False)
        
        if success:
            messagebox.showinfo("Success", f"Added '{name}' to gallery")
            self.callback()  # Refresh parent
            self.dialog.destroy()
        else:
            messagebox.showerror("Error", "Failed to add person to gallery")


def main():
    """Main entry point."""
    # Set up PyTorch for Pi 5
    import torch
    torch.set_num_threads(config.TORCH_THREADS)
    
    # Create main window
    root = tk.Tk()
    app = LRFRApp(root)
    
    # Run
    root.mainloop()


if __name__ == "__main__":
    main()
