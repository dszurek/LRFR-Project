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
        self.verification_mode = tk.StringVar(value="1:N")
        self.selected_person = tk.StringVar(value="")
        
        # Model paths (defaults from config)
        self.dsr_model_path = tk.StringVar(value="")
        self.edgeface_model_path = tk.StringVar(value="")
        
        # Pipeline lock to serialize all pipeline access
        self.pipeline_lock = threading.Lock()
        
        # Processing state
        self.is_processing = False
        self.is_capturing = False
        self.last_result: Optional[Dict] = None
        
        # References to prevent image garbage collection
        self.input_tk_image: Optional[ImageTk.PhotoImage] = None
        self.upscaled_tk_image: Optional[ImageTk.PhotoImage] = None
        self.matched_tk_image: Optional[ImageTk.PhotoImage] = None
        
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
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(0, weight=2)
        main_frame.rowconfigure(1, weight=1)
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
        
        # Action buttons - SHORTENED TEXT
        ttk.Button(
            control_frame,
            text="📷 Webcam",
            command=self._on_capture_clicked
        ).grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        ttk.Button(
            control_frame,
            text="📁 Load File",
            command=self._on_load_file_clicked
        ).grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        ttk.Button(
            control_frame,
            text="👥 Gallery",
            command=self._on_manage_gallery_clicked
        ).grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        ttk.Button(
            control_frame,
            text="⚙️ Models",
            command=self._on_model_settings_clicked
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
        image_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Three image panels: Input, Upscaled, Matched
        image_subframe = ttk.Frame(image_frame)
        image_subframe.pack(fill=tk.BOTH, expand=True)
        
        # Configure column weights to distribute space evenly
        for i in range(3):
            image_subframe.columnconfigure(i, weight=1)
        image_subframe.rowconfigure(0, weight=1)
        
        for i, title in enumerate(["Input (VLR)", "Upscaled (DSR)", "Matched Identity"]):
            panel = ttk.Frame(image_subframe)
            panel.grid(row=0, column=i, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
            panel.rowconfigure(1, weight=1)
            panel.columnconfigure(0, weight=1)
            
            ttk.Label(panel, text=title, font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, sticky=tk.W)
            
            # Use tk.Label with proper configuration
            label = tk.Label(
                panel,
                text="No image",
                relief=tk.SUNKEN,
                bg="gray90",
                compound=tk.CENTER,
                anchor=tk.CENTER
            )
            label.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
            
            # Set minimum size for the label
            label.configure(width=150, height=150)
            
            if i == 0:
                self.input_image_label = label
            elif i == 1:
                self.upscaled_image_label = label
            else:
                self.matched_image_label = label
        
        # ===== Middle Right: Results =====
        results_frame = ttk.LabelFrame(main_frame, text="Top-5 Predictions", padding="10")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Results text area
        self.results_text = tk.Text(results_frame, height=6, width=60, wrap=tk.WORD)
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
        metrics_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.metrics_text = tk.Text(metrics_frame, height=4, width=60, wrap=tk.WORD)
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
                
                # Use the lock during initialization
                with self.pipeline_lock:
                    # Initialize pipeline with default resolution
                    dsr_path = self.dsr_model_path.get() if self.dsr_model_path.get() else None
                    edgeface_path = self.edgeface_model_path.get() if self.edgeface_model_path.get() else None
                    
                    self.pipeline = LRFRPipeline(
                        vlr_size=self.current_vlr_size,
                        dsr_model_path=dsr_path,
                        edgeface_model_path=edgeface_path
                    )
                    
                    # Compute embeddings if gallery exists
                    if self.gallery.size() > 0:
                        # Progress callback to update status
                        def update_progress(current, total, name):
                            msg = f"Computing embeddings: {name} ({current}/{total})"
                            self.root.after(0, lambda: self._set_status(msg, "blue"))
                        
                        # Force recomputation based on config
                        force = config.FORCE_RECOMPUTE_EMBEDDINGS_ON_STARTUP
                        self.gallery.compute_all_embeddings(
                            self.pipeline, 
                            force=force,
                            progress_callback=update_progress
                        )
                
                # Update UI
                self.root.after(0, self._on_initialization_complete)
                
            except Exception as e:
                self.root.after(0, lambda err=str(e): self._on_initialization_error(err))
        
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
                
                # Use the lock during reload
                with self.pipeline_lock:
                    # Load new pipeline
                    dsr_path = self.dsr_model_path.get() if self.dsr_model_path.get() else None
                    edgeface_path = self.edgeface_model_path.get() if self.edgeface_model_path.get() else None
                    
                    self.pipeline = LRFRPipeline(
                        vlr_size=self.current_vlr_size,
                        dsr_model_path=dsr_path,
                        edgeface_model_path=edgeface_path
                    )
                    
                    # Recompute embeddings with progress callback
                    if self.gallery.size() > 0:
                        def update_progress(current, total, name):
                            msg = f"Recomputing embeddings: {name} ({current}/{total})"
                            self.root.after(0, lambda: self._set_status(msg, "blue"))
                        
                        self.gallery.compute_all_embeddings(
                            self.pipeline,
                            force=True,  # Always force on reload
                            progress_callback=update_progress
                        )
                
                self.root.after(0, self._on_reload_complete)
                
            except Exception as e:
                self.root.after(0, lambda err=str(e): self._on_reload_error(err))
        
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
    
    def _on_model_settings_clicked(self):
        """Open model settings dialog."""
        ModelSettingsDialog(self.root, self.dsr_model_path, self.edgeface_model_path, self._reload_pipeline)
    
    def _on_capture_clicked(self):
        """Handle webcam capture button click."""
        if self.is_processing:
            messagebox.showwarning("Busy", "Processing in progress, please wait...")
            return
        
        if self.is_capturing:
            messagebox.showwarning("Busy", "Webcam capture already in progress...")
            return
        
        # Gallery check for 1:N mode
        if self.verification_mode.get() == "1:N" and self.gallery.size() == 0:
            messagebox.showwarning("No Gallery", "Please add people to gallery first for 1:N identification.")
            return
        
        # Open webcam and capture - RUN ON MAIN THREAD
        self._capture_from_webcam_main_thread()
    
    def _capture_from_webcam_main_thread(self):
        """Capture face from webcam - runs entirely on main thread."""
        if self.is_capturing:
            return
        
        self.is_capturing = True
        self._set_status("Opening webcam...", "blue")
        
        # Create webcam
        self.webcam = WebcamCapture()
        if not self.webcam.open():
            self.is_capturing = False
            self._set_status("Failed to open webcam", "red")
            messagebox.showerror("Webcam Error", "Failed to open webcam")
            return
        
        self._set_status("Position yourself and press SPACE to capture", "blue")
        
        # Start polling for frames using Tkinter's after method
        self._webcam_poll()
    
    def _webcam_poll(self):
        """Poll webcam for frames - called repeatedly by Tkinter."""
        if not self.is_capturing or self.webcam is None:
            return
        
        # Read and display frame
        face = self.webcam.poll_for_capture()
        
        if face is not None:
            # Capture successful or cancelled
            self.webcam.close()
            self.webcam = None
            self.is_capturing = False
            
            if isinstance(face, np.ndarray):
                # Got a face image
                self._set_status("Face captured!", "green")
                self._process_face(face.copy())
            else:
                # Cancelled
                self._set_status("Capture cancelled", "orange")
        else:
            # Continue polling
            self.root.after(10, self._webcam_poll)
    
    def _on_load_file_clicked(self):
        """Handle load from file button click."""
        if self.is_processing:
            messagebox.showwarning("Busy", "Processing in progress, please wait...")
            return
        
        if self.verification_mode.get() == "1:N" and self.gallery.size() == 0:
            messagebox.showwarning("No Gallery", "Please add people to gallery first for 1:N identification.")
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
                # Use the lock during processing
                with self.pipeline_lock:
                    # Run pipeline
                    result = self.pipeline.process_image(face_image, return_intermediate=True)
                    
                    # Get gallery embeddings
                    gallery_embeddings, gallery_names = self.gallery.get_gallery_embeddings()
                    
                    # Perform identification
                    mode = self.verification_mode.get()
                    
                    if mode == "1:1":
                        # Verify against selected person
                        selected = self.selected_person.get()
                        if not selected:
                            raise ValueError("No person selected for 1:1 verification")
                        if not gallery_embeddings:
                             raise ValueError("Gallery is empty, cannot verify")
                        
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
                        if not gallery_embeddings:
                            raise ValueError("No embeddings in gallery, cannot identify")
                        
                        predictions = self.pipeline.identify_1_to_n(
                            result["embedding"],
                            gallery_embeddings,
                            gallery_names
                        )
                        
                        result["mode"] = "1:N"
                        result["predictions"] = predictions
                
                # Update UI
                self.root.after(0, lambda res=result: self._on_processing_complete(res))
                
            except Exception as e:
                self.root.after(0, lambda err=str(e): self._on_processing_error(err))
        
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
            self._display_image(result["vlr_image"], "input")
        else:
            self._display_image(None, "input")
        
        # Upscaled (DSR output)
        if "sr_image" in result:
            self._display_image(result["sr_image"], "upscaled")
        else:
            self._display_image(None, "upscaled")
        
        # Matched identity (rank-1 prediction)
        thumbnail = None
        if "predictions" in result and len(result["predictions"]) > 0:
            top_match_name = result["predictions"][0][0]
            # Check if match is valid before showing thumbnail
            if result["predictions"][0][2]:
                thumbnail = self.gallery.get_person_thumbnail(top_match_name)
            
        if thumbnail is not None:
            self._display_image(thumbnail, "matched")
        else:
            self._display_image(None, "matched")

    def _display_image(self, cv_image: Optional[np.ndarray], image_type: str):
        """Convert OpenCV image to Tkinter and display."""
        
        label = None
        if image_type == "input":
            label = self.input_image_label
        elif image_type == "upscaled":
            label = self.upscaled_image_label
        elif image_type == "matched":
            label = self.matched_image_label
        
        if cv_image is None:
            # Clear the image
            label.configure(image='', text="No image")
            if image_type == "input":
                self.input_tk_image = None
            elif image_type == "upscaled":
                self.upscaled_tk_image = None
            elif image_type == "matched":
                self.matched_tk_image = None
            return

        # Convert BGR to RGB
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Resize to fit display
        display_size = (200, 200)
        rgb = cv2.resize(rgb, display_size, interpolation=cv2.INTER_AREA)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb)
        
        # Convert to Tkinter PhotoImage
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Update label
        label.configure(image=tk_image, text="")
        
        # Keep reference on self to prevent garbage collection
        if image_type == "input":
            self.input_tk_image = tk_image
        elif image_type == "upscaled":
            self.upscaled_tk_image = tk_image
        elif image_type == "matched":
            self.matched_tk_image = tk_image
    
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
            self.results_text.insert(tk.END, f"Similarity Score: {similarity:.4f} ({similarity:.2%})\n")
            
            if is_match:
                self.results_text.insert(tk.END, f"✓ MATCH\n", "match")
            else:
                self.results_text.insert(tk.END, f"✗ NO MATCH\n", "nomatch")
            
            threshold = config.VERIFICATION_THRESHOLD_1_1
            self.results_text.insert(tk.END, f"\nThreshold: {threshold:.4f} ({threshold:.2%})\n", "timing")
            self.results_text.insert(tk.END, f"Note: Similarity = cosine similarity of embeddings (range: -1 to 1)\n", "timing")
            
        else:
            # 1:N Identification
            self.results_text.insert(tk.END, "1:N Identification Results\n", "header")
            
            if not predictions:
                self.results_text.insert(tk.END, "\nNo one in gallery to identify.", "nomatch")
                return

            self.results_text.insert(tk.END, f"\nTop-{len(predictions)} Matches:\n\n")
            
            for rank, (name, similarity, is_valid) in enumerate(predictions, 1):
                prefix = "✓" if is_valid else "✗"
                tag = "match" if is_valid else "nomatch"
                
                self.results_text.insert(tk.END, f"{rank}. {prefix} {name}\n", tag)
                self.results_text.insert(tk.END, f"   Similarity: {similarity:.4f} ({similarity:.2%})\n\n")
            
            threshold = config.IDENTIFICATION_THRESHOLD_1_N
            self.results_text.insert(tk.END, f"\nMinimum threshold: {threshold:.4f} ({threshold:.2%})\n", "timing")
            self.results_text.insert(tk.END, f"Note: Similarity = cosine similarity (range: -1 to 1, higher is better)\n", "timing")
    
    def _update_metrics_display(self, result: Dict):
        """Update performance metrics display."""
        self.metrics_text.delete(1.0, tk.END)
        
        timings = result.get("timings", {})
        total_time = result.get("total_time", 0)
        
        self.metrics_text.insert(tk.END, "Processing Time Breakdown:\n\n", "metric")
        
        # Per-stage timings
        if timings:
            for stage, time_ms in timings.items():
                self.metrics_text.insert(tk.END, f"  {stage.capitalize()}: {time_ms:.1f} ms\n", "metric")
            
            self.metrics_text.insert(tk.END, f"\nTotal Time: {total_time:.1f} ms\n", "metric")
            if total_time > 0:
                self.metrics_text.insert(tk.END, f"Throughput: {1000/total_time:.1f} FPS\n\n", "metric")
        else:
             self.metrics_text.insert(tk.END, "  No timing data available.\n", "metric")

        # Model info
        self.metrics_text.insert(tk.END, f"Resolution: {self.current_vlr_size}×{self.current_vlr_size} → 112×112\n", "metric")
        if self.gallery:
            self.metrics_text.insert(tk.END, f"Gallery Size: {self.gallery.size()} people\n", "metric")
    
    def _on_manage_gallery_clicked(self):
        """Open gallery management dialog."""
        GalleryDialog(self.root, self.gallery, self.pipeline, self.pipeline_lock, self._update_person_list)


class ModelSettingsDialog:
    """Dialog for configuring model paths."""
    
    def __init__(self, parent, dsr_path_var: tk.StringVar, edgeface_path_var: tk.StringVar, callback):
        """Initialize model settings dialog."""
        self.dsr_path_var = dsr_path_var
        self.edgeface_path_var = edgeface_path_var
        self.callback = callback
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Model Settings")
        self.dialog.geometry("600x300")
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # DSR Model Path
        ttk.Label(main_frame, text="DSR Model Path:", font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        dsr_frame = ttk.Frame(main_frame)
        dsr_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.dsr_entry = ttk.Entry(dsr_frame, textvariable=self.dsr_path_var, width=50)
        self.dsr_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(dsr_frame, text="Browse...", command=self._browse_dsr).pack(side=tk.LEFT)
        ttk.Button(dsr_frame, text="Clear", command=lambda: self.dsr_path_var.set("")).pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Label(main_frame, text="(Leave empty to use default)", font=('TkDefaultFont', 8, 'italic')).pack(anchor=tk.W, pady=(0, 15))
        
        # EdgeFace Model Path
        ttk.Label(main_frame, text="EdgeFace Model Path:", font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        edgeface_frame = ttk.Frame(main_frame)
        edgeface_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.edgeface_entry = ttk.Entry(edgeface_frame, textvariable=self.edgeface_path_var, width=50)
        self.edgeface_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(edgeface_frame, text="Browse...", command=self._browse_edgeface).pack(side=tk.LEFT)
        ttk.Button(edgeface_frame, text="Clear", command=lambda: self.edgeface_path_var.set("")).pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Label(main_frame, text="(Leave empty to use default)", font=('TkDefaultFont', 8, 'italic')).pack(anchor=tk.W, pady=(0, 20))
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Apply & Reload", command=self._on_apply).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=self.dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _browse_dsr(self):
        """Browse for DSR model file."""
        filename = filedialog.askopenfilename(
            title="Select DSR Model",
            filetypes=[("PyTorch Model", "*.pth *.pt"), ("All files", "*.*")]
        )
        if filename:
            self.dsr_path_var.set(filename)
    
    def _browse_edgeface(self):
        """Browse for EdgeFace model file."""
        filename = filedialog.askopenfilename(
            title="Select EdgeFace Model",
            filetypes=[("PyTorch Model", "*.pth *.pt"), ("All files", "*.*")]
        )
        if filename:
            self.edgeface_path_var.set(filename)
    
    def _on_apply(self):
        """Apply settings and reload pipeline."""
        if messagebox.askyesno("Confirm", "This will reload the models with the new paths.\nContinue?"):
            self.dialog.destroy()
            self.callback()


class GalleryDialog:
    """Dialog for managing gallery."""
    
    def __init__(self, parent, gallery: GalleryManager, pipeline: LRFRPipeline, pipeline_lock: threading.Lock, callback):
        """Initialize gallery management dialog."""
        self.gallery = gallery
        self.pipeline = pipeline
        self.pipeline_lock = pipeline_lock
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
        
        AddPersonDialog(self.dialog, self.gallery, self.pipeline, self.pipeline_lock, self._refresh_list)
    
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
            self.callback()
    
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
    
    def __init__(self, parent, gallery: GalleryManager, pipeline: LRFRPipeline, pipeline_lock: threading.Lock, callback):
        """Initialize add person dialog."""
        self.gallery = gallery
        self.pipeline = pipeline
        self.pipeline_lock = pipeline_lock
        self.callback = callback
        self.images = []
        self.is_capturing = False
        self.is_auto_capturing = False
        self.webcam = None
        self.face_detector = FaceDetector()
        self.auto_capture_start_time = None
        
        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Add Person")
        self.dialog.geometry("500x450")
        
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
        
        self.auto_capture_btn = ttk.Button(button_frame, text="Auto Capture (100 photos)", command=self._auto_capture)
        self.auto_capture_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Manual Capture", command=self._capture_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load from File", command=self._load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear All", command=self._clear_images).pack(side=tk.LEFT, padx=5)
        
        self.images_label = ttk.Label(images_frame, text=f"0/{config.MAX_IMAGES_PER_PERSON} images")
        self.images_label.pack(pady=5)
        
        self.status_label = ttk.Label(images_frame, text="", foreground="blue")
        self.status_label.pack(pady=5)
        
        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(action_frame, text="Add to Gallery", command=self._on_add).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _capture_image(self):
        """Capture image from webcam (manual mode - single capture)."""
        if len(self.images) >= config.MAX_IMAGES_PER_PERSON:
            messagebox.showwarning("Limit Reached", f"Maximum {config.MAX_IMAGES_PER_PERSON} images per person")
            return
        
        if self.is_capturing or self.is_auto_capturing:
            messagebox.showwarning("Busy", "Webcam capture in progress...")
            return
        
        self.is_capturing = True
        
        # Create webcam
        self.webcam = WebcamCapture()
        if not self.webcam.open():
            self.is_capturing = False
            messagebox.showerror("Webcam Error", "Failed to open webcam")
            return
        
        # Start polling
        self._webcam_poll()
    
    def _auto_capture(self):
        """Start automatic capture mode - captures 100 photos automatically."""
        if self.is_capturing or self.is_auto_capturing:
            messagebox.showwarning("Busy", "Webcam capture already in progress...")
            return
        
        # Clear existing images
        if self.images:
            if not messagebox.askyesno("Clear Images?", f"This will clear the existing {len(self.images)} images. Continue?"):
                return
            self.images.clear()
            self._update_images_label()
        
        self.is_auto_capturing = True
        self.auto_capture_start_time = time.time()
        
        # Disable auto capture button
        self.auto_capture_btn.config(state='disabled')
        
        # Update status
        self.status_label.config(text="Starting auto capture... Position your face in frame and move slowly.", foreground="blue")
        
        # Create webcam with custom window name
        self.webcam = WebcamCapture()
        self.webcam.window_name = "Auto Capture - Move your head slowly around. ESC to stop"
        
        if not self.webcam.open():
            self.is_auto_capturing = False
            self.auto_capture_btn.config(state='normal')
            messagebox.showerror("Webcam Error", "Failed to open webcam")
            return
        
        # Start auto-capture polling
        self._auto_capture_poll()
    
    def _auto_capture_poll(self):
        """Poll webcam for automatic capture."""
        if not self.is_auto_capturing or self.webcam is None:
            return
        
        # Check if we've reached the limit
        if len(self.images) >= config.MAX_IMAGES_PER_PERSON:
            elapsed = time.time() - self.auto_capture_start_time
            self.status_label.config(
                text=f"Auto capture complete! Captured {len(self.images)} images in {elapsed:.1f} seconds.",
                foreground="green"
            )
            self.webcam.close()
            self.webcam = None
            self.is_auto_capturing = False
            self.auto_capture_btn.config(state='normal')
            messagebox.showinfo("Complete", f"Successfully captured {len(self.images)} images!")
            return
        
        # Read frame
        if not self.webcam.is_opened or self.webcam.cap is None:
            self._stop_auto_capture("Webcam closed unexpectedly")
            return
        
        ret, frame = self.webcam.cap.read()
        
        if not ret or frame is None:
            self._stop_auto_capture("Failed to read frame")
            return
        
        # Update FPS
        self.webcam.fps_frame_count += 1
        elapsed = time.time() - self.webcam.fps_start_time
        if elapsed >= 1.0:
            self.webcam.current_fps = self.webcam.fps_frame_count / elapsed
            self.webcam.fps_frame_count = 0
            self.webcam.fps_start_time = time.time()
        
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        
        # Create display frame
        display_frame = frame.copy()
        
        # If face detected, capture it
        face_detected = False
        if faces:
            face_detected = True
            x, y, w, h = faces[0]
            
            # Draw rectangle
            cv2.rectangle(
                display_frame,
                (x, y),
                (x + w, y + h),
                config.COLOR_GREEN,
                2
            )
            
            # Capture the face
            captured_face = self.face_detector.detect_and_crop(
                frame,
                output_size=config.HR_SIZE
            )
            
            if captured_face is not None:
                self.images.append(captured_face.copy())
                self.dialog.after(0, self._update_images_label)
                
                # Update status
                elapsed = time.time() - self.auto_capture_start_time
                self.status_label.config(
                    text=f"Capturing... {len(self.images)}/{config.MAX_IMAGES_PER_PERSON} images ({elapsed:.1f}s)",
                    foreground="blue"
                )
        
        # Add overlay text
        cv2.putText(
            display_frame,
            f"FPS: {self.webcam.current_fps:.1f}",
            (10, 30),
            config.FONT_FACE,
            config.FONT_SCALE,
            config.COLOR_GREEN,
            config.FONT_THICKNESS
        )
        
        status_text = f"Captured: {len(self.images)}/{config.MAX_IMAGES_PER_PERSON}"
        cv2.putText(
            display_frame,
            status_text,
            (10, 60),
            config.FONT_FACE,
            config.FONT_SCALE,
            config.COLOR_GREEN if face_detected else config.COLOR_RED,
            config.FONT_THICKNESS
        )
        
        instruction_text = "Move your head slowly - looking left, right, up, down"
        cv2.putText(
            display_frame,
            instruction_text,
            (10, 90),
            config.FONT_FACE,
            config.FONT_SCALE * 0.7,
            config.COLOR_YELLOW,
            config.FONT_THICKNESS
        )
        
        if not face_detected:
            cv2.putText(
                display_frame,
                "No face detected - position yourself in frame",
                (10, 120),
                config.FONT_FACE,
                config.FONT_SCALE * 0.7,
                config.COLOR_RED,
                config.FONT_THICKNESS
            )
        
        cv2.putText(
            display_frame,
            "ESC to stop",
            (10, display_frame.shape[0] - 20),
            config.FONT_FACE,
            config.FONT_SCALE,
            config.COLOR_YELLOW,
            config.FONT_THICKNESS
        )
        
        # Show frame
        cv2.imshow(self.webcam.window_name, display_frame)
        
        # Check for ESC key
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            self._stop_auto_capture(f"Stopped by user. Captured {len(self.images)} images.")
            return
        
        # Continue polling
        self.dialog.after(10, self._auto_capture_poll)
    
    def _stop_auto_capture(self, message: str):
        """Stop auto capture and clean up."""
        if self.webcam:
            self.webcam.close()
            self.webcam = None
        
        self.is_auto_capturing = False
        self.auto_capture_btn.config(state='normal')
        self.status_label.config(text=message, foreground="orange")
    
    def _webcam_poll(self):
        """Poll webcam for frames (manual mode)."""
        if not self.is_capturing or self.webcam is None:
            return
        
        # Read and display frame
        face = self.webcam.poll_for_capture()
        
        if face is not None:
            # Capture successful or cancelled
            self.webcam.close()
            self.webcam = None
            self.is_capturing = False
            
            if isinstance(face, np.ndarray):
                # Got a face image
                self.images.append(face.copy())
                self.dialog.after(0, self._update_images_label)
        else:
            # Continue polling
            self.dialog.after(10, self._webcam_poll)
    
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
        success = False
        try:
            with self.pipeline_lock:
                success = self.gallery.add_person(name, self.images, pipeline=self.pipeline, overwrite=False)
        except Exception as e:
            messagebox.showerror("Gallery Error", f"Failed to add person:\n{str(e)}")
            return

        if success:
            messagebox.showinfo("Success", f"Added '{name}' to gallery")
            self.callback()
            self.dialog.destroy()
        else:
            messagebox.showerror("Error", "Failed to add person to gallery (may already exist)")


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
