"""Webcam capture interface for Raspberry Pi 5.

Handles camera initialization, frame capture, and face detection overlay.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import time

import config
from face_detector import FaceDetector


class WebcamCapture:
    """Webcam interface optimized for Pi 5."""
    
    def __init__(
        self,
        camera_index: int = None,
        width: int = None,
        height: int = None,
        fps: int = None
    ):
        """Initialize webcam.
        
        Args:
            camera_index: Camera device index (default from config)
            width: Frame width (default from config)
            height: Frame height (default from config)
            fps: Target FPS (default from config)
        """
        self.camera_index = camera_index if camera_index is not None else config.WEBCAM_INDEX
        self.width = width if width is not None else config.WEBCAM_WIDTH
        self.height = height if height is not None else config.WEBCAM_HEIGHT
        self.fps = fps if fps is not None else config.WEBCAM_FPS
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        self.face_detector = FaceDetector()
        
        # FPS tracking
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
    
    def open(self) -> bool:
        """Open camera and configure settings.
        
        Returns:
            True if camera opened successfully
        """
        if self.is_opened:
            print("[Webcam] Already opened")
            return True
        
        print(f"[Webcam] Opening camera {self.camera_index}...")
        
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"[Webcam] Failed to open camera {self.camera_index}")
            return False
        
        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Try to set MJPEG format for better performance (if supported)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Get actual settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"[Webcam] Opened: {actual_width}×{actual_height} @ {actual_fps} FPS")
        
        # Warmup camera (skip first frames for auto-exposure)
        for _ in range(config.WEBCAM_WARMUP_FRAMES):
            self.cap.read()
        
        self.is_opened = True
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        
        return True
    
    def close(self):
        """Release camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.is_opened = False
        print("[Webcam] Closed")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read single frame from camera.
        
        Returns:
            Frame (H, W, 3) BGR or None if failed
        """
        if not self.is_opened:
            return None
        
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            print("[Webcam] Failed to read frame")
            return None
        
        # Update FPS counter
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = time.time()
        
        return frame
    
    def read_with_face_detection(
        self,
        draw_boxes: bool = True,
        show_fps: bool = None
    ) -> Tuple[Optional[np.ndarray], list]:
        """Read frame and detect faces.
        
        Args:
            draw_boxes: Draw bounding boxes on frame
            show_fps: Show FPS counter (default from config)
        
        Returns:
            (frame, faces) where faces is list of (x, y, w, h) bounding boxes
        """
        if show_fps is None:
            show_fps = config.SHOW_FPS
        
        frame = self.read_frame()
        
        if frame is None:
            return None, []
        
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        
        # Draw visualization
        if draw_boxes and len(faces) > 0:
            frame = self.face_detector.draw_detections(frame, faces)
        
        # Draw FPS
        if show_fps:
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(
                frame,
                fps_text,
                (10, 30),
                config.FONT_FACE,
                config.FONT_SCALE,
                config.COLOR_GREEN,
                config.FONT_THICKNESS
            )
        
        return frame, faces
    
    def capture_face(
        self,
        output_size: Tuple[int, int] = None,
        max_attempts: int = 30,
        show_preview: bool = True,
        window_name: str = "Capture Face - Press SPACE"
    ) -> Optional[np.ndarray]:
        """Interactive face capture with live preview.
        
        Args:
            output_size: Resize captured face to (w, h)
            max_attempts: Maximum frames to try before giving up
            show_preview: Show live camera preview
            window_name: Window name for preview
        
        Returns:
            Cropped face image or None if failed
        """
        if not self.is_opened and not self.open():
            return None
        
        if show_preview:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        captured_face = None
        attempts = 0
        
        print("[Webcam] Press SPACE to capture, ESC to cancel...")
        
        while attempts < max_attempts:
            frame, faces = self.read_with_face_detection()
            
            if frame is None:
                break
            
            attempts += 1
            
            # Show preview
            if show_preview:
                # Add instructions
                instructions = "SPACE: Capture | ESC: Cancel"
                cv2.putText(
                    frame,
                    instructions,
                    (10, frame.shape[0] - 10),
                    config.FONT_FACE,
                    config.FONT_SCALE,
                    config.COLOR_YELLOW,
                    config.FONT_THICKNESS
                )
                
                # Show face count
                if len(faces) > 0:
                    status = f"{len(faces)} face(s) detected - Ready!"
                    color = config.COLOR_GREEN
                else:
                    status = "No face detected - Position yourself"
                    color = config.COLOR_RED
                
                cv2.putText(
                    frame,
                    status,
                    (10, 60),
                    config.FONT_FACE,
                    config.FONT_SCALE,
                    color,
                    config.FONT_THICKNESS
                )
                
                cv2.imshow(window_name, frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC - cancel
                print("[Webcam] Capture cancelled")
                break
            
            elif key == ord(' '):  # SPACE - capture
                if len(faces) == 0:
                    print("[Webcam] No face detected, try again")
                    continue
                
                # Crop largest face
                captured_face = self.face_detector.detect_and_crop(
                    frame,
                    output_size=output_size
                )
                
                if captured_face is not None:
                    print("[Webcam] Face captured!")
                    break
        
        if show_preview:
            cv2.destroyWindow(window_name)
        
        return captured_face
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return self.current_fps
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


if __name__ == "__main__":
    # Test webcam capture
    print("Testing Webcam Capture...")
    
    with WebcamCapture() as cam:
        if not cam.is_opened:
            print("Failed to open camera")
        else:
            print("Camera opened successfully!")
            
            # Capture 10 frames
            for i in range(10):
                frame, faces = cam.read_with_face_detection()
                if frame is not None:
                    print(f"Frame {i+1}: {frame.shape}, {len(faces)} faces")
                time.sleep(0.1)
            
            # Test interactive capture (comment out for automated testing)
            # face = cam.capture_face(output_size=(112, 112))
            # if face is not None:
            #     print(f"Captured face: {face.shape}")
    
    print("Webcam test complete!")
