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
        
        # For polling mode
        self.window_name = "Webcam Capture - Press SPACE to capture, ESC to cancel"
        self.last_face_rect = None
        self.captured_face = None
    
    def open(self) -> bool:
        """Open camera and configure settings.
        
        Returns:
            True if camera opened successfully
        """
        # Force close if already opened
        if self.is_opened or self.cap is not None:
            print("[Webcam] Forcing close before reopen...")
            self.close()
            time.sleep(0.3)
        
        print(f"[Webcam] Opening camera {self.camera_index}...")
        
        # Try multiple times
        for attempt in range(3):
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if self.cap.isOpened():
                break
            
            print(f"[Webcam] Attempt {attempt + 1} failed, retrying...")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            time.sleep(0.3)
        
        if self.cap is None or not self.cap.isOpened():
            print(f"[Webcam] Failed to open camera {self.camera_index}")
            return False
        
        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Try to set MJPEG format
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Get actual settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"[Webcam] Opened: {actual_width}×{actual_height} @ {actual_fps} FPS")
        
        # Warmup camera
        for _ in range(config.WEBCAM_WARMUP_FRAMES):
            self.cap.read()
        
        self.is_opened = True
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        
        # Create window for polling mode
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 480)
        
        return True
    
    def close(self):
        """Release camera and destroy windows."""
        print("[Webcam] Closing...")
        
        self.is_opened = False
        
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                print(f"[Webcam] Error releasing camera: {e}")
            finally:
                self.cap = None
        
        try:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
        except Exception as e:
            print(f"[Webcam] Error destroying window: {e}")
        
        print("[Webcam] Closed")
    
    def poll_for_capture(self) -> Optional[np.ndarray]:
        """Poll for a single frame and handle user input.
        
        Called repeatedly by GUI's event loop.
        
        Returns:
            None if still waiting for capture
            np.ndarray if face captured
            False if cancelled
        """
        if not self.is_opened or self.cap is None:
            return False
        
        # Read frame
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            print("[Webcam] Failed to read frame")
            return False
        
        # Update FPS
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = time.time()
        
        # Detect faces (every 3 frames)
        if self.fps_frame_count % 3 == 0:
            faces = self.face_detector.detect_faces(frame)
            if faces:
                self.last_face_rect = faces[0]
        
        # Create display frame
        display_frame = frame.copy()
        
        # Draw face rectangle
        if self.last_face_rect is not None:
            x, y, w, h = self.last_face_rect
            
            cv2.rectangle(
                display_frame,
                (x, y),
                (x + w, y + h),
                config.COLOR_GREEN,
                2
            )
            
            cv2.putText(
                display_frame,
                "Face detected - Ready to capture!",
                (10, 60),
                config.FONT_FACE,
                config.FONT_SCALE,
                config.COLOR_GREEN,
                config.FONT_THICKNESS
            )
        else:
            cv2.putText(
                display_frame,
                "No face detected - position yourself in frame",
                (10, 60),
                config.FONT_FACE,
                config.FONT_SCALE,
                config.COLOR_RED,
                config.FONT_THICKNESS
            )
        
        # Add FPS
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(
            display_frame,
            fps_text,
            (10, 30),
            config.FONT_FACE,
            config.FONT_SCALE,
            config.COLOR_GREEN,
            config.FONT_THICKNESS
        )
        
        # Add instructions
        cv2.putText(
            display_frame,
            "SPACE: Capture | ESC: Cancel",
            (10, display_frame.shape[0] - 20),
            config.FONT_FACE,
            config.FONT_SCALE,
            config.COLOR_YELLOW,
            config.FONT_THICKNESS
        )
        
        # Show frame
        cv2.imshow(self.window_name, display_frame)
        
        # Check for key press (non-blocking)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("[Webcam] Capture cancelled by user")
            return False
        
        elif key == ord(' ') or key == 32:  # SPACE
            if self.last_face_rect is None:
                print("[Webcam] No face detected, try again")
                return None  # Continue polling
            
            # Crop face
            captured_face = self.face_detector.detect_and_crop(
                frame,
                output_size=config.HR_SIZE
            )
            
            if captured_face is not None:
                print("[Webcam] Face captured successfully!")
                
                # Flash screen green
                flash_frame = display_frame.copy()
                cv2.rectangle(
                    flash_frame, (0, 0),
                    (flash_frame.shape[1], flash_frame.shape[0]),
                    config.COLOR_GREEN, 10
                )
                cv2.imshow(self.window_name, flash_frame)
                cv2.waitKey(200)
                
                return captured_face
            else:
                print("[Webcam] Failed to crop face, try again")
                return None  # Continue polling
        
        # Still waiting for input
        return None
    
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
    
    # Test multiple captures
    print("\nTesting multiple sequential captures...")
    for i in range(3):
        print(f"\n--- Capture {i+1} ---")
        cam = WebcamCapture()
        if cam.open():
            # Simulate polling loop
            while True:
                face = cam.poll_for_capture()
                if face is not None:
                    if isinstance(face, np.ndarray):
                        print(f"Captured face: {face.shape}")
                        cv2.imwrite(f"test_capture_{i+1}.jpg", face)
                        print(f"Saved to test_capture_{i+1}.jpg")
                    break
                time.sleep(0.01)  # Simulate GUI event loop delay
            cam.close()
        
        time.sleep(1)
    
    print("\nWebcam test complete!")
