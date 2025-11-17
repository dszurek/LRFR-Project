"""Face detection and cropping using OpenCV Haar Cascades.

Lightweight face detection optimized for Raspberry Pi 5.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import config


class FaceDetector:
    """Haar Cascade face detector for real-time detection on Pi 5."""
    
    def __init__(self):
        """Initialize face detector with Haar Cascade."""
        # Load Haar Cascade (built into OpenCV)
        cascade_path = cv2.data.haarcascades + config.HAAR_CASCADE_FACE
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar Cascade from {cascade_path}. "
                "Ensure OpenCV is installed correctly."
            )
        
        print("[FaceDetector] Initialized with Haar Cascade")
    
    def detect_faces(
        self, 
        image: np.ndarray,
        min_size: Tuple[int, int] = None,
        scale_factor: float = None,
        min_neighbors: int = None
    ) -> List[Tuple[int, int, int, int]]:
        """Detect all faces in image.
        
        Args:
            image: Input image (H, W, 3) BGR format
            min_size: Minimum face size (w, h)
            scale_factor: Detection scale factor (lower = more sensitive)
            min_neighbors: Minimum neighbors for detection (higher = stricter)
        
        Returns:
            List of face bounding boxes [(x, y, w, h), ...]
        """
        if min_size is None:
            min_size = config.FACE_MIN_SIZE
        if scale_factor is None:
            scale_factor = config.FACE_DETECTION_SCALE_FACTOR
        if min_neighbors is None:
            min_neighbors = config.FACE_DETECTION_MIN_NEIGHBORS
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
    
    def detect_largest_face(
        self, 
        image: np.ndarray,
        **kwargs
    ) -> Optional[Tuple[int, int, int, int]]:
        """Detect the largest face in image (typically the closest/most prominent).
        
        Args:
            image: Input image (H, W, 3) BGR format
            **kwargs: Passed to detect_faces()
        
        Returns:
            Face bounding box (x, y, w, h) or None if no face found
        """
        faces = self.detect_faces(image, **kwargs)
        
        if len(faces) == 0:
            return None
        
        # Return largest face by area
        largest = max(faces, key=lambda f: f[2] * f[3])
        return largest
    
    def crop_face(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: float = None,
        output_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Crop face from image with optional padding and resizing.
        
        Args:
            image: Input image (H, W, 3) BGR format
            bbox: Face bounding box (x, y, w, h)
            padding: Fraction of bbox size to add as padding (default from config)
            output_size: Resize cropped face to (w, h) if specified
        
        Returns:
            Cropped face image (H', W', 3) BGR format
        """
        if padding is None:
            padding = config.FACE_PADDING
        
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)
        
        # Crop
        face = image[y1:y2, x1:x2]
        
        # Resize if requested
        if output_size is not None:
            face = cv2.resize(face, output_size, interpolation=cv2.INTER_AREA)
        
        return face
    
    def detect_and_crop(
        self,
        image: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None,
        padding: float = None,
        return_bbox: bool = False
    ) -> Optional[np.ndarray] | Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Detect largest face and crop it in one step.
        
        Args:
            image: Input image (H, W, 3) BGR format
            output_size: Resize cropped face to (w, h)
            padding: Padding fraction around face
            return_bbox: If True, return (cropped_face, bbox) tuple
        
        Returns:
            Cropped face image or None if no face detected
            If return_bbox=True: (cropped_face, bbox) tuple
        """
        bbox = self.detect_largest_face(image)
        
        if bbox is None:
            if return_bbox:
                return None, None
            return None
        
        face = self.crop_face(image, bbox, padding=padding, output_size=output_size)
        
        if return_bbox:
            return face, bbox
        return face
    
    def draw_detections(
        self,
        image: np.ndarray,
        faces: List[Tuple[int, int, int, int]],
        color: Tuple[int, int, int] = None,
        thickness: int = 2,
        label: bool = True
    ) -> np.ndarray:
        """Draw bounding boxes around detected faces.
        
        Args:
            image: Input image (will be copied, not modified)
            faces: List of face bounding boxes
            color: Box color in BGR (default green)
            thickness: Line thickness
            label: Draw "Face" label above box
        
        Returns:
            Image with drawn boxes
        """
        if color is None:
            color = config.COLOR_GREEN
        
        output = image.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            # Draw rectangle
            cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label
            if label:
                label_text = f"Face {i+1}"
                label_size = cv2.getTextSize(
                    label_text, 
                    config.FONT_FACE, 
                    config.FONT_SCALE, 
                    config.FONT_THICKNESS
                )[0]
                
                # Background for text
                cv2.rectangle(
                    output,
                    (x, y - label_size[1] - 10),
                    (x + label_size[0], y),
                    color,
                    -1  # Filled
                )
                
                # Text
                cv2.putText(
                    output,
                    label_text,
                    (x, y - 5),
                    config.FONT_FACE,
                    config.FONT_SCALE,
                    config.COLOR_WHITE,
                    config.FONT_THICKNESS
                )
        
        return output


if __name__ == "__main__":
    # Test face detector
    print("Testing Face Detector...")
    
    detector = FaceDetector()
    
    # Create test image with random pattern
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Detect (won't find anything in random noise, but tests the API)
    faces = detector.detect_faces(test_img)
    print(f"Detected {len(faces)} faces in random image (expected: 0)")
    
    # Test with actual camera if available
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                faces = detector.detect_faces(frame)
                print(f"Detected {len(faces)} faces in camera frame")
                
                if len(faces) > 0:
                    face = detector.detect_and_crop(frame, output_size=(112, 112))
                    if face is not None:
                        print(f"Cropped face shape: {face.shape}")
            cap.release()
    except Exception as e:
        print(f"Camera test skipped: {e}")
    
    print("Face detector test complete!")
