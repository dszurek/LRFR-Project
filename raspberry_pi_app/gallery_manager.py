"""Gallery management for storing and retrieving person embeddings.

Handles:
- Adding/removing people from gallery
- Storing multiple images per person
- Computing and caching embeddings
- Persistence to disk (JSON + images)
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
from dataclasses import dataclass, asdict

import config
from face_detector import FaceDetector


@dataclass
class PersonEntry:
    """Single person in gallery."""
    name: str
    image_paths: List[str]  # Relative to gallery dir
    embedding: Optional[List[float]] = None  # Average embedding
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "PersonEntry":
        """Load from dict."""
        return cls(**data)


class GalleryManager:
    """Manage gallery of enrolled identities."""
    
    def __init__(self, gallery_dir: Path = None, metadata_path: Path = None):
        """Initialize gallery manager.
        
        Args:
            gallery_dir: Directory to store gallery images
            metadata_path: Path to gallery metadata JSON
        """
        self.gallery_dir = gallery_dir or config.GALLERY_DIR
        self.metadata_path = metadata_path or config.GALLERY_METADATA
        
        # Create directories
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing gallery
        self.people: Dict[str, PersonEntry] = {}
        self._load_metadata()
        
        # Face detector for processing new images
        self.face_detector = FaceDetector()
        
        print(f"[Gallery] Initialized with {len(self.people)} people")
    
    def _load_metadata(self):
        """Load gallery metadata from disk."""
        if not self.metadata_path.exists():
            return
        
        try:
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
            
            for person_data in data.get("people", []):
                person = PersonEntry.from_dict(person_data)
                self.people[person.name] = person
            
            print(f"[Gallery] Loaded {len(self.people)} people from {self.metadata_path.name}")
        except Exception as e:
            print(f"[Gallery] Error loading metadata: {e}")
    
    def _save_metadata(self):
        """Save gallery metadata to disk."""
        data = {
            "people": [person.to_dict() for person in self.people.values()]
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[Gallery] Saved {len(self.people)} people to {self.metadata_path.name}")
    
    def add_person(
        self,
        name: str,
        images: List[np.ndarray],
        pipeline=None,
        overwrite: bool = False
    ) -> bool:
        """Add a person to gallery with their images.
        
        Args:
            name: Person's name (unique identifier)
            images: List of face images (BGR format)
            pipeline: LRFRPipeline for computing embeddings (optional)
            overwrite: Replace existing person if True
        
        Returns:
            True if successfully added, False otherwise
        """
        # Check if exists
        if name in self.people and not overwrite:
            print(f"[Gallery] Person '{name}' already exists. Use overwrite=True to replace.")
            return False
        
        # Validate image count
        if len(images) < config.MIN_IMAGES_PER_PERSON:
            print(f"[Gallery] Need at least {config.MIN_IMAGES_PER_PERSON} images, got {len(images)}")
            return False
        
        if len(images) > config.MAX_IMAGES_PER_PERSON:
            print(f"[Gallery] Maximum {config.MAX_IMAGES_PER_PERSON} images allowed, got {len(images)}")
            images = images[:config.MAX_IMAGES_PER_PERSON]
        
        # Check gallery size limit
        if len(self.people) >= config.MAX_GALLERY_SIZE and name not in self.people:
            print(f"[Gallery] Gallery full ({config.MAX_GALLERY_SIZE} people max)")
            return False
        
        # Create person directory
        person_dir = self.gallery_dir / name.replace(" ", "_")
        if person_dir.exists():
            shutil.rmtree(person_dir)  # Remove old images
        person_dir.mkdir(parents=True, exist_ok=True)
        
        # Process and save images
        image_paths = []
        processed_images = []
        
        for i, img in enumerate(images):
            # Detect and crop face
            face = self.face_detector.detect_and_crop(
                img, 
                output_size=config.GALLERY_IMAGE_SIZE
            )
            
            if face is None:
                print(f"[Gallery] Warning: No face detected in image {i+1}, skipping")
                continue
            
            # Save image
            img_filename = f"{i:03d}.jpg"
            img_path = person_dir / img_filename
            cv2.imwrite(str(img_path), face)
            
            # Store relative path
            rel_path = str(img_path.relative_to(self.gallery_dir))
            image_paths.append(rel_path)
            processed_images.append(face)
        
        if len(image_paths) < config.MIN_IMAGES_PER_PERSON:
            print(f"[Gallery] Not enough valid faces detected ({len(image_paths)}/{config.MIN_IMAGES_PER_PERSON})")
            shutil.rmtree(person_dir)
            return False
        
        # Compute average embedding if pipeline provided
        embedding = None
        if pipeline is not None:
            embeddings = []
            for img in processed_images:
                result = pipeline.process_image(img, return_intermediate=False)
                embeddings.append(result["embedding"])
            
            # Average embeddings and re-normalize
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            embedding = avg_embedding.tolist()
            
            # Clean up embeddings list
            del embeddings
            del avg_embedding
        
        # Create entry
        person = PersonEntry(
            name=name,
            image_paths=image_paths,
            embedding=embedding
        )
        
        self.people[name] = person
        self._save_metadata()
        
        print(f"[Gallery] Added '{name}' with {len(image_paths)} images")
        return True
    
    def remove_person(self, name: str) -> bool:
        """Remove a person from gallery.
        
        Args:
            name: Person's name
        
        Returns:
            True if removed, False if not found
        """
        if name not in self.people:
            print(f"[Gallery] Person '{name}' not found")
            return False
        
        # Remove images
        person_dir = self.gallery_dir / name.replace(" ", "_")
        if person_dir.exists():
            shutil.rmtree(person_dir)
        
        # Remove from metadata
        del self.people[name]
        self._save_metadata()
        
        print(f"[Gallery] Removed '{name}'")
        return True
    
    def get_person(self, name: str) -> Optional[PersonEntry]:
        """Get person entry by name."""
        return self.people.get(name)
    
    def get_all_names(self) -> List[str]:
        """Get list of all person names in gallery."""
        return list(self.people.keys())
    
    def get_person_images(self, name: str) -> List[np.ndarray]:
        """Load all images for a person.
        
        Args:
            name: Person's name
        
        Returns:
            List of images (BGR format)
        """
        person = self.get_person(name)
        if person is None:
            return []
        
        images = []
        for rel_path in person.image_paths:
            img_path = self.gallery_dir / rel_path
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append(img)
        
        return images
    
    def get_person_thumbnail(self, name: str) -> Optional[np.ndarray]:
        """Get first image of person as thumbnail.
        
        Args:
            name: Person's name
        
        Returns:
            Thumbnail image (BGR) or None
        """
        images = self.get_person_images(name)
        return images[0] if images else None
    
    def compute_all_embeddings(self, pipeline) -> int:
        """Compute embeddings for all people in gallery.
        
        Args:
            pipeline: LRFRPipeline instance
        
        Returns:
            Number of people processed
        """
        count = 0
        
        for name in self.get_all_names():
            images = self.get_person_images(name)
            
            if not images:
                continue
            
            # Compute embeddings
            embeddings = []
            for img in images:
                result = pipeline.process_image(img, return_intermediate=False)
                embeddings.append(result["embedding"])
            
            # Average and normalize
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            
            # Update person
            self.people[name].embedding = avg_embedding.tolist()
            count += 1
        
        self._save_metadata()
        print(f"[Gallery] Computed embeddings for {count} people")
        
        return count
    
    def get_gallery_embeddings(self) -> Tuple[List[np.ndarray], List[str]]:
        """Get all embeddings and names for identification.
        
        Returns:
            (embeddings, names) where embeddings[i] corresponds to names[i]
        """
        embeddings = []
        names = []
        
        for name, person in self.people.items():
            if person.embedding is not None:
                embeddings.append(np.array(person.embedding, dtype=np.float32))
                names.append(name)
        
        return embeddings, names
    
    def size(self) -> int:
        """Get number of people in gallery."""
        return len(self.people)
    
    def is_full(self) -> bool:
        """Check if gallery is at maximum capacity."""
        return self.size() >= config.MAX_GALLERY_SIZE
    
    def clear(self) -> bool:
        """Remove all people from gallery.
        
        Returns:
            True if cleared successfully
        """
        # Remove all person directories
        for person_dir in self.gallery_dir.iterdir():
            if person_dir.is_dir():
                shutil.rmtree(person_dir)
        
        # Clear metadata
        self.people.clear()
        self._save_metadata()
        
        print("[Gallery] Cleared all entries")
        return True


if __name__ == "__main__":
    # Test gallery manager
    print("Testing Gallery Manager...")
    
    # Create test gallery in temp location
    test_dir = Path("test_gallery")
    test_metadata = Path("test_gallery.json")
    
    gallery = GalleryManager(test_dir, test_metadata)
    
    # Create dummy images
    dummy_imgs = [
        np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        for _ in range(5)
    ]
    
    # Test add (without pipeline, won't compute embeddings)
    success = gallery.add_person("Test Person", dummy_imgs)
    print(f"Add person: {success}")
    
    # Test get
    print(f"Gallery size: {gallery.size()}")
    print(f"Names: {gallery.get_all_names()}")
    
    # Cleanup
    gallery.clear()
    shutil.rmtree(test_dir, ignore_errors=True)
    test_metadata.unlink(missing_ok=True)
    
    print("Gallery manager test complete!")
