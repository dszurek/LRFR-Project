# LRFR Application

This application performs Low-Resolution Facial Recognition (LRFR) using a Hybrid DSR (Super-Resolution) model and EdgeFace for recognition. It is designed to run on a Raspberry Pi 5 but also supports Windows and other Linux systems.

## Prerequisites

- Python 3.12+

## Installation

### Raspberry Pi 5 / Linux (Automated)

The included `install.sh` script handles everything: managing system dependencies, creating a virtual environment, and downloading models.

1. **Run the Installer**:
   ```bash
   cd implementation_app
   chmod +x install.sh
   ./install.sh
   ```

2. **Activate the Environment**:
   ```bash
   source venv/bin/activate
   ```

### Windows (Manual Setup)

1. **Create Virtual Environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```

2. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

## Running the Application

Ensure your virtual environment is activated before running the application.

### Start the GUI

```bash
# Assuming you are in the implementation_app directory and venv is active
python app.py
```

### Windows Shortcut

You can also use the helper script (if available) or create a simple batch file to launch it:
```batch
cd implementation_app
venv\Scripts\python app.py
```

## Structure

- **implementation_app/**: Contains the main application code.
  - `app.py`: Main GUI entry point.
  - `pipeline.py`: Logic for DSR upscaling and facial recognition.
  - `config.py`: Configuration settings (paths, thresholds, etc.).
  - `install.sh`: Setup script for Raspberry Pi/Linux.
  - `requirements.txt`: Python dependencies.
- **technical/**: Contains model architecture definitions and weights.

## Troubleshooting

- **Models Not Found**: Run `git lfs pull` to ensure model weights are downloaded correctly.
- **Camera Error**: Ensure your camera is connected and accessible. On Linux, check permissions with `sudo usermod -a -G video $USER`.
- **Missing cv2**: If you encounter `ModuleNotFoundError: No module named 'cv2'`, ensure `opencv-python` or `opencv-python-headless` is installed via pip.
- **Poor recognition**: Ensure that the user is well-lit. Gallery photos should also be well-lit and of good quality. Try using the built-in automatic gallery photo mode to quickly capture 100 photos of the user.


## Additional Information 

The full repository, including training code, can be found on GitHub at https://github.com/dszurek/LRFR-Project