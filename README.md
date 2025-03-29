# Brain Tumor Segmentation Web App

This web application performs brain tumor segmentation from MRI scans using a pre-trained MONAI-based UNet model. The app allows users to upload four NIfTI (.nii.gz) files corresponding to different imaging modalities, process the segmentation, and visualize the results.

## Features
- Upload four NIfTI files for tumor segmentation.
- Predicts and analyzes tumor regions (Necrotic, Edema, Enhancing Tumor).
- Provides visual previews of the input MRI and predicted segmentation mask.
- Clinical validation of the segmentation results.
- Downloadable segmented output.

## Requirements
Ensure you have the following installed before running the app:
- Python 3.8+
- Flask
- MONAI
- PyTorch
- Nibabel
- NumPy
- Matplotlib

## Installation
Clone the repository and install the required dependencies:
```bash
# Clone the repository
git clone https://github.com/Deepak-Menan-R/BraTS-2021-Tumor.git
cd BraTS-2021-Tumor

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Model Setup
Ensure you have the pre-trained model weights in the `model/` directory:
```bash
model/best_model.pth  # Place your trained UNet weights here
```

## Running the App
Start the Flask server:
```bash
python app.py
```

The app will run on `http://0.0.0.0:5000/` by default.

## Usage
1. Open the web interface in a browser.
2. Upload **exactly four** NIfTI files (T1, T1c, T2, FLAIR).
3. Click 'Submit' to start segmentation.
4. View and download the segmented results.

## Project Structure
```
BraTS-2021-Tumor/
│── model/
│   ├── best_model.pth    # Pretrained UNet weights
│── templates/
│   ├── index.html        # Upload page
│   ├── results.html      # Result display page
│   ├── error.html        # Error handling page
│── static/               # Stores CSS, JS, images
│── app.py                # Flask backend
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
```

## API Endpoints
### `GET /`
Loads the file upload page.

### `POST /upload`
Processes the uploaded NIfTI files and returns segmentation results.

### `GET /download/<filename>`
Allows downloading segmented results.

## Notes
- Ensure input files are properly formatted before uploading.
- If the model produces empty predictions, verify the model weights.
- The segmentation visualization is a middle slice for preview purposes.

## Acknowledgments
This project is built using [MONAI](https://monai.io/) and [BraTS 2021](https://www.med.upenn.edu/cbica/brats2021/) dataset methodology.

## License
This project is licensed under the MIT License.

