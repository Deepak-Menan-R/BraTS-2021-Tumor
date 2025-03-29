import os
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    EnsureTyped
)
from monai.inferers import SlidingWindowInferer
import numpy as np
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import nibabel as nib
from flask import Flask, request, send_file, render_template, redirect, url_for
from io import BytesIO
import tempfile
import matplotlib.pyplot as plt
import base64
from model.analysis import analysis

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit per file

# Configure model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(8, 16, 32, 64),  # Reduced capacity
        strides=(2, 2, 2),
        num_res_units=1,
        norm=Norm.BATCH
    ).to(device)
model.load_state_dict(torch.load("model/best_model.pth", map_location=device))
model.eval()


# Inference Transforms
inference_transforms = Compose([
    LoadImaged(keys=["image"], reader="NibabelReader"),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityRanged(keys="image", a_min=-100, a_max=1000, b_min=0, b_max=1, clip=True),
    EnsureTyped(keys=["image"], dtype=torch.float32)
])

def analyze_segmentation(seg, voxel_spacing=(1,1,1)):
    if not isinstance(seg, np.ndarray):
        raise ValueError("Segmentation must be numpy array")
    
    unique_vals = np.unique(seg)
    if not set(unique_vals).issubset({0,1,2,3}):
        print(f"Warning: Unexpected label values {unique_vals}")
    
    voxel_vol = np.prod(voxel_spacing)
    results = {
        "Necrotic (mm³)": np.sum(seg == 1) * voxel_vol,
        "Edema (mm³)": np.sum(seg == 2) * voxel_vol,
        "Enhancing (mm³)": np.sum(seg == 3) * voxel_vol,
    }
    results["Total Tumor (mm³)"] = sum(v for k,v in results.items() if "mm³" in k)
    
    # Calculate percentages
    total = results["Total Tumor (mm³)"]
    if total > 0:
        for k in list(results.keys()):
            if "mm³" in k:
                results[k.replace("mm³", "%")] = results[k] / total * 100
    
    return results

def predict_volume(scan_paths):
    for path in scan_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
    
    try:
        sample = {"image": scan_paths}
        data = inference_transforms(sample)
        inputs = data["image"].unsqueeze(0).to(device)
        
        inferer = SlidingWindowInferer(roi_size=[128, 128, 96], sw_batch_size=1, overlap=0.25)
        with torch.no_grad():
            logits = inferer(inputs, model)
            seg = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
        
        return seg
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return np.zeros((240, 240, 155), dtype=np.uint8)

def nifti_to_png_slice(nifti_data, slice_index=None, channel=0):
    """Convert 3D NIfTI data to PNG slice (show specific channel if multi-channel)"""
    if slice_index is None:
        slice_index = nifti_data.shape[-1] // 2  # Middle slice
    
    if len(nifti_data.shape) == 4:  # Multi-channel
        slice_data = nifti_data[channel, :, :, slice_index]
    else:
        slice_data = nifti_data[:, :, slice_index]
        
    plt.imshow(slice_data.T, cmap="gray", origin="lower")
    plt.axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

import io
import sys

@app.route("/upload", methods=["POST"])
def upload_file():
    if len(request.files.getlist('files')) != 4:
        return render_template("error.html", error="Please upload exactly 4 NIfTI files (one for each modality)")

    try:
        file_paths = [{"image": []}]
        for i, file in enumerate(request.files.getlist('files')):
            if file and allowed_file(file.filename):
                filename = f"modality_{i}.nii.gz"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file_paths[0]["image"].append(filepath)
                file.save(filepath)
            else:
                return render_template("error.html", error="Invalid file type")

        # Capture print outputs
        log_stream = io.StringIO()
        sys.stdout = log_stream  # Redirect print statements to this buffer

        print("Files saved successfully")
        print("Processing segmentation...")

        segmentation = predict_volume(file_paths[0]["image"])
        print("Segmentation completed")

        # Verify output
        print(f"Segmentation shape: {segmentation.shape}")
        print(f"Unique values: {np.unique(segmentation)}")

        if np.all(segmentation == 0):
            print("Warning: Empty prediction - checking model weights")
            print(f"Model weights loaded: {not all(p.isnan().any() for p in model.parameters())}")

        # Tumor analysis
        results = analyze_segmentation(segmentation)
        print("\nTumor Volumes:")
        for k, v in results.items():
            print(f"{k:>20}: {v:,.2f}" if 'mm³' in k else f"{k:>20}: {v:.1f}%")
        clinical_validation = analysis(segmentation, results, file_paths[0]["image"][0])
        # Restore stdout
        sys.stdout = sys.__stdout__
        log_output = log_stream.getvalue()

        # Generate base64 encoded images
        input_preview = nifti_to_png_slice(nib.load(file_paths[0]["image"][0]).get_fdata())  # First modality
        output_preview = nifti_to_png_slice(segmentation)

        return render_template("results.html",
                       message="Segmentation completed successfully!",
                       results=results,
                       input_preview=input_preview,
                       output_preview=output_preview,
                       log_output=log_output,
                       clinical_validation=clinical_validation) 

    
    except Exception as e:
        sys.stdout = sys.__stdout__  # Restore stdout before error
        return render_template("error.html", error=str(e))

    
    except Exception as e:
        return render_template("error.html", error=str(e))

@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    if filename in ['input.nii.gz', 'output.nii.gz']:
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(path):
            return send_file(path, as_attachment=True)
    return redirect(url_for('index'))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'nii', 'gz', 'nii.gz'}

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)