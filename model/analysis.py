import numpy as np
import nibabel as nib

def analysis(segmentation, results, path):
    # Compare with BraTS expected volumes (healthy brain ~1.5M mm³)
    img = nib.load(path)
    voxel_vol = np.prod(img.header.get_zooms())
    total_vol = np.prod(segmentation.shape) * voxel_vol
    
    tumor_brain_ratio = results['Total Tumor (mm³)'] / total_vol * 100
    
    # Expected ranges (BraTS 2021 averages)
    expected_ranges = {
        "Necrotic": (15, 25),  # % of tumor
        "Edema": (55, 65),
        "Enhancing": (15, 25)
    }
    
    validation_results = {
        "Total Brain Volume (ml)": total_vol / 1e6,
        "Tumor/Brain Ratio (%)": tumor_brain_ratio
    }
    
    for region, (low, high) in expected_ranges.items():
        val = results[f"{region} (%)"]
        status = "NORMAL" if low <= val <= high else "WARNING"
        validation_results[f"{region} (%)"] = {"value": val, "status": status}
    
    return validation_results
