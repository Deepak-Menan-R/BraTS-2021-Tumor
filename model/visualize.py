import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import io
import base64

def visualize_tumor(scan_path, segmentation, slice_idx=None):
    """Create clinical-grade visualization with tumor overlay."""
    
    # Load FLAIR scan
    flair = nib.load(scan_path).get_fdata()
    
    # Define tumor colormap
    tumor_cmap = ListedColormap(['black', 'darkred', 'dodgerblue', 'yellow'])
    
    # Auto-select center slice if not given
    if slice_idx is None:
        slice_idx = [s // 2 for s in segmentation.shape]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    views = ["Axial", "Coronal", "Sagittal"]
    slices = [
        (flair[slice_idx[0]], segmentation[slice_idx[0]]),
        (flair[:, slice_idx[1]], segmentation[:, slice_idx[1]]),
        (flair[:, :, slice_idx[2]], segmentation[:, :, slice_idx[2]])
    ]
    
    for i, (flair_slice, seg_slice) in enumerate(slices):
        axes[i].imshow(np.rot90(flair_slice), cmap='gray', interpolation='nearest')
        axes[i].imshow(np.rot90(seg_slice), cmap=tumor_cmap, alpha=0.5, vmin=0, vmax=3)
        axes[i].set_title(f"{views[i]} Slice {slice_idx[i]}")
        axes[i].axis("off")

    # Create legend
    legend_labels = ["Background", "Necrotic Core", "Edema", "Enhancing Tumor"]
    patches = [plt.plot([], [], marker="s", ls="", ms=10, color=tumor_cmap(i), 
                         label=legend_labels[i])[0] for i in range(4)]
    axes[-1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Save figure to memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # Convert to base64
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_base64  # Return base64-encoded image for HTML display
