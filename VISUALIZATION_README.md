# Grain Visualization Script

This script (`visualize_grain.py`) provides 3D visualization of the EXPoly grain carving process for a single grain.

## Features

The script visualizes the following stages:

1. **Stage 1: Original Voxel Structure**
   - Shows the original experimental voxel data
   - Displays mesh boundaries (convex hull)
   - Color-codes voxels by margin type:
     - Green: Interior (grain core)
     - Blue/Orange: Outer shell (neighbor grains)
     - Orange: Inner margin (touching neighbors)

2. **Stage 2: Grid Ball Generation**
   - Shows the initial simple cubic (SC) grid ball
   - Displays the grain center

3. **Stage 3: Lattice Transformation**
   - Shows three steps:
     - FCC/BCC/DIA lattice conversion
     - Rotation by grain Euler angles
     - Translation to grain center with overlap

4. **Stage 4: Carving Process**
   - Shows the filtering steps:
     - All lattice points before carving
     - After distance filtering (within ci_radius)
     - Final points after margin filtering

4. **Stage 5: Final Result**
   - Shows the final selected lattice points
   - Color-coded by margin type (interior vs. margin)

## Requirements

Install the required dependencies:

```bash
pip install matplotlib scipy pillow
```

Or if you have the project installed:

```bash
pip install -e ".[dev]"
```

## Usage

### Basic Usage

```bash
python visualize_grain.py \
    --dream3d An0new6.dream3d \
    --grain-id 111 \
    --lattice FCC \
    --ratio 1.5
```

### With Custom HDF5 Dataset Names

If your HDF5 file uses different dataset names (like `An0new6.dream3d`):

```bash
python visualize_grain.py \
    --dream3d An0new6.dream3d \
    --grain-id 111 \
    --lattice FCC \
    --ratio 1.5 \
    --h5-grain-dset FeatureIds \
    --h5-euler-dset EulerAngles \
    --h5-numneighbors-dset NumNeighbors \
    --h5-neighborlist-dset NeighborList2 \
    --h5-dimensions-dset DIMENSIONS
```

### Custom Camera Settings

Control the camera angle and rotation speed:

```bash
python visualize_grain.py \
    --dream3d An0new6.dream3d \
    --grain-id 111 \
    --lattice FCC \
    --ratio 1.5 \
    --camera-elevation 30 \
    --camera-azimuth 60 \
    --rotation-speed 1.5
```

### Custom Output Directory

```bash
python visualize_grain.py \
    --dream3d An0new6.dream3d \
    --grain-id 111 \
    --lattice FCC \
    --ratio 1.5 \
    --output-dir my_visualizations
```

### Skip GIF Generation

If you only want static images:

```bash
python visualize_grain.py \
    --dream3d An0new6.dream3d \
    --grain-id 111 \
    --lattice FCC \
    --ratio 1.5 \
    --no-gif
```

## Output Files

The script generates the following files in `visualizations/grain_<ID>/`:

- `stage1_original_voxels.png` - Original voxel structure
- `stage2_grid_ball.png` - Grid ball generation
- `stage3_lattice_transformation.png` - Lattice transformation
- `stage4_carving.png` - Carving process
- `stage5_final_result.png` - Final result
- `animation_original.gif` - Rotating animation of original voxels
- `animation_final.gif` - Rotating animation of final result

## Camera Configuration

The camera can be configured via command-line arguments:

- `--camera-elevation`: Elevation angle in degrees (default: 20)
- `--camera-azimuth`: Initial azimuth angle in degrees (default: 45)
- `--rotation-speed`: Rotation speed for animations in degrees per frame (default: 2.0)

## Example Workflow

1. **Extract and visualize a specific grain:**

```bash
python visualize_grain.py \
    --dream3d An0new6.dream3d \
    --grain-id 111 \
    --lattice FCC \
    --ratio 1.5 \
    --h5-grain-dset FeatureIds \
    --h5-euler-dset EulerAngles \
    --h5-numneighbors-dset NumNeighbors \
    --h5-neighborlist-dset NeighborList2 \
    --h5-dimensions-dset DIMENSIONS
```

2. **Check the output directory:**

```bash
ls -la visualizations/grain_111/
```

3. **View the generated images and GIFs**

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running from the project root:

```bash
cd /path/to/EXPoly
python visualize_grain.py ...
```

### GIF Generation Fails

If GIF generation fails, install pillow:

```bash
pip install pillow
```

Or use imageio:

```bash
pip install imageio
```

### Memory Issues

For very large grains, you may need to reduce the number of points visualized. The script automatically samples large datasets, but you can modify the sampling in the code if needed.

## Future Enhancements

This is a basic version. Future enhancements could include:

- Interactive 3D plots (using plotly)
- Animation of the carving process step-by-step
- Custom color schemes
- Export to other formats (OBJ, PLY)
- Batch processing for multiple grains
- More detailed margin visualization
