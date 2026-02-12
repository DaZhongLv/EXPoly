#!/usr/bin/env python3
"""
Blender Animation Script: Multi-Stage Workflow Visualization

This script creates a timeline-based animation showing:
- Shot 0: Grid Ball with Local Coordinate System (frames 1-40)
- Shot 1: FCC Lattice Generation (frames 41-80)
- Shot 2: FCC Grid Ball Rotation (frames 81-160)
- Shot 3: Overlap with Experimental Grain (frames 161-220)
- Shot 4: Final Overlap Result (frames 221-260)

All structures are represented as point clouds (instanced spheres).
Camera is fixed, all rotations are on objects.

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python blender_animation_workflow.py -- --grain-ply grain_111_points.ply
"""

import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Parse command line arguments
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

parser = argparse.ArgumentParser(description="Blender Workflow Animation")
parser.add_argument(
    "--grain-ply",
    type=str,
    required=True,
    help="Path to grain points PLY file"
)
parser.add_argument(
    "--output-blend",
    type=str,
    default="workflow_animation.blend",
    help="Output .blend file path (default: workflow_animation.blend)"
)
parser.add_argument(
    "--point-size",
    type=float,
    default=0.5,
    help="Point sphere size (default: 0.5)"
)
parser.add_argument(
    "--grid-ball-radius",
    type=float,
    default=20.0,
    help="Grid ball radius (default: 20.0)"
)
parser.add_argument(
    "--grid-spacing",
    type=float,
    default=1.0,
    help="Grid spacing (default: 1.0)"
)
parser.add_argument(
    "--fcc-lattice-constant",
    type=float,
    default=3.524,
    help="FCC lattice constant (default: 3.524)"
)
parser.add_argument(
    "--use-scipy",
    action="store_true",
    help="Use scipy for point filtering (requires scipy package)"
)
parser.add_argument(
    "--resolution-x",
    type=int,
    default=1920,
    help="Render resolution X (default: 1920)"
)
parser.add_argument(
    "--resolution-y",
    type=int,
    default=1080,
    help="Render resolution Y (default: 1080)"
)

args = parser.parse_args(argv)

# ==================== Animation Frame Settings ====================

SHOT_0_START = 1
SHOT_0_END = 40
SHOT_1_START = 41
SHOT_1_END = 80
SHOT_2_START = 81
SHOT_2_END = 160
SHOT_3_START = 161
SHOT_3_END = 220
SHOT_4_START = 221
SHOT_4_END = 260

TOTAL_FRAMES = SHOT_4_END

# ==================== Scene Setup ====================

# Set render settings
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = TOTAL_FRAMES
scene.render.resolution_x = args.resolution_x
scene.render.resolution_y = args.resolution_y
scene.render.fps = 24
scene.render.engine = 'BLENDER_EEVEE'

# Enable transparency
scene.render.film_transparent = True

# Set background to white
world = bpy.data.worlds.new("World")
world.use_nodes = True
bg_node = world.node_tree.nodes.new(type='ShaderNodeBackground')
bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # White
world.node_tree.links.new(bg_node.outputs['Background'], world.node_tree.nodes['World Output'].inputs['Surface'])
scene.world = world

# ==================== Helper Functions ====================

def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)) + (1.0,)

def read_ply_file(ply_path: Path):
    """Read PLY file and return vertices as list of tuples."""
    vertices = []
    in_vertex_section = False
    num_vertices = 0
    vertex_count = 0

    with open(ply_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line == 'end_header':
                in_vertex_section = True
                continue
            elif in_vertex_section and line:
                parts = line.split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    vertices.append((x, y, z))
                    vertex_count += 1
                    if vertex_count >= num_vertices:
                        break

    return vertices

def create_point_cloud_mesh(vertices, name: str):
    """Create a mesh object from vertices (point cloud)."""
    mesh = bpy.data.meshes.new(name=f"{name}_Mesh")
    mesh.from_pydata(vertices, [], [])  # vertices, edges, faces
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj

def setup_point_cloud_geometry_nodes(obj, point_size: float, material=None):
    """Setup Geometry Nodes to render points as spheres."""
    # Add Geometry Nodes modifier
    mod = obj.modifiers.new(name="PointCloudSpheres", type='NODES')
    node_group = bpy.data.node_groups.new(name=f"{obj.name}_PointCloud", type='GeometryNodeTree')
    mod.node_group = node_group

    # Clear default nodes
    nodes = node_group.nodes
    links = node_group.links
    nodes.clear()

    # Add input/output interfaces
    node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

    # Input
    group_input = nodes.new(type='NodeGroupInput')
    group_input.location = (-400, 0)

    # Mesh to Points
    mesh_to_points = nodes.new(type='GeometryNodeMeshToPoints')
    mesh_to_points.location = (-200, 0)
    mesh_to_points.mode = 'VERTICES'

    # Create sphere mesh
    sphere_mesh = nodes.new(type='GeometryNodeMeshIcoSphere')
    sphere_mesh.location = (-200, -200)
    sphere_mesh.inputs['Radius'].default_value = point_size
    sphere_mesh.inputs['Subdivisions'].default_value = 1

    # Instance on Points
    instance_on_points = nodes.new(type='GeometryNodeInstanceOnPoints')
    instance_on_points.location = (0, 0)

    # Realize Instances
    realize_instances = nodes.new(type='GeometryNodeRealizeInstances')
    realize_instances.location = (200, 0)

    # Set Material (if provided)
    set_material = None
    if material:
        set_material = nodes.new(type='GeometryNodeSetMaterial')
        set_material.location = (400, 0)
        set_material.inputs['Material'].default_value = material

    # Output
    group_output = nodes.new(type='NodeGroupOutput')
    group_output.location = (600, 0)

    # Connect nodes
    links.new(group_input.outputs[0], mesh_to_points.inputs['Mesh'])
    links.new(mesh_to_points.outputs['Points'], instance_on_points.inputs['Points'])
    links.new(sphere_mesh.outputs['Mesh'], instance_on_points.inputs['Instance'])
    links.new(instance_on_points.outputs['Instances'], realize_instances.inputs['Geometry'])

    if set_material:
        links.new(realize_instances.outputs['Geometry'], set_material.inputs['Geometry'])
        links.new(set_material.outputs['Geometry'], group_output.inputs[0])
    else:
        links.new(realize_instances.outputs['Geometry'], group_output.inputs[0])

def create_material(name: str, color: tuple, alpha: float = 1.0, emission_strength: float = 0.0):
    """Create a material with given color and transparency."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Principled BSDF
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Alpha'].default_value = alpha
    bsdf.inputs['Emission Strength'].default_value = emission_strength
    if emission_strength > 0:
        bsdf.inputs['Emission Color'].default_value = color

    # Material Output
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (200, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    # Set blend mode
    if alpha < 1.0:
        mat.blend_method = 'BLEND'
        mat.shadow_method = 'NONE'

    return mat

def create_axis_helper(name: str, scale: float = 1.0):
    """Create XYZ axis helper using colored cylinders and cones."""
    # Create empty for pivot/organization
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = 'CUBE'
    empty.empty_display_size = 0.1
    bpy.context.collection.objects.link(empty)

    # Create materials for axes
    mat_x = create_material(f"{name}_X_Mat", (1.0, 0.0, 0.0, 1.0), alpha=1.0)  # Red
    mat_y = create_material(f"{name}_Y_Mat", (0.0, 1.0, 0.0, 1.0), alpha=1.0)  # Green
    mat_z = create_material(f"{name}_Z_Mat", (0.0, 0.0, 1.0, 1.0), alpha=1.0)  # Blue

    # X axis (Red) - cylinder + cone
    bpy.ops.mesh.primitive_cylinder_add(radius=scale * 0.05, depth=scale * 0.8, location=(scale * 0.4, 0, 0))
    x_shaft = bpy.context.active_object
    x_shaft.name = f"{name}_X_Shaft"
    x_shaft.rotation_euler = (0, math.pi / 2, 0)
    x_shaft.data.materials.append(mat_x)
    x_shaft.parent = empty

    bpy.ops.mesh.primitive_cone_add(radius1=scale * 0.1, depth=scale * 0.2, location=(scale * 0.9, 0, 0))
    x_arrow = bpy.context.active_object
    x_arrow.name = f"{name}_X_Arrow"
    x_arrow.rotation_euler = (0, math.pi / 2, 0)
    x_arrow.data.materials.append(mat_x)
    x_arrow.parent = empty

    # Y axis (Green) - cylinder + cone
    bpy.ops.mesh.primitive_cylinder_add(radius=scale * 0.05, depth=scale * 0.8, location=(0, scale * 0.4, 0))
    y_shaft = bpy.context.active_object
    y_shaft.name = f"{name}_Y_Shaft"
    y_shaft.rotation_euler = (math.pi / 2, 0, 0)
    y_shaft.data.materials.append(mat_y)
    y_shaft.parent = empty

    bpy.ops.mesh.primitive_cone_add(radius1=scale * 0.1, depth=scale * 0.2, location=(0, scale * 0.9, 0))
    y_arrow = bpy.context.active_object
    y_arrow.name = f"{name}_Y_Arrow"
    y_arrow.rotation_euler = (math.pi / 2, 0, 0)
    y_arrow.data.materials.append(mat_y)
    y_arrow.parent = empty

    # Z axis (Blue) - cylinder + cone
    bpy.ops.mesh.primitive_cylinder_add(radius=scale * 0.05, depth=scale * 0.8, location=(0, 0, scale * 0.4))
    z_shaft = bpy.context.active_object
    z_shaft.name = f"{name}_Z_Shaft"
    z_shaft.data.materials.append(mat_z)
    z_shaft.parent = empty

    bpy.ops.mesh.primitive_cone_add(radius1=scale * 0.1, depth=scale * 0.2, location=(0, 0, scale * 0.9))
    z_arrow = bpy.context.active_object
    z_arrow.name = f"{name}_Z_Arrow"
    z_arrow.data.materials.append(mat_z)
    z_arrow.parent = empty

    return empty

def generate_grid_ball_points(radius: float, spacing: float):
    """Generate points in a spherical grid."""
    points = []
    r = 0
    while r <= radius:
        for theta in np.arange(0, 2 * math.pi, spacing / max(r, 0.1)):
            for phi in np.arange(0, math.pi, spacing / max(r, 0.1)):
                x = r * math.sin(phi) * math.cos(theta)
                y = r * math.sin(phi) * math.sin(theta)
                z = r * math.cos(phi)
                points.append((x, y, z))
        r += spacing
    return points

def generate_fcc_points(grid_points, lattice_constant: float):
    """Transform grid points to FCC lattice positions."""
    # FCC basis vectors (normalized)
    fcc_basis = np.array([
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]) * lattice_constant

    fcc_points = []
    # Use subset of grid points for performance
    subset_points = grid_points[:len(grid_points)//4] if len(grid_points) > 100 else grid_points

    for point in subset_points:
        # For each grid point, create FCC lattice points at the corners
        base = np.array(point)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    offset = i * fcc_basis[0] + j * fcc_basis[1] + k * fcc_basis[2]
                    fcc_point = base + offset
                    fcc_points.append(tuple(fcc_point))

    return fcc_points

def filter_points_inside_grain(fcc_points, grain_points, threshold: float = 1.0, use_scipy: bool = False):
    """Filter FCC points that are inside the grain (within threshold distance)."""
    if not grain_points or not fcc_points:
        return []

    grain_array = np.array(grain_points)
    fcc_array = np.array(fcc_points)

    if use_scipy:
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(grain_array)
            distances, _ = tree.query(fcc_array)
            inside_mask = distances <= threshold
            inside_points = fcc_array[inside_mask].tolist()
            return [tuple(p) for p in inside_points]
        except ImportError:
            print("Warning: scipy not available, using simple distance calculation")

    # Fallback: simple distance calculation (slower but no dependencies)
    inside_points = []
    for fcc_point in fcc_points:
        fcc_arr = np.array(fcc_point)
        distances = np.linalg.norm(grain_array - fcc_arr, axis=1)
        if np.min(distances) <= threshold:
            inside_points.append(fcc_point)

    return inside_points

# ==================== Create Materials ====================

# Grid/FCC default material (light gray)
mat_grid_default = create_material("MAT_GridDefault", (0.7, 0.7, 0.7, 1.0), alpha=0.8)

# Grid/FCC highlighted material (warm color)
mat_grid_highlight = create_material("MAT_GridHighlight", (1.0, 0.6, 0.2, 1.0), alpha=1.0, emission_strength=0.3)

# Grain material (neutral light gray, semi-transparent)
mat_grain = create_material("MAT_Grain", (0.6, 0.6, 0.6, 1.0), alpha=0.4)

# ==================== Create Point Clouds ====================

print("Generating grid ball points...")
grid_points = generate_grid_ball_points(args.grid_ball_radius, args.grid_spacing)
print(f"Generated {len(grid_points)} grid points")

print("Generating FCC lattice points...")
fcc_points = generate_fcc_points(grid_points[:len(grid_points)//4], args.fcc_lattice_constant)  # Use subset for performance
print(f"Generated {len(fcc_points)} FCC points")

print("Loading grain points...")
grain_vertices = read_ply_file(Path(args.grain_ply))
print(f"Loaded {len(grain_vertices)} grain points")

# Create point cloud objects
grid_ball_obj = create_point_cloud_mesh(grid_points, "GridBallPoints")
fcc_ball_obj = create_point_cloud_mesh(fcc_points, "FCCBallPoints")
grain_obj = create_point_cloud_mesh(grain_vertices, "GrainPointCloud")

# Filter FCC points inside grain for Shot 4
print("Filtering FCC points inside grain...")
fcc_inside_points = filter_points_inside_grain(
    fcc_points,
    grain_vertices,
    threshold=args.point_size * 2,
    use_scipy=args.use_scipy
)
print(f"Found {len(fcc_inside_points)} FCC points inside grain")

# Create highlighted FCC points object (for Shot 4)
fcc_inside_obj = None
if fcc_inside_points:
    fcc_inside_obj = create_point_cloud_mesh(fcc_inside_points, "FCCInsideGrain")
    setup_point_cloud_geometry_nodes(fcc_inside_obj, args.point_size, mat_grid_highlight)

# Setup Geometry Nodes for point rendering
setup_point_cloud_geometry_nodes(grid_ball_obj, args.point_size, mat_grid_default)
setup_point_cloud_geometry_nodes(fcc_ball_obj, args.point_size, mat_grid_default)
setup_point_cloud_geometry_nodes(grain_obj, args.point_size, mat_grain)

# ==================== Create Object Hierarchy ====================

# Create RotationPivot (empty)
rotation_pivot = bpy.data.objects.new("RotationPivot", None)
rotation_pivot.empty_display_type = 'CUBE'
rotation_pivot.empty_display_size = 1.0
bpy.context.collection.objects.link(rotation_pivot)

# Parent all objects to RotationPivot
grid_ball_obj.parent = rotation_pivot
fcc_ball_obj.parent = rotation_pivot
grain_obj.parent = rotation_pivot
if fcc_inside_obj:
    fcc_inside_obj.parent = rotation_pivot

# Create AxisHelper
axis_helper = create_axis_helper("AxisHelper", scale=5.0)
axis_helper.parent = rotation_pivot
# Position axis helper at lower-left of grid ball
axis_helper.location = (-args.grid_ball_radius * 0.8, -args.grid_ball_radius * 0.8, -args.grid_ball_radius * 0.8)

# ==================== Setup Camera ====================

# Create camera
cam = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam)
bpy.context.collection.objects.link(cam_obj)

# Position camera (fixed for entire animation)
# Camera looks at the center of the scene
cam_obj.location = (50, -50, 30)
cam_obj.rotation_euler = (math.radians(60), 0, math.radians(45))

# Set camera as active
bpy.context.scene.camera = cam_obj

# ==================== Animation Keyframes ====================

def set_visibility_keyframe(obj, frame: int, visible: bool):
    """Set visibility keyframe for object."""
    obj.hide_viewport = not visible
    obj.hide_render = not visible
    obj.keyframe_insert(data_path="hide_viewport", frame=frame)
    obj.keyframe_insert(data_path="hide_render", frame=frame)

def set_alpha_keyframe(material, frame: int, alpha: float):
    """Set alpha keyframe for material."""
    if material.node_tree.nodes.get("Principled BSDF"):
        material.node_tree.nodes["Principled BSDF"].inputs['Alpha'].default_value = alpha
        material.node_tree.nodes["Principled BSDF"].inputs['Alpha'].keyframe_insert(data_path="default_value", frame=frame)

def fade_in(obj, start_frame: int, end_frame: int):
    """Fade in object by animating material alpha."""
    mat = obj.data.materials[0] if obj.data.materials else None
    if mat:
        set_alpha_keyframe(mat, start_frame, 0.0)
        set_alpha_keyframe(mat, end_frame, mat.node_tree.nodes["Principled BSDF"].inputs['Alpha'].default_value)

def fade_out(obj, start_frame: int, end_frame: int):
    """Fade out object by animating material alpha."""
    mat = obj.data.materials[0] if obj.data.materials else None
    if mat:
        current_alpha = mat.node_tree.nodes["Principled BSDF"].inputs['Alpha'].default_value
        set_alpha_keyframe(mat, start_frame, current_alpha)
        set_alpha_keyframe(mat, end_frame, 0.0)

print("\nSetting up animation keyframes...")

# Shot 0: Grid Ball with Local Coordinate System (frames 1-40)
print("Shot 0: Grid Ball with Local Coordinate System")
set_visibility_keyframe(grid_ball_obj, SHOT_0_START, True)
set_visibility_keyframe(fcc_ball_obj, SHOT_0_START, False)
set_visibility_keyframe(grain_obj, SHOT_0_START, False)
set_visibility_keyframe(axis_helper, SHOT_0_START, True)

# Shot 1: FCC Lattice Generation (frames 41-80)
print("Shot 1: FCC Lattice Generation")
set_visibility_keyframe(fcc_ball_obj, SHOT_1_START, True)
fade_in(fcc_ball_obj, SHOT_1_START, SHOT_1_END)
# Optionally fade out grid ball slightly
fade_out(grid_ball_obj, SHOT_1_START, SHOT_1_END)

# Shot 2: FCC Grid Ball Rotation (frames 81-160)
print("Shot 2: FCC Grid Ball Rotation")
# Rotate RotationPivot around Z axis (180 degrees)
rotation_pivot.rotation_euler = (0, 0, 0)
rotation_pivot.keyframe_insert(data_path="rotation_euler", frame=SHOT_2_START, index=2)  # Z rotation

rotation_pivot.rotation_euler = (0, 0, math.radians(180))
rotation_pivot.keyframe_insert(data_path="rotation_euler", frame=SHOT_2_END, index=2)

# Shot 3: Overlap with Experimental Grain (frames 161-220)
print("Shot 3: Overlap with Experimental Grain")
set_visibility_keyframe(grain_obj, SHOT_3_START, True)
fade_in(grain_obj, SHOT_3_START, SHOT_3_END)

# Move FCC ball slightly to overlap with grain (translation only, no rotation)
fcc_ball_obj.location = (0, 0, 0)
fcc_ball_obj.keyframe_insert(data_path="location", frame=SHOT_2_END)

# Calculate grain centroid for alignment
if grain_vertices:
    grain_centroid = np.mean(grain_vertices, axis=0)
    # Adjust FCC ball position to overlap
    fcc_ball_obj.location = tuple(grain_centroid - np.array([0, 0, 0]))  # Adjust as needed
    fcc_ball_obj.keyframe_insert(data_path="location", frame=SHOT_3_END)

# Shot 4: Final Overlap Result (frames 221-260)
print("Shot 4: Final Overlap Result")
# Hide original FCC ball, show only highlighted points inside grain
fade_out(grid_ball_obj, SHOT_4_START, SHOT_4_END)
fade_out(fcc_ball_obj, SHOT_4_START, SHOT_4_END)

# Show highlighted FCC points inside grain
if fcc_inside_obj:
    set_visibility_keyframe(fcc_inside_obj, SHOT_4_START, True)
    fade_in(fcc_inside_obj, SHOT_4_START, SHOT_4_START + 10)

# Reduce grain opacity to show FCC points better
fade_out(grain_obj, SHOT_4_START + 10, SHOT_4_END)

# Reduce axis helper visibility
for child in axis_helper.children:
    if child.data and child.data.materials:
        for mat in child.data.materials:
            if mat:
                set_alpha_keyframe(mat, SHOT_4_START, 1.0)
                set_alpha_keyframe(mat, SHOT_4_END, 0.5)

print("\nAnimation setup complete!")

# ==================== Save File ====================

output_path = Path(args.output_blend)
bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
print(f"\nSaved Blender file: {output_path}")
print(f"\nAnimation frames: {SHOT_0_START} to {SHOT_4_END}")
print("You can now render the animation or adjust settings in Blender.")
