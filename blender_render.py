#!/usr/bin/env python3
"""
Blender Script for Rendering Grain Point Cloud

This script:
1. Loads point cloud files (PLY format)
2. Creates point cloud meshes
3. Uses Geometry Nodes to instance cubes on points
4. Assigns materials (gray for grain, light blue for margin)
5. Renders turntable animation (61 frames, 180° rotation)

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python blender_render.py -- --grain-ply grain_111_points.ply --margin-ply grain_111_margin_points.ply
"""

import argparse
import math
import sys
from pathlib import Path

import bpy
import mathutils

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Parse command line arguments
# Blender passes arguments after '--'
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

parser = argparse.ArgumentParser(description="Blender Point Cloud Renderer")
parser.add_argument(
    "--grain-ply",
    type=str,
    required=True,
    help="Path to grain points PLY file"
)
parser.add_argument(
    "--margin-ply",
    type=str,
    default=None,
    help="Path to margin points PLY file (optional)"
)
parser.add_argument(
    "--voxel-size",
    type=float,
    default=1.0,
    help="Voxel cube size (default: 1.0)"
)
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Output directory for .blend file (default: current directory)"
)
parser.add_argument(
    "--output-blend",
    type=str,
    default=None,
    help="Output .blend file path (default: grain_<ID>_render.blend)"
)
parser.add_argument(
    "--grain-color",
    type=str,
    default="#8E8E93",
    help="Grain color in hex format (default: #8E8E93)"
)
parser.add_argument(
    "--margin-color",
    type=str,
    default="#A7C7E7",
    help="Margin color in hex format (default: #A7C7E7)"
)
parser.add_argument(
    "--n-frames",
    type=int,
    default=61,
    help="Number of frames (default: 61)"
)
parser.add_argument(
    "--step-deg",
    type=float,
    default=3.0,
    help="Rotation step per frame in degrees (default: 3.0)"
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
parser.add_argument(
    "--geometry-nodes-config",
    type=str,
    default=None,
    help="JSON file with Geometry Nodes configuration from manual file (optional)"
)

args = parser.parse_args(argv)

# Load Geometry Nodes config if provided
geometry_nodes_config = None
if args.geometry_nodes_config:
    import json
    config_path = Path(args.geometry_nodes_config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            all_configs = json.load(f)
            # Use the first available config (or you can specify which one)
            if all_configs:
                first_key = list(all_configs.keys())[0]
                geometry_nodes_config = all_configs[first_key]
                print(f"Loaded Geometry Nodes config from: {config_path}")
                print(f"  Using config for: {geometry_nodes_config.get('object_name', 'unknown')}")
    else:
        print(f"Warning: Geometry Nodes config file not found: {config_path}")

# ==================== Helper Functions ====================

def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b, 1.0)

# ==================== Setup Scene ====================

# Set render engine to Cycles (for better transparency)
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 128  # Adjust for quality/speed

# Set render resolution
bpy.context.scene.render.resolution_x = args.resolution_x
bpy.context.scene.render.resolution_y = args.resolution_y

# Set output format (for future rendering)
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.image_settings.color_mode = 'RGBA'

# ==================== Create Materials ====================

# Grain material (Apple gray, semi-transparent)
grain_mat = bpy.data.materials.new(name="GrainMaterial")
grain_mat.use_nodes = True
# Clear existing nodes (Blender 4.0+ API)
grain_mat.node_tree.nodes.clear()

# Add Principled BSDF
bsdf = grain_mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
grain_color = hex_to_rgb(args.grain_color)
bsdf.inputs['Base Color'].default_value = grain_color
bsdf.inputs['Alpha'].default_value = 0.4  # Semi-transparent
bsdf.inputs['Roughness'].default_value = 0.5

# Add Material Output
output = grain_mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
grain_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

# Enable transparency (Blender 4.0+ handles alpha through BSDF)
grain_mat.blend_method = 'BLEND'

# Margin material (Light blue, semi-transparent)
margin_mat = bpy.data.materials.new(name="MarginMaterial")
margin_mat.use_nodes = True
# Clear existing nodes (Blender 4.0+ API)
margin_mat.node_tree.nodes.clear()

bsdf_margin = margin_mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
margin_color = hex_to_rgb(args.margin_color)
bsdf_margin.inputs['Base Color'].default_value = margin_color
bsdf_margin.inputs['Alpha'].default_value = 0.17  # More transparent
bsdf_margin.inputs['Roughness'].default_value = 0.5

output_margin = margin_mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
margin_mat.node_tree.links.new(bsdf_margin.outputs['BSDF'], output_margin.inputs['Surface'])

margin_mat.blend_method = 'BLEND'

# Enable transparency in render settings
bpy.context.scene.render.film_transparent = True

# ==================== Load Point Clouds ====================

def read_ply_file(ply_path: Path):
    """Read PLY file and return vertices as numpy array."""
    vertices = []
    in_vertex_section = False

    with open(ply_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('element vertex'):
                int(line.split()[-1])
                in_vertex_section = False
            elif line == 'end_header':
                in_vertex_section = True
                continue
            elif in_vertex_section and line:
                # Parse vertex line: x y z
                parts = line.split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    vertices.append([x, y, z])

    return vertices


def create_point_cloud_from_ply(ply_path: Path, name: str):
    """Load PLY file and create mesh object."""
    # Read PLY file manually
    vertices = read_ply_file(ply_path)

    if len(vertices) == 0:
        print(f"Warning: No vertices found in {ply_path}")
        return None

    # Create mesh from vertices
    mesh = bpy.data.meshes.new(name=name)
    mesh.from_pydata(vertices, [], [])  # vertices, edges, faces
    mesh.update()

    # Create object
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # Make it active
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    return obj


def setup_geometry_nodes(obj, voxel_size: float, geometry_nodes_config: dict = None):
    """
    Setup Geometry Nodes for instancing cubes on points.

    If geometry_nodes_config is provided, it will use those settings.
    Otherwise, creates default setup.

    Node tree:
    - Mesh to Points
    - Instance on Points (Cube, size=voxel_size)
    - Realize Instances
    - Group Output
    """
    # Add Geometry Nodes modifier
    mod = obj.modifiers.new(name="GeometryNodes", type='NODES')

    # Create node group
    node_group = bpy.data.node_groups.new(name="VoxelInstancer", type='GeometryNodeTree')
    mod.node_group = node_group

    # Add input and output sockets explicitly
    node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

    # If config provided, use it to recreate the node setup
    if geometry_nodes_config:
        print("  Applying Geometry Nodes configuration from manual file...")
        # Create nodes from config
        nodes_map = {}  # Map node names to actual node objects

        # Create all nodes first
        for node_data in geometry_nodes_config.get('nodes', []):
            node_type = node_data['type']
            node_name = node_data['name']

            # Create node
            node = node_group.nodes.new(type=node_type)
            node.name = node_name
            node.location = node_data['location']
            nodes_map[node_name] = node

            # Set input values
            for input_name, input_value in node_data.get('inputs', {}).items():
                if input_name in node.inputs:
                    try:
                        if isinstance(input_value, list) and len(input_value) <= 4:
                            # Vector/color input
                            node.inputs[input_name].default_value = tuple(input_value)
                        else:
                            # Scalar input
                            node.inputs[input_name].default_value = input_value
                    except Exception as e:
                        print(f"    Warning: Could not set {input_name} = {input_value}: {e}")

        # Create links
        for link_data in geometry_nodes_config.get('links', []):
            from_node = nodes_map.get(link_data['from_node'])
            to_node = nodes_map.get(link_data['to_node'])

            if from_node and to_node:
                try:
                    from_socket = from_node.outputs.get(link_data['from_socket'])
                    to_socket = to_node.inputs.get(link_data['to_socket'])

                    if from_socket and to_socket:
                        node_group.links.new(from_socket, to_socket)
                except Exception as e:
                    print(f"    Warning: Could not create link {link_data['from_node']}.{link_data['from_socket']} → {link_data['to_node']}.{link_data['to_socket']}: {e}")

        print(f"  Applied {len(nodes_map)} nodes and {len(geometry_nodes_config.get('links', []))} links")
        return

    # Default setup (if no config provided)
    # Add nodes
    # Input
    group_input = node_group.nodes.new(type='NodeGroupInput')
    group_input.location = (-400, 0)

    # Mesh to Points
    mesh_to_points = node_group.nodes.new(type='GeometryNodeMeshToPoints')
    mesh_to_points.location = (-200, 0)
    mesh_to_points.mode = 'VERTICES'

    # Create cube mesh
    cube_mesh = node_group.nodes.new(type='GeometryNodeMeshCube')
    cube_mesh.location = (-200, -200)
    cube_mesh.inputs['Size'].default_value = (voxel_size, voxel_size, voxel_size)

    # Instance on Points
    instance_on_points = node_group.nodes.new(type='GeometryNodeInstanceOnPoints')
    instance_on_points.location = (0, 0)

    # Realize Instances (convert instances to actual geometry)
    realize_instances = node_group.nodes.new(type='GeometryNodeRealizeInstances')
    realize_instances.location = (200, 0)

    # Output
    group_output = node_group.nodes.new(type='NodeGroupOutput')
    group_output.location = (400, 0)

    # Connect nodes
    node_group.links.new(group_input.outputs[0], mesh_to_points.inputs['Mesh'])
    node_group.links.new(mesh_to_points.outputs['Points'], instance_on_points.inputs['Points'])
    node_group.links.new(cube_mesh.outputs['Mesh'], instance_on_points.inputs['Instance'])
    node_group.links.new(instance_on_points.outputs['Instances'], realize_instances.inputs['Geometry'])
    node_group.links.new(realize_instances.outputs['Geometry'], group_output.inputs[0])


# Load grain points
print(f"Loading grain points from: {args.grain_ply}")
grain_obj = create_point_cloud_from_ply(Path(args.grain_ply), "GrainPoints")
if grain_obj is None:
    print("Error: Failed to load grain points")
    sys.exit(1)

# Use Geometry Nodes config if available, otherwise use default
grain_gn_config = None
margin_gn_config = None

if geometry_nodes_config:
    # Try to find configs for GrainPoints and MarginPoints
    if 'GrainPoints' in geometry_nodes_config.get('object_name', ''):
        grain_gn_config = geometry_nodes_config
    elif 'MarginPoints' in geometry_nodes_config.get('object_name', ''):
        margin_gn_config = geometry_nodes_config

setup_geometry_nodes(grain_obj, args.voxel_size, grain_gn_config)
grain_obj.data.materials.append(grain_mat)

# Load margin points if provided
margin_obj = None
if args.margin_ply and Path(args.margin_ply).exists():
    print(f"Loading margin points from: {args.margin_ply}")
    margin_obj = create_point_cloud_from_ply(Path(args.margin_ply), "MarginPoints")
    if margin_obj is not None:
        setup_geometry_nodes(margin_obj, args.voxel_size, margin_gn_config)
        margin_obj.data.materials.append(margin_mat)

# ==================== Center Objects and Setup Camera ====================

# Calculate bounding box center of all objects (grain + margin)
all_objects = [grain_obj]
if margin_obj:
    all_objects.append(margin_obj)

# Get combined bounding box
min_coords = [float('inf')] * 3
max_coords = [float('-inf')] * 3

for obj in all_objects:
    # Update object to get correct bounding box
    bpy.context.view_layer.update()
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    for corner in bbox_corners:
        for i in range(3):
            min_coords[i] = min(min_coords[i], corner[i])
            max_coords[i] = max(max_coords[i], corner[i])

center = mathutils.Vector([
    (min_coords[0] + max_coords[0]) / 2,
    (min_coords[1] + max_coords[1]) / 2,
    (min_coords[2] + max_coords[2]) / 2,
])

# Move all objects to center at origin (0, 0, 0)
# This makes them easier to control and view
for obj in all_objects:
    obj.location = obj.location - center

# Update center to origin
center = mathutils.Vector((0, 0, 0))

# Calculate camera distance based on bounding box
max_range = max(
    max_coords[0] - min_coords[0],
    max_coords[1] - min_coords[1],
    max_coords[2] - min_coords[2],
)
camera_distance = max_range * 2.5

# Create camera
bpy.ops.object.camera_add(location=(camera_distance, 0, 0))
camera = bpy.context.active_object
camera.name = "Camera"
camera.data.type = 'PERSP'

# Set camera to look at origin (where objects are now centered)
camera.location = (camera_distance, 0, 0)
# Camera on XY plane, looking at origin, Z-up
direction = mathutils.Vector((0, 0, 0)) - camera.location
rot_quat = direction.to_track_quat('-Z', 'Y')
camera.rotation_euler = rot_quat.to_euler()

# Alternative: explicit rotation for XY plane view
camera.rotation_euler = (math.radians(90), 0, math.radians(90))

bpy.context.scene.camera = camera

# ==================== Setup Lighting ====================

# Add area light
bpy.ops.object.light_add(type='AREA', location=(camera_distance * 0.8, 0, center.z))
light = bpy.context.active_object
light.data.energy = 100
light.data.size = max_range

# Add world lighting
bpy.context.scene.world.use_nodes = True
world_bg = bpy.context.scene.world.node_tree.nodes['Background']
world_bg.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # White background
world_bg.inputs['Strength'].default_value = 1.0

# ==================== Animation Setup ====================

# Set frame range (for animation preview in Blender)
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = args.n_frames

# Create empty at origin for rotation pivot
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
pivot = bpy.context.active_object
pivot.name = "RotationPivot"

# ==================== Create GrainVoxel (Outer Surface + Inner Volume) ====================

def create_grain_voxel_surface_and_volume(grain_obj, voxel_size: float):
    """
    Create GrainVoxel object with:
    1. Outer surface only (removed internal faces)
    2. Inner volume (filled mesh representing merged voxels)

    Strategy:
    - Duplicate grain object and apply Geometry Nodes to get voxel mesh
    - Remove internal faces to get outer surface
    - Create a filled volume mesh for the interior
    """
    print("Creating GrainVoxel object (outer surface + inner volume)...")

    if len(grain_obj.data.vertices) == 0:
        print("Warning: No vertices in grain object")
        return None

    # Create a copy of the grain object for processing
    bpy.context.view_layer.objects.active = grain_obj
    grain_obj.select_set(True)
    bpy.ops.object.duplicate()
    temp_obj = bpy.context.active_object
    temp_obj.name = "GrainVoxel_Temp"

    # Apply Geometry Nodes modifier to convert points to voxel cubes
    bpy.ops.object.select_all(action='DESELECT')
    temp_obj.select_set(True)
    bpy.context.view_layer.objects.active = temp_obj

    # Apply all modifiers to get actual geometry
    if temp_obj.modifiers:
        for mod in list(temp_obj.modifiers):
            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
            except:
                print(f"  Warning: Could not apply modifier {mod.name}")

    # Now we have a mesh with all voxel cubes
    # Extract outer surface by removing internal faces

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    # Method 1: Try to select and delete interior faces
    # This removes faces that are shared by multiple voxels (internal faces)
    try:
        # Select faces by number of connections (interior faces have more connections)
        # First, deselect all
        bpy.ops.mesh.select_all(action='DESELECT')

        # Select non-manifold edges (edges shared by more than 2 faces)
        bpy.ops.mesh.select_mode(type='EDGE')
        bpy.ops.mesh.select_non_manifold(extend=False, use_wire=False, use_boundary=True, use_multi_face=False, use_non_contiguous=False, use_verts=False)

        # Convert edge selection to face selection
        bpy.ops.mesh.select_mode(type='FACE')
        bpy.ops.mesh.select_linked()

        # Invert selection to get interior faces
        bpy.ops.mesh.select_all(action='INVERT')

        # Delete interior faces
        bpy.ops.mesh.delete(type='FACE')

        print("  Removed interior faces using non-manifold edge method")
    except Exception as e:
        print(f"  Method 1 failed: {e}, trying alternative method...")
        # Method 2: Use "Select Interior Faces" if available
        try:
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.select_interior_faces()
            bpy.ops.mesh.delete(type='FACE')
            print("  Removed interior faces using select_interior_faces")
        except:
            # Method 3: Fallback - use convex hull (simpler but less accurate)
            print("  Using convex hull as fallback for outer surface")
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.convex_hull()

    bpy.ops.object.mode_set(mode='OBJECT')

    # Create inner volume mesh (filled solid)
    # Duplicate the outer surface mesh
    bpy.context.view_layer.objects.active = temp_obj
    temp_obj.select_set(True)
    bpy.ops.object.duplicate()
    volume_obj = bpy.context.active_object
    volume_obj.name = "GrainVoxel_Volume"

    # Fill the volume mesh to create a solid
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    # Try to fill holes and create a solid mesh
    try:
        # Fill any holes in the mesh
        bpy.ops.mesh.fill_holes(sides=0)
        # Make the mesh manifold (solid)
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
    except:
        pass

    bpy.ops.object.mode_set(mode='OBJECT')

    # Join the outer surface and inner volume into one object
    bpy.ops.object.select_all(action='DESELECT')
    temp_obj.select_set(True)
    volume_obj.select_set(True)
    bpy.context.view_layer.objects.active = temp_obj
    bpy.ops.object.join()

    # Rename to GrainVoxel
    temp_obj.name = "GrainVoxel"

    # Create material for GrainVoxel
    voxel_mat = bpy.data.materials.new(name="GrainVoxelMaterial")
    voxel_mat.use_nodes = True
    voxel_mat.node_tree.nodes.clear()

    bsdf_voxel = voxel_mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    voxel_color = hex_to_rgb(args.grain_color)
    bsdf_voxel.inputs['Base Color'].default_value = voxel_color
    bsdf_voxel.inputs['Alpha'].default_value = 0.3  # More transparent for volume effect
    bsdf_voxel.inputs['Roughness'].default_value = 0.5

    output_voxel = voxel_mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    voxel_mat.node_tree.links.new(bsdf_voxel.outputs['BSDF'], output_voxel.inputs['Surface'])
    voxel_mat.blend_method = 'BLEND'

    temp_obj.data.materials.append(voxel_mat)

    print(f"  Created GrainVoxel: {len(temp_obj.data.vertices)} vertices, {len(temp_obj.data.polygons)} faces")

    return temp_obj

# Create GrainVoxel object
grain_voxel_obj = create_grain_voxel_surface_and_volume(grain_obj, args.voxel_size)

# Parent all objects to pivot (so they rotate together but remain separate objects)
for obj in all_objects:
    obj.parent = pivot

# Parent GrainVoxel to pivot as well
if grain_voxel_obj:
    grain_voxel_obj.parent = pivot
    print(f"Created GrainVoxel with {len(grain_voxel_obj.data.vertices)} vertices")

# Animate rotation (360 degrees for full rotation)
pivot.rotation_euler = (0, 0, 0)
pivot.keyframe_insert(data_path="rotation_euler", frame=1, index=2)  # Z rotation only

# 360 degrees rotation
total_rotation_360 = math.radians(360.0)
pivot.rotation_euler = (0, 0, total_rotation_360)
pivot.keyframe_insert(data_path="rotation_euler", frame=args.n_frames, index=2)

# Set interpolation to linear for smooth rotation
if pivot.animation_data and pivot.animation_data.action:
    for fcurve in pivot.animation_data.action.fcurves:
        if fcurve.data_path == "rotation_euler" and fcurve.array_index == 2:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'

# ==================== Save Blender File ====================

# Extract grain ID from filename if not provided, or use default
grain_id = "111"  # Default
if args.grain_ply:
    # Try to extract grain ID from filename (e.g., "grain_111_points.ply" -> "111")
    import re
    match = re.search(r'grain_(\d+)', args.grain_ply)
    if match:
        grain_id = match.group(1)

# Determine output .blend file path
if args.output_blend:
    blend_output = Path(args.output_blend)
elif args.output_dir:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blend_output = output_dir / f"grain_{grain_id}_render.blend"
else:
    blend_output = Path(f"grain_{grain_id}_render.blend")

print(f"\nSaving Blender file: {blend_output}")
bpy.ops.wm.save_as_mainfile(filepath=str(blend_output))

print("\nBlender file saved!")
print(f"File: {blend_output}")
print("\nYou can now open this file in Blender to:")
print("  - View the 3D scene")
print("  - Adjust materials, lighting, camera")
print("  - Render individual frames or animation")
print("  - Export to other formats")
