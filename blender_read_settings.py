#!/usr/bin/env python3
"""
Blender Script: Read and Display .blend File Settings

This script reads a Blender file and displays:
- Scene settings (frame range, resolution, render engine)
- Objects and their properties
- Materials and their settings
- Animation keyframes
- Camera settings
- World/background settings
- And more...

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python blender_read_settings.py -- --blend-file workflow_animation.blend
"""

import argparse
import json
import sys
from pathlib import Path

import bpy

# Parse command line arguments
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

parser = argparse.ArgumentParser(description="Read Blender File Settings")
parser.add_argument(
    "--blend-file",
    type=str,
    required=True,
    help="Path to .blend file to read"
)
parser.add_argument(
    "--output-json",
    type=str,
    default=None,
    help="Output JSON file path (optional)"
)
parser.add_argument(
    "--detailed",
    action="store_true",
    help="Show detailed information (materials, keyframes, etc.)"
)

args = parser.parse_args(argv)

blend_path = Path(args.blend_file)
if not blend_path.exists():
    print(f"Error: File not found: {blend_path}")
    sys.exit(1)

# Load the .blend file
print(f"Loading Blender file: {blend_path}")
bpy.ops.wm.open_mainfile(filepath=str(blend_path))

# ==================== Collect Information ====================

info = {
    "file_path": str(blend_path),
    "scene": {},
    "objects": [],
    "materials": [],
    "cameras": [],
    "world": {},
    "animation": {},
}

# Scene Settings
scene = bpy.context.scene
info["scene"] = {
    "name": scene.name,
    "frame_start": scene.frame_start,
    "frame_end": scene.frame_end,
    "frame_current": scene.frame_current,
    "fps": scene.render.fps,
    "resolution_x": scene.render.resolution_x,
    "resolution_y": scene.render.resolution_y,
    "resolution_percentage": scene.render.resolution_percentage,
    "render_engine": scene.render.engine,
    "film_transparent": scene.render.film_transparent,
    "active_camera": scene.camera.name if scene.camera else None,
}

# World/Background Settings
world = scene.world
if world:
    info["world"] = {
        "name": world.name,
        "use_nodes": world.use_nodes,
    }

    if world.use_nodes:
        nodes_info = []
        for node in world.node_tree.nodes:
            node_info = {
                "name": node.name,
                "type": node.type,
                "location": list(node.location),
            }
            if node.type == 'BACKGROUND':
                node_info["color"] = list(node.inputs['Color'].default_value)
                node_info["strength"] = node.inputs['Strength'].default_value
            nodes_info.append(node_info)
        info["world"]["nodes"] = nodes_info

# Objects
for obj in scene.objects:
    obj_info = {
        "name": obj.name,
        "type": obj.type,
        "location": list(obj.location),
        "rotation_euler": list(obj.rotation_euler),
        "scale": list(obj.scale),
        "hide_viewport": obj.hide_viewport,
        "hide_render": obj.hide_render,
        "parent": obj.parent.name if obj.parent else None,
        "children": [child.name for child in obj.children],
    }

    # Object-specific information
    if obj.type == 'MESH':
        obj_info["mesh"] = {
            "vertices_count": len(obj.data.vertices),
            "edges_count": len(obj.data.edges),
            "faces_count": len(obj.data.faces) if hasattr(obj.data, 'faces') else len(obj.data.polygons),
            "materials": [mat.name for mat in obj.data.materials] if obj.data.materials else [],
        }

        # Geometry Nodes modifiers
        obj_info["modifiers"] = []
        for mod in obj.modifiers:
            if mod.type == 'NODES':
                mod_info = {
                    "name": mod.name,
                    "type": mod.type,
                    "node_group": mod.node_group.name if mod.node_group else None,
                }
                obj_info["modifiers"].append(mod_info)

    elif obj.type == 'CAMERA':
        cam = obj.data
        obj_info["camera"] = {
            "type": cam.type,
            "lens": cam.lens,
            "sensor_width": cam.sensor_width,
            "clip_start": cam.clip_start,
            "clip_end": cam.clip_end,
        }
        info["cameras"].append(obj_info)

    elif obj.type == 'EMPTY':
        obj_info["empty_display_type"] = obj.empty_display_type
        obj_info["empty_display_size"] = obj.empty_display_size

    # Animation keyframes (if detailed)
    if args.detailed:
        obj_info["keyframes"] = {}
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                data_path = fcurve.data_path
                if data_path not in obj_info["keyframes"]:
                    obj_info["keyframes"][data_path] = []
                for keyframe in fcurve.keyframe_points:
                    obj_info["keyframes"][data_path].append({
                        "frame": int(keyframe.co[0]),
                        "value": keyframe.co[1],
                    })

    info["objects"].append(obj_info)

# Materials
for mat in bpy.data.materials:
    mat_info = {
        "name": mat.name,
        "use_nodes": mat.use_nodes,
        "blend_method": mat.blend_method,
        "shadow_method": mat.shadow_method,
    }

    if mat.use_nodes and args.detailed:
        nodes_info = []
        for node in mat.node_tree.nodes:
            node_info = {
                "name": node.name,
                "type": node.type,
                "location": list(node.location),
            }

            # Node-specific information
            if node.type == 'BSDF_PRINCIPLED':
                node_info["inputs"] = {}
                for input_socket in node.inputs:
                    if input_socket.is_linked:
                        node_info["inputs"][input_socket.name] = "linked"
                    else:
                        value = input_socket.default_value
                        if isinstance(value, (list, tuple)) and len(value) <= 4:
                            node_info["inputs"][input_socket.name] = list(value)
                        else:
                            node_info["inputs"][input_socket.name] = value

            nodes_info.append(node_info)
        mat_info["nodes"] = nodes_info

    info["materials"].append(mat_info)

# Animation Summary
if args.detailed:
    info["animation"]["total_keyframes"] = 0
    info["animation"]["objects_with_animation"] = []
    for obj in scene.objects:
        if obj.animation_data and obj.animation_data.action:
            keyframe_count = sum(len(fcurve.keyframe_points) for fcurve in obj.animation_data.action.fcurves)
            if keyframe_count > 0:
                info["animation"]["total_keyframes"] += keyframe_count
                info["animation"]["objects_with_animation"].append({
                    "name": obj.name,
                    "keyframe_count": keyframe_count,
                })

# ==================== Print Information ====================

print("\n" + "=" * 80)
print("BLENDER FILE INFORMATION")
print("=" * 80)

print(f"\n📁 File: {info['file_path']}")

print("\n🎬 Scene Settings:")
print(f"  Name: {info['scene']['name']}")
print(f"  Frame Range: {info['scene']['frame_start']} - {info['scene']['frame_end']} ({info['scene']['frame_end'] - info['scene']['frame_start'] + 1} frames)")
print(f"  FPS: {info['scene']['fps']}")
print(f"  Resolution: {info['scene']['resolution_x']} x {info['scene']['resolution_y']} ({info['scene']['resolution_percentage']}%)")
print(f"  Render Engine: {info['scene']['render_engine']}")
print(f"  Film Transparent: {info['scene']['film_transparent']}")
print(f"  Active Camera: {info['scene']['active_camera']}")

if info['world']:
    print("\n🌍 World/Background:")
    print(f"  Name: {info['world']['name']}")
    print(f"  Use Nodes: {info['world']['use_nodes']}")
    if info['world'].get('nodes'):
        for node in info['world']['nodes']:
            if node['type'] == 'BACKGROUND':
                color = node.get('color', [1, 1, 1, 1])
                strength = node.get('strength', 1.0)
                print(f"  Background Color: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f}), Strength: {strength:.2f}")

print(f"\n📦 Objects ({len(info['objects'])} total):")
for obj in info['objects']:
    print(f"  • {obj['name']} ({obj['type']})")
    print(f"    Location: ({obj['location'][0]:.2f}, {obj['location'][1]:.2f}, {obj['location'][2]:.2f})")
    if obj['parent']:
        print(f"    Parent: {obj['parent']}")
    if obj['children']:
        print(f"    Children: {', '.join(obj['children'])}")
    if obj['type'] == 'MESH' and 'mesh' in obj:
        mesh = obj['mesh']
        print(f"    Vertices: {mesh['vertices_count']}, Edges: {mesh['edges_count']}, Faces: {mesh['faces_count']}")
        if mesh['materials']:
            print(f"    Materials: {', '.join(mesh['materials'])}")
    if obj['type'] == 'CAMERA' and 'camera' in obj:
        cam = obj['camera']
        print(f"    Lens: {cam['lens']}mm, Type: {cam['type']}")
    if args.detailed and obj.get('keyframes'):
        total_kf = sum(len(kfs) for kfs in obj['keyframes'].values())
        print(f"    Keyframes: {total_kf} total")

print(f"\n🎨 Materials ({len(info['materials'])} total):")
for mat in info['materials']:
    print(f"  • {mat['name']}")
    print(f"    Blend Method: {mat['blend_method']}, Shadow Method: {mat['shadow_method']}")
    if args.detailed and mat.get('nodes'):
        node_count = len(mat['nodes'])
        print(f"    Nodes: {node_count} total")
        for node in mat['nodes']:
            if node['type'] == 'BSDF_PRINCIPLED':
                print(f"      - {node['name']} ({node['type']})")

if info['cameras']:
    print(f"\n📷 Cameras ({len(info['cameras'])} total):")
    for cam in info['cameras']:
        print(f"  • {cam['name']}")
        print(f"    Location: ({cam['location'][0]:.2f}, {cam['location'][1]:.2f}, {cam['location'][2]:.2f})")
        if 'camera' in cam:
            print(f"    Lens: {cam['camera']['lens']}mm")

if args.detailed and info['animation']:
    print("\n🎞️  Animation:")
    print(f"  Total Keyframes: {info['animation']['total_keyframes']}")
    if info['animation']['objects_with_animation']:
        print(f"  Animated Objects: {len(info['animation']['objects_with_animation'])}")
        for anim_obj in info['animation']['objects_with_animation']:
            print(f"    • {anim_obj['name']}: {anim_obj['keyframe_count']} keyframes")

# ==================== Save JSON (if requested) ====================

if args.output_json:
    output_path = Path(args.output_json)
    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2, default=str)
    print(f"\n💾 Saved detailed information to: {output_path}")

print("\n" + "=" * 80)
print("Done!")
print("=" * 80)
