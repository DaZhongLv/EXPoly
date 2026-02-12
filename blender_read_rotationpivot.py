#!/usr/bin/env python3
"""
Read RotationPivot settings from a Blender file
"""

import json
import sys

import bpy

blend_file = sys.argv[-1] if len(sys.argv) > 1 else "grain_111_render_manual.blend"

print(f"Loading: {blend_file}")
bpy.ops.wm.open_mainfile(filepath=blend_file)

scene = bpy.context.scene
rotation_pivot = scene.objects.get("RotationPivot")

if not rotation_pivot:
    print("ERROR: RotationPivot not found!")
    print("Available objects:", [obj.name for obj in scene.objects])
    sys.exit(1)

print("\n" + "=" * 80)
print("ROTATIONPIVOT SETTINGS")
print("=" * 80)

info = {
    "name": rotation_pivot.name,
    "type": rotation_pivot.type,
    "location": list(rotation_pivot.location),
    "rotation_euler": list(rotation_pivot.rotation_euler),
    "rotation_quaternion": list(rotation_pivot.rotation_quaternion),
    "scale": list(rotation_pivot.scale),
    "parent": rotation_pivot.parent.name if rotation_pivot.parent else None,
    "children": [child.name for child in rotation_pivot.children],
}

# Animation data
if rotation_pivot.animation_data and rotation_pivot.animation_data.action:
    action = rotation_pivot.animation_data.action
    info["animation"] = {
        "action_name": action.name,
        "fcurves": [],
    }

    for fcurve in action.fcurves:
        fcurve_info = {
            "data_path": fcurve.data_path,
            "array_index": fcurve.array_index,
            "keyframes": [],
        }

        for kf in fcurve.keyframe_points:
            fcurve_info["keyframes"].append({
                "frame": int(kf.co[0]),
                "value": kf.co[1],
                "interpolation": kf.interpolation,
                "easing": kf.easing,
            })

        info["animation"]["fcurves"].append(fcurve_info)

print(f"\nObject: {info['name']}")
print(f"Type: {info['type']}")
print(f"Location: {info['location']}")
print(f"Rotation Euler: {info['rotation_euler']}")
print(f"Scale: {info['scale']}")
print(f"Parent: {info['parent']}")
print(f"Children: {', '.join(info['children'])}")

if "animation" in info:
    print(f"\nAnimation: {info['animation']['action_name']}")
    for fcurve in info["animation"]["fcurves"]:
        print(f"\n  {fcurve['data_path']}[{fcurve['array_index']}]:")
        for kf in fcurve["keyframes"]:
            print(f"    Frame {kf['frame']}: {kf['value']:.6f} ({kf['interpolation']})")

# Save to JSON
with open("rotationpivot_settings.json", "w") as f:
    json.dump(info, f, indent=2)

print("\n\nSaved to: rotationpivot_settings.json")
