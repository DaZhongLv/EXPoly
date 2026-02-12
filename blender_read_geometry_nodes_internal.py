"""
Blender Internal Script: Read Geometry Nodes Settings

Run this inside Blender:
1. Open grain_111_render_manual.blend
2. Open Scripting workspace
3. Paste this script and run it
4. Check the output in the console
"""

import json
from pathlib import Path

import bpy

scene = bpy.context.scene

# Find objects with Geometry Nodes modifiers
geometry_nodes_info = {}

for obj in scene.objects:
    if obj.type != 'MESH':
        continue

    for mod in obj.modifiers:
        if mod.type == 'NODES' and mod.node_group:
            node_group = mod.node_group

            # Get all nodes and their settings
            nodes_data = []
            for node in node_group.nodes:
                node_data = {
                    "name": node.name,
                    "type": node.type,
                    "location": list(node.location),
                    "inputs": {},
                }

                # Get input values
                for input_socket in node.inputs:
                    if not input_socket.is_linked:
                        value = input_socket.default_value
                        # Convert to serializable format
                        if isinstance(value, (list, tuple)):
                            if len(value) <= 4:
                                node_data["inputs"][input_socket.name] = list(value)
                            else:
                                node_data["inputs"][input_socket.name] = f"<array of {len(value)} elements>"
                        elif isinstance(value, (int, float, str, bool)):
                            node_data["inputs"][input_socket.name] = value
                        else:
                            node_data["inputs"][input_socket.name] = str(value)

                nodes_data.append(node_data)

            # Get links between nodes
            links_data = []
            for link in node_group.links:
                links_data.append({
                    "from_node": link.from_node.name,
                    "from_socket": link.from_socket.name,
                    "to_node": link.to_node.name,
                    "to_socket": link.to_socket.name,
                })

            geometry_nodes_info[f"{obj.name}_{mod.name}"] = {
                "object_name": obj.name,
                "modifier_name": mod.name,
                "node_group_name": node_group.name,
                "nodes": nodes_data,
                "links": links_data,
            }

# Print information
print("\n" + "=" * 80)
print("GEOMETRY NODES INFORMATION")
print("=" * 80)

for key, info in geometry_nodes_info.items():
    print(f"\n📦 Object: {info['object_name']}")
    print(f"   Modifier: {info['modifier_name']}")
    print(f"   Node Group: {info['node_group_name']}")
    print(f"   Nodes: {len(info['nodes'])}")

    for node in info['nodes']:
        print(f"\n     • {node['name']} ({node['type']})")
        if node['inputs']:
            for input_name, input_value in node['inputs'].items():
                print(f"       {input_name}: {input_value}")

    print(f"\n   Links ({len(info['links'])}):")
    for link in info['links']:
        print(f"     {link['from_node']}.{link['from_socket']} → {link['to_node']}.{link['to_socket']}")

# Save to JSON
output_file = Path.home() / "Desktop" / "expoly-with-legacy" / "EXPoly" / "geometry_nodes_settings.json"
try:
    with open(output_file, 'w') as f:
        json.dump(geometry_nodes_info, f, indent=2, default=str)
    print(f"\n\n💾 Saved to: {output_file}")
except Exception as e:
    print(f"\n\n⚠️  Could not save to file: {e}")
    print("Settings printed above - copy manually if needed")
