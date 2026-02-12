#!/usr/bin/env python3
"""
Read Geometry Nodes settings from a Blender file
"""

import json
import sys

import bpy

blend_file = sys.argv[-1] if len(sys.argv) > 1 else "grain_111_render_manual.blend"

print(f"Loading: {blend_file}")
try:
    bpy.ops.wm.open_mainfile(filepath=blend_file)
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(1)

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
                    "outputs": {},
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

                # Get output connections (just the names, not the actual connections)
                for output_socket in node.outputs:
                    node_data["outputs"][output_socket.name] = output_socket.type

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

            # Get interface (input/output sockets)
            interface_data = {
                "inputs": [],
                "outputs": [],
            }
            for socket in node_group.interface.items_tree:
                socket_data = {
                    "name": socket.name,
                    "type": socket.socket_type,
                    "default_value": socket.default_value if hasattr(socket, 'default_value') else None,
                }
                if socket.in_out == 'INPUT':
                    interface_data["inputs"].append(socket_data)
                else:
                    interface_data["outputs"].append(socket_data)

            geometry_nodes_info[f"{obj.name}_{mod.name}"] = {
                "object_name": obj.name,
                "modifier_name": mod.name,
                "node_group_name": node_group.name,
                "nodes": nodes_data,
                "links": links_data,
                "interface": interface_data,
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
    print(f"   Links: {len(info['links'])}")

    print("\n   Nodes:")
    for node in info['nodes']:
        print(f"     • {node['name']} ({node['type']})")
        if node['inputs']:
            for input_name, input_value in node['inputs'].items():
                print(f"       Input '{input_name}': {input_value}")

    print("\n   Links:")
    for link in info['links']:
        print(f"     {link['from_node']}.{link['from_socket']} → {link['to_node']}.{link['to_socket']}")

# Save to JSON
output_file = "geometry_nodes_settings.json"
with open(output_file, 'w') as f:
    json.dump(geometry_nodes_info, f, indent=2, default=str)

print(f"\n\n💾 Saved to: {output_file}")
