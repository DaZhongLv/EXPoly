#!/usr/bin/env python3
"""
Blender Script: Setup Glass Materials with Wireframe Edges for Voxel Visualization

This script creates glass materials with wireframe edge enhancement for:
- GrainPoints: Light gray glass with dark gray edges
- MarginPoints: Light blue glass with dark blue-gray edges

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender -P blender_setup_glass_materials.py
    Or in Blender: Scripting workspace → Run Script
"""

import bpy

# ==================== Configuration ====================

# Glass material parameters
GRAIN_GLASS_COLOR = (0.557, 0.557, 0.576, 1.0)  # Light gray (#8E8E93)
MARGIN_GLASS_COLOR = (0.655, 0.780, 0.906, 1.0)  # Light blue (#A7C7E7)

# Edge colors
GRAIN_EDGE_COLOR = (0.227, 0.227, 0.235, 1.0)  # Dark gray (#3A3A3C)
MARGIN_EDGE_COLOR = (0.231, 0.416, 0.549, 1.0)  # Dark blue-gray (#3B6A8C)

# Transparency
GRAIN_ALPHA = 0.38
MARGIN_ALPHA = 0.14

# Glass properties
GLASS_IOR = 1.45
GLASS_ROUGHNESS = 0.1

# Wireframe edge parameters
WIRE_SIZE = 0.03  # Adjustable: 0.02~0.05
EDGE_STRENGTH = 1.0  # Edge color intensity

# Volume absorption (for Cycles, optional)
USE_VOLUME_ABSORPTION = True
ABSORPTION_DENSITY = 0.1  # Adjustable

# ==================== Helper Functions ====================

def setup_eevee_settings():
    """Configure Eevee renderer for glass materials."""
    scene = bpy.context.scene
    if scene.render.engine == 'BLENDER_EEVEE':
        eevee = scene.eevee
        # Enable screen space reflections
        eevee.use_ssr = True
        eevee.use_ssr_refraction = True
        eevee.ssr_quality = 'HIGH'
        eevee.ssr_max_roughness = 1.0
        # Enable screen space refractions
        eevee.use_ssr_refraction = True
        print("Eevee settings configured for glass materials")


def create_glass_material_with_edges(
    name: str,
    glass_color: tuple,
    edge_color: tuple,
    alpha: float,
    wire_size: float,
    edge_strength: float = 1.0,
) -> bpy.types.Material:
    """
    Create a glass material with wireframe edge enhancement.

    Node setup:
    - Principled BSDF (glass: Transmission=1.0, IOR, Roughness)
    - Wireframe node for edge detection
    - ColorRamp to control edge width
    - Mix node to blend edge color with base color
    - Optional: Volume Absorption (for Cycles)
    """
    # Get or create material
    if name in bpy.data.materials:
        mat = bpy.data.materials[name]
        mat.use_nodes = True
        mat.node_tree.nodes.clear()
    else:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # ========== Wireframe Edge Setup (First) ==========
    # Set up edge detection first, then mix with glass color

    # Geometry node (for wireframe)
    geometry = nodes.new(type='ShaderNodeNewGeometry')
    geometry.location = (-600, -200)

    # Wireframe node
    wireframe = nodes.new(type='ShaderNodeWireframe')
    wireframe.location = (-400, -200)
    wireframe.inputs['Size'].default_value = wire_size

    # ColorRamp to control edge width and intensity
    colorramp = nodes.new(type='ShaderNodeValToRGB')
    colorramp.location = (-200, -200)
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (0, 0, 0, 1)  # No edge (glass color)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (1, 1, 1, 1)  # Full edge (edge color)

    # Connect wireframe to colorramp
    links.new(wireframe.outputs['Fac'], colorramp.inputs['Fac'])

    # Edge color
    edge_color_node = nodes.new(type='ShaderNodeRGB')
    edge_color_node.location = (-200, -400)
    edge_color_node.outputs[0].default_value = edge_color

    # Glass base color
    glass_color_node = nodes.new(type='ShaderNodeRGB')
    glass_color_node.location = (-200, -600)
    glass_color_node.outputs[0].default_value = glass_color

    # Mix edge color with glass color based on wireframe
    mix_color = nodes.new(type='ShaderNodeMixRGB')
    mix_color.location = (0, -200)
    mix_color.blend_type = 'MIX'
    mix_color.inputs['Color1'].default_value = glass_color
    mix_color.inputs['Color2'].default_value = edge_color

    # Connect colorramp to mix factor (edge areas = 1.0, non-edge = 0.0)
    links.new(colorramp.outputs['Color'], mix_color.inputs['Fac'])

    # ========== Glass BSDF Setup ==========

    # Glass BSDF (dedicated glass shader - produces true glass effect)
    glass_bsdf = nodes.new(type='ShaderNodeBsdfGlass')
    glass_bsdf.location = (200, 0)

    # Apply mixed color (glass color + edge color) to glass
    links.new(mix_color.outputs['Color'], glass_bsdf.inputs['Color'])
    glass_bsdf.inputs['IOR'].default_value = GLASS_IOR
    glass_bsdf.inputs['Roughness'].default_value = GLASS_ROUGHNESS

    # Mix with Transparent BSDF for alpha control
    transparent_bsdf = nodes.new(type='ShaderNodeBsdfTransparent')
    transparent_bsdf.location = (200, -200)
    transparent_bsdf.inputs['Color'].default_value = (1, 1, 1, 1)

    # Mix shader for alpha blending
    mix_shader = nodes.new(type='ShaderNodeMixShader')
    mix_shader.location = (400, 0)
    mix_shader.inputs['Fac'].default_value = alpha

    # Connect: Glass and Transparent to Mix
    links.new(glass_bsdf.outputs['BSDF'], mix_shader.inputs[1])
    links.new(transparent_bsdf.outputs['BSDF'], mix_shader.inputs[2])

    # ========== Volume Absorption (Cycles only, optional) ==========

    volume_absorption = None
    if USE_VOLUME_ABSORPTION and bpy.context.scene.render.engine == 'CYCLES':
        volume_absorption = nodes.new(type='ShaderNodeVolumeAbsorption')
        volume_absorption.location = (200, -400)
        volume_absorption.inputs['Density'].default_value = ABSORPTION_DENSITY
        volume_absorption.inputs['Color'].default_value = glass_color

    # ========== Material Output ==========

    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (400, 0)

    # Connect final shader (glass + transparent mix) to output
    links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])

    # Connect volume absorption if enabled
    if volume_absorption:
        links.new(volume_absorption.outputs['Volume'], output.inputs['Volume'])

    # ========== Material Settings ==========

    mat.blend_method = 'HASHED'  # Alpha Hashed for better transparency
    mat.shadow_method = 'NONE'  # No shadows for transparent objects

    # Enable screen space refraction for Eevee
    if bpy.context.scene.render.engine == 'BLENDER_EEVEE':
        mat.use_screen_refraction = True

    print(f"Created glass material: {name}")
    return mat


def apply_material_to_geometry_nodes(obj, material):
    """
    Ensure material is applied to Geometry Nodes output.
    This adds Set Material node after Realize Instances.
    """
    if not obj.modifiers:
        print(f"Warning: {obj.name} has no modifiers")
        return

    # Find Geometry Nodes modifier
    gn_modifier = None
    for mod in obj.modifiers:
        if mod.type == 'NODES' and mod.node_group:
            gn_modifier = mod
            break

    if not gn_modifier:
        print(f"Warning: {obj.name} has no Geometry Nodes modifier")
        return

    node_group = gn_modifier.node_group
    nodes = node_group.nodes
    links = node_group.links

    # Find Realize Instances node
    realize_node = None
    for node in nodes:
        if node.type == 'GEOMETRY_NODE_REALIZE_INSTANCES':
            realize_node = node
            break

    if not realize_node:
        print(f"Warning: No Realize Instances node found in {obj.name}")
        return

    # Find or create Set Material node
    set_material_node = None
    for node in nodes:
        if node.type == 'GEOMETRY_NODE_SET_MATERIAL':
            set_material_node = node
            break

    if not set_material_node:
        # Create Set Material node
        set_material_node = nodes.new(type='GeometryNodeSetMaterial')
        set_material_node.location = (realize_node.location.x + 200, realize_node.location.y)
        set_material_node.inputs['Material'].default_value = material

    # Connect Realize Instances to Set Material
    if 'Geometry' in realize_node.outputs and 'Geometry' in set_material_node.inputs:
        # Check if already connected
        if not set_material_node.inputs['Geometry'].is_linked:
            links.new(realize_node.outputs['Geometry'], set_material_node.inputs['Geometry'])

    # Find Group Output and connect Set Material to it
    group_output = None
    for node in nodes:
        if node.type == 'NODE_GROUP_OUTPUT':
            group_output = node
            break

    if group_output:
        # Disconnect old connection from Realize Instances to Output
        if realize_node.outputs['Geometry'].is_linked:
            for link in realize_node.outputs['Geometry'].links:
                if link.to_node == group_output:
                    links.remove(link)

        # Connect Set Material to Output
        if 'Geometry' in set_material_node.outputs and 'Geometry' in group_output.inputs:
            if not group_output.inputs[0].is_linked:
                links.new(set_material_node.outputs['Geometry'], group_output.inputs[0])

    # Set material in Set Material node
    set_material_node.inputs['Material'].default_value = material

    print(f"Applied material {material.name} to {obj.name} Geometry Nodes")


# ==================== Main Script ====================

def check_material_exists(name: str) -> bool:
    """Check if material exists and has been manually modified."""
    if name not in bpy.data.materials:
        return False
    mat = bpy.data.materials[name]
    # Check if material has custom properties or is linked to objects
    # (indicating it might have been manually modified)
    if mat.users > 0:  # Material is in use
        return True
    return False


def main():
    print("=" * 60)
    print("Setting up glass materials with wireframe edges")
    print("=" * 60)

    # Check if materials already exist
    grain_exists = check_material_exists("MAT_Grain")
    margin_exists = check_material_exists("MAT_Margin")

    if grain_exists or margin_exists:
        print("\n⚠️  Warning: Materials already exist!")
        print("  - MAT_Grain exists: ", grain_exists)
        print("  - MAT_Margin exists: ", margin_exists)
        print("\nOptions:")
        print("  1. Script will recreate materials (overwrites manual changes)")
        print("  2. Save your manual edits to a different file first")
        print("  3. Rename existing materials to preserve them")
        print("\nTo preserve your manual edits:")
        print("  - Save as: grain_111_render_manual.blend")
        print("  - Or rename materials: MAT_Grain_Manual, MAT_Margin_Manual")
        print("\nContinuing with material creation...\n")

    # Configure renderer settings
    scene = bpy.context.scene
    if scene.render.engine == 'BLENDER_EEVEE':
        setup_eevee_settings()

    # Create materials
    print("Creating glass materials...")
    grain_mat = create_glass_material_with_edges(
        name="MAT_Grain",
        glass_color=GRAIN_GLASS_COLOR,
        edge_color=GRAIN_EDGE_COLOR,
        alpha=GRAIN_ALPHA,
        wire_size=WIRE_SIZE,
        edge_strength=EDGE_STRENGTH,
    )

    margin_mat = create_glass_material_with_edges(
        name="MAT_Margin",
        glass_color=MARGIN_GLASS_COLOR,
        edge_color=MARGIN_EDGE_COLOR,
        alpha=MARGIN_ALPHA,
        wire_size=WIRE_SIZE,
        edge_strength=EDGE_STRENGTH,
    )

    # Find objects and apply materials
    print("\nApplying materials to objects...")
    grain_obj = bpy.data.objects.get("GrainPoints")
    margin_obj = bpy.data.objects.get("MarginPoints")

    if grain_obj:
        apply_material_to_geometry_nodes(grain_obj, grain_mat)
        # Also set material directly (fallback)
        grain_obj.data.materials.clear()
        grain_obj.data.materials.append(grain_mat)
        print(f"Applied {grain_mat.name} to GrainPoints")
    else:
        print("Warning: GrainPoints object not found")

    if margin_obj:
        apply_material_to_geometry_nodes(margin_obj, margin_mat)
        # Also set material directly (fallback)
        margin_obj.data.materials.clear()
        margin_obj.data.materials.append(margin_mat)
        print(f"Applied {margin_mat.name} to MarginPoints")
    else:
        print("Warning: MarginPoints object not found")

    print("\n" + "=" * 60)
    print("Material setup complete!")
    print("=" * 60)
    print("\nMaterials created:")
    print(f"  - {grain_mat.name}: Light gray glass with dark gray edges")
    print(f"  - {margin_mat.name}: Light blue glass with dark blue-gray edges")
    print("\nIMPORTANT: Glass effect is only visible in Rendered view!")
    print("  - Switch to 'Rendered' viewport shading (top-right viewport menu)")
    print("  - Or render an image to see the glass effect")
    print("\nYou can adjust parameters in the script:")
    print(f"  - WIRE_SIZE: {WIRE_SIZE} (0.02~0.05)")
    print(f"  - EDGE_STRENGTH: {EDGE_STRENGTH}")
    print(f"  - GLASS_ROUGHNESS: {GLASS_ROUGHNESS} (lower = clearer glass)")
    print(f"  - GRAIN_ALPHA: {GRAIN_ALPHA}")
    print(f"  - MARGIN_ALPHA: {MARGIN_ALPHA}")
    print(f"  - GLASS_IOR: {GLASS_IOR} (1.0 = no refraction, 1.5 = typical glass)")


if __name__ == "__main__":
    main()
