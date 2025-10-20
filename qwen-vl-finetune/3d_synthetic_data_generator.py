import os
from datetime import datetime

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath


class EngineeringDrawing:
    def __init__(self, drawing_info=None):
        """Initialize with drawing information."""
        self.drawing_info = drawing_info or self.get_default_drawing_info()

    def get_default_drawing_info(self):
        """Default drawing information template."""
        return {
            "part_name": "PART NAME",
            "part_number": "PN-000-000",
            "material": "MATERIAL TBD",
            "scale": "1:1",
            "projection_method": "THIRD ANGLE",
            "drawing_number": "DWG-001",
            "revision": "A",
            "drawn_by": "ENGINEER",
            "checked_by": "CHECKER",
            "approved_by": "APPROVER",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "company": "COMPANY NAME",
            "title": "ENGINEERING DRAWING",
            "units": "mm",
            "surface_finish": "3.2 μm",
            "tolerances": {"linear": "±0.1", "angular": "±0.5°"},
        }


def load_and_center_mesh(file_path):
    """Load STL file and center its geometry."""
    mesh = trimesh.load_mesh(file_path)
    mesh.apply_translation(-mesh.center_mass)
    return mesh


def get_cross_section(mesh, origin, normal):
    """Get 2D cross-section of the mesh."""
    section = mesh.section(plane_origin=origin, plane_normal=normal)
    if section is None:
        return None
    return section.to_2D()[0]


def get_mesh_dimensions(mesh):
    """Get overall dimensions of the mesh."""
    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]  # max - min
    return {
        "length": dimensions[0],
        "width": dimensions[1],
        "height": dimensions[2],
        "bounds": bounds,
    }


def add_dimension_annotations(ax, path2D, view_name):
    """Add horizontal and vertical dimension lines with extension lines (dotted)."""
    bounds = path2D.bounds.reshape(2, 2)
    min_pt, max_pt = bounds[0], bounds[1]
    width = max_pt[0] - min_pt[0]
    height = max_pt[1] - min_pt[1]

    offset = max(width, height) * 0.15
    arrowprops = dict(arrowstyle="<->", color="blue", linewidth=1)

    # --- Horizontal dimension line and extension lines ---
    y_dim = min_pt[1] - offset

    # Extension lines (vertical dashed lines from corners)
    ax.plot(
        [min_pt[0], min_pt[0]],
        [min_pt[1], y_dim],
        linestyle="--",
        color="gray",
        linewidth=0.8,
    )
    ax.plot(
        [max_pt[0], max_pt[0]],
        [max_pt[1], y_dim],
        linestyle="--",
        color="gray",
        linewidth=0.8,
    )

    # Dimension arrow
    ax.annotate(
        "", xy=(min_pt[0], y_dim), xytext=(max_pt[0], y_dim), arrowprops=arrowprops
    )
    ax.text(
        (min_pt[0] + max_pt[0]) / 2,
        y_dim - offset * 0.1,
        f"{width:.2f} mm",
        ha="center",
        va="top",
        fontsize=8,
        color="blue",
    )

    # --- Vertical dimension line and extension lines ---
    x_dim = max_pt[0] + offset

    # Extension lines (horizontal dashed lines from top and bottom)
    ax.plot(
        [x_dim, max_pt[0]],
        [max_pt[1], max_pt[1]],
        linestyle="--",
        color="gray",
        linewidth=0.8,
    )
    ax.plot(
        [x_dim, min_pt[0]],
        [min_pt[1], min_pt[1]],
        linestyle="--",
        color="gray",
        linewidth=0.8,
    )

    # Dimension arrow
    ax.annotate(
        "", xy=(x_dim, min_pt[1]), xytext=(x_dim, max_pt[1]), arrowprops=arrowprops
    )
    ax.text(
        x_dim + offset * 0.1,
        (min_pt[1] + max_pt[1]) / 2,
        f"{height:.2f} mm",
        ha="left",
        va="center",
        fontsize=8,
        color="blue",
        rotation=90,
    )


def is_path_closed(path, tol=1e-2):
    """Check if a path is closed (first and last point match within tolerance)."""
    return np.allclose(path[0], path[-1], atol=tol)


def plot_cross_section(ax, path2D, title, show_dimensions=True):
    """Plot the 2D cross-section with dimension lines and no axes border."""
    if path2D is None:
        ax.set_title(f"{title} (No section)", fontsize=10, weight="bold")
        ax.axis("off")
        return

    for path in path2D.discrete:
        path = np.asarray(path)

        # Force close path if not closed
        if not is_path_closed(path):
            path = np.vstack([path, path[0]])  # append start point to close loop

        # Create closed polygon path manually
        codes = (
            [MplPath.MOVETO] + [MplPath.LINETO] * (len(path) - 2) + [MplPath.CLOSEPOLY]
        )
        poly_path = MplPath(path, codes)
        patch = PathPatch(poly_path, facecolor="none", edgecolor="black", lw=1.5)
        ax.add_patch(patch)

    # Add dimensions
    if show_dimensions:
        add_dimension_annotations(ax, path2D, title)

    # Set consistent plot limits
    bounds = path2D.bounds.reshape(2, 2)
    min_pt, max_pt = bounds[0], bounds[1]
    padding = 0.2 * max(max_pt[0] - min_pt[0], max_pt[1] - min_pt[1])
    ax.set_xlim(min_pt[0] - padding, max_pt[0] + padding)
    ax.set_ylim(min_pt[1] - padding, max_pt[1] + padding)

    ax.set_aspect("equal")
    ax.axis("off")


def create_title_block(fig, drawing_info):
    """Create a professional title block at the bottom right of the drawing."""
    # Title block dimensions (as fraction of figure)
    tb_width = 0.35
    tb_height = 0.25
    tb_x = 1 - tb_width - 0.05
    tb_y = 0.02

    # Create title block background
    title_block_ax = fig.add_axes([tb_x, tb_y, tb_width, tb_height])
    title_block_ax.set_xlim(0, 10)
    title_block_ax.set_ylim(0, 10)
    title_block_ax.axis("off")

    # Draw title block border
    border = Rectangle((0, 0), 10, 10, linewidth=2, edgecolor="black", facecolor="none")
    title_block_ax.add_patch(border)

    # Add horizontal dividers
    for y in [2, 4, 6, 8]:
        title_block_ax.plot([0, 10], [y, y], "k-", linewidth=1)

    # Add vertical dividers
    for x in [3, 6]:
        title_block_ax.plot([x, x], [0, 6], "k-", linewidth=1)

    # Add text fields
    fields = [
        # (text, x, y, fontsize, weight)
        (drawing_info["company"], 5, 9, 12, "bold"),
        (drawing_info["title"], 5, 8.3, 10, "bold"),
        ("PART NAME:", 0.2, 7.3, 8, "bold"),
        (drawing_info["part_name"], 3.2, 7.3, 8, "normal"),
        ("PART NUMBER:", 6.2, 7.3, 8, "bold"),
        (drawing_info["part_number"], 6.2, 6.7, 8, "normal"),
        ("MATERIAL:", 0.2, 5.3, 8, "bold"),
        (drawing_info["material"], 3.2, 5.3, 8, "normal"),
        ("SCALE:", 6.2, 5.3, 8, "bold"),
        (drawing_info["scale"], 6.2, 4.7, 8, "normal"),
        ("PROJECTION:", 0.2, 3.3, 8, "bold"),
        (drawing_info["projection_method"], 3.2, 3.3, 8, "normal"),
        ("UNITS:", 6.2, 3.3, 8, "bold"),
        (drawing_info["units"], 6.2, 2.7, 8, "normal"),
        ("DRAWN BY:", 0.2, 1.3, 7, "bold"),
        (drawing_info["drawn_by"], 0.2, 0.7, 7, "normal"),
        ("DATE:", 3.2, 1.3, 7, "bold"),
        (drawing_info["date"], 3.2, 0.7, 7, "normal"),
        ("DWG NO:", 6.2, 1.3, 7, "bold"),
        (drawing_info["drawing_number"], 6.2, 0.7, 7, "normal"),
    ]

    for text, x, y, fontsize, weight in fields:
        title_block_ax.text(
            x,
            y,
            text,
            fontsize=fontsize,
            weight=weight,
            ha="left" if x < 5 else "left",
            va="center",
        )


def create_notes_section(fig, drawing_info):
    """Add general notes section to the drawing."""
    notes_ax = fig.add_axes([0.05, 0.02, 0.4, 0.25])
    notes_ax.set_xlim(0, 10)
    notes_ax.set_ylim(0, 10)
    notes_ax.axis("off")

    # Notes border
    border = Rectangle(
        (0, 0), 10, 10, linewidth=1.5, edgecolor="black", facecolor="none"
    )
    notes_ax.add_patch(border)

    # Notes title
    notes_ax.text(5, 9, "GENERAL NOTES", fontsize=10, weight="bold", ha="center")
    notes_ax.plot([0, 10], [8.5, 8.5], "k-", linewidth=1)

    # Add notes
    notes = [
        "1. ALL DIMENSIONS IN " + drawing_info["units"].upper(),
        "2. UNLESS OTHERWISE SPECIFIED:",
        f"   LINEAR TOL: {drawing_info['tolerances']['linear']}",
        f"   ANGULAR TOL: {drawing_info['tolerances']['angular']}",
        f"3. SURFACE FINISH: {drawing_info['surface_finish']}",
        "4. REMOVE ALL BURRS AND SHARP EDGES",
        "5. MATERIAL: " + drawing_info["material"],
    ]

    for i, note in enumerate(notes):
        notes_ax.text(0.2, 7.5 - i * 0.7, note, fontsize=8, ha="left", va="center")


def create_engineering_drawing(
    mesh,
    views,
    drawing_info=None,
    output_path="engineering_drawing.png",
    show_dimensions=True,
):
    """Create a complete engineering drawing with title block and notes."""

    # Initialize drawing info
    eng_drawing = EngineeringDrawing(drawing_info)

    # Get mesh dimensions for reference
    mesh_dims = get_mesh_dimensions(mesh)

    # Create figure with appropriate size for engineering drawing
    fig = plt.figure(figsize=(16, 11))  # A3 landscape proportions
    fig.patch.set_facecolor("white")

    # Calculate layout for views
    n_views = len(views)
    if n_views == 1:
        view_positions = [(0.15, 0.4, 0.3, 0.4)]
    elif n_views == 2:
        view_positions = [(0.1, 0.4, 0.25, 0.4), (0.4, 0.4, 0.25, 0.4)]
    else:  # 3 views
        view_positions = [
            (0.05, 0.4, 0.25, 0.4),
            (0.35, 0.4, 0.25, 0.4),
            (0.65, 0.4, 0.25, 0.4),
        ]

    # Create subplots for each view
    for i, ((title, plane), pos) in enumerate(zip(views.items(), view_positions)):
        ax = fig.add_axes(pos)
        path2D = get_cross_section(mesh, plane["origin"], plane["normal"])
        plot_cross_section(ax, path2D, title, show_dimensions=show_dimensions)

    # Add title block
    create_title_block(fig, eng_drawing.drawing_info)

    # Add notes section
    create_notes_section(fig, eng_drawing.drawing_info)

    # Add drawing border
    border_ax = fig.add_axes([0, 0, 1, 1])
    border_ax.set_xlim(0, 1)
    border_ax.set_ylim(0, 1)
    border_ax.axis("off")
    border = Rectangle(
        (0.02, 0.02), 0.96, 0.96, linewidth=3, edgecolor="black", facecolor="none"
    )
    border_ax.add_patch(border)

    # Save the drawing
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.show()

    return output_path


if __name__ == "__main__":
    STL_FILE = "ball-bearing.stl"
    OUTPUT_FILE = "engineering_drawing.png"

    # Define your drawing information
    drawing_info = {
        "part_name": "BALL BEARING",
        "part_number": "BB-6000-2RS",
        "material": "CHROME STEEL",
        "scale": "2:1",
        "projection_method": "THIRD ANGLE",
        "drawing_number": "DWG-BB-001",
        "revision": "A",
        "drawn_by": "J. SMITH",
        "checked_by": "M. JOHNSON",
        "approved_by": "R. WILSON",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "company": "PRECISION BEARINGS INC.",
        "title": "BALL BEARING ASSEMBLY",
        "units": "mm",
        "surface_finish": "1.6 μm",
        "tolerances": {"linear": "±0.05", "angular": "±0.25°"},
    }

    # Define slicing views (standard orthographic projections)
    slicing_planes = {
        "FRONT VIEW": {"origin": [0, 0, 0], "normal": [0, 1, 0]},
        "TOP VIEW": {"origin": [0, 0, 0], "normal": [0, 0, 1]},
        "RIGHT VIEW": {"origin": [0, 0, 0], "normal": [1, 0, 0]},
    }

    # Load mesh and create engineering drawing
    try:
        mesh = load_and_center_mesh(STL_FILE)
        result = create_engineering_drawing(
            mesh, slicing_planes, drawing_info, OUTPUT_FILE, show_dimensions=True
        )
        print(f"✅ Engineering drawing saved to: {result}")

        # Print mesh information
        dims = get_mesh_dimensions(mesh)
        print(f"\n📐 Part Dimensions:")
        print(f"   Length: {dims['length']:.2f} mm")
        print(f"   Width:  {dims['width']:.2f} mm")
        print(f"   Height: {dims['height']:.2f} mm")

    except FileNotFoundError:
        print(f"❌ STL file '{STL_FILE}' not found. Please check the file path.")
    except Exception as e:
        print(f"❌ Error: {e}")
