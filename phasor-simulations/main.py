"""
Python framework for generating publication-quality phasor diagrams.

This module contains functions to generate various types of phasor diagrams
for publication, including single phasors, phasor summations, amplitude and
phase modulation, and optical cavity field patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Arc
import json
import os


def load_style():
    """Load style configuration from style.json"""
    style_path = os.path.join(os.path.dirname(__file__), 'style.json')
    if os.path.exists(style_path):
        with open(style_path, 'r') as f:
            return json.load(f)
    return {}


def add_text(ax, position, text, fontsize=20, color='black'):
    """
    Add text to the plot using serif font.
    
    Args:
        ax: Matplotlib axes object
        position: Tuple (x, y) for text position
        text: Text string (supports LaTeX)
        fontsize: Font size
        color: Text color
    """
    ax.text(position[0], position[1], text, fontsize=fontsize, 
            color=color, ha='center', va='center',
            fontfamily='serif', zorder=4)


def setup_plot(xlim=None, ylim=None, aspect='equal', axis_linewidth=4.0):
    """
    Set up a clean, minimal plot for phasor diagrams.
    
    Args:
        xlim: Tuple of (xmin, xmax) for x-axis limits
        ylim: Tuple of (ymin, ymax) for y-axis limits
        aspect: Aspect ratio of the plot
    
    Returns:
        fig, ax: Figure and axes objects
    """
    style = load_style()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect(aspect)
    
    # Remove all ticks, labels, and grid
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw axes lines at the origin
    ax.axhline(0, color=style.get('axis_color', "#949494"), linewidth=axis_linewidth, zorder=0)
    ax.axvline(0, color=style.get('axis_color', "#949494"), linewidth=axis_linewidth, zorder=0)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    # Add "Re" and "Im" labels on the axes
    if xlim:
        ax.text(xlim[1]*0.95, 0.05*(ylim[1]-ylim[0])+ylim[0], 'Re', 
                fontsize=30, color=style.get('axis_label_color', 'black'),
                ha='right', va='bottom', fontfamily='serif', zorder=4)
    if ylim:
        ax.text(0.03*(xlim[1]-xlim[0])+xlim[0], ylim[1]*0.95, 'Im', 
                fontsize=30, color=style.get('axis_label_color', 'black'),
                ha='left', va='top', fontfamily='serif', zorder=4)

    return fig, ax

def draw_square_brace(ax, center, length, angle, color='black', linewidth=2.0):
    """
    Draw a square brace (like a '[') at a specified position, length, and angle.
    
    Args:
        ax: Matplotlib axes object
        center: Tuple (x, y) for the center of the brace
        length: Total length of the brace
        angle: Angle in degrees for the orientation of the brace
        color: Brace color
        linewidth: Line width
    """
    # Calculate the main line of the brace
    angle_rad = np.radians(angle)
    dx = (length / 2) * np.cos(angle_rad)
    dy = (length / 2) * np.sin(angle_rad)
    
    start = (center[0] - dx, center[1] - dy)
    end = (center[0] + dx, center[1] + dy)
    
    # Draw the main line of the brace
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=linewidth, zorder=2)
    
    # Draw the orthogonal lines at both ends to mimic the square brace shape
    orthogonal_angle_rad = angle_rad + np.pi / 2
    orthogonal_length = length * 0.1  # Adjust this for the size of the orthogonal lines
    
    orthogonal_dx = orthogonal_length * np.cos(orthogonal_angle_rad)
    orthogonal_dy = orthogonal_length * np.sin(orthogonal_angle_rad)
    
    # Start end orthogonal line
    ax.plot([start[0], start[0] + orthogonal_dx], [start[1], start[1] + orthogonal_dy], 
            color=color, linewidth=linewidth, zorder=2)
    
    # End end orthogonal line
    ax.plot([end[0], end[0] + orthogonal_dx], [end[1], end[1] + orthogonal_dy], 
            color=color, linewidth=linewidth, zorder=2)

def draw_phasor(ax, start, end, color='#2E86AB', linewidth=2.5, label=None):
    """
    Draw a phasor arrow from start to end.
    
    Args:
        ax: Matplotlib axes object
        start: Tuple (x, y) for arrow start position
        end: Tuple (x, y) for arrow end position
        color: Arrow color
        linewidth: Arrow line width
        label: Optional label for the arrow
    """
    arrow = FancyArrowPatch(
        start, end,  # Extend slightly beyond the end point for better arrowhead visibility
        arrowstyle='->,head_width=0.4,head_length=0.6',
        shrinkA=0,
        shrinkB=0,
        color=color,
        linewidth=linewidth,
        mutation_scale=20,
        zorder=3
    )
    ax.add_patch(arrow)
    return arrow


def draw_curved_arrow(ax, center, radius, start_angle, end_angle, color='black', 
                      linewidth=1.5, direction='ccw'):
    """
    Draw a curved arrow to indicate rotation direction.
    
    Args:
        ax: Matplotlib axes object
        center: Tuple (x, y) for arc center
        radius: Arc radius
        start_angle: Starting angle in degrees
        end_angle: Ending angle in degrees
        color: Arrow color
        linewidth: Line width
        direction: 'ccw' for counterclockwise or 'cw' for clockwise
    """
    arc = patches.FancyArrowPatch(
        posA=(center[0] + radius * np.cos(np.radians(start_angle)),
              center[1] + radius * np.sin(np.radians(start_angle))),
        posB=(center[0] + radius * np.cos(np.radians(end_angle)),
              center[1] + radius * np.sin(np.radians(end_angle))),
        arrowstyle='->,head_width=0.3,head_length=0.4',
        connectionstyle=f"arc3,rad={'0.3' if direction == 'ccw' else '-0.3'}",
        color=color,
        linewidth=linewidth,
        mutation_scale=15,
        zorder=2
    )
    ax.add_patch(arc)


def draw_double_headed_arrow(ax, center, direction, length, color='black', linewidth=1.5):
    """
    Draw a double-headed arrow for indicating variation.
    
    Args:
        ax: Matplotlib axes object
        center: Tuple (x, y) for arrow center
        direction: Tuple (dx, dy) for arrow direction (will be normalized)
        length: Total length of the arrow
        color: Arrow color
        linewidth: Line width
    """
    # Normalize direction
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)
    
    # Calculate start and end points
    start = center - direction * length / 2
    end = center + direction * length / 2
    
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='<->,head_width=0.3,head_length=0.4',
        color=color,
        linewidth=linewidth,
        mutation_scale=15,
        zorder=2
    )
    ax.add_patch(arrow)


def draw_circle(ax, center, radius, color='black', linestyle='--', linewidth=1.0):
    """
    Draw a circle.
    
    Args:
        ax: Matplotlib axes object
        center: Tuple (x, y) for circle center
        radius: Circle radius
        color: Circle color
        linestyle: Line style
        linewidth: Line width
    """
    circle = plt.Circle(center, radius, fill=False, color=color, 
                       linestyle=linestyle, linewidth=linewidth, zorder=1)
    ax.add_patch(circle)

def save_figure(fig, filename):
    """
    Save figure as both PNG and EPS.
    
    Args:
        fig: Figure object
        filename: Base filename (without extension)
    """
    output_dir = os.path.join(os.path.dirname(__file__), 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    png_path = os.path.join(output_dir, f'{filename}.png')
    eps_path = os.path.join(output_dir, f'{filename}.eps')
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(eps_path, format='eps', bbox_inches='tight')
    
    print(f"Saved: {png_path}")
    print(f"Saved: {eps_path}")


def illustration_1_single_phasor():
    """
    Illustration 1: A single phasor diagram with one phasor at 45 degrees.
    Shows rotation direction and angle label.
    """
    fig, ax = setup_plot(xlim=(-0.2, 1.3), ylim=(-0.2, 1.3))
    
    # Phasor at 45 degrees with magnitude 1
    angle = np.radians(45)
    magnitude = 1.2
    end_point = (magnitude * np.cos(angle), magnitude * np.sin(angle))
    
    # Draw the phasor
    draw_phasor(ax, (0, 0), end_point, color='#2E86AB', linewidth=5)
    
    # Draw curved arrow indicating counterclockwise rotation
    # Position the arc at radius 0.5 from origin
    draw_curved_arrow(ax, (0, 0), 0.4, 0, 45, color='#333333', linewidth=3, direction='ccw')
    draw_curved_arrow(ax, (0, 0), magnitude, 45, 80, color='#333333', linewidth=3, direction='ccw')
    # Add angle label
    label_pos = (0.55, 0.18)
    add_text(ax, label_pos, r'$\phi = \omega t$', fontsize=30)
    
    plt.tight_layout()
    save_figure(fig, 'illustration_1_single_phasor')
    plt.close()


def illustration_2_phasor_summation():
    """
    Illustration 2: Three diagrams showing phasor summation.
    a) Constructive interference (both at 45°)
    b) Destructive interference (first at 45°, second at 225°)
    c) Orthogonal phasors (first at 45°, second at 135°)
    """
    # Part a: Constructive interference
    fig, ax = setup_plot(xlim=(-0.2, 1.3), ylim=(-0.2, 1.3))
    
    # First phasor: magnitude 1, angle 45°
    angle1 = np.radians(45)
    end1 = (np.cos(angle1), np.sin(angle1))
    draw_phasor(ax, (0, 0), end1, color='#2E86AB', linewidth=5)
    
    # Second phasor: magnitude 0.5, angle 45°, starts from end of first
    angle2 = np.radians(45)
    start2 = end1
    end2 = (end1[0] + 0.5 * np.cos(angle2), end1[1] + 0.5 * np.sin(angle2))
    draw_phasor(ax, start2, end2, color='#A23B72', linewidth=5)
    
    # Resultant phasor
    draw_phasor(ax, (0.03, -0.03), (end2[0]+0.05, end2[1]-0.05), color="#000000", linewidth=3.0)
    
    # Add labels
    add_text(ax, (0.25,0.5), '$|A_1|=1.0$', fontsize=25, color='#2E86AB')
    add_text(ax, (0.65,0.9), '$|A_2|=0.5$', fontsize=25, color='#A23B72')
    add_text(ax, (0.75,0.4), '$|A_R|=1.5$', fontsize=25, color='#000000')

    plt.tight_layout()
    save_figure(fig, 'illustration_2a_constructive_interference')
    plt.close()
    
    # Part b: Destructive interference
    fig, ax = setup_plot(xlim=(-0.2, 1.3), ylim=(-0.2, 1.3))
    
    # First phasor: magnitude 1, angle 45°
    end1 = (1.5 * np.cos(np.radians(45)), 1.5 * np.sin(np.radians(45)))
    draw_phasor(ax, (0, 0), end1, color='#2E86AB', linewidth=5)
    
    # Second phasor: magnitude 0.5, angle 225°, starts from end of first
    # Offset amount to avoid overlap / for visual clarity
    offset = 0.03
    angle2 = np.radians(225)
    start2 = (end1[0] + offset, end1[1] - offset)
    end2 = (end1[0] + 0.75 * np.cos(angle2) + offset, end1[1] + 0.75 * np.sin(angle2) - offset)
    draw_phasor(ax, start2, end2, color='#A23B72', linewidth=5)
    
    # Resultant phasor
    draw_phasor(ax, (offset, -offset), (end2[0], end2[1]), color="#000000", linewidth=3.0)
    
    # Add labels
    add_text(ax, (0.4,0.65), '$|A_1|=1.0$', fontsize=25, color='#2E86AB')
    add_text(ax, (1.0,0.7), '$|A_2|=0.5$', fontsize=25, color='#A23B72')
    add_text(ax, (0.5,0.2), '$|A_R|=0.5$', fontsize=25, color='#000000')

    plt.tight_layout()
    save_figure(fig, 'illustration_2b_destructive_interference')
    plt.close()
    
    # Part c: Orthogonal phasors
    fig, ax = setup_plot(xlim=(-0.2, 1.3), ylim=(-0.2, 1.3))
    
    
    # First phasor: magnitude 1, angle 45°
    magnitude1 = 0.75
    end1 = (magnitude1 * np.cos(np.radians(20)), magnitude1 * np.sin(np.radians(20)))
    draw_phasor(ax, (0, 0), end1, color='#2E86AB', linewidth=5)
    
    # Second phasor: magnitude 1, angle 135°, starts from end of first
    magnitude2 = 0.9
    angle2 = np.radians(60)
    start2 = end1
    end2 = (end1[0] + magnitude2 * np.cos(angle2), end1[1] + magnitude2 * np.sin(angle2))
    draw_phasor(ax, start2, end2, color='#A23B72', linewidth=5)
    
    # Resultant phasor
    draw_phasor(ax, (0, 0), end2, color="#000000", linewidth=3.0)
    
    # Add a thin dashed line at the same angle as the first phasor for reference
    dashed_end = (1.3 * np.cos(np.radians(20)), 1.3 * np.sin(np.radians(20)))
    ax.plot([0, dashed_end[0]], [0, dashed_end[1]], linestyle='--', color='#666666', linewidth=1.5, zorder=1)

    # Draw angle arc between the two phasors centered at the end of the first phasor
    arc_radius = 0.2
    draw_curved_arrow(ax, end1, arc_radius, 20, 60, color='#333333', linewidth=2.0, direction='ccw')
    add_text(ax, (end1[0] + arc_radius * 1, end1[1] + arc_radius * 0.9), r'$\phi$', fontsize=25, color='#333333')

    # Add labels
    add_text(ax, (0.4,0.05), '$|A_1|$', fontsize=25, color='#2E86AB')
    add_text(ax, (1.1,0.7), '$|A_2|$', fontsize=25, color='#A23B72')
    add_text(ax, (0.35,0.85), '$|A_R|=\sqrt{A_1^2 + A_2^2 + 2A_1A_2\cos \\phi }$', fontsize=25, color='#000000')

    plt.tight_layout()
    save_figure(fig, 'illustration_2c_orthogonal_phasors')
    plt.close()


def illustration_3_amplitude_modulation():
    """
    Illustration 3: Amplitude modulation diagrams.
    Shows carrier with amplitude variation and modulation sidebands at different times.
    """
    # Part 1: Carrier with amplitude variation indicator
    fig, ax = setup_plot(xlim=(-0.2, 1.3), ylim=(-0.2, 1.3))
    
    # For visual purposes, draw a faint carrier behind the first one which extends to 1.0. Don't use transparency, but rather a lighter/more desaturated color.
    draw_phasor(ax, (0, 0), (1.25 * np.cos(np.radians(45)), 1.25 * np.sin(np.radians(45))), 
                color='#AED6F1', linewidth=5)

    # Carrier phasor at 45 degrees, magnitude 1
    angle = np.radians(45)
    magnitude = 1.0
    end_point = (magnitude * np.cos(angle), magnitude * np.sin(angle))
    draw_phasor(ax, (0, 0), end_point, color='#2E86AB', linewidth=5)
    

    # Double-headed arrow along the phasor direction, offset slightly
    # Center near the head of the carrier
    offset = 0.025
    center = (end_point[0] - 2*offset, end_point[1] + offset)
    direction = (np.cos(angle), np.sin(angle))
    draw_double_headed_arrow(ax, center, direction, length=0.55, color='#333333', linewidth=3.5)
    
    # Add label for the amplitude of the carrier and of the modulated signal
    add_text(ax, (0.5, 0.25), r'$A_0$', fontsize=24, color='#2E86AB')
    add_text(ax, (0.4, 0.9), r'$A(t) = A_0 \left( 1 + m \cos \omega_m t \right)$', fontsize=24, color='#333333')
    add_text(ax, (1.0, 0.65), r'$m \cdot A_0$', fontsize=24, color='#333333')

    # Add a square brace '[' to indicate the amplitude variation, angled at 45 degrees to run parallel with the modulation direction
    brace_center = (end_point[0] + 0.125, end_point[1] + 0.025)
    draw_square_brace(ax, brace_center, length=0.27, angle=45, color='#333333', linewidth=2.0)


    plt.tight_layout()
    save_figure(fig, 'illustration_3_carrier_amplitude_variation')
    plt.close()
    
    # Part 2: Modulation sidebands at different times
    times = [
        ('t=0', 0, [45, 45]),           # a) Both sidebands aligned with carrier
        ('t=T/4', np.pi/2, [135, 315]), # b) Sidebands perpendicular
        ('t=T/2', np.pi, [225, 225]),   # c) Sidebands opposite to carrier
        ('t=3T/4', 3*np.pi/2, [315, 135]) # d) Sidebands perpendicular again
    ]
    
    # Offsets for sidebands: [(upper_x, upper_y), (lower_x, lower_y)] for each time
    # These offsets are applied to the start position of each sideband
    offsets = [
        [(0, 0), (0, 0)],  # t=0
        [(0, 0), (0.02, 0.02)],  # t=T/4
        [(0.02, -0.02), (0, 0)],  # t=T/2
        [(0, 0), (0.02, 0.02)]   # t=3T/4
    ]
    
    for idx, (time_label, phase_offset, angles) in enumerate(times):
        fig, ax = setup_plot(xlim=(-0.2, 1.3), ylim=(-0.2, 1.3))
        
        # Get offsets for this time position
        upper_offset = offsets[idx][0]
        lower_offset = offsets[idx][1]
        
        # Carrier phasor at 45 degrees, magnitude 1
        carrier_angle = np.radians(45)
        carrier_end = (np.cos(carrier_angle), np.sin(carrier_angle))
        draw_phasor(ax, (0, 0), carrier_end, color='#2E86AB', linewidth=5)
        
        # Upper sideband: magnitude 0.2, starts from carrier end (with offset)
        upper_angle = np.radians(angles[0])
        upper_start = (carrier_end[0] + upper_offset[0], carrier_end[1] + upper_offset[1])
        upper_end = (upper_start[0] + 0.2 * np.cos(upper_angle),
                     upper_start[1] + 0.2 * np.sin(upper_angle))
        draw_phasor(ax, upper_start, upper_end, color='#A23B72', linewidth=3.5)
        
        # Calculate un-offset end position for upper sideband (for circle/arrow center)
        upper_end_no_offset = (carrier_end[0] + 0.2 * np.cos(upper_angle),
                               carrier_end[1] + 0.2 * np.sin(upper_angle))
        
        # Draw circle for upper sideband (centered at un-offset position)
        draw_circle(ax, carrier_end, 0.2, color='#A23B72', linestyle=':', linewidth=1.0)
        
        # Curved arrow for upper sideband (counterclockwise, centered at un-offset position)
        arrow_radius = 0.2
        arrow_angle = angles[0]
        draw_curved_arrow(ax, carrier_end, arrow_radius, arrow_angle, 
                         arrow_angle + 50, color='#A23B72', linewidth=2, direction='ccw')
        
        # Lower sideband: magnitude 0.2, starts from upper sideband end (with offset)
        lower_angle = np.radians(angles[1])
        lower_start = (upper_end[0] + lower_offset[0], upper_end[1] + lower_offset[1])
        lower_end = (lower_start[0] + 0.2 * np.cos(lower_angle),
                     lower_start[1] + 0.2 * np.sin(lower_angle))
        draw_phasor(ax, lower_start, lower_end, color='#C73E1D', linewidth=3.5)
        
        # Calculate un-offset end position for lower sideband (for resultant vector)
        lower_end_no_offset = (upper_end_no_offset[0] + 0.2 * np.cos(lower_angle),
                               upper_end_no_offset[1] + 0.2 * np.sin(lower_angle))
        
        # Draw circle for lower sideband (centered at un-offset position)
        draw_circle(ax, upper_end_no_offset, 0.2, color='#C73E1D', linestyle=':', linewidth=1.0)
        
        # Curved arrow for lower sideband (clockwise, centered at un-offset position)
        arrow_angle_lower = angles[1]
        draw_curved_arrow(ax, upper_end_no_offset, arrow_radius, arrow_angle_lower ,
                         arrow_angle_lower - 50, color='#C73E1D', linewidth=2, direction='cw')
        
        # Resultant phasor (using un-offset end position)
        offset = 0.03
        draw_phasor(ax, (offset, -offset), (lower_end_no_offset[0] + offset, lower_end_no_offset[1] - offset), color="#000000", linewidth=3.0)

        # Add label for the time
        add_text(ax, (0.7, 0.25), time_label, fontsize=32, color='#333333')
        
        plt.tight_layout()
        save_figure(fig, f'illustration_3_modulation_{chr(97+idx)}_{time_label.replace("/", "_")}')
        plt.close()


def illustration_4_phase_modulation():
    """
    Illustration 4: Phase modulation diagram.
    Shows carrier with phase variation indicator (curved double-headed arrow).
    """
    fig, ax = setup_plot(xlim=(-0.2, 1.2), ylim=(-0.2, 1.2))
    
    # Carrier phasor at 45 degrees, magnitude 1
    angle = np.radians(45)
    end_point = (np.cos(angle), np.sin(angle))
    draw_phasor(ax, (0, 0), end_point, color='#2E86AB', linewidth=2.5)
    
    # Curved double-headed arrow perpendicular to the carrier
    # Position near the tip of the arrow
    # The arrow should be curved with radius equal to carrier magnitude (1)
    carrier_magnitude = 1.0
    
    # Create a curved double-headed arrow
    # Position it tangent to a circle of radius 1 at angle 45°
    arc_center = (0, 0)
    arc_radius = carrier_magnitude
    
    # Arc spans about ±10 degrees from the carrier angle
    start_angle_deg = 45 - 8
    end_angle_deg = 45 + 8
    
    # Create the curved double-headed arrow using a custom approach
    start_pos = (arc_radius * np.cos(np.radians(start_angle_deg)),
                 arc_radius * np.sin(np.radians(start_angle_deg)))
    end_pos = (arc_radius * np.cos(np.radians(end_angle_deg)),
               arc_radius * np.sin(np.radians(end_angle_deg)))
    
    # Draw two curved arrows facing opposite directions
    arc1 = patches.FancyArrowPatch(
        posA=start_pos, posB=end_pos,
        arrowstyle='->,head_width=0.3,head_length=0.4',
        connectionstyle="arc3,rad=0.3",
        color='#333333',
        linewidth=2.0,
        mutation_scale=15,
        zorder=2
    )
    ax.add_patch(arc1)
    
    arc2 = patches.FancyArrowPatch(
        posA=end_pos, posB=start_pos,
        arrowstyle='->,head_width=0.3,head_length=0.4',
        connectionstyle="arc3,rad=0.3",
        color='#333333',
        linewidth=2.0,
        mutation_scale=15,
        zorder=2
    )
    ax.add_patch(arc2)
    
    plt.tight_layout()
    save_figure(fig, 'illustration_4_phase_modulation')
    plt.close()


def illustration_5_optical_cavity():
    """
    Illustration 5: Optical cavity field patterns.
    Two illustrations showing chains of phasors with different parameters.
    """
    
    def draw_phasor_chain(ax, r, phi_deg, n, color, label=None):
        """
        Draw a chain of n phasors, each with magnitude r^k and angle increment phi.
        
        Returns the final endpoint and all endpoints for drawing the resultant.
        """
        phi = np.radians(phi_deg)
        current_pos = np.array([0.0, 1.0])  # Start pointing up (positive imaginary)
        current_angle = np.pi / 2  # 90 degrees (pointing up)
        
        endpoints = [(0.0, 0.0)]
        
        for k in range(n):
            magnitude = r ** k
            next_angle = current_angle + k * phi
            
            # Calculate next position
            dx = magnitude * np.cos(next_angle)
            dy = magnitude * np.sin(next_angle)
            next_pos = current_pos + np.array([dx, dy])
            
            # Draw phasor
            draw_phasor(ax, tuple(current_pos), tuple(next_pos), 
                       color=color, linewidth=1.0)
            
            endpoints.append(tuple(next_pos))
            current_pos = next_pos
        
        return current_pos, endpoints
    
    # Illustration 1: Three chains with different phi values
    fig, ax = setup_plot(xlim=(-2, 8), ylim=(-2, 8))
    
    # Chain a: r=0.8, phi=-10°, n=100, blue
    final_a, endpoints_a = draw_phasor_chain(ax, 0.8, -10, 100, '#4A90E2')
    draw_phasor(ax, (0, 0), tuple(final_a), color='#1E5A8E', linewidth=2.5)
    
    # Chain b: r=0.8, phi=-3°, n=100, green
    final_b, endpoints_b = draw_phasor_chain(ax, 0.8, -3, 100, '#7CB342')
    draw_phasor(ax, (0, 0), tuple(final_b), color='#4A7C28', linewidth=2.5)
    
    # Chain c: r=0.8, phi=0°, n=100, orange
    final_c, endpoints_c = draw_phasor_chain(ax, 0.8, 0, 100, '#FF9800')
    draw_phasor(ax, (0, 0), tuple(final_c), color='#CC7700', linewidth=2.5)
    
    # Add a dotted circle - adjust these parameters as needed
    circle_center = (2.5, 2.5)
    circle_radius = 3.0
    draw_circle(ax, circle_center, circle_radius, color='#666666', 
               linestyle='--', linewidth=1.5)
    
    plt.tight_layout()
    save_figure(fig, 'illustration_5a_optical_cavity_varying_phi')
    plt.close()
    
    # Illustration 2: Three chains with different r values
    fig, ax = setup_plot(xlim=(-2, 15), ylim=(-2, 15))
    
    # Chain a: r=0.8, phi=-3°, n=100, blue
    final_a, endpoints_a = draw_phasor_chain(ax, 0.8, -3, 100, '#4A90E2')
    draw_phasor(ax, (0, 0), tuple(final_a), color='#1E5A8E', linewidth=2.5)
    
    # Chain b: r=0.9, phi=-3°, n=100, green
    final_b, endpoints_b = draw_phasor_chain(ax, 0.9, -3, 100, '#7CB342')
    draw_phasor(ax, (0, 0), tuple(final_b), color='#4A7C28', linewidth=2.5)
    
    # Chain c: r=0.95, phi=-3°, n=100, orange
    final_c, endpoints_c = draw_phasor_chain(ax, 0.95, -3, 100, '#FF9800')
    draw_phasor(ax, (0, 0), tuple(final_c), color='#CC7700', linewidth=2.5)
    
    # Add a dotted circle - adjust these parameters as needed
    circle_center = (5.0, 5.0)
    circle_radius = 7.0
    draw_circle(ax, circle_center, circle_radius, color='#666666', 
               linestyle='--', linewidth=1.5)
    
    plt.tight_layout()
    save_figure(fig, 'illustration_5b_optical_cavity_varying_r')
    plt.close()


def main():
    """
    Generate all phasor diagram illustrations.
    """
    print("Generating phasor diagrams...")
    #print("\nIllustration 1: Single phasor")
    #illustration_1_single_phasor()
    
    #print("\nIllustration 2: Phasor summation")
    #illustration_2_phasor_summation()
    
    print("\nIllustration 3: Amplitude modulation")
    illustration_3_amplitude_modulation()
    
    print("\nIllustration 4: Phase modulation")
    illustration_4_phase_modulation()
    
    #print("\nIllustration 5: Optical cavity fields")
    #illustration_5_optical_cavity()
    
    print("\nAll diagrams generated successfully!")


if __name__ == "__main__":
    main()
