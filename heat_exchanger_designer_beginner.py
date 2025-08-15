import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, Ellipse, Polygon
import pandas as pd
from io import BytesIO, StringIO
import pickle
import random

# Page configuration
st.set_page_config(
    page_title="🌡️ Heat Exchanger 2D Designer (Beginner-Friendly)", 
    page_icon="🌡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

class HeatExchangerDesigner:
    def __init__(self, width=300, height=200):
        self.width = width
        self.height = height
        self.grid_size = 0.2  # mm per grid unit
        self.nz_layers = 31   # z-direction layers for extrusion
        self.reset_canvas()
        
    def reset_canvas(self):
        """Reset canvas to all white"""
        self.canvas = np.ones((self.height, self.width), dtype=int)  # 1=white, 0=black
        
    def get_real_dimensions(self):
        """Get real world dimensions in mm"""
        real_width = self.width * self.grid_size
        real_height = self.height * self.grid_size
        return real_width, real_height
        
    def add_rectangular_fin(self, x, y, width, height, angle=0):
        """Add rectangular fin"""
        # Create four corner points of rectangle
        corners = np.array([
            [-width/2, -height/2],
            [width/2, -height/2], 
            [width/2, height/2],
            [-width/2, height/2]
        ])
        
        # Rotation
        if angle != 0:
            angle_rad = np.radians(angle)
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)]
            ])
            corners = corners @ rotation_matrix.T
        
        # Translate to specified position
        corners[:, 0] += x
        corners[:, 1] += y
        
        # Fill rectangular area
        self._fill_polygon(corners)
    
    def add_needle_fin(self, x, y, length, thickness, angle=0):
        """Add needle fin (elongated rectangle)"""
        self.add_rectangular_fin(x, y, length, thickness, angle)
    
    def add_circular_fin(self, x, y, radius):
        """Add circular fin"""
        y_indices, x_indices = np.ogrid[:self.height, :self.width]
        mask = (x_indices - x)**2 + (y_indices - y)**2 <= radius**2
        self.canvas[mask] = 0  # 0=black
    
    def add_elliptical_fin(self, x, y, a, b, angle=0):
        """Add elliptical fin"""
        y_indices, x_indices = np.ogrid[:self.height, :self.width]
        
        # Coordinates relative to ellipse center
        x_rel = x_indices - x
        y_rel = y_indices - y
        
        if angle != 0:
            angle_rad = np.radians(angle)
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)
            x_rot = x_rel * cos_angle + y_rel * sin_angle
            y_rot = -x_rel * sin_angle + y_rel * cos_angle
        else:
            x_rot, y_rot = x_rel, y_rel
            
        mask = (x_rot/a)**2 + (y_rot/b)**2 <= 1
        self.canvas[mask] = 0
    
    def add_l_shaped_fin(self, x, y, length1, width1, length2, width2, angle=0):
        """Add L-shaped fin"""
        # L-shape consists of two rectangles
        # First rectangle (horizontal part)
        self.add_rectangular_fin(x, y, length1, width1, angle)
        
        # Calculate position of second rectangle (vertical part)
        if angle == 0:
            x2 = x + (length1 - width2) / 2
            y2 = y + (width1 + length2) / 2
        else:
            # For rotated L-shape, more complex calculation needed
            angle_rad = np.radians(angle)
            dx = (length1 - width2) / 2 * np.cos(angle_rad) - (width1 + length2) / 2 * np.sin(angle_rad)
            dy = (length1 - width2) / 2 * np.sin(angle_rad) + (width1 + length2) / 2 * np.cos(angle_rad)
            x2 = x + dx
            y2 = y + dy
            
        self.add_rectangular_fin(x2, y2, width2, length2, angle + 90)
    
    def add_wave_fin(self, x_start, y, length, amplitude, frequency, thickness):
        """Add wave-shaped fin"""
        x_points = np.linspace(x_start, x_start + length, int(length))
        
        for x in x_points:
            wave_y = y + amplitude * np.sin(2 * np.pi * frequency * (x - x_start) / length)
            self.add_circular_fin(int(x), int(wave_y), thickness//2)
    
    def add_fin_array(self, start_x, start_y, fin_type, spacing_x, spacing_y, 
                      rows, cols, **fin_params):
        """Add fin array"""
        for row in range(rows):
            for col in range(cols):
                x = start_x + col * spacing_x
                y = start_y + row * spacing_y
                
                if fin_type == "rectangular":
                    self.add_rectangular_fin(x, y, **fin_params)
                elif fin_type == "needle":
                    self.add_needle_fin(x, y, **fin_params)
                elif fin_type == "circular":
                    self.add_circular_fin(x, y, **fin_params)
                elif fin_type == "elliptical":
                    self.add_elliptical_fin(x, y, **fin_params)
    
    def draw_flow_wall_line(self, x0, y0, length, angle_deg, width):
        """Draw a flow guide wall line with specified parameters"""
        angle_rad = np.radians(angle_deg)
        width = max(1, width)
        
        # Calculate the effective length to avoid going out of bounds
        effective_length = length
        for i in range(length):
            end_x = int(x0 + i * np.cos(angle_rad))
            end_y = int(y0 + i * np.sin(angle_rad))
            
            # Check if we're approaching boundaries
            if (end_x < width//2 or end_x >= self.width - width//2 or 
                end_y < width//2 or end_y >= self.height - width//2):
                effective_length = max(i, 10)  # Stop at boundary with 0 margin
                break
        
        # Draw the line with boundary-safe length
        for i in range(effective_length):
            x = int(x0 + i * np.cos(angle_rad))
            y = int(y0 + i * np.sin(angle_rad))
            
            # Ensure we're within canvas bounds
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                continue
            
            # Draw thick line by expanding around the center point with strict bounds checking
            y_start = max(y - width//2, 0)
            y_end = min(y + width//2 + 1, self.height)
            x_start = max(x - width//2, 0)  
            x_end = min(x + width//2 + 1, self.width)
            
            # Additional safety check
            if y_start < self.height and x_start < self.width and y_end > 0 and x_end > 0:
                self.canvas[y_start:y_end, x_start:x_end] = 0
    
    def add_inlet_center_walls(self, angle=20, length=60, thickness=2):
        """Add center inlet flow guide walls (left side, middle pair)"""
        # Proportional positioning based on domain size (reference: 300*200)
        # Original: 4mm from left = 20 grid units = 6.67% of 300
        inlet_x = int(self.width * 0.0667)  # 6.67% from left edge
        
        # Original: 4mm from center = 20 grid units = 10% of 200 height
        wall_offset_y = int(self.height * 0.10)  # 10% of height from center
        wall_upper_y = self.height//2 - wall_offset_y
        wall_lower_y = self.height//2 + wall_offset_y
        
        # Draw center inlet walls with same angle relative to horizontal
        self.draw_flow_wall_line(inlet_x, wall_upper_y, length, -angle, thickness)  # Upper: negative angle
        self.draw_flow_wall_line(inlet_x, wall_lower_y, length, angle, thickness)   # Lower: positive angle
    
    def add_inlet_outer_walls(self, angle=60, length=60, thickness=2):
        """Add outer inlet flow guide walls (left side, outer pair)"""
        # Proportional positioning based on domain size (reference: 300*200)
        # Original: 2mm from left = 10 grid units = 3.33% of 300
        inlet_x = int(self.width * 0.0333)  # 3.33% from left edge
        
        # Original: 8mm from center = 40 grid units = 20% of 200 height
        wall_offset_y = int(self.height * 0.20)  # 20% of height from center
        wall_upper_y = self.height//2 - wall_offset_y
        wall_lower_y = self.height//2 + wall_offset_y
        
        # Draw outer inlet walls with same angle relative to horizontal
        self.draw_flow_wall_line(inlet_x, wall_upper_y, length, -angle, thickness)  # Upper: negative angle
        self.draw_flow_wall_line(inlet_x, wall_lower_y, length, angle, thickness)   # Lower: positive angle
    
    def add_outlet_walls(self, angle=10, length=150, thickness=3):
        """Add outlet flow guide walls (right side fixed)"""
        # Proportional positioning based on domain size (reference: 300*200)
        # Original: 2mm from right = 10 grid units = 3.33% of 300
        outlet_x = self.width - int(self.width * 0.0333)  # 3.33% from right edge
        
        # New positioning: upper wall at 1/8 from top, lower wall at 1/8 from bottom
        upper_wall_y = int(self.height * 0.125)  # 1/8 from top
        lower_wall_y = int(self.height * 0.875)  # 1/8 from bottom (7/8 from top)
        
        # Draw outlet walls with unified angle control (symmetric, pointing left)
        # Angles adjusted for right-to-left direction
        self.draw_flow_wall_line(outlet_x, upper_wall_y, length, 180 - angle, thickness)  # Upper: angled down-left
        self.draw_flow_wall_line(outlet_x, lower_wall_y, length, 180 + angle, thickness)  # Lower: angled up-left
    
    def add_horizontal_fins(self, fin_length=120, fin_thickness=3, fin_spacing=5, fin_count=7):
        """Add horizontal rectangular fins with fixed right end"""
        # Fixed right end point (same as outlet walls)
        outlet_x = self.width - int(self.width * 0.0333)  # Right end of outlet walls
        
        # Calculate left start position based on fin length
        start_x = outlet_x - fin_length
        
        # Upper region: from top to 1/8 height (lower_count controls this - corrected)
        upper_boundary = int(self.height * 0.125)
        
        # Lower region: from 7/8 height to bottom (upper_count controls this - corrected)
        lower_boundary = int(self.height * 0.875)
        
        # Add fins in upper region (top area) - start from top boundary with spacing
        if fin_count > 0:
            for i in range(fin_count):
                # First fin: spacing distance from top, then each subsequent fin goes down
                fin_y = fin_spacing + i * (fin_thickness + fin_spacing) + fin_thickness // 2
                fin_center_x = start_x + fin_length // 2
                self.add_rectangular_fin(fin_center_x, fin_y, fin_length, fin_thickness, 0)
        
        # Add fins in lower region (bottom area) - start from bottom boundary with spacing
        if fin_count > 0:
            for i in range(fin_count):
                # First fin: spacing distance from bottom, then each subsequent fin goes up
                fin_y = self.height - fin_spacing - i * (fin_thickness + fin_spacing) - fin_thickness // 2
                fin_center_x = start_x + fin_length // 2
                self.add_rectangular_fin(fin_center_x, fin_y, fin_length, fin_thickness, 0)
    
    def add_circular_fins_array(self, diameter=2, h_spacing=15, v_spacing=15, min_distance=3):
        """Add array of circular fins in middle region with edge margins (1/40 from each edge)"""
        radius = diameter / 2
        
        # Middle region boundaries (smaller margins for more circle space)
        upper_boundary = int(self.height * 0.05)   # 1/20 from top
        lower_boundary = int(self.height * 0.95)   # 1/20 from bottom
        
        # Available middle region with left/right boundaries (1/40 from each edge)
        middle_region_height = lower_boundary - upper_boundary
        left_boundary = int(self.width * 0.025)   # 1/40 from left
        right_boundary = int(self.width * 0.975)  # 39/40 from left (1/40 from right)
        middle_region_width = right_boundary - left_boundary
        
        # Calculate how many circles can fit in each direction
        # For columns: (width - diameter) / (diameter + h_spacing) + 1
        cols = max(1, int((middle_region_width - diameter) / (diameter + h_spacing)) + 1)
        
        # For rows: (height - diameter) / (diameter + v_spacing) + 1  
        rows = max(1, int((middle_region_height - diameter) / (diameter + v_spacing)) + 1)
        
        # Calculate actual spacing to center the array
        if cols > 1:
            actual_h_spacing = (middle_region_width - cols * diameter) / (cols - 1)
        else:
            actual_h_spacing = 0
            
        if rows > 1:
            actual_v_spacing = (middle_region_height - rows * diameter) / (rows - 1)
        else:
            actual_v_spacing = 0
        
        # Add circular fins array to fill the entire middle region
        for row in range(rows):
            for col in range(cols):
                # Calculate position (center of each circle) within the middle region boundaries
                fin_x = left_boundary + radius + col * (diameter + actual_h_spacing)
                fin_y = upper_boundary + radius + row * (diameter + actual_v_spacing)
                
                # Add circular fin if within bounds and not too close to existing structures
                if (fin_x - radius >= left_boundary and fin_x + radius <= right_boundary and 
                    fin_y - radius >= upper_boundary and fin_y + radius <= lower_boundary):
                    
                    # Check if this circular fin is too close to existing structures
                    if self._is_position_safe_for_circle(int(fin_x), int(fin_y), int(radius), min_distance):
                        self.add_circular_fin(int(fin_x), int(fin_y), int(radius))
    
    def _is_position_safe_for_circle(self, center_x, center_y, radius, min_distance=3):
        """Check if a circular fin position is safe (not too close to existing structures)"""
        # Check multiple layers around the circle for more robust detection
        check_radius = int(radius + min_distance)
        
        # Method 1: Dense grid sampling within the check area
        overlap_pixels = 0
        total_pixels = 0
        
        # Sample a grid around the circle center
        for dx in range(-check_radius, check_radius + 1):
            for dy in range(-check_radius, check_radius + 1):
                # Only check points within the "safety zone" circle
                distance = np.sqrt(dx*dx + dy*dy)
                if distance <= check_radius:
                    check_x = center_x + dx
                    check_y = center_y + dy
                    
                    # If check point is within canvas bounds
                    if (0 <= check_x < self.width and 0 <= check_y < self.height):
                        total_pixels += 1
                        # If this point is occupied by existing structure (black pixel = 0)
                        if self.canvas[check_y, check_x] == 0:
                            overlap_pixels += 1
                    else:
                        # If check point is outside canvas, treat as obstacle
                        total_pixels += 1
                        overlap_pixels += 1
        
        # If we couldn't sample enough points or too much overlap, reject
        if total_pixels == 0:
            return False
        
        overlap_ratio = overlap_pixels / total_pixels
        # Use stricter threshold: reject if more than 10% overlap
        return overlap_ratio < 0.1
    
    def add_parallel_needle_array(self, length=30, thickness=3, h_spacing=20, v_spacing=25):
        """Add dense parallel needle fin array filling the middle region"""
        # Use same boundaries as circular fins: 1/20 top/bottom, 1/40 left/right
        top_boundary = self.height * 0.05  # 1/20 from top
        bottom_boundary = self.height * 0.95  # 1/20 from bottom
        left_boundary = self.width * 0.025  # 1/40 from left
        right_boundary = self.width * 0.975  # 1/40 from right
        
        # Calculate array dimensions
        middle_region_width = right_boundary - left_boundary
        middle_region_height = bottom_boundary - top_boundary
        
        # Calculate number of rows and columns to fill the region
        cols = max(1, int(middle_region_width / h_spacing))
        rows = max(1, int(middle_region_height / v_spacing))
        
        # Centering start positions
        mid_x = (left_boundary + right_boundary) / 2.0
        mid_y = (top_boundary + bottom_boundary) / 2.0
        start_x = mid_x - ((cols - 1) * h_spacing) / 2.0
        start_y = mid_y - ((rows - 1) * v_spacing) / 2.0
        
        # Add needle fins in centered grid pattern
        for row in range(rows):
            for col in range(cols):
                fin_x = start_x + col * h_spacing
                fin_y = start_y + row * v_spacing
                
                # Ensure within boundaries
                if (left_boundary <= fin_x <= right_boundary and 
                    top_boundary <= fin_y <= bottom_boundary):
                    self.add_needle_fin(int(fin_x), int(fin_y), length, thickness, 0)
    
    def add_staggered_circle_array(self, radius=8, h_spacing=25, v_spacing=20):
        """Add dense staggered circular fin array filling the middle region"""
        # Use same boundaries as circular fins
        top_boundary = self.height * 0.05
        bottom_boundary = self.height * 0.95
        left_boundary = self.width * 0.025
        right_boundary = self.width * 0.975
        
        middle_region_width = right_boundary - left_boundary
        middle_region_height = bottom_boundary - top_boundary
        
        cols = max(1, int(middle_region_width / h_spacing))
        rows = max(1, int(middle_region_height / v_spacing))
        
        # Center grid start positions
        mid_x = (left_boundary + right_boundary) / 2.0
        mid_y = (top_boundary + bottom_boundary) / 2.0
        start_x = mid_x - ((cols - 1) * h_spacing) / 2.0
        start_y = mid_y - ((rows - 1) * v_spacing) / 2.0
        
        for row in range(rows):
            for col in range(cols):
                # Stagger every other row
                offset_x = (h_spacing // 2) if row % 2 == 1 else 0
                fin_x = start_x + col * h_spacing + offset_x
                fin_y = start_y + row * v_spacing
                
                if (left_boundary <= fin_x <= right_boundary and 
                    top_boundary <= fin_y <= bottom_boundary):
                    self.add_circular_fin(int(fin_x), int(fin_y), radius)
    
    def add_wave_heat_sink_array(self, wavelength=40, amplitude=8, thickness=3, spacing=30):
        """Add dense wave heat sink array filling the middle region"""
        top_boundary = self.height * 0.05
        bottom_boundary = self.height * 0.95
        left_boundary = self.width * 0.025
        right_boundary = self.width * 0.975
        
        middle_region_height = bottom_boundary - top_boundary
        rows = max(1, int(middle_region_height / spacing))
        
        # Center rows vertically
        mid_y = (top_boundary + bottom_boundary) / 2.0
        start_y = mid_y - ((rows - 1) * spacing) / 2.0
        total_len = int(right_boundary - left_boundary)
        
        for row in range(rows):
            y_center = start_y + row * spacing
            if top_boundary <= y_center <= bottom_boundary:
                n_cycles = max(1, int(round(total_len / wavelength)))
                self.add_wave_fin(
                    x_start=int(left_boundary),
                    y=int(y_center),
                    length=total_len,
                    amplitude=amplitude,
                    frequency=n_cycles,
                    thickness=thickness,
                )
    
    def add_rectangular_grid_array(self, width=15, height=20, h_spacing=25, v_spacing=30):
        """Add dense rectangular grid array filling the middle region"""
        top_boundary = self.height * 0.05
        bottom_boundary = self.height * 0.95
        left_boundary = self.width * 0.025
        right_boundary = self.width * 0.975
        
        middle_region_width = right_boundary - left_boundary
        middle_region_height = bottom_boundary - top_boundary
        
        cols = max(1, int(middle_region_width / h_spacing))
        rows = max(1, int(middle_region_height / v_spacing))
        
        # Center grid start positions
        mid_x = (left_boundary + right_boundary) / 2.0
        mid_y = (top_boundary + bottom_boundary) / 2.0
        start_x = mid_x - ((cols - 1) * h_spacing) / 2.0
        start_y = mid_y - ((rows - 1) * v_spacing) / 2.0
        
        for row in range(rows):
            for col in range(cols):
                fin_x = start_x + col * h_spacing
                fin_y = start_y + row * v_spacing
                
                if (left_boundary <= fin_x <= right_boundary and 
                    top_boundary <= fin_y <= bottom_boundary):
                    self.add_rectangular_fin(int(fin_x), int(fin_y), width, height)
    
    def clear_outlet_walls_right_side(self, angle=18, length=170, thickness=3):
        """Clear (set to white) the right side of outlet walls to remove overlapping fins"""
        # Get outlet wall parameters (same as add_outlet_walls)
        outlet_x = self.width - int(self.width * 0.0333)  # 3.33% from right edge
        upper_wall_y = int(self.height * 0.125)  # 1/8 from top
        lower_wall_y = int(self.height * 0.875)  # 1/8 from bottom
        
        # Calculate the angle in radians
        angle_rad = np.radians(angle)
        
        # Clear right side of upper wall
        self._clear_wall_right_side(outlet_x, upper_wall_y, length, 180 - angle, thickness)
        
        # Clear right side of lower wall  
        self._clear_wall_right_side(outlet_x, lower_wall_y, length, 180 + angle, thickness)
    
    def _clear_wall_right_side(self, x0, y0, length, angle_deg, width):
        """Clear the right side area of a wall line"""
        angle_rad = np.radians(angle_deg)
        width = max(1, width)
        
        # Calculate the effective length (same logic as draw_flow_wall_line)
        effective_length = length
        for i in range(length):
            end_x = int(x0 + i * np.cos(angle_rad))
            end_y = int(y0 + i * np.sin(angle_rad))
            
            if (end_x < width//2 or end_x >= self.width - width//2 or 
                end_y < width//2 or end_y >= self.height - width//2):
                effective_length = max(i, 10)
                break
        
        # For each point along the wall line, clear everything to the right
        for i in range(effective_length):
            x = int(x0 + i * np.cos(angle_rad))
            y = int(y0 + i * np.sin(angle_rad))
            
            # Ensure we're within canvas bounds
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                continue
            
            # Clear a rectangular area around this point, extending to the right edge
            y_start = max(y - width//2, 0)
            y_end = min(y + width//2 + 1, self.height)
            
            # Clear from the RIGHT EDGE of the wall (not center) to avoid deleting the wall itself
            # Calculate thickness projection in X direction considering wall angle
            angle_sin = abs(np.sin(angle_rad))
            if angle_sin > 0.01:  # Avoid division by very small values
                wall_right_edge = int(x + width / angle_sin)
            else:
                wall_right_edge = x + width//2 + 1  # Fallback for nearly horizontal walls
            
            # Clear from wall's right edge to the right edge of canvas
            if y_start < self.height and y_end > 0 and wall_right_edge < self.width:
                self.canvas[y_start:y_end, wall_right_edge:self.width] = 1  # Set to white (1)
    
    def clear_outlet_walls_upward(self, angle=18, length=170, thickness=3, clear_height=4):
        """Clear areas away from outlet walls: upper wall upward, lower wall downward"""
        # Get outlet wall parameters (same as add_outlet_walls)
        outlet_x = self.width - int(self.width * 0.0333)  # 3.33% from right edge
        upper_wall_y = int(self.height * 0.125)  # 1/8 from top
        lower_wall_y = int(self.height * 0.875)  # 1/8 from bottom
        
        # Clear upward from upper wall (toward smaller Y values)
        self._clear_wall_directional(outlet_x, upper_wall_y, length, 180 - angle, thickness, clear_height, direction="up")
        
        # Clear downward from lower wall (toward larger Y values)
        self._clear_wall_directional(outlet_x, lower_wall_y, length, 180 + angle, thickness, clear_height, direction="down")
    
    def _clear_wall_directional(self, x0, y0, length, angle_deg, width, clear_height=4, direction="up"):
        """Clear the area around a wall line by specified height in given direction"""
        angle_rad = np.radians(angle_deg)
        width = max(1, width)
        
        # Calculate the effective length (same logic as draw_flow_wall_line)
        effective_length = length
        for i in range(length):
            end_x = int(x0 + i * np.cos(angle_rad))
            end_y = int(y0 + i * np.sin(angle_rad))
            
            if (end_x < width//2 or end_x >= self.width - width//2 or 
                end_y < width//2 or end_y >= self.height - width//2):
                effective_length = max(i, 10)
                break
        
        # For each point along the wall line, clear specified height in given direction
        for i in range(effective_length):
            x = int(x0 + i * np.cos(angle_rad))
            y = int(y0 + i * np.sin(angle_rad))
            
            # Ensure we're within canvas bounds
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                continue
            
            # Clear a rectangular area around this point
            x_start = max(x - width//2, 0)
            x_end = min(x + width//2 + 1, self.width)
            
            # Calculate thickness projection in Y direction considering wall angle
            angle_cos = abs(np.cos(angle_rad))
            if angle_cos > 0.01:  # Avoid division by very small values
                wall_half_thickness_y = int(width / (2 * angle_cos))
            else:
                wall_half_thickness_y = width//2  # Fallback for nearly vertical walls
            
            if direction == "up":
                # Clear upward from upper wall (toward smaller Y values)
                wall_top_edge = y - wall_half_thickness_y - 1
                clear_top_boundary = max(wall_top_edge - clear_height, 0)
                
                # Clear from clear_top_boundary to wall's top edge
                if x_start < self.width and x_end > 0 and clear_top_boundary >= 0 and wall_top_edge > clear_top_boundary:
                    self.canvas[clear_top_boundary:wall_top_edge, x_start:x_end] = 1  # Set to white (1)
                    
            elif direction == "down":
                # Clear downward from lower wall (toward larger Y values)
                wall_bottom_edge = y + wall_half_thickness_y + 1
                clear_bottom_boundary = min(wall_bottom_edge + clear_height, self.height)
                
                # Clear from wall's bottom edge to clear_bottom_boundary
                if x_start < self.width and x_end > 0 and wall_bottom_edge < clear_bottom_boundary and wall_bottom_edge >= 0:
                    self.canvas[wall_bottom_edge:clear_bottom_boundary, x_start:x_end] = 1  # Set to white (1)
    
    def _fill_polygon(self, corners):
        """Fill polygon area"""
        from matplotlib.path import Path
        
        # Create grid points
        y, x = np.mgrid[:self.height, :self.width]
        points = np.vstack((x.ravel(), y.ravel())).T
        
        # Create path and check which points are inside
        path = Path(corners)
        mask = path.contains_points(points)
        mask = mask.reshape(self.height, self.width)
        
        self.canvas[mask] = 0  # 0=black
    
    def get_canvas(self):
        """Get current canvas"""
        return self.canvas.copy()
    
    def export_geometry(self, filename):
        """Export geometry data in LBM simulation format"""
        # Convert to LBM format: 0=fluid, 1=solid
        lbm_data = 1 - self.canvas  # Invert: black fins become solid(1)
        
        # Extend to 3D (assuming single layer)
        layers = int(getattr(self, 'nz_layers', 31))
        layers = max(1, layers)
        lbm_3d = np.tile(lbm_data[:, :, np.newaxis], (1, 1, layers))
        
        # Save in LBM format with solver-friendly ordering (x fastest, then y, then z)
        flat = self._flatten_for_dat(lbm_3d)
        np.savetxt(filename, flat, fmt='%d')
        return lbm_3d

    def _flatten_for_dat(self, lbm_3d: np.ndarray) -> np.ndarray:
        """Flatten array to 1D in the order expected by typical LBM loaders:
        x varies fastest, then y, then z.
        Input lbm_3d is shaped (ny, nx, nz). We first transpose to (nx, ny, nz),
        then use Fortran-order ravel so the first axis (x) varies fastest.
        """
        ny, nx, nz = lbm_3d.shape
        arr_xynz = np.transpose(lbm_3d, (1, 0, 2))  # (nx, ny, nz)
        flat = arr_xynz.ravel(order='F')
        return flat

    def _vtk_from_array(self, lbm_3d, spacing_mm=(0.2, 0.2, 0.2)):
        """Create VTK legacy (STRUCTURED_POINTS, ASCII) content string from 3D numpy array.
        Values should be 0/1 indicating fluid/solid.
        spacing_mm: tuple of (sx, sy, sz) in mm.
        """
        ny, nx, nz = lbm_3d.shape  # array is (y, x, z)
        sx, sy, sz = spacing_mm
        sio = StringIO()
        sio.write("# vtk DataFile Version 3.0\n")
        sio.write("LBM geometry\n")
        sio.write("ASCII\n")
        sio.write("DATASET STRUCTURED_POINTS\n")
        sio.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        sio.write("ORIGIN 0 0 0\n")
        sio.write(f"SPACING {sx} {sy} {sz}\n")
        sio.write(f"POINT_DATA {nx*ny*nz}\n")
        sio.write("SCALARS solid int 1\n")
        sio.write("LOOKUP_TABLE default\n")
        for k in range(nz):
            for j in range(ny):
                row_vals = " ".join(str(int(lbm_3d[j, i, k])) for i in range(nx))
                sio.write(row_vals + "\n")
        return sio.getvalue()

    def export_geometry_vtk(self, filename, lbm_3d=None, spacing_mm=None):
        """Export geometry to a VTK legacy .vtk file (ASCII STRUCTURED_POINTS)."""
        if lbm_3d is None:
            # Generate from current canvas
            lbm_data = 1 - self.canvas
            layers = int(getattr(self, 'nz_layers', 31))
            layers = max(1, layers)
            lbm_3d = np.tile(lbm_data[:, :, np.newaxis], (1, 1, layers))
        if spacing_mm is None:
            s = float(self.grid_size)
            spacing_mm = (s, s, s)
        vtk_str = self._vtk_from_array(lbm_3d, spacing_mm)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(vtk_str)
        return vtk_str

    def export_geometry_vtk_to_buffer(self, lbm_3d=None, spacing_mm=None):
        """Return BytesIO buffer containing VTK legacy ASCII content for download."""
        vtk_str = self.export_geometry_vtk("geometry.vtk", lbm_3d=lbm_3d, spacing_mm=spacing_mm)
        buf = BytesIO()
        buf.write(vtk_str.encode('utf-8'))
        buf.seek(0)
        return buf

    def _stl_from_array(self, lbm_3d: np.ndarray, spacing_mm=(0.2, 0.2, 0.2), solid_value: int = 1, name: str = "lbm_solid") -> str:
        """Create ASCII STL content from a 3D voxel array (ny, nx, nz).
        Only exposed faces of voxels equal to solid_value are exported.
        spacing_mm: (sx, sy, sz) voxel edge lengths in mm.
        """
        ny, nx, nz = lbm_3d.shape  # (y, x, z)
        sx, sy, sz = spacing_mm

        def facet(normal, v1, v2, v3) -> str:
            return (
                f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n"
                "    outer loop\n"
                f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n"
                f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n"
                f"      vertex {v3[0]:.6e} {v3[1]:.6e} {v3[2]:.6e}\n"
                "    endloop\n"
                "  endfacet\n"
            )

        chunks = [f"solid {name}\n"]
        # Iterate voxels
        for j in range(ny):
            y0 = j * sy
            y1 = (j + 1) * sy
            for i in range(nx):
                x0 = i * sx
                x1 = (i + 1) * sx
                for k in range(nz):
                    if lbm_3d[j, i, k] != solid_value:
                        continue
                    z0 = k * sz
                    z1 = (k + 1) * sz

                    # neighbor checks (fluid/out of bounds → emit face)
                    # -X face
                    if i == 0 or lbm_3d[j, i - 1, k] != solid_value:
                        n = (-1.0, 0.0, 0.0)
                        v = [
                            (x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)
                        ]
                        chunks.append(facet(n, v[0], v[1], v[2]))
                        chunks.append(facet(n, v[0], v[2], v[3]))
                    # +X face
                    if i == nx - 1 or lbm_3d[j, i + 1, k] != solid_value:
                        n = (1.0, 0.0, 0.0)
                        v = [
                            (x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)
                        ]
                        chunks.append(facet(n, v[0], v[1], v[2]))
                        chunks.append(facet(n, v[0], v[2], v[3]))
                    # -Y face
                    if j == 0 or lbm_3d[j - 1, i, k] != solid_value:
                        n = (0.0, -1.0, 0.0)
                        v = [
                            (x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)
                        ]
                        chunks.append(facet(n, v[0], v[1], v[2]))
                        chunks.append(facet(n, v[0], v[2], v[3]))
                    # +Y face
                    if j == ny - 1 or lbm_3d[j + 1, i, k] != solid_value:
                        n = (0.0, 1.0, 0.0)
                        v = [
                            (x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)
                        ]
                        chunks.append(facet(n, v[0], v[1], v[2]))
                        chunks.append(facet(n, v[0], v[2], v[3]))
                    # -Z face
                    if k == 0 or lbm_3d[j, i, k - 1] != solid_value:
                        n = (0.0, 0.0, -1.0)
                        v = [
                            (x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)
                        ]
                        chunks.append(facet(n, v[0], v[1], v[2]))
                        chunks.append(facet(n, v[0], v[2], v[3]))
                    # +Z face
                    if k == nz - 1 or lbm_3d[j, i, k + 1] != solid_value:
                        n = (0.0, 0.0, 1.0)
                        v = [
                            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)
                        ]
                        chunks.append(facet(n, v[0], v[1], v[2]))
                        chunks.append(facet(n, v[0], v[2], v[3]))

        chunks.append(f"endsolid {name}\n")
        return "".join(chunks)

    def export_geometry_stl(self, filename: str, lbm_3d=None, spacing_mm=None):
        """Export geometry to ASCII STL file (surface of solid voxels)."""
        if lbm_3d is None:
            lbm_data = 1 - self.canvas  # (y,x) fluid/solid -> solid=1
            layers = int(getattr(self, 'nz_layers', 31))
            layers = max(1, layers)
            lbm_3d = np.tile(lbm_data[:, :, np.newaxis], (1, 1, layers))
        if spacing_mm is None:
            s = float(self.grid_size)
            spacing_mm = (s, s, s)
        stl_str = self._stl_from_array(lbm_3d, spacing_mm)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(stl_str)
        return stl_str

    def export_geometry_stl_to_buffer(self, lbm_3d=None, spacing_mm=None):
        """Return BytesIO buffer containing ASCII STL for download."""
        stl_str = self.export_geometry_stl("geometry.stl", lbm_3d=lbm_3d, spacing_mm=spacing_mm)
        buf = BytesIO()
        buf.write(stl_str.encode('utf-8'))
        buf.seek(0)
        return buf

    def apply_preset_configuration(self, preset_name):
        """Apply predefined configurations"""
        self.reset_canvas()
        
        if preset_name == "Hybrid Finned Cold Plate":
            # Add 6 flow guide walls in 3 categories
            # Category 1: Inlet Center Walls (2 walls - middle pair)
            self.add_inlet_center_walls(angle=20, length=60, thickness=2)
            
            # Category 2: Inlet Outer Walls (2 walls - outer pair)  
            self.add_inlet_outer_walls(angle=60, length=60, thickness=2)
            
            # Category 3: Outlet Walls (2 walls)
            self.add_outlet_walls(angle=18, length=170, thickness=3)
            
            # Category 4: Horizontal Rectangular Fins
            self.add_horizontal_fins(fin_length=120, fin_thickness=3, fin_spacing=5, fin_count=7)
            
            # Category 5: Clear outlet walls areas (remove overlapping fins)
            self.clear_outlet_walls_right_side(angle=18, length=170, thickness=3)
            self.clear_outlet_walls_upward(angle=18, length=170, thickness=3, clear_height=4)
            
            # Category 6: Circular Fins Array (added after clearing)
            self.add_circular_fins_array(diameter=2, h_spacing=15, v_spacing=15, min_distance=3)
            
        elif preset_name == "Parallel Needle Array":
            self.add_parallel_needle_array(length=30, thickness=3, h_spacing=20, v_spacing=25)
        elif preset_name == "Staggered Circle Array":
            self.add_staggered_circle_array(radius=8, h_spacing=25, v_spacing=20)
        elif preset_name == "Wave Heat Sink":
            self.add_wave_heat_sink_array(wavelength=40, amplitude=8, thickness=3, spacing=30)
        elif preset_name == "Rectangular Grid":
            self.add_rectangular_grid_array(width=15, height=20, h_spacing=25, v_spacing=30)

def evaluate_design_quality(canvas: np.ndarray):
    """Evaluate design quality and return (score, metrics dict).
    Metrics include density (solid ratio), quadrant uniformity (std), and edge usage ratio.
    """
    if canvas is None or canvas.size == 0:
        return 0.0, {"density": 0.0, "uniformity": 1.0, "edge_usage": 1.0}

    height, width = canvas.shape
    total_pixels = canvas.size
    solid_pixels = int(np.sum(canvas == 0))
    density = solid_pixels / total_pixels if total_pixels > 0 else 0.0

    # Quadrant uniformity: std of quadrant solid densities
    mid_x, mid_y = width // 2, height // 2
    q1 = canvas[:mid_y, :mid_x]
    q2 = canvas[:mid_y, mid_x:]
    q3 = canvas[mid_y:, :mid_x]
    q4 = canvas[mid_y:, mid_x:]
    quadrants = [q for q in (q1, q2, q3, q4) if q.size > 0]
    quad_densities = [np.sum(q == 0) / q.size for q in quadrants] if quadrants else [0.0]
    uniformity_std = float(np.std(quad_densities))

    # Edge usage: fraction of solid pixels lying on the borders
    top_edge = np.sum(canvas[0, :] == 0) if height > 0 else 0
    bottom_edge = np.sum(canvas[-1, :] == 0) if height > 0 else 0
    left_edge = np.sum(canvas[:, 0] == 0) if width > 0 else 0
    right_edge = np.sum(canvas[:, -1] == 0) if width > 0 else 0
    edge_total = top_edge + bottom_edge + left_edge + right_edge
    perimeter = max(1, 2 * (width + height))
    edge_usage_ratio = edge_total / perimeter

    # Composite score (0-100): proxy for j/f^(1/3) with penalties
    # 1) Density window target 15%~55% → peak around 35% (weight 0.45)
    target_mid = 0.35
    if 0.15 <= density <= 0.55:
        density_score = 45.0 - abs(density - target_mid) * 100.0 * 0.75
    else:
        density_score = max(0.0, 45.0 - abs(density - target_mid) * 100.0)
    density_score = max(0.0, min(45.0, density_score))

    # 2) Uniformity (std): lower is better → up to 25 points
    uniformity_score = max(0.0, min(25.0, 25.0 - uniformity_std * 120.0))

    # 3) Edge usage: lower is better → up to 15 points
    edge_score = max(0.0, min(15.0, 15.0 - edge_usage_ratio * 150.0))

    # 4) j/f^(1/3) proxy (15 points): encourage moderate density and low edge usage as proxy for better j/f^(1/3)
    # proxy_j ~ (1 + 0.8*mix) / (1 + 0.6*edge), here mix≈(1 - uniformity_std), edge=edge_usage_ratio
    mix = max(0.0, 1.0 - uniformity_std)
    proxy_j = (1.0 + 0.8 * mix) / (1.0 + 0.6 * edge_usage_ratio)
    jf_score = max(0.0, min(15.0, 15.0 * (proxy_j / 1.6)))

    total_score = max(0.0, min(100.0, density_score + uniformity_score + edge_score + jf_score))
    metrics = {
        "density": float(density),
        "uniformity": float(uniformity_std),
        "edge_usage": float(edge_usage_ratio),
    }
    return float(total_score), metrics

def generate_random_candidates(designer: HeatExchangerDesigner):
    """Generate 2 random candidates using current Random Explore settings."""
    preserved_grid_size = designer.grid_size

    def rand_domain():
        # Optimized ranges for better design proportions
        # Length: 200-500 (more practical range)
        # Width: 100-300 (better aspect ratio)
        length = random.randint(200, 500)
        width = random.randint(100, 300)
        # Ensure reasonable aspect ratio (Length > Width, but not too extreme)
        if width >= length:
            width = max(80, length - random.randint(30, 120))
        elif length > width * 4:
            width = max(80, length // random.randint(2, 4))
        return int(length), int(width)

    def ir(v, lo, hi):
        return int(max(lo, min(hi, v)))

    def gen_candidate_once():
        # Randomize or preserve domain size based on settings, keep grid unit size
        if st.session_state.get("randomize_lw", False):
            L, W = rand_domain()
        else:
            L, W = designer.width, designer.height
        designer.width = L
        designer.height = W
        designer.grid_size = preserved_grid_size
        designer.reset_canvas()

        # Choose types: smart presets or free mix
        allowed = st.session_state.get(
            "random_allowed_types", ["needle", "circle", "wave", "rect", "ellipse"]
        ) or ["needle", "circle", "wave", "rect", "ellipse"]
        array_types_all = [t for t in ["needle", "circle", "wave", "rect", "ellipse"] if t in allowed]

        preset_mode = st.session_state.get("random_preset_mode", "Free mix")
        complexity = st.session_state.get("random_complexity", "Random")

        def pick_k_for_random(num_types: int) -> int:
            # Weighted choice for 1..num_types, bias towards middle values
            # Reduce probability of single-type; increase 2+ types (while supporting up to num_types)
            # Index: [k=1, k=2, k=3, k=4, k=5]
            base_weights = [0.08, 0.34, 0.30, 0.18, 0.10]
            weights = base_weights[:max(1, min(num_types, len(base_weights)))]
            s = sum(weights)
            weights = [w / s for w in weights]
            choices = list(range(1, len(weights) + 1))
            return random.choices(choices, weights=weights, k=1)[0]

        if preset_mode != "Free mix" and len(array_types_all) > 0:
            label_to_combo = {
                "Needle + Circle (high heat transfer/mixing)": ["needle", "circle"],
                "Rect + Circle (balanced grid + mixing)": ["rect", "circle"],
                "Wave + Needle (complex flow + conduction)": ["wave", "needle"],
                "Ellipse + Needle (smooth footprint + needles)": ["ellipse", "needle"],
                "Rect + Ellipse + Circle (diverse mix)": ["rect", "ellipse", "circle"],
            }
            target_combo = label_to_combo.get(preset_mode, [array_types_all[0]])
            chosen = [t for t in target_combo if t in array_types_all]
            if not chosen:
                chosen = [array_types_all[0]]
        else:
            if complexity == "Simple":
                k = 1 if len(array_types_all) >= 1 else 0
            elif complexity == "Medium":
                k = min(2, max(1, len(array_types_all)))
            elif complexity == "Complex":
                k = min(3, max(2, len(array_types_all)))
            else:
                k = pick_k_for_random(len(array_types_all)) if len(array_types_all) > 0 else 0
            chosen = random.sample(array_types_all, k) if k > 0 else []

        # Decide dropout mode per array instance
        def choose_dropout_mode() -> str:
            # Force uniform arrays: no dropout of elements within an array
            return "none"

        def should_keep_cell(row_idx: int, col_idx: int, mode: str) -> bool:
            if mode == "none":
                return True
            if mode == "regular":
                return ((row_idx + col_idx) % 3) != 0
            # irregular
            return random.random() < 0.85

        def should_keep_row(row_idx: int, mode: str) -> bool:
            if mode == "none":
                return True
            if mode == "regular":
                return (row_idx % 3) != 0
            return random.random() < 0.85

        # Add selected arrays with smart, scaled parameters
        circles = []   # list of (x,y,r)
        rects = []     # list of (x,y,w,h)
        ellipses = []  # list of (x,y,a,b)
        for at in chosen:
            if at == "needle":
                dropout_mode = choose_dropout_mode()
                
                # 15% 概率生成"直线"效果（长针+小间距）
                create_line_effect = random.random() < 0.15
                
                if create_line_effect:
                    # 直线效果：长针 + 极小间距
                    base_len = max(20, min(int(0.15 * L), 50))  # 较长的针
                    length = ir(base_len, 15, min(60, L // 6))
                    thickness = ir(int(max(2, min(0.015 * min(L, W), 6))), 2, 8)
                    
                    # 极小的水平间距（让针几乎连接）
                    min_h_spacing = int(max(length * 0.2, 5))  # 只有长度的0.2倍
                    min_v_spacing = int(max(thickness * 4, 20))  # 垂直间距保持较大
                    
                    h_spacing = ir(min_h_spacing, min_h_spacing, min_h_spacing + 5)  # 几乎固定的小间距
                    v_spacing = ir(int(max(min_v_spacing, min(0.12 * W, 60))), min_v_spacing, W)
                else:
                    # 正常效果：短针 + 合理间距（85%概率）
                    base_len = max(10, min(int(0.08 * L), 35))
                    length = ir(base_len, 8, min(40, L // 8))
                    thickness = ir(int(max(2, min(0.02 * min(L, W), 8))), 2, 10)
                    
                    # 正常间距，确保针之间有明显间隔
                    min_h_spacing = int(max(length * 1.2, 20))
                    min_v_spacing = int(max(thickness * 3, 15))
                    
                    h_spacing = ir(int(max(min_h_spacing, min(0.15 * L, 80))), min_h_spacing, L)
                    v_spacing = ir(int(max(min_v_spacing, min(0.12 * W, 60))), min_v_spacing, W)
                
                # Optional overlap mitigation
                if st.session_state.get("random_auto_reduce_overlap", True):
                    current_density = np.sum(designer.canvas == 0) / (designer.width * designer.height)
                    if current_density > 0.30:
                        h_spacing = ir(int(h_spacing * 1.20), min_h_spacing, L)
                        v_spacing = ir(int(v_spacing * 1.20), min_v_spacing, W)
                        # 同时减小长度以降低密度
                        if not create_line_effect:  # 直线效果不参与密度调整
                            length = ir(int(length * 0.85), 8, min(35, L // 8))
                
                # Manual placement with optional dropout
                top_boundary = designer.height * 0.05
                bottom_boundary = designer.height * 0.95
                left_boundary = designer.width * 0.025
                right_boundary = designer.width * 0.975
                region_w = right_boundary - left_boundary
                region_h = bottom_boundary - top_boundary
                
                spacing_scale = float(st.session_state.get("_random_global_spacing_scale", 1.0))
                
                # Large array-level randomness range (keep wide spread)
                rand_scale_x = random.uniform(0.6, 1.4)
                rand_scale_y = random.uniform(0.6, 1.4)
                
                cols = max(1, int(region_w / (h_spacing * spacing_scale * rand_scale_x)))
                rows = max(1, int(region_h / (v_spacing * spacing_scale * rand_scale_y)))
                
                # 限制最大行列数，避免太密集
                cols = min(cols, int(region_w / min_h_spacing))
                rows = min(rows, int(region_h / min_v_spacing))
                
                mid_x = (left_boundary + right_boundary) / 2.0
                mid_y = (top_boundary + bottom_boundary) / 2.0
                start_x = mid_x - ((cols - 1) * h_spacing) / 2.0
                start_y = mid_y - ((rows - 1) * v_spacing) / 2.0
                
                # 确保间距不会太小
                h_spacing = max(h_spacing, int(length * 1.2))
                
                for r in range(rows):
                    for c in range(cols):
                        if not should_keep_cell(r, c, dropout_mode):
                            continue
                        # 精确的网格位置，无抖动
                        fin_x = start_x + c * h_spacing
                        fin_y = start_y + r * v_spacing
                        if (left_boundary <= fin_x <= right_boundary and top_boundary <= fin_y <= bottom_boundary):
                            # 使用一致的长度和厚度
                            designer.add_needle_fin(int(fin_x), int(fin_y), length, thickness, 0)
            elif at == "circle":
                dropout_mode = choose_dropout_mode()
                # Allow array-level size shrink with fixed spacing
                radius = ir(int(max(3, min(0.04 * W, 12))), 3, 18)
                if random.random() < 0.35:  # 35% chance to shrink element size while keeping spacing
                    radius = max(2, int(radius * random.uniform(0.6, 0.9)))
                min_hs = radius * 3
                min_vs = radius * 3
                h_spacing = ir(int(max(min_hs, min(0.06 * L, 30))), min_hs, L)
                v_spacing = ir(int(max(min_vs, min(0.06 * W, 25))), min_vs, W)
                if st.session_state.get("random_auto_reduce_overlap", True):
                    current_density = np.sum(designer.canvas == 0) / (designer.width * designer.height)
                    if current_density > 0.30:
                        h_spacing = ir(int(h_spacing * 1.10), min_hs, L)
                        v_spacing = ir(int(v_spacing * 1.10), min_vs, W)
                # Manual placement (staggered) with optional dropout
                top_boundary = designer.height * 0.05
                bottom_boundary = designer.height * 0.95
                left_boundary = designer.width * 0.025
                right_boundary = designer.width * 0.975
                region_w = right_boundary - left_boundary
                region_h = bottom_boundary - top_boundary
                spacing_scale = float(st.session_state.get("_random_global_spacing_scale", 1.0))
                cols = max(1, int(region_w / (h_spacing * spacing_scale * random.uniform(0.6, 1.4))))
                rows = max(1, int(region_h / (v_spacing * spacing_scale * random.uniform(0.6, 1.4))))
                mid_x = (left_boundary + right_boundary) / 2.0
                mid_y = (top_boundary + bottom_boundary) / 2.0
                start_x = mid_x - ((cols - 1) * h_spacing) / 2.0
                start_y = mid_y - ((rows - 1) * v_spacing) / 2.0
                for r in range(rows):
                    for c in range(cols):
                        if not should_keep_cell(r, c, dropout_mode):
                            continue
                        offset_x = (h_spacing // 2) if r % 2 == 1 else 0
                        fin_x = start_x + c * h_spacing + offset_x
                        fin_y = start_y + r * v_spacing
                        if (left_boundary <= fin_x <= right_boundary and top_boundary <= fin_y <= bottom_boundary):
                            designer.add_circular_fin(int(fin_x), int(fin_y), int(radius))
                            circles.append((int(fin_x), int(fin_y), int(radius)))
            elif at == "wave":
                dropout_mode = choose_dropout_mode()
                wavelength = ir(int(max(12, min(0.16 * L, 80))), 12, L)
                amplitude = ir(int(max(3, min(0.06 * W, max(6, W // 3)))), 3, max(6, W // 3))
                thickness = ir(int(max(2, min(0.02 * min(L, W), 6))), 2, 8)
                spacing = ir(int(max(12, min(0.14 * W, 50))), 8, W)
                if st.session_state.get("random_auto_reduce_overlap", True):
                    current_density = np.sum(designer.canvas == 0) / (designer.width * designer.height)
                    if current_density > 0.30:
                        spacing = ir(int(spacing * 1.10), 8, W)
                # Manual placement (rows of waves) with optional dropout
                top_boundary = designer.height * 0.05
                bottom_boundary = designer.height * 0.95
                left_boundary = designer.width * 0.025
                right_boundary = designer.width * 0.975
                region_h = bottom_boundary - top_boundary
                spacing_scale = float(st.session_state.get("_random_global_spacing_scale", 1.0))
                rows = max(1, int(region_h / (spacing * spacing_scale * random.uniform(0.9, 1.1))))
                mid_y = (top_boundary + bottom_boundary) / 2.0
                start_y = mid_y - ((rows - 1) * spacing) / 2.0
                total_len = int(right_boundary - left_boundary)
                for r in range(rows):
                    if not should_keep_row(r, dropout_mode):
                        continue
                    y_center = start_y + r * spacing
                    if top_boundary <= y_center <= bottom_boundary:
                        n_cycles = max(1, int(round(total_len / wavelength)))
                        designer.add_wave_fin(
                            x_start=int(left_boundary),
                            y=int(y_center),
                            length=total_len,
                            amplitude=amplitude,
                            frequency=n_cycles,
                            thickness=thickness,
                        )
            elif at == "rect":
                dropout_mode = choose_dropout_mode()
                elem_w = ir(int(max(6, min(0.06 * L, L // 5))), 5, max(6, L // 5))
                elem_h = ir(int(max(6, min(0.08 * W, W // 4))), 5, max(6, W // 4))
                # Allow array-level size shrink with fixed spacing
                if random.random() < 0.35:
                    scale_rect = random.uniform(0.6, 0.9)
                    elem_w = max(5, int(elem_w * scale_rect))
                    elem_h = max(5, int(elem_h * scale_rect))
                h_spacing = ir(int(max(12, min(0.10 * L, 50))), 8, L)
                v_spacing = ir(int(max(15, min(0.12 * W, 60))), 8, W)
                if st.session_state.get("random_auto_reduce_overlap", True):
                    current_density = np.sum(designer.canvas == 0) / (designer.width * designer.height)
                    if current_density > 0.30:
                        h_spacing = ir(int(h_spacing * 1.10), 8, L)
                        v_spacing = ir(int(v_spacing * 1.10), 8, W)
                        elem_w = ir(max(5, int(elem_w * 0.95)), 5, max(6, L // 5))
                        elem_h = ir(max(5, int(elem_h * 0.95)), 5, max(6, W // 4))
                # Manual placement with optional dropout
                top_boundary = designer.height * 0.05
                bottom_boundary = designer.height * 0.95
                left_boundary = designer.width * 0.025
                right_boundary = designer.width * 0.975
                region_w = right_boundary - left_boundary
                region_h = bottom_boundary - top_boundary
                spacing_scale = float(st.session_state.get("_random_global_spacing_scale", 1.0))
                cols = max(1, int(region_w / (h_spacing * spacing_scale * random.uniform(0.6, 1.4))))
                rows = max(1, int(region_h / (v_spacing * spacing_scale * random.uniform(0.6, 1.4))))
                mid_x = (left_boundary + right_boundary) / 2.0
                mid_y = (top_boundary + bottom_boundary) / 2.0
                start_x = mid_x - ((cols - 1) * h_spacing) / 2.0
                start_y = mid_y - ((rows - 1) * v_spacing) / 2.0
                for r in range(rows):
                    for c in range(cols):
                        if not should_keep_cell(r, c, dropout_mode):
                            continue
                        fin_x = start_x + c * h_spacing
                        fin_y = start_y + r * v_spacing
                        if (left_boundary <= fin_x <= right_boundary and top_boundary <= fin_y <= bottom_boundary):
                            designer.add_rectangular_fin(int(fin_x), int(fin_y), int(elem_w), int(elem_h), 0)
                            rects.append((int(fin_x), int(fin_y), int(elem_w), int(elem_h)))
            elif at == "ellipse":
                dropout_mode = choose_dropout_mode()
                # Build a sparser elliptical fin arrangement with optional dropout
                top_boundary = designer.height * 0.05
                bottom_boundary = designer.height * 0.95
                left_boundary = designer.width * 0.025
                right_boundary = designer.width * 0.975
                region_w = right_boundary - left_boundary
                region_h = bottom_boundary - top_boundary

                # Larger base spacing to reduce density
                denom_y = ir(int(max(24, min(0.18 * W, 120))), 12, W)
                denom_x = ir(int(max(24, min(0.16 * L, 120))), 12, L)
                if st.session_state.get("random_auto_reduce_overlap", True):
                    current_density = np.sum(designer.canvas == 0) / (designer.width * designer.height)
                    if current_density > 0.30:
                        denom_x = ir(int(denom_x * 1.10), 12, L)
                        denom_y = ir(int(denom_y * 1.10), 12, W)
                spacing_scale = float(st.session_state.get("_random_global_spacing_scale", 1.0))
                rows = max(1, int(region_h / (denom_y * spacing_scale * random.uniform(0.6, 1.4))))
                cols = max(1, int(region_w / (denom_x * spacing_scale * random.uniform(0.6, 1.4))))
                spacing_x = region_w / max(1, cols)
                spacing_y = region_h / max(1, rows)

                # Centered start
                mid_x = (left_boundary + right_boundary) / 2.0
                mid_y = (top_boundary + bottom_boundary) / 2.0
                start_x = mid_x - ((cols - 1) * spacing_x) / 2.0
                start_y = mid_y - ((rows - 1) * spacing_y) / 2.0

                # Smaller ellipse axes and clamp (with chance to shrink while keeping spacing)
                a_axis = ir(int(max(5, min(0.05 * L, L // 6))), 5, L // 5)
                b_axis = ir(int(max(5, min(0.05 * W, W // 6))), 5, W // 5)
                if random.random() < 0.35:
                    scale_el = random.uniform(0.6, 0.9)
                    a_axis = max(4, int(a_axis * scale_el))
                    b_axis = max(4, int(b_axis * scale_el))

                for r in range(int(rows)):
                    for c in range(int(cols)):
                        if not should_keep_cell(r, c, dropout_mode):
                            continue
                        fin_x = int(start_x + c * spacing_x)
                        fin_y = int(start_y + r * spacing_y)
                        designer.add_elliptical_fin(fin_x, fin_y, int(a_axis), int(b_axis), 0)
                        ellipses.append((int(fin_x), int(fin_y), int(a_axis), int(b_axis)))

        # Pairwise overlap mitigation between circle/rect/ellipse (no canvas clearing)
        if st.session_state.get("random_auto_reduce_overlap", True):
            def circles_overlap(c1, c2):
                (x1,y1,r1),(x2,y2,r2)=c1,c2
                return (x1-x2)**2 + (y1-y2)**2 < (r1+r2)**2 * 0.85  # 15% slack
            def rects_overlap(a,b):
                (x1,y1,w1,h1),(x2,y2,w2,h2)=a,b
                # axis-aligned approximate bbox overlap for our placed rectangles
                return (abs(x1-x2) < (w1+w2)//2*0.9) and (abs(y1-y2) < (h1+h2)//2*0.9)
            def ellipse_overlap(e1,e2):
                (x1,y1,a1,b1),(x2,y2,a2,b2)=e1,e2
                # approximate via bounding boxes
                return (abs(x1-x2) < (a1+a2)*0.9) and (abs(y1-y2) < (b1+b2)*0.9)

            # simple pass: if too many overlaps, slightly inflate spacing factors for future arrays (no redraw)
            circle_overlaps = sum(
                1 for i in range(len(circles)) for j in range(i+1,len(circles)) if circles_overlap(circles[i],circles[j])
            )
            rect_overlaps = sum(
                1 for i in range(len(rects)) for j in range(i+1,len(rects)) if rects_overlap(rects[i],rects[j])
            )
            ellipse_overlaps = sum(
                1 for i in range(len(ellipses)) for j in range(i+1,len(ellipses)) if ellipse_overlap(ellipses[i],ellipses[j])
            )
            total_pairs = max(1, len(circles)*(len(circles)-1)//2 + len(rects)*(len(rects)-1)//2 + len(ellipses)*(len(ellipses)-1)//2)
            overlap_ratio = (circle_overlaps + rect_overlaps + ellipse_overlaps) / total_pairs
            # if heavy pairwise overlap detected, nudge global spacing hints in session for next candidates
            if overlap_ratio > 0.15:
                # store mild global spacing scale to be used implicitly next run
                st.session_state["_random_global_spacing_scale"] = min(1.25, float(st.session_state.get("_random_global_spacing_scale", 1.0)) * 1.05)
            else:
                st.session_state["_random_global_spacing_scale"] = float(st.session_state.get("_random_global_spacing_scale", 1.0)) * 0.995

        return designer.canvas.copy()

    def gen_candidate_with_density_bounds(max_attempts: int = 30):
        min_target = float(st.session_state.get("random_min_density_pct", 15)) / 100.0
        max_target = float(st.session_state.get("random_max_density_pct", 45)) / 100.0
        last_canvas = None
        for _ in range(max_attempts):
            canvas = gen_candidate_once()
            last_canvas = canvas
            if canvas is None or canvas.size == 0:
                continue
            density = np.sum(canvas == 0) / canvas.size
            if min_target <= density <= max_target:
                return canvas
        return last_canvas

    return [gen_candidate_with_density_bounds() for _ in range(2)]

def main():
    st.title("🌡️ Heat Exchanger 2D Designer (Beginner-Friendly)")
    st.markdown("""
    **Easy-to-use Heat Exchanger Design Tool**
    
    🎯 **Quick Start**: Choose a preset design below and modify as needed  
    📏 **Real Scale**: Set grid size (mm per unit) in the sidebar  
    🔧 **Advanced Options**: Expand the advanced section for custom designs
    """)
    
    # Initialize designer
    if 'designer' not in st.session_state:
        st.session_state.designer = HeatExchangerDesigner()
        # Apply default preset on first load
        st.session_state.designer.apply_preset_configuration("Hybrid Finned Cold Plate")
        st.session_state.first_load = True
    
    designer = st.session_state.designer
    
    # Sidebar controls
    with st.sidebar:
        st.header("🔧 Design Controls")
        
        # Reset button (moved to top like original code)
        rcol1, rcol2, rcol3 = st.columns([1, 1, 0.25])
        with rcol1:
            if st.button("🔄 Reset Canvas", type="primary"):
                # Exit random explore mode and clear candidates
                st.session_state.is_random_explore = False
                st.session_state.random_candidates = []
                designer.reset_canvas()
                st.rerun()
        with rcol2:
            if st.button("🎲 Random Explore", help="Generate 2 random candidates + 2 processed (flow-unblocked & smoothed)"):
                st.session_state.is_random_explore = True
                st.session_state.random_candidates = generate_random_candidates(designer)
                st.rerun()
        with rcol3:
            st.markdown(
                """
                <div style='margin-top: 6px;'>
                  <span title='Random Explore notes:\n- Generates 2 original candidates + 2 processed (flow-unblocked & smoothed)\n- Length/Width randomized by default (configurable below)\n- Length range: 200–500, Width range: 100–300\n- Ensures Length > Width and limits extreme aspect ratios\n- Uses current grid unit size (mm/unit)'
                        style='display:inline-block;width:22px;height:22px;line-height:22px;border-radius:50%;
                               background:#f3f3f3;text-align:center;border:1px solid #ccc;cursor:help;'>
                    ?
                  </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Random Explore Settings dropdown
        with st.expander("❔ Random Explore Settings", expanded=False):
            st.markdown("*Adjust essential options for Random Explore*")
            st.checkbox(
                "Randomize length and width",
                value=st.session_state.get("randomize_lw", False),
                key="randomize_lw",
                help="When checked, each Random Explore will randomize Length (200–500) and Width (100–300). If unchecked, current dimensions are used.",
            )
            st.selectbox(
                "Random preset mode",
                [
                    "Free mix",
                    "Needle + Circle (high heat transfer/mixing)",
                    "Rect + Circle (balanced grid + mixing)",
                    "Wave + Needle (complex flow + conduction)",
                    "Ellipse + Needle (smooth footprint + needles)",
                    "Rect + Ellipse + Circle (diverse mix)",
                ],
                index={
                    "Free mix": 0,
                    "Needle + Circle (high heat transfer/mixing)": 1,
                    "Rect + Circle (balanced grid + mixing)": 2,
                    "Wave + Needle (complex flow + conduction)": 3,
                    "Ellipse + Needle (smooth footprint + needles)": 4,
                    "Rect + Ellipse + Circle (diverse mix)": 5,
                }.get(st.session_state.get("random_preset_mode", "Free mix"), 0),
                key="random_preset_mode",
                help="Choose a curated combo or free mix with complexity settings.",
            )
            st.selectbox(
                "Complexity",
                ["Random", "Simple", "Medium", "Complex"],
                index={"Random":0, "Simple":1, "Medium":2, "Complex":3}.get(st.session_state.get("random_complexity", "Random"), 0),
                key="random_complexity",
                help="Random: weighted selection of 1 to all array types (biased towards medium). Simple: 1 type; Medium: 2 types; Complex: 3 types.",
            )
            st.multiselect(
                "Allowed types",
                ["needle", "circle", "wave", "rect", "ellipse"],
                default=st.session_state.get("random_allowed_types", ["needle", "circle", "wave", "rect", "ellipse"]),
                key="random_allowed_types",
                help="Limit array types allowed during Random Explore. Default: all.",
            )
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                min_d_pct = st.slider(
                    "Min target density (%)",
                    min_value=0,
                    max_value=80,
                    value=int(st.session_state.get("random_min_density_pct", 15)),
                    step=1,
                    key="random_min_density_pct",
                )
            with dcol2:
                max_d_pct = st.slider(
                    "Max target density (%)",
                    min_value=10,
                    max_value=90,
                    value=int(st.session_state.get("random_max_density_pct", 45)),
                    step=1,
                    key="random_max_density_pct",
                )
            st.checkbox(
                "Allow partial arrays",
                value=st.session_state.get("random_allow_partial", True),
                key="random_allow_partial",
                help="When enabled: each array is mostly complete with small chance of regular dropout, and a smaller chance of irregular dropout.",
            )
            st.checkbox(
                "Auto reduce density when arrays overlap too much",
                value=st.session_state.get("random_auto_reduce_overlap", True),
                key="random_auto_reduce_overlap",
                help="When enabled, if circle/rectangle/ellipse elements heavily overlap with each other, spacing and/or element size are adjusted to reduce pairwise overlaps.",
            )
            # Flow unblocking and smoothing controls
            st.slider(
                "Flow corridor count (if blocked)",
                min_value=1,
                max_value=8,
                value=int(st.session_state.get("random_flow_corridors", 4)),
                step=1,
                key="random_flow_corridors",
                help="Number of horizontal corridors to open only when the domain is not passable from left to right (checked in middle 80% height).",
            )
            st.slider(
                "Smoothing strength",
                min_value=1,
                max_value=15,
                value=int(st.session_state.get("random_smoothing_strength", 9)),
                step=1,
                key="random_smoothing_strength",
                help="Higher = much stronger edge smoothing applied to processed images.",
            )
            if st.button("🔄 Update", key="update_random_settings", type="secondary"):
                if st.session_state.get("is_random_explore", False):
                    st.session_state.random_candidates = generate_random_candidates(designer)
                    st.rerun()
        
        # Domain Settings Section
        st.subheader("📏 Domain Settings")
        # Grid unit size control (mm per grid unit) – half width like width/height inputs
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            grid_unit_mm = st.number_input(
                "Grid unit size (mm per unit)",
                min_value=0.01,
                max_value=10.0,
                value=float(designer.grid_size),
                step=0.01,
                format="%.3f",
            )
        with gcol2:
            st.write("")
        if grid_unit_mm != designer.grid_size:
            designer.grid_size = float(grid_unit_mm)
            st.rerun()
        st.markdown(f"*Grid units (1 unit = {designer.grid_size:.3f} mm)*")
        
        col1, col2 = st.columns(2)
        with col1:
            domain_width = st.number_input("Length (grid units)", 
                                         min_value=100, max_value=800, 
                                         value=300, step=10)
        with col2:
            domain_height = st.number_input("Width (grid units)", 
                                          min_value=50, max_value=400, 
                                          value=200, step=10)
        
        # Calculate and display real dimensions
        real_width = domain_width * designer.grid_size
        real_height = domain_height * designer.grid_size
        
        st.info(f"""
        **Real Dimensions:**
        - Length: {real_width:.1f} mm ({real_width/10:.1f} cm)
        - Width: {real_height:.1f} mm ({real_height/10:.1f} cm)
        - Area: {(real_width * real_height)/100:.1f} cm²
        """)
        
        # Update domain size if changed
        if domain_width != designer.width or domain_height != designer.height:
            designer.width = domain_width
            designer.height = domain_height
            designer.reset_canvas()
            st.rerun()
        
        st.divider()
        # Preset Configurations (moved to top)
        st.subheader("🎨 Quick Design Presets")
        st.markdown("*Start with a preset and customize*")
        
        preset_options = [
            "Hybrid Finned Cold Plate",
            "Parallel Needle Array", 
            "Staggered Circle Array", 
            "Wave Heat Sink", 
            "Rectangular Grid"
        ]
        
        preset = st.selectbox(
            "Select Design Preset",
            preset_options,
            index=0  # Default to first option
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎯 Apply Preset", type="primary"):
                st.session_state.is_random_explore = False
                st.session_state.random_candidates = []
                designer.apply_preset_configuration(preset)
                st.rerun()
        with col2:
            if preset == "Hybrid Finned Cold Plate":
                if st.button("🔄 Update Walls", type="secondary", key="quick_update"):
                    st.session_state.is_random_explore = False
                    # Quick update using current slider values from session state
                    designer.reset_canvas()
                    
                    # Get current slider values from session state, or use defaults
                    center_angle = st.session_state.get("center_angle", 20)
                    center_length = st.session_state.get("center_length", 60)
                    center_thickness = st.session_state.get("center_thickness", 2)
                    
                    outer_angle = st.session_state.get("outer_angle", 60)
                    outer_length = st.session_state.get("outer_length", 60)
                    outer_thickness = st.session_state.get("outer_thickness", 2)
                    
                    outlet_angle = st.session_state.get("outlet_angle", 18)
                    outlet_length = st.session_state.get("outlet_length", 170)
                    outlet_thickness = st.session_state.get("outlet_thickness", 3)
                    
                    designer.add_inlet_center_walls(angle=center_angle, length=center_length, thickness=center_thickness)
                    designer.add_inlet_outer_walls(angle=outer_angle, length=outer_length, thickness=outer_thickness)
                    designer.add_outlet_walls(angle=outlet_angle, length=outlet_length, thickness=outlet_thickness)
                    
                    # Also add horizontal fins with current settings
                    fin_length = st.session_state.get("fin_length", 120)
                    fin_thickness = st.session_state.get("fin_thickness", 3)
                    fin_spacing = st.session_state.get("fin_spacing", 5)
                    fin_count = st.session_state.get("fin_count", 7)
                    
                    designer.add_horizontal_fins(fin_length=fin_length, fin_thickness=fin_thickness, 
                                               fin_spacing=fin_spacing, fin_count=fin_count)
                    
                    # Finally clear outlet walls (right side + directional areas)
                    designer.clear_outlet_walls_right_side(
                        angle=outlet_angle, length=outlet_length, thickness=outlet_thickness
                    )
                    designer.clear_outlet_walls_upward(
                        angle=outlet_angle, length=outlet_length, thickness=outlet_thickness, clear_height=4
                    )
                    
                    # Add circular fins after clearing (Update Walls button)
                    circle_diameter = st.session_state.get("circle_diameter", 2)
                    h_spacing = st.session_state.get("h_spacing", 15)
                    v_spacing = st.session_state.get("v_spacing", 15)
                    min_distance = st.session_state.get("min_distance", 3)
                    
                    designer.add_circular_fins_array(diameter=circle_diameter, h_spacing=h_spacing, 
                                                   v_spacing=v_spacing, min_distance=min_distance)
                    st.rerun()
        
        # Show preset description
        preset_descriptions = {
            "Hybrid Finned Cold Plate": "Advanced cold plate with flow guide walls",
            "Parallel Needle Array": "Parallel needle fins for high heat transfer",
            "Staggered Circle Array": "Staggered circular fins for enhanced mixing",
            "Wave Heat Sink": "Wavy fins for complex flow patterns",
            "Mixed Configuration": "Combination of circles and needles",
            "Rectangular Grid": "Simple rectangular fin grid"
        }
        
        if preset in preset_descriptions:
            st.markdown(f"*{preset_descriptions[preset]}*")
        
        # Flow Guide Wall Controls (for Hybrid Finned Cold Plate)
        if preset == "Hybrid Finned Cold Plate":
            with st.expander("🌊 Flow Guide Wall Settings", expanded=False):
                st.markdown("**Control 3 categories of flow guide walls:**")
                
                # Category 1: Inlet Center Walls (middle pair)
                st.markdown("**1️⃣ Inlet Center Walls** *(inner pair)*")
                col1, col2 = st.columns(2)
                with col1:
                    center_angle = st.slider("Center Walls Angle", 5, 45, 20, key="center_angle", 
                                           help="Control the angle of inner pair flow guide walls relative to horizontal")
                with col2:
                    center_length = st.slider("Center Walls Length", 30, 100, 60, key="center_length")
                    center_thickness = st.slider("Center Walls Thickness", 1, 6, 2, key="center_thickness")
                
                # Category 2: Inlet Outer Walls (outer pair)
                st.markdown("**2️⃣ Inlet Outer Walls** *(outer pair)*")
                col1, col2 = st.columns(2)
                with col1:
                    outer_angle = st.slider("Outer Walls Angle", 30, 80, 60, key="outer_angle",
                                          help="Control the angle of outer pair flow guide walls relative to horizontal")
                with col2:
                    outer_length = st.slider("Outer Walls Length", 30, 100, 60, key="outer_length")
                    outer_thickness = st.slider("Outer Walls Thickness", 1, 6, 2, key="outer_thickness")
                
                # Category 3: Outlet Walls
                st.markdown("**3️⃣ Outlet Walls** *(converging outlet walls)*")
                col1, col2 = st.columns(2)
                with col1:
                    outlet_angle = st.slider("Outlet Walls Angle", 5, 30, 18, key="outlet_angle",
                                           help="Control the angle of outlet converging walls relative to horizontal")
                with col2:
                    outlet_length = st.slider("Outlet Walls Length", 80, 200, 170, key="outlet_length")
                    outlet_thickness = st.slider("Outlet Walls Thickness", 2, 8, 3, key="outlet_thickness")
                
                if st.button("🔄 Update Flow Walls", key="update_walls"):
                    designer.reset_canvas()
                    # Apply flow walls
                    designer.add_inlet_center_walls(
                        angle=center_angle, length=center_length, thickness=center_thickness
                    )
                    designer.add_inlet_outer_walls(
                        angle=outer_angle, length=outer_length, thickness=outer_thickness
                    )
                    designer.add_outlet_walls(
                        angle=outlet_angle, length=outlet_length, thickness=outlet_thickness
                    )
                    
                    # Also re-add horizontal fins with current settings
                    fin_length = st.session_state.get("fin_length", 120)
                    fin_thickness = st.session_state.get("fin_thickness", 3)
                    fin_spacing = st.session_state.get("fin_spacing", 5)
                    fin_count = st.session_state.get("fin_count", 7)
                    
                    designer.add_horizontal_fins(fin_length=fin_length, fin_thickness=fin_thickness, 
                                               fin_spacing=fin_spacing, fin_count=fin_count)
                    
                    # Finally clear outlet walls (right side + directional areas)
                    designer.clear_outlet_walls_right_side(
                        angle=outlet_angle, length=outlet_length, thickness=outlet_thickness
                    )
                    designer.clear_outlet_walls_upward(
                        angle=outlet_angle, length=outlet_length, thickness=outlet_thickness, clear_height=4
                    )
                    
                    # Add circular fins after clearing (Update Flow Walls button)
                    circle_diameter = st.session_state.get("circle_diameter", 2)
                    h_spacing = st.session_state.get("h_spacing", 15)
                    v_spacing = st.session_state.get("v_spacing", 15)
                    min_distance = st.session_state.get("min_distance", 3)
                    
                    designer.add_circular_fins_array(diameter=circle_diameter, h_spacing=h_spacing, 
                                                   v_spacing=v_spacing, min_distance=min_distance)
                    st.rerun()
        
        # Horizontal Fins Controls (dropdown expander)
        if preset == "Hybrid Finned Cold Plate":
            with st.expander("🔲 Horizontal Fins", expanded=False):
                st.markdown("*Right end fixed to outlet wall position*")
                
                col1, col2 = st.columns(2)
                with col1:
                    fin_length = st.slider("Length", 50, 200, 120, key="fin_length")
                    fin_thickness = st.slider("Thickness", 2, 6, 3, key="fin_thickness")
                with col2:
                    fin_spacing = st.slider("Spacing", 1, 10, 5, key="fin_spacing")
                    fin_count = st.slider("Fin Count", 2, 10,7, key="fin_count")
                
                if st.button("🔄 Update Horizontal Fins", key="update_fins"):
                    designer.reset_canvas()
                    # Apply flow walls first
                    center_angle = st.session_state.get("center_angle", 20)
                    center_length = st.session_state.get("center_length", 60)
                    center_thickness = st.session_state.get("center_thickness", 2)
                    
                    outer_angle = st.session_state.get("outer_angle", 60)
                    outer_length = st.session_state.get("outer_length", 60)
                    outer_thickness = st.session_state.get("outer_thickness", 2)
                    
                    outlet_angle = st.session_state.get("outlet_angle", 18)
                    outlet_length = st.session_state.get("outlet_length", 170)
                    outlet_thickness = st.session_state.get("outlet_thickness", 3)
                    
                    designer.add_inlet_center_walls(angle=center_angle, length=center_length, thickness=center_thickness)
                    designer.add_inlet_outer_walls(angle=outer_angle, length=outer_length, thickness=outer_thickness)
                    designer.add_outlet_walls(angle=outlet_angle, length=outlet_length, thickness=outlet_thickness)
                    
                    # Then add horizontal fins
                    designer.add_horizontal_fins(fin_length=fin_length, fin_thickness=fin_thickness, 
                                               fin_spacing=fin_spacing, fin_count=fin_count)
                    
                    # Finally clear outlet walls (right side + directional areas)
                    designer.clear_outlet_walls_right_side(
                        angle=outlet_angle, length=outlet_length, thickness=outlet_thickness
                    )
                    designer.clear_outlet_walls_upward(
                        angle=outlet_angle, length=outlet_length, thickness=outlet_thickness, clear_height=4
                    )
                    
                    # Add circular fins after clearing (Update Horizontal Fins button)
                    circle_diameter = st.session_state.get("circle_diameter", 2)
                    h_spacing = st.session_state.get("h_spacing", 15)
                    v_spacing = st.session_state.get("v_spacing", 15)
                    min_distance = st.session_state.get("min_distance", 3)
                    
                    designer.add_circular_fins_array(diameter=circle_diameter, h_spacing=h_spacing, 
                                                   v_spacing=v_spacing, min_distance=min_distance)
                    st.rerun()
        
        # Circular Fins Controls (dropdown expander)  
        if preset == "Hybrid Finned Cold Plate":
            with st.expander("🟡 Circular Fins", expanded=False):
                st.markdown("*Circular fins array in middle region with edge margins*")
                st.markdown("*Auto-removes circles too close to walls/fins*")
                
                col1, col2 = st.columns(2)
                with col1:
                    circle_diameter = st.slider("Diameter", 2, 16, 2, key="circle_diameter")
                    h_spacing = st.slider("Horizontal Spacing", 5, 30, 15, key="h_spacing")
                with col2:
                    v_spacing = st.slider("Vertical Spacing", 5, 30, 15, key="v_spacing")
                    min_distance = st.slider("Min Distance to Structures", 1, 10, 3, key="min_distance",
                                           help="Minimum distance from circles to walls/fins")
                
                if st.button("🔄 Update Circular Fins", key="update_circles"):
                    designer.reset_canvas()
                    # Apply all components in order
                    center_angle = st.session_state.get("center_angle", 20)
                    center_length = st.session_state.get("center_length", 60)
                    center_thickness = st.session_state.get("center_thickness", 2)
                    
                    outer_angle = st.session_state.get("outer_angle", 60)
                    outer_length = st.session_state.get("outer_length", 60)
                    outer_thickness = st.session_state.get("outer_thickness", 2)
                    
                    outlet_angle = st.session_state.get("outlet_angle", 18)
                    outlet_length = st.session_state.get("outlet_length", 170)
                    outlet_thickness = st.session_state.get("outlet_thickness", 3)
                    
                    fin_length = st.session_state.get("fin_length", 120)
                    fin_thickness = st.session_state.get("fin_thickness", 3)
                    fin_spacing = st.session_state.get("fin_spacing", 5)
                    fin_count = st.session_state.get("fin_count", 7)
                    
                    designer.add_inlet_center_walls(angle=center_angle, length=center_length, thickness=center_thickness)
                    designer.add_inlet_outer_walls(angle=outer_angle, length=outer_length, thickness=outer_thickness)
                    designer.add_outlet_walls(angle=outlet_angle, length=outlet_length, thickness=outlet_thickness)
                    designer.add_horizontal_fins(fin_length=fin_length, fin_thickness=fin_thickness, 
                                               fin_spacing=fin_spacing, fin_count=fin_count)
                    designer.clear_outlet_walls_right_side(angle=outlet_angle, length=outlet_length, thickness=outlet_thickness)
                    designer.clear_outlet_walls_upward(angle=outlet_angle, length=outlet_length, thickness=outlet_thickness, clear_height=4)
                    
                    # Add circular fins
                    designer.add_circular_fins_array(diameter=circle_diameter, h_spacing=h_spacing, 
                                                   v_spacing=v_spacing, min_distance=min_distance)
                    st.rerun()
        
        # Other Preset Controls
        if preset == "Parallel Needle Array":
            with st.expander("🔹 Parallel Needle Settings", expanded=False):
                st.markdown("**Adjust parallel needle fin parameters**")
                
                needle_length = st.slider("Needle Length", min_value=15, max_value=60, value=30, key="needle_length")
                needle_thickness = st.slider("Needle Thickness", min_value=1, max_value=8, value=3, key="needle_thickness") 
                needle_h_spacing = st.slider("Horizontal Spacing", min_value=10, max_value=40, value=20, key="needle_h_spacing")
                needle_v_spacing = st.slider("Vertical Spacing", min_value=15, max_value=50, value=25, key="needle_v_spacing")
                
                if st.button("🔄 Update Needle Array", type="secondary"):
                    designer.reset_canvas()
                    designer.add_parallel_needle_array(length=needle_length, thickness=needle_thickness, 
                                                     h_spacing=needle_h_spacing, v_spacing=needle_v_spacing)
                    st.rerun()
        
        elif preset == "Staggered Circle Array":
            with st.expander("🟢 Staggered Circle Settings", expanded=False):
                st.markdown("**Adjust staggered circular fin parameters**")
                
                circle_radius = st.slider("Circle Radius", min_value=4, max_value=20, value=8, key="stagger_radius")
                stagger_h_spacing = st.slider("Horizontal Spacing", min_value=15, max_value=50, value=25, key="stagger_h_spacing")
                stagger_v_spacing = st.slider("Vertical Spacing", min_value=10, max_value=40, value=20, key="stagger_v_spacing")
                
                if st.button("🔄 Update Circle Array", type="secondary"):
                    designer.reset_canvas()
                    designer.add_staggered_circle_array(radius=circle_radius, h_spacing=stagger_h_spacing, 
                                                       v_spacing=stagger_v_spacing)
                    st.rerun()
        
        elif preset == "Wave Heat Sink":
            with st.expander("🌊 Wave Heat Sink Settings", expanded=False):
                st.markdown("**Adjust wave pattern parameters**")
                
                wave_wavelength = st.slider("Wavelength", min_value=20, max_value=80, value=40, key="wave_wavelength")
                wave_amplitude = st.slider("Wave Amplitude", min_value=4, max_value=20, value=8, key="wave_amplitude")
                wave_thickness = st.slider("Wave Thickness", min_value=1, max_value=8, value=3, key="wave_thickness")
                wave_spacing = st.slider("Row Spacing", min_value=15, max_value=60, value=30, key="wave_spacing")
                
                if st.button("🔄 Update Wave Pattern", type="secondary"):
                    designer.reset_canvas()
                    designer.add_wave_heat_sink_array(wavelength=wave_wavelength, amplitude=wave_amplitude, 
                                                     thickness=wave_thickness, spacing=wave_spacing)
                    st.rerun()
        
        elif preset == "Rectangular Grid":
            with st.expander("⬜ Rectangular Grid Settings", expanded=False):
                st.markdown("**Adjust rectangular grid parameters**")
                
                rect_width = st.slider("Rectangle Width", min_value=8, max_value=30, value=15, key="rect_width")
                rect_height = st.slider("Rectangle Height", min_value=10, max_value=40, value=20, key="rect_height")
                rect_h_spacing = st.slider("Horizontal Spacing", min_value=15, max_value=50, value=25, key="rect_h_spacing")
                rect_v_spacing = st.slider("Vertical Spacing", min_value=20, max_value=60, value=30, key="rect_v_spacing")
                
                if st.button("🔄 Update Grid Pattern", type="secondary"):
                    designer.reset_canvas()
                    designer.add_rectangular_grid_array(width=rect_width, height=rect_height, 
                                                       h_spacing=rect_h_spacing, v_spacing=rect_v_spacing)
                    st.rerun()
        
        st.divider()
        # Advanced Design Controls as independent module
        st.subheader("🔧 Advanced Design (High Freedom)")
        with st.expander("Manual Fin Design Controls", expanded=False):
            # Fin type selection
            fin_type = st.selectbox(
                "Fin Type",
                ["Rectangular Fin", "Needle Fin", "Circular Fin", "Elliptical Fin", "L-shaped Fin", "Wave Fin", "Fin Array"],
                key="advanced_fin_type"
            )

            st.markdown(f"**📐 {fin_type} Parameters**")

            # Common parameters
            col1, col2 = st.columns(2)
            with col1:
                x_pos = st.slider("X Position", 0, domain_width, domain_width//2, key="adv_x")
            with col2:
                y_pos = st.slider("Y Position", 0, domain_height, domain_height//2, key="adv_y")

            # Display different parameters based on fin type
            if fin_type == "Rectangular Fin":
                col1, col2 = st.columns(2)
                with col1:
                    width = st.slider("Width", 5, 100, 30, key="adv_rect_w")
                with col2:
                    height = st.slider("Height", 5, 100, 60, key="adv_rect_h")
                angle = st.slider("Rotation Angle", -180, 180, 0, key="adv_rect_angle")

                if st.button("➕ Add Rectangular Fin", key="add_rect"):
                    designer.add_rectangular_fin(x_pos, y_pos, width, height, angle)
                    st.rerun()

            elif fin_type == "Needle Fin":
                col1, col2 = st.columns(2)
                with col1:
                    length = st.slider("Length", 10, 150, 80, key="adv_needle_l")
                with col2:
                    thickness = st.slider("Thickness", 2, 20, 5, key="adv_needle_t")
                angle = st.slider("Angle", -180, 180, 0, key="adv_needle_angle")

                if st.button("➕ Add Needle Fin", key="add_needle"):
                    designer.add_needle_fin(x_pos, y_pos, length, thickness, angle)
                    st.rerun()

            elif fin_type == "Circular Fin":
                radius = st.slider("Radius", 5, 80, 20, key="adv_circle_r")

                if st.button("➕ Add Circular Fin", key="add_circle"):
                    designer.add_circular_fin(x_pos, y_pos, radius)
                    st.rerun()

            elif fin_type == "Elliptical Fin":
                col1, col2 = st.columns(2)
                with col1:
                    a_axis = st.slider("Major Axis", 10, 100, 40, key="adv_ellipse_a")
                with col2:
                    b_axis = st.slider("Minor Axis", 5, 80, 20, key="adv_ellipse_b")
                angle = st.slider("Rotation Angle", -180, 180, 0, key="adv_ellipse_angle")

                if st.button("➕ Add Elliptical Fin", key="add_ellipse"):
                    designer.add_elliptical_fin(x_pos, y_pos, a_axis, b_axis, angle)
                    st.rerun()

            elif fin_type == "L-shaped Fin":
                st.markdown("**Horizontal Part**")
                col1, col2 = st.columns(2)
                with col1:
                    length1 = st.slider("Horizontal Length", 10, 100, 50, key="adv_l_l1")
                with col2:
                    width1 = st.slider("Horizontal Width", 5, 30, 10, key="adv_l_w1")

                st.markdown("**Vertical Part**")
                col1, col2 = st.columns(2)
                with col1:
                    length2 = st.slider("Vertical Length", 10, 100, 40, key="adv_l_l2")
                with col2:
                    width2 = st.slider("Vertical Width", 5, 30, 10, key="adv_l_w2")

                angle = st.slider("Overall Rotation", -180, 180, 0, key="adv_l_angle")

                if st.button("➕ Add L-shaped Fin", key="add_l"):
                    designer.add_l_shaped_fin(x_pos, y_pos, length1, width1, length2, width2, angle)
                    st.rerun()

            elif fin_type == "Wave Fin":
                col1, col2 = st.columns(2)
                with col1:
                    wave_length = st.slider("Wave Length", 20, 150, 80, key="adv_wave_l")
                    amplitude = st.slider("Amplitude", 5, 50, 20, key="adv_wave_a")
                with col2:
                    frequency = st.slider("Frequency", 0.5, 5.0, 2.0, 0.1, key="adv_wave_f")
                    thickness = st.slider("Thickness", 2, 15, 6, key="adv_wave_t")

                if st.button("➕ Add Wave Fin", key="add_wave"):
                    designer.add_wave_fin(x_pos, y_pos, wave_length, amplitude, frequency, thickness)
                    st.rerun()

            elif fin_type == "Fin Array":
                array_type = st.selectbox("Array Type", ["rectangular", "needle", "circular"], key="adv_array_type")

                col1, col2 = st.columns(2)
                with col1:
                    rows = st.slider("Rows", 1, 10, 3, key="adv_array_rows")
                    spacing_x = st.slider("X Spacing", 10, 80, 40, key="adv_array_sx")
                with col2:
                    cols = st.slider("Columns", 1, 10, 3, key="adv_array_cols")
                    spacing_y = st.slider("Y Spacing", 10, 80, 40, key="adv_array_sy")

                # Display parameters based on array type
                if array_type == "rectangular":
                    col1, col2 = st.columns(2)
                    with col1:
                        arr_width = st.slider("Element Width", 5, 50, 15, key="adv_array_w")
                    with col2:
                        arr_height = st.slider("Element Height", 5, 50, 25, key="adv_array_h")
                    fin_params = {"width": arr_width, "height": arr_height}
                elif array_type == "needle":
                    col1, col2 = st.columns(2)
                    with col1:
                        arr_length = st.slider("Element Length", 10, 60, 30, key="adv_array_l")
                    with col2:
                        arr_thickness = st.slider("Element Thickness", 2, 15, 5, key="adv_array_t")
                    fin_params = {"length": arr_length, "thickness": arr_thickness}
                else:  # circular
                    arr_radius = st.slider("Element Radius", 3, 25, 8, key="adv_array_r")
                    fin_params = {"radius": arr_radius}

                if st.button("➕ Add Fin Array", key="add_array"):
                    designer.add_fin_array(x_pos, y_pos, array_type, spacing_x, spacing_y, 
                                         rows, cols, **fin_params)
                    st.rerun()
        
        st.divider()
        # Export functionality
        st.subheader("💾 Export")
        # Z-direction layers control
        zcol1, zcol2 = st.columns(2)
        with zcol1:
            nz_layers = st.number_input(
                "Thickness (layers)",
                min_value=1,
                max_value=512,
                value=int(designer.nz_layers),
                step=1
            )
        with zcol2:
            st.markdown(f"<div style='margin-top: 28px;'>Thickness: {nz_layers * designer.grid_size:.3f} mm</div>", unsafe_allow_html=True)
        if nz_layers != designer.nz_layers:
            designer.nz_layers = int(nz_layers)

        # Add solid boundary shell (1-cell thick) option
        st.checkbox(
            "Add solid boundary shell (1-cell thick)",
            value=st.session_state.get("export_add_boundary", True),
            key="export_add_boundary",
            help="When enabled, the outermost cells on all 6 faces (±X, ±Y, ±Z) are set to solid to form a closed boundary."
        )

        if st.button("📁 Export LBM Geometry (.dat & .vtk & .stl)"):
            try:
                geometry_3d = designer.export_geometry("geometry.dat")
                # Optionally set the outermost layer to solid on top/bottom (±Y) and front/back (±Z) only (no left/right)
                if st.session_state.get("export_add_boundary", True) and geometry_3d.size > 0:
                    gy, gx, gz = geometry_3d.shape
                    g = geometry_3d.copy()
                    # ±Y faces (top/bottom)
                    g[0, :, :] = 1
                    g[gy - 1, :, :] = 1
                    # ±Z faces (front/back along thickness)
                    g[:, :, 0] = 1
                    g[:, :, gz - 1] = 1
                    geometry_3d = g
                
                # Prepare .dat buffer (use same flattening order)
                dat_buffer = BytesIO()
                np.savetxt(dat_buffer, designer._flatten_for_dat(geometry_3d), fmt='%d')
                
                # Prepare .vtk buffer
                vtk_buffer = designer.export_geometry_vtk_to_buffer(
                    lbm_3d=geometry_3d,
                    spacing_mm=(designer.grid_size, designer.grid_size, designer.grid_size)
                )
                
                col_db1, col_db2, col_db3 = st.columns(3)
                with col_db1:
                    st.download_button(
                        label="⬇️ Download geometry.dat",
                        data=dat_buffer.getvalue(),
                        file_name="geometry.dat",
                        mime="text/plain"
                    )
                with col_db2:
                    st.download_button(
                        label="⬇️ Download geometry.vtk",
                        data=vtk_buffer.getvalue(),
                        file_name="geometry.vtk",
                        mime="application/vnd.vtk"
                    )
                with col_db3:
                    stl_buffer = designer.export_geometry_stl_to_buffer(
                        lbm_3d=geometry_3d,
                        spacing_mm=(designer.grid_size, designer.grid_size, designer.grid_size)
                    )
                    st.download_button(
                        label="⬇️ Download geometry.stl",
                        data=stl_buffer.getvalue(),
                        file_name="geometry.stl",
                        mime="application/sla"
                    )
                
                # Display statistics
                solid_cells = np.sum(geometry_3d == 1)
                fluid_cells = np.sum(geometry_3d == 0)
                total_cells = geometry_3d.size
                
                st.success(f"""
                ✅ Export successful!
                - Total cells: {total_cells:,}
                - Solid cells: {solid_cells:,} ({solid_cells/total_cells*100:.1f}%)
                - Fluid cells: {fluid_cells:,} ({fluid_cells/total_cells*100:.1f}%)
                """)
                
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    
    # Main display area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.session_state.get("is_random_explore", False) and st.session_state.get("random_candidates"):
            candidates = st.session_state.get("random_candidates", [])[:2]
            # compute processed variants (flow-unblock + smoothing)
            originals = candidates
            processed = []

            def has_left_right_path(canvas_np: np.ndarray) -> bool:
                # BFS on fluid cells (1) from left to right but only within middle 80% height window
                H, W = canvas_np.shape
                from collections import deque
                y0 = int(H * 0.10)
                y1 = int(H * 0.90)
                visited = np.zeros_like(canvas_np, dtype=bool)
                q = deque()
                for y in range(y0, y1):
                    if canvas_np[y, 0] == 1:
                        q.append((y, 0))
                        visited[y, 0] = True
                dirs = [(1,0),(-1,0),(0,1),(0,-1)]
                while q:
                    y, x = q.popleft()
                    if x == W - 1:
                        return True
                    for dy, dx in dirs:
                        ny, nx = y + dy, x + dx
                        if y0 <= ny < y1 and 0 <= nx < W and not visited[ny, nx] and canvas_np[ny, nx] == 1:
                            visited[ny, nx] = True
                            q.append((ny, nx))
                return False

            def strong_smooth(canvas_np: np.ndarray) -> np.ndarray:
                # Strong, very visible smoothing using multiple majority passes and a 5x5 pass
                H, W = canvas_np.shape
                out = canvas_np.copy()
                # several 3x3 majority passes
                for _ in range(2):
                    new_out = out.copy()
                    for y in range(1, H-1):
                        for x in range(1, W-1):
                            block = out[y-1:y+2, x-1:x+2]
                            solid_count = np.sum(block == 0)
                            fluid_count = 9 - solid_count
                            if solid_count >= 6:
                                new_out[y, x] = 0
                            elif fluid_count >= 6:
                                new_out[y, x] = 1
                    out = new_out
                # one 5x5 pass for stronger smoothing
                new_out = out.copy()
                for y in range(2, H-2):
                    for x in range(2, W-2):
                        block5 = out[y-2:y+3, x-2:x+3]
                        solid_count = np.sum(block5 == 0)
                        if solid_count >= 15:
                            new_out[y, x] = 0
                        elif solid_count <= 10:
                            new_out[y, x] = 1
                out = new_out
                # final 3x3 settle
                new_out = out.copy()
                for y in range(1, H-1):
                    for x in range(1, W-1):
                        block = out[y-1:y+2, x-1:x+2]
                        solid_count = np.sum(block == 0)
                        fluid_count = 9 - solid_count
                        if solid_count >= 6:
                            new_out[y, x] = 0
                        elif fluid_count >= 6:
                            new_out[y, x] = 1
                return new_out

            def open_gaps(canvas_np: np.ndarray) -> np.ndarray:
                # Create multiple horizontal corridors if blocked (count controlled by settings)
                H, W = canvas_np.shape
                out = canvas_np.copy()
                if has_left_right_path(out):
                    return out
                bands = int(st.session_state.get("random_flow_corridors", 4))
                corridor_h = max(2, H // 150 + 2)
                centers = np.linspace(int(H*0.25), int(H*0.75), bands, dtype=int)
                for cy in centers:
                    top = max(1, cy - corridor_h//2)
                    bot = min(H-2, top + corridor_h - 1)
                    out[top:bot+1, 1:W-2] = 1
                # if still blocked, add an extra corridor near mid
                if not has_left_right_path(out):
                    mid = H // 2
                    top = max(1, mid - corridor_h)
                    bot = min(H-2, mid + corridor_h)
                    out[top:bot+1, 1:W-2] = 1
                return out

            for c in originals:
                # Step 1: ensure left-right flow path with multiple corridors
                c1 = open_gaps(c)
                # Step 2: apply very strong smoothing (intensity from settings)
                strength = int(st.session_state.get("random_smoothing_strength", 9))
                # run multiple rounds of strong smoothing to amplify effect
                c2 = c1
                for _ in range(max(1, strength)):
                    c2 = strong_smooth(c2)
                processed.append(c2)

            # store processed for stats
            st.session_state["random_candidates_processed"] = processed

            st.subheader("🖼️ Random Explore — Left: Originals, Right: Flow-unblocked + Strong smoothing applied")
            left_col, right_col = st.columns(2)
            with left_col:
                st.markdown("##### Originals")
                for i, canv in enumerate(originals, start=1):
                    fig_i, ax_i = plt.subplots(figsize=(6, 3))
                    ax_i.imshow(canv, cmap='gray', vmin=0, vmax=1, origin='lower')
                    ax_i.set_xticks([])
                    ax_i.set_yticks([])
                    ax_i.set_title(f"Original #{i}")
                    st.pyplot(fig_i)
                    dens = float(np.sum(canv == 0)) / canv.size if canv.size > 0 else 0.0
                    poro = (1.0 - dens) * 100.0
                    ok = has_left_right_path(canv)
                    st.caption(f"Unmodified random candidate.  Porosity: {poro:.1f}% | Passable (L→R): {'Yes' if ok else 'No'}")
            with right_col:
                st.markdown("##### Processed")
                for i, canv in enumerate(processed, start=1):
                    fig_i, ax_i = plt.subplots(figsize=(6, 3))
                    ax_i.imshow(canv, cmap='gray', vmin=0, vmax=1, origin='lower')
                    ax_i.set_xticks([])
                    ax_i.set_yticks([])
                    ax_i.set_title(f"Processed #{i}")
                    st.pyplot(fig_i)
                    dens = float(np.sum(canv == 0)) / canv.size if canv.size > 0 else 0.0
                    poro = (1.0 - dens) * 100.0
                    ok = has_left_right_path(canv)
                    st.caption(f"Flow-unblocked + strong smoothing.  Porosity: {poro:.1f}% | Passable (L→R): {'Yes' if ok else 'No'}")
        else:
            st.subheader("🖼️ Design Preview")
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Display canvas (white background, black fins)
            canvas_display = designer.get_canvas()
            im = ax.imshow(canvas_display, cmap='gray', vmin=0, vmax=1, origin='lower')
            
            # Add grid (default enabled)
            show_grid = st.checkbox("Show Grid", value=True)  # Default True
            if show_grid:
                ax.grid(True, alpha=0.3, linewidth=0.5)
                # Set grid ticks based on domain size
                x_ticks = np.arange(0, designer.width + 1, max(1, designer.width // 20))
                y_ticks = np.arange(0, designer.height + 1, max(1, designer.height // 10))
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
            
            ax.set_xlabel("Length (grid units)")
            ax.set_ylabel("Width (grid units)") 
            ax.set_title(f"Heat Exchanger 2D Design (White=Fluid, Black=Fins) | Real size (L×W): {real_width:.1f}×{real_height:.1f} mm")
            
            # Set coordinate axes
            ax.set_xlim(0, designer.width)
            ax.set_ylim(0, designer.height)
            
            st.pyplot(fig)
    
    with col2:
        # Prepare stats
        canvas = designer.get_canvas()
        solid_pixels = np.sum(canvas == 0)
        fluid_pixels = np.sum(canvas == 1)

        # Show 3D model image and academic reference only for Hybrid Finned Cold Plate when canvas is not empty
        if (
            preset == "Hybrid Finned Cold Plate"
            and not st.session_state.get("is_random_explore", False)
            and solid_pixels > 0
        ):
            st.markdown("#### 🏗️ 3D Model Reference")
            try:
                st.image(r"C:\\Users\\13497\\OneDrive\\Desktop\\1.png", 
                        caption="3D Model of Hybrid Finned Cold Plate with Flow Regulation",
                        width=300)
                st.markdown("*Three-dimensional visualization showing flow guide walls, horizontal fins, and circular fins integration*")
            except FileNotFoundError:
                st.warning("📋 3D model image not found. Please ensure the file exists at the specified path.")
            st.markdown("---")
        
        # Stats panel: in Random Explore mode we do not show per-candidate scores anymore
        if st.session_state.get("is_random_explore", False) and st.session_state.get("random_candidates"):
            st.markdown("#### 📊 Stats")
            originals = candidates
            processed = st.session_state.get("random_candidates_processed", [])
            def _pp(canvas_np: np.ndarray):
                if canvas_np is None or canvas_np.size == 0:
                    return 0.0, False
                dens = float(np.sum(canvas_np == 0)) / canvas_np.size
                poro = (1.0 - dens) * 100.0
                ok = has_left_right_path(canvas_np)
                return poro, ok
            if len(originals) == 2:
                p1, ok1 = _pp(originals[0])
                p2, ok2 = _pp(originals[1])
                st.markdown(f"Original #1 — Porosity: {p1:.1f}% | Passable: {'Yes' if ok1 else 'No'}")
                st.markdown(f"Original #2 — Porosity: {p2:.1f}% | Passable: {'Yes' if ok2 else 'No'}")
            if len(processed) == 2:
                p3, ok3 = _pp(processed[0])
                p4, ok4 = _pp(processed[1])
                st.markdown(f"Processed #1 — Porosity: {p3:.1f}% | Passable: {'Yes' if ok3 else 'No'}")
                st.markdown(f"Processed #2 — Porosity: {p4:.1f}% | Passable: {'Yes' if ok4 else 'No'}")
        else:
            st.markdown("#### 📊 Stats")
            total_pixels = canvas.size
            real_area = (real_width * real_height) / 100
            solid_area_ratio = solid_pixels / total_pixels if total_pixels > 0 else 0
            solid_real_area = real_area * solid_area_ratio
            density = solid_pixels / total_pixels if total_pixels > 0 else 0

            # Passability: whether there exists a fluid path from left to right
            def _has_left_right_path(canvas_np: np.ndarray) -> bool:
                H, W = canvas_np.shape
                from collections import deque
                visited = np.zeros_like(canvas_np, dtype=bool)
                q = deque()
                for y in range(H):
                    if canvas_np[y, 0] == 1:
                        q.append((y, 0))
                        visited[y, 0] = True
                dirs = [(1,0),(-1,0),(0,1),(0,-1)]
                while q:
                    y, x = q.popleft()
                    if x == W - 1:
                        return True
                    for dy, dx in dirs:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and canvas_np[ny, nx] == 1:
                            visited[ny, nx] = True
                            q.append((ny, nx))
                return False
            passable_lr = _has_left_right_path(canvas)
            st.markdown(f"""
            <div style="font-size: 12px;">
            <b>Porosity: {(1-density)*100:.1f}%</b><br>
            Density: {density*100:.1f}%<br>
            Passable (L→R): {'Yes' if passable_lr else 'No'}<br>
            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Compact legend
        st.markdown("#### 🎨 Legend")
        st.markdown(
            f"""
        <div style="font-size: 11px;">
        ⬜ <b>White:</b> Fluid region<br>
        ⬛ <b>Black:</b> Solid fins<br>
        📏 <b>Scale:</b> 1 grid = {designer.grid_size:.3f} mm
        </div>
        """,
            unsafe_allow_html=True,
        )
        
        # Academic Reference only for Hybrid Finned Cold Plate (not in random explore) and when canvas not empty
        if (
            preset == "Hybrid Finned Cold Plate"
            and not st.session_state.get("is_random_explore", False)
            and solid_pixels > 0
        ):
            st.markdown("---")
            st.markdown("#### 📚 Reference")
            st.markdown("""
            <div style="font-size: 10px; color: #666;">
            <b>Original Model Authors:</b><br>
            "A hybrid finned cold plate with flow regulation"<br><br>
            <b>Researchers:</b><br>
            Xiaotong Wang, Zijun Gao<br>
            <i>Department of Mechanical and Automation Engineering</i><br>
            <i>The Chinese University of Hong Kong</i><br>
            Hong Kong, China<br>
            wangxt@link.cuhk.edu.hk
            </div>
            """, unsafe_allow_html=True)
        
        # Design Tool Developer (always shown at the bottom)
        st.markdown("---")
        st.markdown("#### 👨‍💻 Design Tool Developer")
        st.markdown("""
        <div style="font-size: 10px; color: #666;">
        <b>Developer:</b><br>
        Ying Li<br>
        John.li.697@cranfield.ac.uk
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()