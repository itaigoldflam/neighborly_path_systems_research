import tkinter as tk
from tkinter import simpledialog, messagebox
import math
import heapq

# Constants
NODE_RADIUS = 12
CLICK_TOLERANCE = 8
EPSILON = 0.01
PATH_OFFSET_STEP = 4  # Distance between parallel colored lines
COLORS = ["red", "green", "blue", "magenta", "orange", "cyan", "purple", "brown"]

class WheelGraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Wheel Graph Interface")
        
        # Control Frame setup
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Vertex count input
        tk.Label(control_frame, text="Vertices (n):").pack(side=tk.LEFT)
        self.n_entry = tk.Entry(control_frame, width=5)
        self.n_entry.insert(0, "5")
        self.n_entry.pack(side=tk.LEFT, padx=5)

        # --- NEW: Spoke Weight Input ---
        # Allows user to define initial weight for spokes (radius lines)
        tk.Label(control_frame, text="Spoke Weight:").pack(side=tk.LEFT)
        self.spoke_entry = tk.Entry(control_frame, width=5)
        self.spoke_entry.insert(0, "1") # Default value
        self.spoke_entry.pack(side=tk.LEFT, padx=5)
        # -------------------------------
        
        # Buttons
        tk.Button(control_frame, text="Create Graph", command=self.create_graph).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Reset Colors", command=self.reset_colors).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Show All Paths", command=self.generate_auto_paths).pack(side=tk.LEFT, padx=5)
        
        # Canvas setup
        self.canvas = tk.Canvas(root, bg="white", width=900, height=700)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Graph State Initialization
        self.nodes = {}       
        self.edges = {}       
        self.rim_nodes = []   
        self.center_node_id = 0
        self.next_node_id = 1
        self.k_counter = 0    
        
        self.selected_nodes = [] 
        self.active_paths = []   
        self.color_index = 0
        
        # Geometry Settings
        self.width = 900
        self.height = 700
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        self.radius = 250

    def create_graph(self):
        try:
            n = int(self.n_entry.get())
            # --- NEW: Retrieve spoke weight from input ---
            spoke_weight = float(self.spoke_entry.get())
            
            if n < 3:
                messagebox.showerror("Error", "n must be >= 3")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")
            return

        # Reset all graph state
        self.nodes = {}
        self.edges = {}
        self.rim_nodes = []
        self.active_paths = []
        self.k_counter = 0
        self.selected_nodes = []
        self.color_index = 0
        self.next_node_id = 1
        self.center_node_id = 0
        
        # Create Center Node
        self.nodes[0] = (self.center_x, self.center_y)
        
        # Create Rim Nodes
        for i in range(n):
            node_id = self.next_node_id
            self.rim_nodes.append(node_id)
            self.next_node_id += 1
            
            # Connect to center using the CUSTOM spoke weight
            self.add_edge(0, node_id, spoke_weight)
            
        # Connect Rim Edges (Default weight 1.0)
        for i in range(n):
            u = self.rim_nodes[i]
            v = self.rim_nodes[(i + 1) % n]
            self.add_edge(u, v, 1.0)
            
        # Calculate visual positions and draw
        self.recalculate_layout()
        self.draw_graph()

    def recalculate_layout(self):
        """
        Redistributes all rim nodes to form a perfect regular polygon
        based on the current number of nodes in the rim.
        """
        n = len(self.rim_nodes)
        if n == 0: return
        
        angle_step = 2 * math.pi / n
        
        for i, node_id in enumerate(self.rim_nodes):
            # Start from -PI/2 to have the first node at the top
            angle = i * angle_step - math.pi / 2
            x = self.center_x + self.radius * math.cos(angle)
            y = self.center_y + self.radius * math.sin(angle)
            self.nodes[node_id] = (x, y)
            
        # Ensure center is still center
        self.nodes[self.center_node_id] = (self.center_x, self.center_y)

    def add_edge(self, u, v, weight):
        key = tuple(sorted((u, v)))
        self.edges[key] = weight

    def remove_edge(self, u, v):
        key = tuple(sorted((u, v)))
        if key in self.edges:
            del self.edges[key]

    def get_weight(self, u, v):
        key = tuple(sorted((u, v)))
        return self.edges.get(key, float('inf'))

    def draw_graph(self):
        self.canvas.delete("all")
        
        # 1. Draw Edges (Black base lines)
        for (u, v), weight in self.edges.items():
            x1, y1 = self.nodes[u]
            x2, y2 = self.nodes[v]
            self.canvas.create_line(x1, y1, x2, y2, fill="black", width=1, tags="edge")

        # 2. Draw Paths (Colored lines side-by-side)
        # First, map each edge to the list of colors passing through it
        edge_colors = {} # {(u,v): [color1, color2]}
        
        for path_data in self.active_paths:
            color = path_data['color']
            for edge in path_data['edges']:
                key = tuple(sorted(edge))
                if key not in edge_colors:
                    edge_colors[key] = []
                edge_colors[key].append(color)
        
        # Draw the colored offsets
        for (u, v), colors in edge_colors.items():
            if not colors: continue
            
            x1, y1 = self.nodes[u]
            x2, y2 = self.nodes[v]
            
            # Vector math for parallel lines
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length == 0: continue
            
            # Unit normal vector (perpendicular)
            # (-dy, dx) is rotated 90 degrees
            nx = -dy / length
            ny = dx / length
            
            num_colors = len(colors)
            
            for i, color in enumerate(colors):
                # Calculate offset: center the group of lines around the main edge
                # e.g., if 1 line -> offset 0. If 2 lines -> offsets -step/2, +step/2
                offset_idx = i - (num_colors - 1) / 2
                offset_dist = offset_idx * PATH_OFFSET_STEP * 2 # Multiply for visual gap
                
                ox = nx * offset_dist
                oy = ny * offset_dist
                
                self.canvas.create_line(x1 + ox, y1 + oy, x2 + ox, y2 + oy, 
                                        fill=color, width=3, tags="path_edge")

        # 3. Draw Edge Weights (Top layer)
        for (u, v), weight in self.edges.items():
            x1, y1 = self.nodes[u]
            x2, y2 = self.nodes[v]
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Small background for text
            self.canvas.create_oval(mid_x - 10, mid_y - 8, mid_x + 10, mid_y + 8, 
                                    fill="white", outline="#ddd")
            
            # --- Format Logic: Max 5 decimal places, remove trailing zeros ---
            # 1. Format to fixed 5 decimal places (e.g., 1.0 -> "1.00000", 0.125 -> "0.12500")
            formatted_weight = f"{weight:.5f}"
            
            # 2. Remove trailing zeros and then the decimal point if it's at the end
            # e.g., "1.00000" -> "1.", then "1." -> "1"
            # e.g., "0.12500" -> "0.125"
            label_text = formatted_weight.rstrip('0').rstrip('.')
            
            self.canvas.create_text(mid_x, mid_y, text=label_text, font=("Arial", 8))

        # 4. Draw Nodes
        for node_id, (x, y) in self.nodes.items():
            color = "lightblue"
            outline = "blue"
            if node_id == self.center_node_id:
                color = "yellow"
            if node_id in self.selected_nodes:
                color = "orange"
                outline = "red"
                
            self.canvas.create_oval(x - NODE_RADIUS, y - NODE_RADIUS, 
                                    x + NODE_RADIUS, y + NODE_RADIUS, 
                                    fill=color, outline=outline, width=2, tags="node")
            self.canvas.create_text(x, y, text=str(node_id), font=("Arial", 9, "bold"))

    def on_canvas_click(self, event):
        x, y = event.x, event.y
        
        # Check Node Click
        clicked_node = None
        for node_id, (nx, ny) in self.nodes.items():
            if math.hypot(nx - x, ny - y) <= NODE_RADIUS + 3:
                clicked_node = node_id
                break
        
        if clicked_node is not None:
            self.handle_node_click(clicked_node)
            return

        # Check Edge Click
        clicked_edge = None
        min_dist = CLICK_TOLERANCE
        
        for (u, v), w in self.edges.items():
            x1, y1 = self.nodes[u]
            x2, y2 = self.nodes[v]
            d = self.point_to_line_dist(x, y, x1, y1, x2, y2)
            if d < min_dist:
                clicked_edge = (u, v)
                min_dist = d
        
        if clicked_edge:
            self.handle_edge_click(clicked_edge)

    def handle_node_click(self, node_id):
        # Allow selecting source and target
        if len(self.selected_nodes) >= 2:
            self.selected_nodes = [] # Reset selection for new path, but keep old paths drawn
            
        self.selected_nodes.append(node_id)
        
        if len(self.selected_nodes) == 2:
            start, end = self.selected_nodes
            self.find_shortest_path(start, end)
            # We don't clear selected_nodes immediately so user sees what they clicked,
            # but next click will clear them.
        
        self.draw_graph()

    def handle_edge_click(self, edge):
        u, v = edge
        
        alpha = simpledialog.askfloat("Split Edge", 
                                      f"Enter Alpha (0 < alpha < 1) for edge {u}-{v}:",
                                      minvalue=0.0001, maxvalue=0.9999, initialvalue=0.5)
        
        if alpha is not None:
            self.split_edge(u, v, alpha)

    def update_all_paths(self):
        """
        Re-runs Dijkstra for every active colored path.
        Called after graph topology changes (splitting edges).
        """
        for path_data in self.active_paths:
            start = path_data['start']
            end = path_data['end']
            
            # Recalculate shortest path on the modified graph
            new_cost, new_edges = self.run_dijkstra(start, end)
            
            if new_edges or start == end:
                path_data['edges'] = set(new_edges)
                path_data['cost'] = new_cost
            else:
                # If path becomes impossible (graph disconnected), clear the visual lines
                path_data['edges'] = set()

    def split_edge(self, u, v, alpha):
        # NOTE: 'reset_colors' was removed to persist paths.
        
        old_weight = self.get_weight(u, v)
        self.k_counter += 1
        
        # Calculate new weights based on the formula
        epsilon_k = EPSILON ** self.k_counter
        w_new_1 = alpha * old_weight + epsilon_k
        w_new_2 = (1 - alpha) * old_weight + epsilon_k
        
        # Calculate spoke weight to center
        h1 = self.get_weight(u, self.center_node_id)
        h2 = self.get_weight(v, self.center_node_id)
        if u == self.center_node_id: h1 = 0
        if v == self.center_node_id: h2 = 0
            
        w_spoke = max(abs(h1 - alpha * old_weight), abs(h2 - (1 - alpha) * old_weight))
        
        # Create new node
        new_node_id = self.next_node_id
        self.next_node_id += 1
        
        # Update Topology: Remove old edge, add 3 new edges
        self.remove_edge(u, v)
        self.add_edge(u, new_node_id, w_new_1)
        self.add_edge(new_node_id, v, w_new_2)
        self.add_edge(new_node_id, self.center_node_id, w_spoke)
        
        # --- Relayout Logic ---
        # If splitting a rim edge, insert into rim list and reshape polygon
        if u in self.rim_nodes and v in self.rim_nodes:
            idx_u = self.rim_nodes.index(u)
            idx_v = self.rim_nodes.index(v)
            n = len(self.rim_nodes)
            is_adjacent = False
            insert_index = -1
            
            # Check adjacency (handling circular list)
            if (idx_u + 1) % n == idx_v:
                insert_index = idx_v
                is_adjacent = True
            elif (idx_v + 1) % n == idx_u:
                insert_index = idx_u
                is_adjacent = True
            
            if is_adjacent:
                # Handle wrap-around case
                if insert_index == 0 and max(idx_u, idx_v) == n-1:
                    self.rim_nodes.append(new_node_id)
                else:
                    self.rim_nodes.insert(insert_index, new_node_id)
                
                self.recalculate_layout()
            else:
                # Not adjacent on rim (e.g., chord), just geometric interpolation
                self.set_midpoint_pos(u, v, new_node_id, alpha)
        else:
             self.set_midpoint_pos(u, v, new_node_id, alpha)

        # --- NEW: Update all existing colored paths ---
        self.update_all_paths()
        
        self.draw_graph()

    def set_midpoint_pos(self, u, v, new_id, alpha):
        x1, y1 = self.nodes[u]
        x2, y2 = self.nodes[v]
        self.nodes[new_id] = (x1 + alpha*(x2-x1), y1 + alpha*(y2-y1))

    def run_dijkstra(self, start, end):
        """
        Helper function: Runs Dijkstra's algorithm.
        Returns: (total_cost, list_of_edge_tuples)
        """
        pq = [(0, start, [])] # (cost, current_node, path_so_far)
        visited = set()
        
        while pq:
            cost, u, path = heapq.heappop(pq)
            
            if u in visited: continue
            visited.add(u)
            
            if u == end:
                return cost, path
            
            # Identify neighbors dynamically from edge list
            neighbors = []
            for (n1, n2), w in self.edges.items():
                if n1 == u: neighbors.append((n2, w))
                elif n2 == u: neighbors.append((n1, w))
            
            for v, w in neighbors:
                if v not in visited:
                    edge_key = tuple(sorted((u, v)))
                    heapq.heappush(pq, (cost + w, v, path + [edge_key]))
        
        return None, []

    def find_shortest_path(self, start, end):
        """
        Calculates path and stores it permanently for future updates.
        """
        cost, path_edges = self.run_dijkstra(start, end)

        if path_edges or start == end:
            # Assign a unique color from the list
            color = COLORS[self.color_index % len(COLORS)]
            self.color_index += 1
            
            # --- Store path with start/end nodes to allow recalculation ---
            self.active_paths.append({
                'start': start,
                'end': end,
                'color': color,
                'edges': set(path_edges),
                'cost': cost
            })
        else:
            messagebox.showinfo("Info", "No path found")

    def reset_colors(self):
        self.active_paths = []
        self.selected_nodes = []
        self.color_index = 0
        self.draw_graph()

    def point_to_line_dist(self, px, py, x1, y1, x2, y2):
        l2 = (x1 - x2)**2 + (y1 - y2)**2
        if l2 == 0: return math.hypot(px - x1, py - y1)
        t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2
        t = max(0, min(1, t))
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        return math.hypot(px - proj_x, py - proj_y)

    def generate_auto_paths(self):
        """
        Generates and displays paths between all non-adjacent rim nodes.
        Filters out paths that are subsets of other paths (removes the contained path).
        """
        # 1. Clear existing paths (optional, to avoid visual clutter)
        self.active_paths = [] 
        
        rim = self.rim_nodes
        n = len(rim)
        candidates = []

        # 2. Generate all possible paths between non-neighboring rim nodes
        for i in range(n):
            for j in range(i + 1, n):
                u = rim[i]
                v = rim[j]

                # Check if they are neighbors on the circular rim
                # They are neighbors if index difference is 1 or n-1 (due to wraparound)
                diff = abs(i - j)
                if diff == 1 or diff == n - 1:
                    continue # Skip neighbors

                # Calculate path using Dijkstra
                cost, path_list = self.run_dijkstra(u, v)
                
                # Convert to set to simplify subset checks later
                path_set = set(path_list)
                
                if path_list:
                    candidates.append({
                        'start': u,
                        'end': v,
                        'edges': path_set,
                        'cost': cost,
                        'original_list': path_list 
                    })

        # 3. Filter contained paths (remove the path if it is contained in another)
        final_candidates = []
        
        for i in range(len(candidates)):
            current_edges = candidates[i]['edges']
            is_contained = False
            
            for k in range(len(candidates)):
                if i == k: continue
                
                other_edges = candidates[k]['edges']
                
                # Check: Is the current path a subset of the other path?
                if current_edges.issubset(other_edges):
                    # If current is strictly smaller (proper subset), mark it for removal
                    if len(current_edges) < len(other_edges):
                        is_contained = True
                        break
                    # Tie-breaker: If they are identical sets, remove duplicates based on index
                    elif len(current_edges) == len(other_edges) and i > k:
                        is_contained = True
                        break
            
            # Only add if it is not contained in any other path
            if not is_contained:
                final_candidates.append(candidates[i])

        # 4. Add the surviving candidates to the application's active paths
        for item in final_candidates:
            color = COLORS[self.color_index % len(COLORS)]
            self.color_index += 1
            
            self.active_paths.append({
                'start': item['start'],
                'end': item['end'],
                'color': color,
                'edges': item['edges'],
                'cost': item['cost']
            })

        self.draw_graph()
        
        # Notify user of result
        messagebox.showinfo("Auto Paths", f"Generated {len(final_candidates)} unique non-contained paths.")
        

if __name__ == "__main__":
    root = tk.Tk()
    app = WheelGraphApp(root)
    root.mainloop()