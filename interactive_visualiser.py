import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib import cm
from scipy.spatial import KDTree
import math
import cmath

from hat_tiling import H_init, T_init, P_init, F_init, constructPatch, constructMetatiles
from graph_builder import build_neighbor_graph_fast, analyze_square_frame

# ============================================================
# PENROSE TILING CLASSES
# ============================================================

# Golden ratio
phi = (5 ** 0.5 + 1) / 2

class PenroseTriangle:
    """Robinson's triangles: thick (36, 72, 72) and thin (36, 108, 36)"""
    
    def __init__(self, shape, v1, v2, v3):
        self.shape = shape
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

    def subdivide(self):
        """Subdivide triangles according to Penrose rules"""
        if self.shape == "thin":
            p1 = self.v1 + (self.v2 - self.v1) / phi
            return [
                PenroseTriangle("thin", self.v3, p1, self.v2),
                PenroseTriangle("thick", p1, self.v3, self.v1)
            ]
        else:  # thick
            p2 = self.v2 + (self.v1 - self.v2) / phi
            p3 = self.v2 + (self.v3 - self.v2) / phi
            return [
                PenroseTriangle("thick", p3, self.v3, self.v1),
                PenroseTriangle("thick", p2, p3, self.v2),
                PenroseTriangle("thin", p3, p2, self.v1)
            ]

    def get_all_vertices(self):
        return [self.v1, self.v2, self.v3]

    def get_edges_for_percolation(self):
        """Return edges for percolation (excludes base edges for rhombus formation)"""
        return [
            (self.v1, self.v3),
            (self.v2, self.v3)
        ]

class PenroseTiling:
    """Penrose tiling generator"""
    
    def __init__(self, divisions=4, base=5, scale=200, config=None):
        self.divisions = divisions
        self.base = base
        self.scale = scale
        self.config = config or {}
        self.triangles = []

    def create_initial_tiles(self):
        """Create initial star pattern"""
        initial_scale = self.scale * 0.5
        triangles = []
        
        for i in range(self.base * 2):
            v2 = cmath.rect(initial_scale, (2*i - 1) * math.pi / (self.base * 2))
            v3 = cmath.rect(initial_scale, (2*i + 1) * math.pi / (self.base * 2))
            
            if i % 2 == 0:
                v2, v3 = v3, v2
            
            triangles.append(PenroseTriangle("thin", 0, v2, v3))
        
        self.triangles = triangles

    def subdivide_all(self):
        """Perform all subdivision iterations"""
        for _ in range(self.divisions):
            new_triangles = []
            for triangle in self.triangles:
                new_triangles.extend(triangle.subdivide())
            self.triangles = new_triangles

    def make_tiling(self):
        """Generate complete tiling"""
        self.create_initial_tiles()
        self.subdivide_all()

    def get_statistics(self):
        """Return tiling statistics"""
        thin_count = sum(1 for t in self.triangles if t.shape == "thin")
        thick_count = sum(1 for t in self.triangles if t.shape == "thick")
        total = len(self.triangles)
        
        return {
            'total': total,
            'thin': thin_count,
            'thick': thick_count,
            'thick_to_thin_ratio': thick_count / thin_count if thin_count > 0 else 0
        }

# ============================================================
# UNION-FIND DATA STRUCTURE
# ============================================================

class WeightedQuickUnionUF:
    """Weighted Quick Union with path compression"""
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int32)
        self.count = n
    
    def find(self, p):
        """Find root with path compression"""
        root = p
        while root != self.parent[root]:
            root = self.parent[root]
        while p != root:
            newp = self.parent[p]
            self.parent[p] = root
            p = newp
        return root
    
    def union(self, p, q):
        """Union by size"""
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP == rootQ:
            return
        
        if self.size[rootP] < self.size[rootQ]:
            self.parent[rootP] = rootQ
            self.size[rootQ] += self.size[rootP]
        else:
            self.parent[rootQ] = rootP
            self.size[rootP] += self.size[rootQ]
        
        self.count -= 1
    
    def connected(self, p, q):
        """Check if two nodes are connected"""
        return self.find(p) == self.find(q)

# ============================================================
# GEOMETRIC BOUNDARY EDGE DETECTION
# ============================================================

def classify_boundary_edges(nodes, edges, eps=1e-6):
    xs = nodes[:, 0]
    ys = nodes[:, 1]

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    top, bottom, left, right = set(), set(), set(), set()

    for i, (u, v) in enumerate(edges):
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]

        if abs(y1 - ymax) < eps or abs(y2 - ymax) < eps:
            top.add(i)
        if abs(y1 - ymin) < eps or abs(y2 - ymin) < eps:
            bottom.add(i)
        if abs(x1 - xmin) < eps or abs(x2 - xmin) < eps:
            left.add(i)
        if abs(x1 - xmax) < eps or abs(x2 - xmax) < eps:
            right.add(i)

    return top, bottom, left, right

# ============================================================
# SQUARE LATTICE GENERATOR
# ============================================================

def generate_square_lattice(L):
    """Generate a square lattice of size L x L"""
    nodes = []
    node_map = {}
    
    idx = 0
    for i in range(L + 1):
        for j in range(L + 1):
            nodes.append([float(j), float(i)])
            node_map[(i, j)] = idx
            idx += 1
    
    nodes = np.array(nodes)
    
    edges = []
    for i in range(L + 1):
        for j in range(L + 1):
            if j < L:
                edges.append([node_map[(i, j)], node_map[(i, j + 1)]])
            if i < L:
                edges.append([node_map[(i, j)], node_map[(i + 1, j)]])
    
    edges = np.array(edges)
    
    neighbors = {i: [] for i in range(len(nodes))}
    for u, v in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
    
    return nodes, edges, neighbors

# ============================================================
# PENROSE LATTICE GENERATOR
# ============================================================

def build_penrose_neighbor_graph(tiling):
    """Build neighbor graph from Penrose tiling"""
    TOL = 1e-5
    
    all_vertices = []
    for triangle in tiling.triangles:
        verts = triangle.get_all_vertices()
        for v in verts:
            all_vertices.append([v.real, v.imag])
    
    nodes = np.array(all_vertices, dtype=np.float64)
    
    # Deduplicate vertices
    tree = KDTree(nodes)
    pairs = tree.query_pairs(r=TOL)
    
    parent = np.arange(len(nodes))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    
    for i, j in pairs:
        union(i, j)
    
    mapping = np.array([find(i) for i in range(len(nodes))])
    unique_ids = np.unique(mapping)
    id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
    node_to_unique = np.array([id_to_idx[mapping[i]] for i in range(len(nodes))])
    unique_nodes = nodes[unique_ids]
    
    # Build edges
    vertex_tree = KDTree(nodes)
    rhombus_edge_set = set()
    
    for triangle in tiling.triangles:
        try:
            edges = triangle.get_edges_for_percolation()
            
            for v_start, v_end in edges:
                v_start_arr = np.array([[v_start.real, v_start.imag]])
                v_end_arr = np.array([[v_end.real, v_end.imag]])
                
                _, start_orig_idx = vertex_tree.query(v_start_arr, k=1)
                _, end_orig_idx = vertex_tree.query(v_end_arr, k=1)
                
                start_idx = node_to_unique[start_orig_idx[0]]
                end_idx = node_to_unique[end_orig_idx[0]]
                
                if start_idx != end_idx:
                    edge = (min(start_idx, end_idx), max(start_idx, end_idx))
                    rhombus_edge_set.add(edge)
        except:
            continue
    
    # Convert to edges array and neighbors dict
    edges = np.array(list(rhombus_edge_set))
    neighbors = {i: [] for i in range(len(unique_nodes))}
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    
    return unique_nodes, edges, neighbors

# ============================================================
# TRIPLE PERCOLATION VISUALIZER
# ============================================================

class TriplePercolationVisualizer:
    def __init__(self, hat_data, penrose_data, square_data, mode="bond"):
        self.hat_nodes = np.asarray(hat_data['nodes'])
        self.hat_neighbors = hat_data['neighbors']
        self.hat_edges = hat_data['edges']
        
        self.penrose_nodes = np.asarray(penrose_data['nodes'])
        self.penrose_neighbors = penrose_data['neighbors']
        self.penrose_edges = penrose_data['edges']
        
        self.square_nodes = np.asarray(square_data['nodes'])
        self.square_neighbors = square_data['neighbors']
        self.square_edges = square_data['edges']
        
        self.mode = mode
        
        self.hat_N = len(self.hat_nodes)
        self.hat_M = len(self.hat_edges)
        
        self.penrose_N = len(self.penrose_nodes)
        self.penrose_M = len(self.penrose_edges)
        
        self.square_N = len(self.square_nodes)
        self.square_M = len(self.square_edges)
        
        # Classify boundary edges
        (self.hat_top, self.hat_bottom, 
         self.hat_left, self.hat_right) = classify_boundary_edges(self.hat_nodes, self.hat_edges)
        
        (self.penrose_top, self.penrose_bottom,
         self.penrose_left, self.penrose_right) = classify_boundary_edges(self.penrose_nodes, self.penrose_edges)
        
        (self.square_top, self.square_bottom,
         self.square_left, self.square_right) = classify_boundary_edges(self.square_nodes, self.square_edges)
        
        self.reset_order()
        self.compute_states()
        self.setup_plot()

    def reset_order(self):
        if self.mode == "bond":
            self.hat_order = np.random.permutation(self.hat_M)
            self.penrose_order = np.random.permutation(self.penrose_M)
            self.square_order = np.random.permutation(self.square_M)
        else:
            self.hat_order = np.random.permutation(self.hat_N)
            self.penrose_order = np.random.permutation(self.penrose_N)
            self.square_order = np.random.permutation(self.square_N)

    def compute_states(self):
        if self.mode == "bond":
            self.hat_states = self.compute_bond_states(
                self.hat_N, self.hat_M, self.hat_edges, self.hat_order,
                self.hat_top, self.hat_bottom, self.hat_left, self.hat_right
            )
            self.penrose_states = self.compute_bond_states(
                self.penrose_N, self.penrose_M, self.penrose_edges, self.penrose_order,
                self.penrose_top, self.penrose_bottom, self.penrose_left, self.penrose_right
            )
            self.square_states = self.compute_bond_states(
                self.square_N, self.square_M, self.square_edges, self.square_order,
                self.square_top, self.square_bottom, self.square_left, self.square_right
            )
        else:
            self.hat_states = self.compute_site_states(
                self.hat_N, self.hat_neighbors, self.hat_edges, self.hat_order,
                self.hat_top, self.hat_bottom, self.hat_left, self.hat_right
            )
            self.penrose_states = self.compute_site_states(
                self.penrose_N, self.penrose_neighbors, self.penrose_edges, self.penrose_order,
                self.penrose_top, self.penrose_bottom, self.penrose_left, self.penrose_right
            )
            self.square_states = self.compute_site_states(
                self.square_N, self.square_neighbors, self.square_edges, self.square_order,
                self.square_top, self.square_bottom, self.square_left, self.square_right
            )

    def compute_site_states(self, N, neighbors, edges, order, top_e, bottom_e, left_e, right_e):
        print(f"Computing SITE percolation states for {N} nodes...")
        states = []
        
        uf = WeightedQuickUnionUF(N)
        ufB = WeightedQuickUnionUF(N + 4)
        
        vT, vB, vL, vR = N, N+1, N+2, N+3
        open_site = np.zeros(N, dtype=bool)
        
        for step, s in enumerate(order):
            open_site[s] = True
            
            for nb in neighbors[s]:
                if open_site[nb]:
                    uf.union(s, nb)
                    ufB.union(s, nb)
            
            for ei in top_e:
                u, v = edges[ei]
                if open_site[u] and open_site[v]:
                    ufB.union(u, vT)
                    ufB.union(v, vT)
            
            for ei in bottom_e:
                u, v = edges[ei]
                if open_site[u] and open_site[v]:
                    ufB.union(u, vB)
                    ufB.union(v, vB)
            
            for ei in left_e:
                u, v = edges[ei]
                if open_site[u] and open_site[v]:
                    ufB.union(u, vL)
                    ufB.union(v, vL)
            
            for ei in right_e:
                u, v = edges[ei]
                if open_site[u] and open_site[v]:
                    ufB.union(u, vR)
                    ufB.union(v, vR)
            
            perc_TB = ufB.connected(vT, vB)
            perc_LR = ufB.connected(vL, vR)
            
            sizes = {}
            node_cluster = {}
            
            for i in range(N):
                if open_site[i]:
                    r = uf.find(i)
                    sizes[r] = sizes.get(r, 0) + 1
                    node_cluster[i] = r
            
            states.append({
                "p": (step + 1) / N,
                "open_sites": open_site.copy(),
                "node_to_cluster": node_cluster,
                "cluster_sizes": sizes,
                "perc_TB": perc_TB,
                "perc_LR": perc_LR,
                "percolates": perc_TB and perc_LR
            })
        
        return states

    def compute_bond_states(self, N, M, edges, order, top_e, bottom_e, left_e, right_e):
        print(f"Computing BOND percolation states for {M} edges...")
        states = []
        
        uf = WeightedQuickUnionUF(N)
        ufB = WeightedQuickUnionUF(N + 4)
        
        vT, vB, vL, vR = N, N+1, N+2, N+3
        open_edge = np.zeros(M, dtype=bool)
        
        for step, e in enumerate(order):
            open_edge[e] = True
            u, v = edges[e]
            
            uf.union(u, v)
            ufB.union(u, v)
            
            if e in top_e:
                ufB.union(u, vT)
                ufB.union(v, vT)
            if e in bottom_e:
                ufB.union(u, vB)
                ufB.union(v, vB)
            if e in left_e:
                ufB.union(u, vL)
                ufB.union(v, vL)
            if e in right_e:
                ufB.union(u, vR)
                ufB.union(v, vR)
            
            perc_TB = ufB.connected(vT, vB)
            perc_LR = ufB.connected(vL, vR)
            
            sizes = {}
            node_cluster = {}
            
            for i in range(N):
                r = uf.find(i)
                sizes[r] = sizes.get(r, 0) + 1
                node_cluster[i] = r
            
            states.append({
                "p": (step + 1) / M,
                "open_edges": open_edge.copy(),
                "node_to_cluster": node_cluster,
                "cluster_sizes": sizes,
                "perc_TB": perc_TB,
                "perc_LR": perc_LR,
                "percolates": perc_TB and perc_LR
            })
        
        return states

    def setup_plot(self):
        self.fig, (self.ax_hat, self.ax_penrose, self.ax_square) = plt.subplots(1, 3, figsize=(24, 8))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.25, top=0.95)
        
        cmap = cm.get_cmap("tab20")
        self.colors = [cmap(i) for i in range(20)]
        
        # Setup hat tiling
        self.hat_lines = []
        for u, v in self.hat_edges:
            p, q = self.hat_nodes[u], self.hat_nodes[v]
            ln, = self.ax_hat.plot([p[0], q[0]], [p[1], q[1]], lw=1.5, alpha=0.2, color="#dddddd")
            self.hat_lines.append(ln)
        
        self.ax_hat.set_aspect("equal")
        self.ax_hat.axis("off")
        xmin, ymin = self.hat_nodes.min(axis=0)
        xmax, ymax = self.hat_nodes.max(axis=0)
        dx, dy = xmax - xmin, ymax - ymin
        self.ax_hat.set_xlim(xmin - 0.1 * dx, xmax + 0.1 * dx)
        self.ax_hat.set_ylim(ymin - 0.1 * dy, ymax + 0.1 * dy)
        
        # Setup Penrose
        self.penrose_lines = []
        for u, v in self.penrose_edges:
            p, q = self.penrose_nodes[u], self.penrose_nodes[v]
            ln, = self.ax_penrose.plot([p[0], q[0]], [p[1], q[1]], lw=1.5, alpha=0.2, color="#dddddd")
            self.penrose_lines.append(ln)
        
        self.ax_penrose.set_aspect("equal")
        self.ax_penrose.axis("off")
        xmin, ymin = self.penrose_nodes.min(axis=0)
        xmax, ymax = self.penrose_nodes.max(axis=0)
        dx, dy = xmax - xmin, ymax - ymin
        self.ax_penrose.set_xlim(xmin - 0.1 * dx, xmax + 0.1 * dx)
        self.ax_penrose.set_ylim(ymin - 0.1 * dy, ymax + 0.1 * dy)
        
        # Setup square
        self.square_lines = []
        for u, v in self.square_edges:
            p, q = self.square_nodes[u], self.square_nodes[v]
            ln, = self.ax_square.plot([p[0], q[0]], [p[1], q[1]], lw=1.5, alpha=0.2, color="#dddddd")
            self.square_lines.append(ln)
        
        self.ax_square.set_aspect("equal")
        self.ax_square.axis("off")
        xmin, ymin = self.square_nodes.min(axis=0)
        xmax, ymax = self.square_nodes.max(axis=0)
        dx, dy = xmax - xmin, ymax - ymin
        self.ax_square.set_xlim(xmin - 0.1 * dx, xmax + 0.1 * dx)
        self.ax_square.set_ylim(ymin - 0.1 * dy, ymax + 0.1 * dy)
        
        self.hat_title = self.ax_hat.set_title("", fontsize=12)
        self.penrose_title = self.ax_penrose.set_title("", fontsize=12)
        self.square_title = self.ax_square.set_title("", fontsize=12)
        
        self.hat_info = self.fig.text(0.05, 0.15, "", family="monospace", fontsize=9)
        self.penrose_info = self.fig.text(0.37, 0.15, "", family="monospace", fontsize=9)
        self.square_info = self.fig.text(0.69, 0.15, "", family="monospace", fontsize=9)
        
        self.slider = Slider(
            plt.axes([0.15, 0.08, 0.7, 0.03]),
            "p", 0, 1, valstep=1/1000
        )
        self.slider.on_changed(self.update)
        
        Button(plt.axes([0.15, 0.02, 0.12, 0.04]), "New Trial").on_clicked(self.new_trial)
        self.anim_btn = Button(plt.axes([0.30, 0.02, 0.12, 0.04]), "Animate")
        self.anim_btn.on_clicked(self.animate)
        
        self.update(0)

    def update(self, val):
        p = self.slider.val
        
        hat_idx = min(int(p * len(self.hat_states)), len(self.hat_states) - 1)
        hat_st = self.hat_states[hat_idx]
        self.update_plot(hat_st, self.hat_lines, self.hat_edges, self.hat_title, self.hat_info, "HAT")
        
        penrose_idx = min(int(p * len(self.penrose_states)), len(self.penrose_states) - 1)
        penrose_st = self.penrose_states[penrose_idx]
        self.update_plot(penrose_st, self.penrose_lines, self.penrose_edges, self.penrose_title, self.penrose_info, "PENROSE")
        
        square_idx = min(int(p * len(self.square_states)), len(self.square_states) - 1)
        square_st = self.square_states[square_idx]
        self.update_plot(square_st, self.square_lines, self.square_edges, self.square_title, self.square_info, "SQUARE")
        
        self.fig.canvas.draw_idle()

    def update_plot(self, st, lines, edges, title_obj, info_obj, name):
        sizes = st["cluster_sizes"]
        roots = sorted(sizes, key=lambda r: sizes[r], reverse=True)
        
        color_map = {}
        for i, r in enumerate(roots):
            color_map[r] = "#dc2626" if i == 0 and st["percolates"] else self.colors[i % len(self.colors)]
        
        for i, ln in enumerate(lines):
            u, v = edges[i]
            show = st["open_edges"][i] if self.mode == "bond" else (
                st["open_sites"][u] and st["open_sites"][v]
            )
            
            if show:
                r = st["node_to_cluster"][u]
                ln.set_color(color_map.get(r, "#999999"))
                ln.set_alpha(0.9)
                ln.set_linewidth(2.5)
            else:
                ln.set_color("#eeeeee")
                ln.set_alpha(0.15)
                ln.set_linewidth(1)
        
        title = f"{name} {self.mode.upper()} p={st['p']:.3f}"
        if st["percolates"]:
            title += " 🔴"
        title_obj.set_text(title)
        
        info_obj.set_text(
            f"Clusters: {len(sizes)}\n"
            f"Largest: {max(sizes.values()) if sizes else 0}\n"
            f"TB: {st['perc_TB']}  LR: {st['perc_LR']}"
        )

    def new_trial(self, event):
        self.reset_order()
        self.compute_states()
        self.slider.set_val(0)

    def animate(self, event):
        if hasattr(self, "timer") and self.timer:
            self.timer.stop()
            self.timer = None
            self.anim_btn.label.set_text("Animate")
            return
        
        self.anim_btn.label.set_text("Stop")
        
        def step():
            v = self.slider.val + 0.005
            if v >= 1:
                self.timer.stop()
                self.anim_btn.label.set_text("Animate")
                return
            self.slider.set_val(v)
        
        self.timer = self.fig.canvas.new_timer(interval=30)
        self.timer.add_callback(step)
        self.timer.start()

    def show(self):
        plt.show()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["site", "bond"], default="bond")
    parser.add_argument("--L", type=int, default=20)
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--penrose-div", type=int, default=5, help="Penrose subdivisions")
    args = parser.parse_args()
    
    # --- Construct hat tiling ---
    print("Building Hat tiling...")
    base = [H_init(), T_init(), P_init(), F_init()]
    cur = base
    for _ in range(args.r):
        p = constructPatch(*cur)
        cur = constructMetatiles(p)
    
    nodes, neighbors = build_neighbor_graph_fast(p, level=args.r + 1)
    fd = analyze_square_frame(nodes, neighbors, L=args.L, boundary_thickness=1.0)
    
    hat_data = {
        'nodes': fd["sub_graph_nodes"],
        'edges': fd["sub_graph_edges"],
        'neighbors': fd["sub_graph_neighbors"] if args.mode == "site" else {}
    }
    
    # --- Generate Penrose tiling ---
    print("Building Penrose tiling...")
    tiling = PenroseTiling(divisions=args.penrose_div, base=5, scale=100, config={})
    tiling.make_tiling()
    penrose_nodes, penrose_edges, penrose_neighbors = build_penrose_neighbor_graph(tiling)
    
    penrose_data = {
        'nodes': penrose_nodes,
        'edges': penrose_edges,
        'neighbors': penrose_neighbors if args.mode == "site" else {}
    }
    print(f"Penrose: {len(penrose_nodes)} nodes, {len(penrose_edges)} edges")
    
    # --- Generate square lattice ---
    print("Building Square lattice...")
    square_nodes, square_edges, square_neighbors = generate_square_lattice(args.L)
    
    square_data = {
        'nodes': square_nodes,
        'edges': square_edges,
        'neighbors': square_neighbors if args.mode == "site" else {}
    }
    
    viz = TriplePercolationVisualizer(hat_data, penrose_data, square_data, mode=args.mode)
    viz.show()