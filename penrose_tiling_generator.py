import math
import cmath

# Golden ratio is an important constant for Penrose
phi = (5 ** 0.5 + 1) / 2

class PenroseTriangle:
    def __init__(self, shape, v1, v2, v3):
        # the 'Robinson-triangle version of the Penrose tiling has two triangles, 'thick' and 'thin' 
        self.shape = shape
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
    def subdivide(self):
        # this recursively divides a thin into a thin and a thick
        if self.shape == "thin":
            # Divide thin triangle into 1 thin + 1 thick
            p1 = self.v1 + (self.v2 - self.v1) / phi
            return [
                PenroseTriangle("thin", self.v3, p1, self.v2),
                PenroseTriangle("thick", p1, self.v3, self.v1)
            ]
        else:
            # this recursively divides a thick into a thin and 2 thicks
            p2 = self.v2 + (self.v1 - self.v2) / phi
            p3 = self.v2 + (self.v3 - self.v2) / phi
            return [
                PenroseTriangle("thick", p3, self.v3, self.v1),
                PenroseTriangle("thick", p2, p3, self.v2),
                PenroseTriangle("thin", p3, p2, self.v1)
            ]
    # returns all the vertices of a given triangle
    def get_all_vertices(self):
        return [self.v1, self.v2, self.v3]
    # returns 2 edges per triangle such that it is globally complete
    def get_edges_for_percolation(self):
        return [
            (self.v1, self.v3),
            (self.v1, self.v2)
        ]

class PenroseTiling:

    # decides the initial conditions, how many divisions etc
    def __init__(self, divisions=4, base=5, scale=200):

        self.divisions = divisions
        self.base = base
        self.scale = scale

        self.triangles = []

    def create_initial_tiles(self):
        # Creates the initial "star" pattern by using 10 thin triangles to form 5 triangles that are placed adjacent to use other 
        initial_scale = self.scale * 0.5
        triangles = []

        for i in range(self.base * 2):
            v2 = cmath.rect(initial_scale, (2*i - 1) * math.pi / (self.base * 2))
            v3 = cmath.rect(initial_scale, (2*i + 1) * math.pi / (self.base * 2))

            # swaps v2 and v3 to reflect the triangle as part of the reflection rules
            if i % 2 == 0:
                v2, v3 = v3, v2

            triangles.append(PenroseTriangle("thin", 0, v2, v3))

        self.triangles = triangles

    def subdivide_all(self):
        # Perform all recursive subdivision iterations
        for _ in range(self.divisions):
            new_triangles = []
            for rhombus in self.triangles:
                new_triangles.extend(rhombus.subdivide())
            self.triangles = new_triangles

    def make_tiling(self):
        # the primary call
        self.create_initial_tiles()
        self.subdivide_all()
