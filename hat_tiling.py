import numpy as np
import math


# Constants
sqrt3 = math.sqrt(3)
ident = [1, 0, 0, 0, 1, 0]
PI = math.pi

# Colours
cols = {
    'H1': [153/255, 100/255, 1],
    'H': [229/255, 205/255, 1],
    'T': [224/255, 224/255, 224/255],
    'P': [250/255, 250/255, 250/255],
    'F': [255/255, 255/255, 198/255],
    'edge': [0, 0, 0]
}

# Grid systems
def pt(x, y):
    return {'x': x, 'y': y}

def hexPt(x, y):
    return pt(x + 0.5*y, (sqrt3/2)*y)

# Affine transform functions
def inv(T):
    det = T[0]*T[4] - T[1]*T[3]
    return [T[4]/det, -T[1]/det, (T[1]*T[5]-T[2]*T[4])/det,
            -T[3]/det, T[0]/det, (T[2]*T[3]-T[0]*T[5])/det]

def mul(A, B):
    return [A[0]*B[0] + A[1]*B[3], 
            A[0]*B[1] + A[1]*B[4],
            A[0]*B[2] + A[1]*B[5] + A[2],
            A[3]*B[0] + A[4]*B[3], 
            A[3]*B[1] + A[4]*B[4],
            A[3]*B[2] + A[4]*B[5] + A[5]]

def padd(p, q):
    return {'x': p['x'] + q['x'], 'y': p['y'] + q['y']}

def psub(p, q):
    return {'x': p['x'] - q['x'], 'y': p['y'] - q['y']}

def trot(ang):
    c = math.cos(ang)
    s = math.sin(ang)
    return [c, -s, 0, s, c, 0]

def ttrans(tx, ty):
    return [1, 0, tx, 0, 1, ty]

def rotAbout(p, ang):
    return mul(ttrans(p['x'], p['y']), mul(trot(ang), ttrans(-p['x'], -p['y'])))

def transPt(M, P):
    return pt(M[0]*P['x'] + M[1]*P['y'] + M[2], M[3]*P['x'] + M[4]*P['y'] + M[5])

def matchSeg(p, q):
    return [q['x']-p['x'], p['y']-q['y'], p['x'], q['y']-p['y'], q['x']-p['x'], p['y']]

def matchTwo(p1, q1, p2, q2):
    return mul(matchSeg(p2, q2), inv(matchSeg(p1, q1)))

def intersect(p1, q1, p2, q2):
    d = (q2['y'] - p2['y']) * (q1['x'] - p1['x']) - (q2['x'] - p2['x']) * (q1['y'] - p1['y'])
    uA = ((q2['x'] - p2['x']) * (p1['y'] - p2['y']) - (q2['y'] - p2['y']) * (p1['x'] - p2['x'])) / d
    return pt(p1['x'] + uA * (q1['x'] - p1['x']), p1['y'] + uA * (q1['y'] - p1['y']))

# Hat outline
hat_outline = [
    hexPt(0, 0), hexPt(-1,-1), hexPt(0,-2), hexPt(2,-2),
    hexPt(2,-1), hexPt(4,-2), hexPt(5,-1), hexPt(4, 0),
    hexPt(3, 0), hexPt(2, 2), hexPt(0, 3), hexPt(0, 2),
    hexPt(-1, 2)]

# Tile classes
class HatTile:
    def __init__(self, label):
        self.label = label
        self.shape = hat_outline
    def draw(self, S, level, ax):
        drawPolygon(hat_outline, S, ax, cols[self.label], cols['edge'])
        return
    
class MetaTile:
    def __init__(self, shape, width):
        self.shape = shape 
        self.width = width
        self.children = [] 
    def addChild(self, T, geom):
        self.children.append({'T': T, 'geom': geom})
        return
    def evalChild(self, n, i):
        return transPt(self.children[n]['T'], self.children[n]['geom'].shape[i])
    def recentre(self):
        cx = sum(p['x'] for p in self.shape) / len(self.shape)
        cy = sum(p['y'] for p in self.shape) / len(self.shape)
        tr = pt(-cx, -cy)
        self.shape = [padd(p, tr) for p in self.shape]
        M = ttrans(-cx, -cy)
        for ch in self.children:
            ch['T'] = mul(M, ch['T'])
        return
    def draw(self, S, level, ax):
        if level > 0:
            for g in self.children:
                g['geom'].draw(mul(S, g['T']), level - 1, ax)
        else:
            drawPolygon(self.shape, S, ax, None, 'black')
        return
    
def drawPolygon(shape, T, ax, f=cols['H'], e=cols['edge'],):
    polygon = [transPt(T, p) for p in shape]
    ax.fill([p['x'] for p in polygon], [p['y'] for p in polygon],
            facecolor=f, edgecolor=e, linewidth=1)
    return


# Initialize tiles
H1_hat = HatTile('H1')
H_hat = HatTile('H')
T_hat = HatTile('T')
P_hat = HatTile('P')
F_hat = HatTile('F')

def H_init():
    H_outline = [
        pt(0, 0), pt(4, 0), pt(4.5, sqrt3/2),
        pt(2.5, 5 * sqrt3/2), pt(1.5, 5 * sqrt3/2), pt(-0.5, sqrt3/2)
    ]
    meta = MetaTile(H_outline, 2)
    meta.addChild(matchTwo(hat_outline[5], hat_outline[7], H_outline[5], H_outline[0]), H_hat)
    meta.addChild(matchTwo(hat_outline[9], hat_outline[11], H_outline[1], H_outline[2]), H_hat)
    meta.addChild(matchTwo(hat_outline[5], hat_outline[7], H_outline[3], H_outline[4]), H_hat)
    meta.addChild(mul(ttrans(2.5, sqrt3/2), mul([-0.5,-sqrt3/2,0,sqrt3/2,-0.5,0], [0.5,0,0,0,-0.5,0])), H1_hat)
    return meta

def T_init():
    T_outline = [pt(0, 0), pt(3, 0), pt(1.5, 3 * sqrt3/2)]
    meta = MetaTile(T_outline, 2)
    meta.addChild([0.5, 0, 0.5, 0, 0.5, sqrt3/2], T_hat)
    return meta

def P_init():
    P_outline = [pt(0, 0), pt(4, 0), pt(3, 2 * sqrt3/2), pt(-1, 2 * sqrt3/2)]
    meta = MetaTile(P_outline, 2)
    meta.addChild([0.5, 0, 1.5, 0, 0.5, sqrt3/2], P_hat)
    meta.addChild(mul(ttrans(0, 2 * sqrt3/2), mul([0.5, sqrt3/2, 0, -sqrt3/2, 0.5, 0], [0.5, 0.0, 0.0, 0.0, 0.5, 0.0])), P_hat)
    return meta

def F_init():
    F_outline = [pt(0, 0), pt(3, 0), pt(3.5, sqrt3/2), pt(3, 2 * sqrt3/2), pt(-1, 2 * sqrt3/2)]
    meta = MetaTile(F_outline, 2)
    meta.addChild([0.5, 0, 1.5, 0, 0.5, sqrt3/2], F_hat)
    meta.addChild(mul(ttrans(0, 2 * sqrt3/2), mul([0.5, sqrt3/2, 0, -sqrt3/2, 0.5, 0], [0.5, 0.0, 0.0, 0.0, 0.5, 0.0])), F_hat)
    return meta

def constructPatch(H,T,P,F):
    rules = [
        ['H'], [0, 0, 'P', 2], [1, 0, 'H', 2], [2, 0, 'P', 2], [3, 0, 'H', 2],
        [4, 4, 'P', 2], [0, 4, 'F', 3], [2, 4, 'F', 3], [4, 1, 3, 2, 'F', 0],
        [8, 3, 'H', 0], [9, 2, 'P', 0], [10, 2, 'H', 0], [11, 4, 'P', 2],
        [12, 0, 'H', 2], [13, 0, 'F', 3], [14, 2, 'F', 1], [15, 3, 'H', 4],
        [8, 2, 'F', 1], [17, 3, 'H', 0], [18, 2, 'P', 0], [19, 2, 'H', 2],
        [20, 4, 'F', 3], [20, 0, 'P', 2], [22, 0, 'H', 2], [23, 4, 'F', 3],
        [23, 0, 'F', 3], [16, 0, 'P', 2], [9, 4, 0, 2, 'T', 2], [4, 0, 'F', 3]
    ]
    
    ret = MetaTile([], H.width)
    shapes = {'H': H, 'T': T, 'P': P, 'F': F}
    
    for r in rules:
        if len(r) == 1:
            ret.addChild(ident, shapes[r[0]])
        elif len(r) == 4:
            poly = ret.children[r[0]]['geom'].shape
            T = ret.children[r[0]]['T']
            P = transPt(T, poly[(r[1]+1)%len(poly)])
            Q = transPt(T, poly[r[1]])
            nshp = shapes[r[2]]
            npoly = nshp.shape
            ret.addChild(matchTwo(npoly[r[3]], npoly[(r[3]+1)%len(npoly)], P, Q), nshp)
        else:
            chP = ret.children[r[0]]
            chQ = ret.children[r[2]]
            P = transPt(chQ['T'], chQ['geom'].shape[r[3]])
            Q = transPt(chP['T'], chP['geom'].shape[r[1]])
            nshp = shapes[r[4]]
            npoly = nshp.shape
            ret.addChild(matchTwo(npoly[r[5]], npoly[(r[5]+1)%len(npoly)], P, Q), nshp)
    return ret

def constructMetatiles(patch):
    bps1 = patch.evalChild(8, 2)
    bps2 = patch.evalChild(21, 2)
    rbps = transPt(rotAbout(bps1, -2.0*PI/3.0), bps2)
    p72 = patch.evalChild(7, 2)
    p252 = patch.evalChild(25, 2)
    llc = intersect(bps1, rbps, patch.evalChild(6, 2), p72)
    w = psub(patch.evalChild(6, 2), llc)
    
    new_H_outline = [llc, bps1]
    w = transPt(trot(-PI/3), w)
    new_H_outline.append(padd(new_H_outline[1], w))
    new_H_outline.append(patch.evalChild(14, 2))
    w = transPt(trot(-PI/3), w)
    new_H_outline.append(psub(new_H_outline[3], w))
    new_H_outline.append(patch.evalChild(6, 2))
    
    new_H = MetaTile(new_H_outline, patch.width * 2)
    for ch in [0, 9, 16, 27, 26, 6, 1, 8, 10, 15]:
        new_H.addChild(patch.children[ch]['T'], patch.children[ch]['geom'])
    
    new_P_outline = [p72, padd(p72, psub(bps1, llc)), bps1, llc]
    new_P = MetaTile(new_P_outline, patch.width * 2)
    for ch in [7, 2, 3, 4, 28]:
        new_P.addChild(patch.children[ch]['T'], patch.children[ch]['geom'])
    
    new_F_outline = [bps2, patch.evalChild(24, 2), patch.evalChild(25, 0), p252, padd(p252, psub(llc, bps1))]
    new_F = MetaTile(new_F_outline, patch.width * 2)
    for ch in [21, 20, 22, 23, 24, 25]:
        new_F.addChild(patch.children[ch]['T'], patch.children[ch]['geom'])
    
    AAA = new_H_outline[2]
    BBB = padd(new_H_outline[1], psub(new_H_outline[4], new_H_outline[5]))
    CCC = transPt(rotAbout(BBB, -PI/3), AAA)
    new_T_outline = [BBB, CCC, AAA]
    new_T = MetaTile(new_T_outline, patch.width * 2)
    new_T.addChild(patch.children[11]['T'], patch.children[11]['geom'])
    
    new_H.recentre()
    new_P.recentre()
    new_F.recentre()
    new_T.recentre()
    
    return [new_H, new_T, new_P, new_F]