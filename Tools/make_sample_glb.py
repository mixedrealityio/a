#!/usr/bin/env python3
"""Generates Samples/GearboxAssembly.glb: a small multi-level product structure.

    Gearbox
    ├── Housing                (assembly)
    │   ├── Base Plate
    │   ├── Side Wall Left
    │   ├── Side Wall Right
    │   └── Cover
    ├── Shaft Assembly         (assembly, rotated so the shafts run along X)
    │   ├── Shaft
    │   ├── Gear A
    │   └── Gear B
    └── Fasteners              (assembly; four instances of one shared mesh)
        ├── Bolt 1 .. Bolt 4

The file exercises: nested nodes, TRS and matrix transforms, mesh instancing,
several materials, uint16 indices and tightly packed accessors.
"""
import json
import math
import struct
import sys
from pathlib import Path


def box(sx, sy, sz):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    faces = [  # normal, four corners (counter-clockwise seen from outside)
        ((0, 0, 1), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
        ((0, 0, -1), [(hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz)]),
        ((1, 0, 0), [(hx, -hy, hz), (hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz)]),
        ((-1, 0, 0), [(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)]),
        ((0, 1, 0), [(-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz), (-hx, hy, -hz)]),
        ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
    ]
    pos, nor, idx = [], [], []
    for normal, corners in faces:
        base = len(pos)
        pos += corners
        nor += [normal] * 4
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return pos, nor, idx


def cylinder(radius, height, segments=32):
    """Axis along +Y, centred on the origin."""
    h = height / 2
    pos, nor, idx = [], [], []
    for i in range(segments):  # side, one quad per segment with flat-per-vertex radial normals
        a0 = 2 * math.pi * i / segments
        a1 = 2 * math.pi * (i + 1) / segments
        c0, s0, c1, s1 = math.cos(a0), math.sin(a0), math.cos(a1), math.sin(a1)
        base = len(pos)
        pos += [(radius * c0, -h, radius * s0), (radius * c1, -h, radius * s1),
                (radius * c1, h, radius * s1), (radius * c0, h, radius * s0)]
        nor += [(c0, 0, s0), (c1, 0, s1), (c1, 0, s1), (c0, 0, s0)]
        # counter-clockwise seen from outside (+X looking down -X, Y up, Z toward the viewer's right)
        idx += [base, base + 2, base + 1, base, base + 3, base + 2]
    for y, ny in ((h, 1), (-h, -1)):  # caps as fans
        centre = len(pos)
        pos.append((0, y, 0))
        nor.append((0, ny, 0))
        for i in range(segments):
            a = 2 * math.pi * i / segments
            pos.append((radius * math.cos(a), y, radius * math.sin(a)))
            nor.append((0, ny, 0))
        for i in range(segments):
            a, b = centre + 1 + i, centre + 1 + (i + 1) % segments
            idx += [centre, b, a] if ny > 0 else [centre, a, b]
    return pos, nor, idx


def quat_z(degrees):
    r = math.radians(degrees) / 2
    return [0, 0, math.sin(r), math.cos(r)]


class Builder:
    def __init__(self):
        self.bin = bytearray()
        self.views, self.accessors, self.meshes, self.materials, self.nodes = [], [], [], [], []

    def _view(self, data, target):
        while len(self.bin) % 4:
            self.bin.append(0)
        self.views.append({"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data), "target": target})
        self.bin += data
        return len(self.views) - 1

    def _accessor(self, view, ctype, count, atype, mn=None, mx=None):
        a = {"bufferView": view, "componentType": ctype, "count": count, "type": atype}
        if mn is not None:
            a["min"], a["max"] = mn, mx
        self.accessors.append(a)
        return len(self.accessors) - 1

    def material(self, name, rgba, metallic, roughness):
        self.materials.append({"name": name, "pbrMetallicRoughness": {
            "baseColorFactor": list(rgba), "metallicFactor": metallic, "roughnessFactor": roughness}})
        return len(self.materials) - 1

    def mesh(self, name, geometry, material):
        pos, nor, idx = geometry
        pv = self._view(b"".join(struct.pack("<3f", *p) for p in pos), 34962)
        nv = self._view(b"".join(struct.pack("<3f", *n) for n in nor), 34962)
        iv = self._view(b"".join(struct.pack("<H", i) for i in idx), 34963)
        mn = [min(p[k] for p in pos) for k in range(3)]
        mx = [max(p[k] for p in pos) for k in range(3)]
        prim = {"attributes": {"POSITION": self._accessor(pv, 5126, len(pos), "VEC3", mn, mx),
                               "NORMAL": self._accessor(nv, 5126, len(nor), "VEC3")},
                "indices": self._accessor(iv, 5123, len(idx), "SCALAR"),
                "material": material, "mode": 4}
        self.meshes.append({"name": name, "primitives": [prim]})
        return len(self.meshes) - 1

    def node(self, name, mesh=None, children=None, **transform):
        n = {"name": name}
        if mesh is not None:
            n["mesh"] = mesh
        if children:
            n["children"] = children
        n.update(transform)
        self.nodes.append(n)
        return len(self.nodes) - 1

    def glb(self, roots):
        doc = {"asset": {"version": "2.0", "generator": "make_sample_glb.py"},
               "scene": 0, "scenes": [{"nodes": roots}], "nodes": self.nodes, "meshes": self.meshes,
               "materials": self.materials, "accessors": self.accessors, "bufferViews": self.views,
               "buffers": [{"byteLength": len(self.bin)}]}
        js = json.dumps(doc, separators=(",", ":")).encode()
        js += b" " * (-len(js) % 4)
        bn = bytes(self.bin) + b"\0" * (-len(self.bin) % 4)
        total = 12 + 8 + len(js) + 8 + len(bn)
        return (struct.pack("<III", 0x46546C67, 2, total)
                + struct.pack("<II", len(js), 0x4E4F534A) + js
                + struct.pack("<II", len(bn), 0x004E4942) + bn)


def main(out):
    b = Builder()
    steel = b.material("Steel", (0.62, 0.64, 0.68, 1), 0.9, 0.35)
    alu = b.material("Aluminium", (0.80, 0.82, 0.85, 1), 0.8, 0.45)
    brass = b.material("Brass", (0.80, 0.60, 0.20, 1), 0.9, 0.30)
    red = b.material("Red anodised", (0.75, 0.10, 0.10, 1), 0.5, 0.40)
    dark = b.material("Black oxide", (0.08, 0.08, 0.09, 1), 0.7, 0.50)

    m_base = b.mesh("Base Plate", box(0.40, 0.03, 0.30), alu)
    m_wall = b.mesh("Side Wall", box(0.03, 0.20, 0.30), alu)
    m_cover = b.mesh("Cover", box(0.40, 0.02, 0.30), red)
    m_shaft = b.mesh("Shaft", cylinder(0.014, 0.42, 32), steel)
    m_gear_a = b.mesh("Gear A", cylinder(0.075, 0.035, 48), brass)
    m_gear_b = b.mesh("Gear B", cylinder(0.050, 0.035, 48), brass)
    m_bolt = b.mesh("Bolt", cylinder(0.010, 0.04, 16), dark)

    housing = b.node("Housing", children=[
        b.node("Base Plate", m_base, translation=[0, 0, 0]),
        b.node("Side Wall Left", m_wall, translation=[-0.185, 0.115, 0]),
        b.node("Side Wall Right", m_wall, translation=[0.185, 0.115, 0]),
        b.node("Cover", m_cover, translation=[0, 0.225, 0]),
    ])

    # The shaft assembly is rotated 90 degrees about Z, so its local Y axis runs along world X.
    shafts = b.node("Shaft Assembly", rotation=quat_z(90), translation=[0, 0.115, 0], children=[
        b.node("Shaft", m_shaft),
        b.node("Gear A", m_gear_a, translation=[0, -0.07, 0]),
        b.node("Gear B", m_gear_b, translation=[0, 0.08, 0.0]),
    ])

    # One mesh instanced four times; the last one uses an explicit column-major matrix.
    def mat_translate(x, y, z):
        return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]

    fasteners = b.node("Fasteners", children=[
        b.node("Bolt 1", m_bolt, translation=[-0.17, 0.035, -0.12]),
        b.node("Bolt 2", m_bolt, translation=[0.17, 0.035, -0.12]),
        b.node("Bolt 3", m_bolt, translation=[-0.17, 0.035, 0.12]),
        b.node("Bolt 4", m_bolt, matrix=mat_translate(0.17, 0.035, 0.12)),
    ])

    gearbox = b.node("Gearbox", children=[housing, shafts, fasteners])
    Path(out).write_bytes(b.glb([gearbox]))
    print(f"wrote {out} ({Path(out).stat().st_size} bytes)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent / "Samples" / "GearboxAssembly.glb"))
