import Foundation
import simd

/// Clips triangle meshes against a plane. Used for the slicing feature.
enum MeshClipper {
    /// Half-space `dot(normal, p) + d >= 0` in the space the plane is defined in (model space).
    struct Plane: Sendable {
        var normal: SIMD3<Float>
        var d: Float

        func distance(to p: SIMD3<Float>) -> Float { simd_dot(normal, p) + d }

        /// Plane through `point` that keeps the side `normal` points to.
        init(point: SIMD3<Float>, normal: SIMD3<Float>) {
            let n = simd_normalize(normal)
            self.normal = n
            self.d = -simd_dot(n, point)
        }
    }

    /// Clips `primitive` (given in node-local space) against a plane defined in model space.
    ///
    /// Instead of transforming every vertex into model space, the plane is pulled back into the
    /// primitive's local space: `n·(A p + t) + d = (Aᵀn)·p + (n·t + d)`. Interpolation parameters are
    /// scale-invariant, so the local plane does not need normalising.
    ///
    /// Returns nil when nothing of the primitive remains.
    static func clip(_ primitive: PrimitiveData, localToModel m: simd_float4x4, plane: Plane) -> PrimitiveData? {
        let a = m.linear
        let localNormal = SIMD3<Float>(
            simd_dot(a.columns.0, plane.normal),
            simd_dot(a.columns.1, plane.normal),
            simd_dot(a.columns.2, plane.normal)
        )
        let localD = simd_dot(plane.normal, m.translationPart) + plane.d
        return clip(primitive, localNormal: localNormal, localD: localD)
    }

    static func clip(_ primitive: PrimitiveData, localNormal: SIMD3<Float>, localD: Float) -> PrimitiveData? {
        var positions = primitive.positions
        var normals = primitive.normals
        var uvs = primitive.uvs
        let hasUV = !uvs.isEmpty
        let dist = positions.map { simd_dot(localNormal, $0) + localD }

        var out: [UInt32] = []
        out.reserveCapacity(primitive.indices.count)

        // One new vertex per cut edge, shared between the triangles on both sides of that edge.
        var edgeCache: [UInt64: UInt32] = [:]

        func cut(_ i: UInt32, _ j: UInt32) -> UInt32 {
            let lo = Swift.min(i, j), hi = Swift.max(i, j)
            let key = UInt64(lo) << 32 | UInt64(hi)
            if let existing = edgeCache[key] { return existing }

            let dLo = dist[Int(lo)], dHi = dist[Int(hi)]
            let t = dLo / (dLo - dHi)
            let pos = positions[Int(lo)] + (positions[Int(hi)] - positions[Int(lo)]) * t
            var nor = normals[Int(lo)] + (normals[Int(hi)] - normals[Int(lo)]) * t
            let len = simd_length(nor)
            nor = len > 1e-12 ? nor / len : normals[Int(lo)]

            let index = UInt32(positions.count)
            positions.append(pos)
            normals.append(nor)
            if hasUV { uvs.append(uvs[Int(lo)] + (uvs[Int(hi)] - uvs[Int(lo)]) * t) }
            edgeCache[key] = index
            return index
        }

        var t = 0
        while t + 2 < primitive.indices.count {
            let tri = [primitive.indices[t], primitive.indices[t + 1], primitive.indices[t + 2]]
            t += 3
            let inside = tri.map { dist[Int($0)] >= 0 }
            let insideCount = inside.filter { $0 }.count

            if insideCount == 3 {
                out += tri
            } else if insideCount > 0 {
                // Sutherland–Hodgman against a single plane: yields a triangle or a quad.
                var polygon: [UInt32] = []
                for k in 0..<3 {
                    let next = (k + 1) % 3
                    if inside[k] { polygon.append(tri[k]) }
                    if inside[k] != inside[next] { polygon.append(cut(tri[k], tri[next])) }
                }
                if polygon.count >= 3 {
                    for k in 1..<(polygon.count - 1) {
                        out += [polygon[0], polygon[k], polygon[k + 1]]
                    }
                }
            }
        }

        guard !out.isEmpty else { return nil }
        // Unreferenced vertices are kept: harmless for RealityKit and avoids a remap pass.
        return PrimitiveData(
            positions: positions,
            normals: normals,
            uvs: uvs,
            indices: out,
            materialIndex: primitive.materialIndex
        )
    }
}
