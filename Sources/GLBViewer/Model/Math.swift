import Foundation
import simd

/// Axis-aligned bounding box. An empty box has `min > max`.
struct AABB: Sendable {
    var min = SIMD3<Float>(repeating: .infinity)
    var max = SIMD3<Float>(repeating: -.infinity)

    var isEmpty: Bool { min.x > max.x }
    var center: SIMD3<Float> { isEmpty ? .zero : (min + max) * 0.5 }
    var size: SIMD3<Float> { isEmpty ? .zero : max - min }
    var diagonal: Float { simd_length(size) }

    mutating func expand(_ p: SIMD3<Float>) {
        min = simd_min(min, p)
        max = simd_max(max, p)
    }

    mutating func formUnion(_ other: AABB) {
        guard !other.isEmpty else { return }
        expand(other.min)
        expand(other.max)
    }
}

extension simd_float4x4 {
    func transformPoint(_ p: SIMD3<Float>) -> SIMD3<Float> {
        let v = self * SIMD4<Float>(p.x, p.y, p.z, 1)
        return SIMD3<Float>(v.x, v.y, v.z)
    }

    /// Upper-left 3x3 block (rotation, scale, shear).
    var linear: simd_float3x3 {
        simd_float3x3(columns: (
            SIMD3<Float>(columns.0.x, columns.0.y, columns.0.z),
            SIMD3<Float>(columns.1.x, columns.1.y, columns.1.z),
            SIMD3<Float>(columns.2.x, columns.2.y, columns.2.z)
        ))
    }

    var translationPart: SIMD3<Float> {
        SIMD3<Float>(columns.3.x, columns.3.y, columns.3.z)
    }

    /// Composes translation * rotation * scale (glTF TRS order).
    init(translation t: SIMD3<Float>, rotation q: simd_quatf, scale s: SIMD3<Float>) {
        let r = simd_float3x3(q)
        self.init(columns: (
            SIMD4<Float>(r.columns.0 * s.x, 0),
            SIMD4<Float>(r.columns.1 * s.y, 0),
            SIMD4<Float>(r.columns.2 * s.z, 0),
            SIMD4<Float>(t, 1)
        ))
    }
}

extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
