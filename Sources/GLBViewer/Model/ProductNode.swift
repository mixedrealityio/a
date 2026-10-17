import Foundation
import Observation
import simd

/// Triangle geometry of one glTF primitive, in the owning node's local space.
struct PrimitiveData: Sendable {
    var positions: [SIMD3<Float>]
    var normals: [SIMD3<Float>]
    /// Empty when the primitive has no TEXCOORD_0. Stored with glTF's top-left origin.
    var uvs: [SIMD2<Float>]
    var indices: [UInt32]
    var materialIndex: Int?

    var triangleCount: Int { indices.count / 3 }

    var bounds: AABB {
        var b = AABB()
        for p in positions { b.expand(p) }
        return b
    }
}

/// One entry of the product structure (assembly, sub-assembly or part).
///
/// A node that carries `primitives` is a part; nodes with children are assemblies.
/// Only `isHidden` is observed by SwiftUI; everything else is fixed after loading.
@Observable
final class ProductNode: Identifiable {
    let id = UUID()
    @ObservationIgnored var name: String
    @ObservationIgnored var localTransform: simd_float4x4
    @ObservationIgnored private(set) var children: [ProductNode] = []
    @ObservationIgnored weak var parent: ProductNode?
    @ObservationIgnored var primitives: [PrimitiveData] = []

    /// Hides this node and its whole subtree.
    var isHidden = false
    /// Hides only this node's own geometry (used to isolate a subtree below an assembly that has geometry).
    @ObservationIgnored var isOwnGeometryHidden = false

    init(name: String, localTransform: simd_float4x4 = matrix_identity_float4x4) {
        self.name = name
        self.localTransform = localTransform
    }

    func addChild(_ child: ProductNode) {
        child.parent = self
        children.append(child)
    }

    /// For `List(_:children:)`; nil marks a leaf.
    var childNodes: [ProductNode]? { children.isEmpty ? nil : children }

    var isPart: Bool { !primitives.isEmpty }

    var worldTransform: simd_float4x4 {
        (parent?.worldTransform ?? matrix_identity_float4x4) * localTransform
    }

    /// This node and all of its descendants, depth first.
    var subtree: [ProductNode] {
        var result: [ProductNode] = []
        func walk(_ n: ProductNode) {
            result.append(n)
            n.children.forEach(walk)
        }
        walk(self)
        return result
    }

    var ancestors: [ProductNode] {
        var result: [ProductNode] = []
        var current = parent
        while let p = current {
            result.append(p)
            current = p.parent
        }
        return result
    }
}
