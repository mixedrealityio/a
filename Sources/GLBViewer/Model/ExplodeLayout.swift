import Foundation
import simd

/// Computes exploded-view displacements from the product structure.
///
/// Every level of the tree spreads its children away from the centre of their parent, in proportion
/// to how far they already sit from it. Displacements are *relative to the parent*, so an assembly that
/// moves outward carries its sub-parts with it and those spread further on top of that.
enum ExplodeLayout {
    /// - Parameters:
    ///   - scope: The node whose subtree is exploded. The scope node itself does not move.
    ///   - bounds: Model-space bounds per node id (from the un-exploded model).
    ///   - amount: 0 = assembled, 1 = fully exploded.
    /// - Returns: Model-space displacement of each affected node relative to its parent.
    static func offsets(scope: ProductNode, bounds: [UUID: AABB], amount: Float) -> [UUID: SIMD3<Float>] {
        var result: [UUID: SIMD3<Float>] = [:]
        guard amount > 0 else { return result }

        func visit(_ parent: ProductNode) {
            if let parentBounds = bounds[parent.id], !parentBounds.isEmpty {
                let kids = parent.children.filter { !(bounds[$0.id]?.isEmpty ?? true) }
                if kids.count > 1 {
                    let radius = Swift.max(parentBounds.diagonal, 1e-6)
                    let size = parentBounds.size
                    var axis = SIMD3<Float>(0, 0, 0)
                    if size.x >= size.y, size.x >= size.z { axis.x = 1 }
                    else if size.y >= size.z { axis.y = 1 }
                    else { axis.z = 1 }

                    for (i, kid) in kids.enumerated() {
                        var d = bounds[kid.id]!.center - parentBounds.center
                        if simd_length(d) < 0.02 * radius {
                            // Concentric parts have no outward direction; fan them out along the longest axis.
                            let slot = Float(i) - Float(kids.count - 1) / 2
                            d = axis * slot * 0.1 * radius
                        }
                        result[kid.id] = d * (2 * amount)
                    }
                }
            }
            parent.children.forEach(visit)
        }

        visit(scope)
        return result
    }
}
