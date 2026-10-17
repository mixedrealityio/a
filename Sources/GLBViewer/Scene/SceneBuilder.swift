import Foundation
import ImageIO
import RealityKit
import simd
import UIKit

/// Tags a RealityKit entity with the product node it renders, so taps can be mapped back to the tree.
struct PartComponent: Component {
    var nodeID: UUID
}

/// RealityKit state for one rendered primitive.
final class PrimitiveRuntime {
    let entity: ModelEntity
    let data: PrimitiveData
    let originalMesh: MeshResource
    let baseMaterials: [any Material]
    /// True while slicing has removed every triangle of this primitive.
    var clippedAway = false

    init(entity: ModelEntity, data: PrimitiveData, originalMesh: MeshResource, baseMaterials: [any Material]) {
        self.entity = entity
        self.data = data
        self.originalMesh = originalMesh
        self.baseMaterials = baseMaterials
    }
}

/// RealityKit state for one product node.
final class NodeRuntime {
    let node: ProductNode
    let entity: Entity
    /// The translation from the file; exploding offsets from here.
    let baseTranslation: SIMD3<Float>
    var primitives: [PrimitiveRuntime] = []

    init(node: ProductNode, entity: Entity) {
        self.node = node
        self.entity = entity
        self.baseTranslation = entity.position
    }
}

@MainActor
enum SceneBuilder {
    /// Creates one entity per product node (mirroring the tree) and one `ModelEntity` per primitive.
    static func build(loaded: LoadedModel, under parent: Entity) throws -> [UUID: NodeRuntime] {
        var runtimes: [UUID: NodeRuntime] = [:]
        var materialCache: [Int: PhysicallyBasedMaterial] = [:]
        var textureCache: [Int: TextureResource] = [:]

        func texture(_ imageIndex: Int) -> TextureResource? {
            if let cached = textureCache[imageIndex] { return cached }
            guard
                let data = loaded.images[imageIndex],
                let source = CGImageSourceCreateWithData(data as CFData, nil),
                let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
                let resource = try? TextureResource.generate(from: image, options: .init(semantic: .color))
            else { return nil }
            textureCache[imageIndex] = resource
            return resource
        }

        func material(_ index: Int?) -> PhysicallyBasedMaterial {
            let key = index ?? -1
            if let cached = materialCache[key] { return cached }

            var m = PhysicallyBasedMaterial()
            if let index, let info = loaded.materials[safe: index] {
                let tex = info.imageIndex.flatMap(texture)
                m.baseColor = .init(
                    tint: color(info.baseColor.x, info.baseColor.y, info.baseColor.z, 1),
                    texture: tex.map { MaterialParameters.Texture($0) }
                )
                m.metallic = .init(scale: info.metallic)
                m.roughness = .init(scale: info.roughness)
                if info.emissive != .zero {
                    m.emissiveColor = .init(color: color(info.emissive.x, info.emissive.y, info.emissive.z, 1))
                    m.emissiveIntensity = 1
                }
                if info.blend, info.baseColor.w < 1 {
                    m.blending = .transparent(opacity: .init(scale: info.baseColor.w))
                }
                m.faceCulling = info.doubleSided ? .none : .back
            } else {
                // Primitive without a material: the glTF default is a white, fully metallic surface,
                // which renders black without an environment. A neutral grey is friendlier.
                m.baseColor = .init(tint: .lightGray)
                m.metallic = .init(scale: 0)
                m.roughness = .init(scale: 0.6)
            }
            materialCache[key] = m
            return m
        }

        func visit(_ node: ProductNode, under parentEntity: Entity) throws {
            let entity = Entity()
            entity.name = node.name
            entity.transform = Transform(matrix: node.localTransform)
            parentEntity.addChild(entity)

            let runtime = NodeRuntime(node: node, entity: entity)
            for data in node.primitives {
                let mesh = try makeMesh(data)
                let mat = material(data.materialIndex)
                let model = ModelEntity(mesh: mesh, materials: [mat])
                model.name = node.name
                model.components.set(PartComponent(nodeID: node.id))
                model.components.set(InputTargetComponent())
                model.components.set(CollisionComponent(shapes: [collisionShape(for: data)]))
                entity.addChild(model)
                runtime.primitives.append(
                    PrimitiveRuntime(entity: model, data: data, originalMesh: mesh, baseMaterials: [mat])
                )
            }
            runtimes[node.id] = runtime
            for child in node.children {
                try visit(child, under: entity)
            }
        }

        try visit(loaded.root, under: parent)
        return runtimes
    }

    static func makeMesh(_ p: PrimitiveData) throws -> MeshResource {
        var descriptor = MeshDescriptor(name: "part")
        descriptor.positions = MeshBuffers.Positions(p.positions)
        descriptor.normals = MeshBuffers.Normals(p.normals)
        if !p.uvs.isEmpty {
            // glTF's texture origin is top-left, RealityKit's is bottom-left.
            descriptor.textureCoordinates = MeshBuffers.TextureCoordinates(p.uvs.map { SIMD2<Float>($0.x, 1 - $0.y) })
        }
        descriptor.primitives = .triangles(p.indices)
        return try MeshResource.generate(from: [descriptor])
    }

    /// A cheap box around the unsliced geometry; good enough for picking and independent of slicing.
    private static func collisionShape(for p: PrimitiveData) -> ShapeResource {
        let b = p.bounds
        let size = simd_max(b.size, SIMD3<Float>(repeating: 1e-4))
        return ShapeResource.generateBox(size: size).offsetBy(translation: b.center)
    }

    static func highlighted(_ material: any Material) -> any Material {
        guard var pbr = material as? PhysicallyBasedMaterial else { return material }
        pbr.emissiveColor = .init(color: UIColor.systemTeal)
        pbr.emissiveIntensity = 0.7
        return pbr
    }

    /// glTF colour factors are linear; `UIColor` expects sRGB.
    private static func color(_ r: Float, _ g: Float, _ b: Float, _ a: Float) -> UIColor {
        func srgb(_ v: Float) -> CGFloat {
            let c = Swift.min(Swift.max(v, 0), 1)
            return CGFloat(c <= 0.0031308 ? c * 12.92 : 1.055 * powf(c, 1 / 2.4) - 0.055)
        }
        return UIColor(red: srgb(r), green: srgb(g), blue: srgb(b), alpha: CGFloat(a))
    }
}
