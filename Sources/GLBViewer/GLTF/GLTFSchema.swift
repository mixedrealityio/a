import Foundation

/// The subset of the glTF 2.0 JSON schema that the viewer reads.
struct GLTFRoot: Decodable {
    struct Scene: Decodable {
        var name: String?
        var nodes: [Int]?
    }

    struct Node: Decodable {
        var name: String?
        var children: [Int]?
        var mesh: Int?
        var skin: Int?
        var matrix: [Float]?
        var translation: [Float]?
        var rotation: [Float]?
        var scale: [Float]?
    }

    struct Primitive: Decodable {
        var attributes: [String: Int]
        var indices: Int?
        var material: Int?
        var mode: Int?
    }

    struct Mesh: Decodable {
        var name: String?
        var primitives: [Primitive]
    }

    struct Sparse: Decodable {
        var count: Int
    }

    struct Accessor: Decodable {
        var bufferView: Int?
        var byteOffset: Int?
        var componentType: Int
        var normalized: Bool?
        var count: Int
        var type: String
        var sparse: Sparse?
    }

    struct BufferView: Decodable {
        var buffer: Int
        var byteOffset: Int?
        var byteLength: Int
        var byteStride: Int?
    }

    struct Buffer: Decodable {
        var byteLength: Int
        var uri: String?
    }

    struct TextureInfo: Decodable {
        var index: Int
        var texCoord: Int?
    }

    struct PBR: Decodable {
        var baseColorFactor: [Float]?
        var baseColorTexture: TextureInfo?
        var metallicFactor: Float?
        var roughnessFactor: Float?
    }

    struct Material: Decodable {
        var name: String?
        var pbrMetallicRoughness: PBR?
        var emissiveFactor: [Float]?
        var alphaMode: String?
        var doubleSided: Bool?
    }

    struct Texture: Decodable {
        var source: Int?
    }

    struct Image: Decodable {
        var bufferView: Int?
        var uri: String?
        var mimeType: String?
    }

    var scene: Int?
    var scenes: [Scene]?
    var nodes: [Node]?
    var meshes: [Mesh]?
    var accessors: [Accessor]?
    var bufferViews: [BufferView]?
    var buffers: [Buffer]?
    var materials: [Material]?
    var textures: [Texture]?
    var images: [Image]?
    var extensionsRequired: [String]?
}
