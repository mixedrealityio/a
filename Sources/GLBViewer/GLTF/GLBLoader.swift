import Foundation
import simd

enum GLBError: LocalizedError {
    case notGLB
    case truncated
    case unsupported(String)
    case invalid(String)

    var errorDescription: String? {
        switch self {
        case .notGLB: return "This file is not a binary glTF (.glb) file."
        case .truncated: return "The GLB file is truncated or corrupt."
        case .unsupported(let what): return "Unsupported glTF feature: \(what)."
        case .invalid(let what): return "Invalid glTF data: \(what)."
        }
    }
}

struct MaterialInfo: Sendable {
    var baseColor = SIMD4<Float>(1, 1, 1, 1)
    var metallic: Float = 1
    var roughness: Float = 1
    var emissive = SIMD3<Float>(0, 0, 0)
    var doubleSided = false
    var blend = false
    var imageIndex: Int?
}

/// Result of parsing a GLB: the product structure plus shared resources.
struct LoadedModel: @unchecked Sendable {
    var root: ProductNode
    var materials: [MaterialInfo]
    var images: [Int: Data]
    var warnings: [String]

    var nodeCount: Int { root.subtree.count }
    var partCount: Int { root.subtree.filter(\.isPart).count }
    var triangleCount: Int {
        root.subtree.reduce(0) { $0 + $1.primitives.reduce(0) { $0 + $1.triangleCount } }
    }
}

/// Splits a GLB container into its JSON and BIN chunks.
struct GLBContainer {
    let json: Data
    let bin: [UInt8]

    init(data: Data) throws {
        let b = [UInt8](data)
        guard b.count >= 12 else { throw GLBError.truncated }

        func u32(_ o: Int) -> Int {
            let b0 = UInt32(b[o])
            let b1 = UInt32(b[o + 1]) << 8
            let b2 = UInt32(b[o + 2]) << 16
            let b3 = UInt32(b[o + 3]) << 24
            return Int(b0 | b1 | b2 | b3)
        }

        guard u32(0) == 0x4654_6C67 else { throw GLBError.notGLB }   // "glTF"
        guard u32(4) == 2 else { throw GLBError.unsupported("glTF container version \(u32(4))") }

        let total = min(u32(8), b.count)
        var offset = 12
        var jsonChunk: Data?
        var binChunk: [UInt8] = []

        while offset + 8 <= total {
            let length = u32(offset)
            let type = u32(offset + 4)
            let start = offset + 8
            guard start + length <= b.count else { throw GLBError.truncated }
            if type == 0x4E4F_534A, jsonChunk == nil {                 // "JSON"
                jsonChunk = Data(b[start..<start + length])
            } else if type == 0x004E_4942, binChunk.isEmpty {          // "BIN\0"
                binChunk = Array(b[start..<start + length])
            }
            offset = start + ((length + 3) & ~3)
        }

        guard let jsonChunk else { throw GLBError.invalid("missing JSON chunk") }
        self.json = jsonChunk
        self.bin = binChunk
    }
}

/// Reads typed data out of accessors backed by the GLB's BIN chunk.
struct GLTFReader {
    let root: GLTFRoot
    let bin: [UInt8]

    static func componentCount(_ type: String) -> Int? {
        switch type {
        case "SCALAR": return 1
        case "VEC2": return 2
        case "VEC3": return 3
        case "VEC4", "MAT2": return 4
        case "MAT3": return 9
        case "MAT4": return 16
        default: return nil
        }
    }

    static func componentSize(_ componentType: Int) -> Int? {
        switch componentType {
        case 5120, 5121: return 1
        case 5122, 5123: return 2
        case 5125, 5126: return 4
        default: return nil
        }
    }

    private static func component(_ p: UnsafeRawBufferPointer, _ off: Int, _ type: Int, _ normalized: Bool) -> Float {
        switch type {
        case 5126:
            return p.loadUnaligned(fromByteOffset: off, as: Float.self)
        case 5120:
            let v = Float(p.loadUnaligned(fromByteOffset: off, as: Int8.self))
            return normalized ? Swift.max(v / 127, -1) : v
        case 5121:
            let v = Float(p.loadUnaligned(fromByteOffset: off, as: UInt8.self))
            return normalized ? v / 255 : v
        case 5122:
            let v = Float(p.loadUnaligned(fromByteOffset: off, as: Int16.self))
            return normalized ? Swift.max(v / 32767, -1) : v
        case 5123:
            let v = Float(p.loadUnaligned(fromByteOffset: off, as: UInt16.self))
            return normalized ? v / 65535 : v
        case 5125:
            return Float(p.loadUnaligned(fromByteOffset: off, as: UInt32.self))
        default:
            return 0
        }
    }

    private func bufferView(_ index: Int) throws -> GLTFRoot.BufferView {
        guard let view = root.bufferViews?[safe: index] else {
            throw GLBError.invalid("bufferView \(index) does not exist")
        }
        guard view.buffer == 0, let buffer = root.buffers?.first, buffer.uri == nil else {
            throw GLBError.unsupported("external or multiple buffers")
        }
        return view
    }

    /// Returns the accessor's values flattened, plus the number of components per element.
    func floats(_ accessorIndex: Int) throws -> (values: [Float], components: Int) {
        guard let a = root.accessors?[safe: accessorIndex] else {
            throw GLBError.invalid("accessor \(accessorIndex) does not exist")
        }
        if a.sparse != nil { throw GLBError.unsupported("sparse accessors") }
        guard let nc = Self.componentCount(a.type), let cs = Self.componentSize(a.componentType) else {
            throw GLBError.invalid("accessor \(accessorIndex) has an unknown type")
        }
        var out = [Float](repeating: 0, count: a.count * nc)
        guard let viewIndex = a.bufferView else { return (out, nc) }   // spec: all zeros

        let view = try bufferView(viewIndex)
        let stride = view.byteStride ?? nc * cs
        let start = (view.byteOffset ?? 0) + (a.byteOffset ?? 0)
        let normalized = a.normalized ?? false

        try bin.withUnsafeBytes { p in
            for i in 0..<a.count {
                let base = start + i * stride
                guard base + nc * cs <= p.count else {
                    throw GLBError.invalid("accessor \(accessorIndex) reads past the end of the buffer")
                }
                for c in 0..<nc {
                    out[i * nc + c] = Self.component(p, base + c * cs, a.componentType, normalized)
                }
            }
        }
        return (out, nc)
    }

    func indices(_ accessorIndex: Int) throws -> [UInt32] {
        guard let a = root.accessors?[safe: accessorIndex] else {
            throw GLBError.invalid("accessor \(accessorIndex) does not exist")
        }
        guard [5121, 5123, 5125].contains(a.componentType), a.type == "SCALAR" else {
            throw GLBError.invalid("indices accessor \(accessorIndex) has an unsupported type")
        }
        let (values, _) = try floats(accessorIndex)
        // Float is exact for integers below 2^24; larger index values fall back to a direct read.
        if let max = values.max(), max >= 16_777_216 {
            return try rawUInt32Indices(accessorIndex)
        }
        return values.map { UInt32($0) }
    }

    private func rawUInt32Indices(_ accessorIndex: Int) throws -> [UInt32] {
        let a = root.accessors![accessorIndex]
        let view = try bufferView(a.bufferView ?? 0)
        let cs = Self.componentSize(a.componentType) ?? 4
        let stride = view.byteStride ?? cs
        let start = (view.byteOffset ?? 0) + (a.byteOffset ?? 0)
        var out = [UInt32](repeating: 0, count: a.count)
        try bin.withUnsafeBytes { p in
            for i in 0..<a.count {
                let o = start + i * stride
                guard o + cs <= p.count else { throw GLBError.invalid("indices read past the end of the buffer") }
                switch a.componentType {
                case 5121: out[i] = UInt32(p.loadUnaligned(fromByteOffset: o, as: UInt8.self))
                case 5123: out[i] = UInt32(p.loadUnaligned(fromByteOffset: o, as: UInt16.self))
                default: out[i] = p.loadUnaligned(fromByteOffset: o, as: UInt32.self)
                }
            }
        }
        return out
    }

    func imageData(_ imageIndex: Int) -> Data? {
        guard let image = root.images?[safe: imageIndex] else { return nil }
        if let viewIndex = image.bufferView, let view = try? bufferView(viewIndex) {
            let start = view.byteOffset ?? 0
            guard start + view.byteLength <= bin.count else { return nil }
            return Data(bin[start..<start + view.byteLength])
        }
        if let uri = image.uri, uri.hasPrefix("data:"), let comma = uri.firstIndex(of: ",") {
            return Data(base64Encoded: String(uri[uri.index(after: comma)...]))
        }
        return nil
    }
}

enum GLBLoader {
    /// Extensions that change how geometry is encoded; without a decoder the meshes would be garbage.
    private static let unsupportedRequired: Set<String> = [
        "KHR_draco_mesh_compression", "EXT_meshopt_compression",
    ]

    static func load(data: Data, name: String) throws -> LoadedModel {
        let container = try GLBContainer(data: data)
        let root = try JSONDecoder().decode(GLTFRoot.self, from: container.json)

        for ext in root.extensionsRequired ?? [] where unsupportedRequired.contains(ext) {
            throw GLBError.unsupported("\(ext) (re-export the model without mesh compression)")
        }

        let reader = GLTFReader(root: root, bin: container.bin)
        var warnings: [String] = []

        let materials = (root.materials ?? []).map { materialInfo($0, root: root) }
        var images: [Int: Data] = [:]
        for imageIndex in Set(materials.compactMap(\.imageIndex)) {
            if let bytes = reader.imageData(imageIndex) {
                images[imageIndex] = bytes
            } else {
                warnings.append("Texture image \(imageIndex) could not be read; using the base color only.")
            }
        }

        var meshCache: [Int: [PrimitiveData]] = [:]
        func primitives(forMesh meshIndex: Int) throws -> [PrimitiveData] {
            if let cached = meshCache[meshIndex] { return cached }
            guard let mesh = root.meshes?[safe: meshIndex] else {
                throw GLBError.invalid("mesh \(meshIndex) does not exist")
            }
            var result: [PrimitiveData] = []
            for prim in mesh.primitives {
                if let p = try makePrimitive(prim, reader: reader, warnings: &warnings) {
                    result.append(p)
                }
            }
            meshCache[meshIndex] = result
            return result
        }

        let sceneRoot = ProductNode(name: name)
        var visiting = Set<Int>()
        var skinWarningIssued = false

        func build(_ index: Int, into parent: ProductNode) throws {
            guard let n = root.nodes?[safe: index] else {
                throw GLBError.invalid("node \(index) does not exist")
            }
            guard visiting.insert(index).inserted else {
                throw GLBError.invalid("node hierarchy contains a cycle")
            }
            defer { visiting.remove(index) }

            let displayName = n.name
                ?? n.mesh.flatMap { root.meshes?[safe: $0]?.name }
                ?? "Node \(index)"
            let node = ProductNode(name: displayName, localTransform: transform(of: n))
            if let meshIndex = n.mesh {
                node.primitives = try primitives(forMesh: meshIndex)
            }
            if n.skin != nil, !skinWarningIssued {
                skinWarningIssued = true
                warnings.append("Skinned meshes are shown in their bind pose.")
            }
            parent.addChild(node)
            for child in n.children ?? [] {
                try build(child, into: node)
            }
        }

        let rootIndices: [Int]
        let sceneIndex = root.scene ?? 0
        if let scene = root.scenes?[safe: sceneIndex], let nodes = scene.nodes {
            rootIndices = nodes
        } else {
            let referenced = Set((root.nodes ?? []).flatMap { $0.children ?? [] })
            rootIndices = (root.nodes ?? []).indices.filter { !referenced.contains($0) }
        }
        for index in rootIndices {
            try build(index, into: sceneRoot)
        }

        guard sceneRoot.subtree.contains(where: \.isPart) else {
            throw GLBError.invalid("the file contains no triangle meshes")
        }

        return LoadedModel(root: sceneRoot, materials: materials, images: images, warnings: warnings)
    }

    // MARK: - Nodes

    private static func transform(of n: GLTFRoot.Node) -> simd_float4x4 {
        if let m = n.matrix, m.count == 16 {
            return simd_float4x4(columns: (
                SIMD4<Float>(m[0], m[1], m[2], m[3]),
                SIMD4<Float>(m[4], m[5], m[6], m[7]),
                SIMD4<Float>(m[8], m[9], m[10], m[11]),
                SIMD4<Float>(m[12], m[13], m[14], m[15])
            ))
        }
        var t = SIMD3<Float>(0, 0, 0)
        var q = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        var s = SIMD3<Float>(1, 1, 1)
        if let v = n.translation, v.count == 3 { t = SIMD3<Float>(v[0], v[1], v[2]) }
        if let v = n.rotation, v.count == 4 {
            q = simd_normalize(simd_quatf(ix: v[0], iy: v[1], iz: v[2], r: v[3]))
        }
        if let v = n.scale, v.count == 3 { s = SIMD3<Float>(v[0], v[1], v[2]) }
        return simd_float4x4(translation: t, rotation: q, scale: s)
    }

    // MARK: - Materials

    private static func materialInfo(_ m: GLTFRoot.Material, root: GLTFRoot) -> MaterialInfo {
        var info = MaterialInfo()
        if let pbr = m.pbrMetallicRoughness {
            if let f = pbr.baseColorFactor, f.count == 4 {
                info.baseColor = SIMD4<Float>(f[0], f[1], f[2], f[3])
            }
            info.metallic = pbr.metallicFactor ?? 1
            info.roughness = pbr.roughnessFactor ?? 1
            if let tex = pbr.baseColorTexture {
                info.imageIndex = root.textures?[safe: tex.index]?.source
            }
        }
        if let e = m.emissiveFactor, e.count == 3 { info.emissive = SIMD3<Float>(e[0], e[1], e[2]) }
        info.doubleSided = m.doubleSided ?? false
        info.blend = (m.alphaMode == "BLEND")
        return info
    }

    // MARK: - Primitives

    private static func makePrimitive(
        _ prim: GLTFRoot.Primitive,
        reader: GLTFReader,
        warnings: inout [String]
    ) throws -> PrimitiveData? {
        let mode = prim.mode ?? 4
        guard [4, 5, 6].contains(mode) else {
            warnings.append("A non-triangle primitive (mode \(mode)) was skipped.")
            return nil
        }
        guard let positionAccessor = prim.attributes["POSITION"] else { return nil }

        let (pf, pc) = try reader.floats(positionAccessor)
        guard pc == 3 else { throw GLBError.invalid("POSITION must be VEC3") }
        let positions = stride(from: 0, to: pf.count, by: 3).map { SIMD3<Float>(pf[$0], pf[$0 + 1], pf[$0 + 2]) }

        var normals: [SIMD3<Float>] = []
        if let accessor = prim.attributes["NORMAL"] {
            let (nf, nc) = try reader.floats(accessor)
            if nc == 3, nf.count == pf.count {
                normals = stride(from: 0, to: nf.count, by: 3).map { SIMD3<Float>(nf[$0], nf[$0 + 1], nf[$0 + 2]) }
            }
        }

        var uvs: [SIMD2<Float>] = []
        if let accessor = prim.attributes["TEXCOORD_0"] {
            let (uf, uc) = try reader.floats(accessor)
            if uc == 2, uf.count / 2 == positions.count {
                uvs = stride(from: 0, to: uf.count, by: 2).map { SIMD2<Float>(uf[$0], uf[$0 + 1]) }
            }
        }

        let raw: [UInt32]
        if let accessor = prim.indices {
            raw = try reader.indices(accessor)
        } else {
            raw = (0..<UInt32(positions.count)).map { $0 }
        }

        var triangles: [UInt32] = []
        switch mode {
        case 4:
            triangles = raw
        case 5:   // strip
            if raw.count >= 3 {
                for i in 0..<(raw.count - 2) {
                    if i % 2 == 0 {
                        triangles += [raw[i], raw[i + 1], raw[i + 2]]
                    } else {
                        triangles += [raw[i + 1], raw[i], raw[i + 2]]
                    }
                }
            }
        default:  // fan
            if raw.count >= 3 {
                for i in 1..<(raw.count - 1) {
                    triangles += [raw[0], raw[i], raw[i + 1]]
                }
            }
        }

        // Drop incomplete or out-of-range triangles instead of crashing later in RealityKit.
        var indices: [UInt32] = []
        indices.reserveCapacity(triangles.count)
        let vertexCount = UInt32(positions.count)
        var t = 0
        while t + 2 < triangles.count {
            let a = triangles[t], b = triangles[t + 1], c = triangles[t + 2]
            if a < vertexCount, b < vertexCount, c < vertexCount {
                indices += [a, b, c]
            }
            t += 3
        }
        guard !indices.isEmpty else { return nil }

        if normals.count != positions.count {
            normals = smoothNormals(positions: positions, indices: indices)
        }

        return PrimitiveData(
            positions: positions,
            normals: normals,
            uvs: uvs,
            indices: indices,
            materialIndex: prim.material
        )
    }

    /// Area-weighted vertex normals for primitives that ship without NORMAL.
    static func smoothNormals(positions: [SIMD3<Float>], indices: [UInt32]) -> [SIMD3<Float>] {
        var normals = [SIMD3<Float>](repeating: .zero, count: positions.count)
        var t = 0
        while t + 2 < indices.count {
            let i0 = Int(indices[t]), i1 = Int(indices[t + 1]), i2 = Int(indices[t + 2])
            let n = simd_cross(positions[i1] - positions[i0], positions[i2] - positions[i0])
            normals[i0] += n
            normals[i1] += n
            normals[i2] += n
            t += 3
        }
        return normals.map { n in
            let len = simd_length(n)
            return len > 1e-12 ? n / len : SIMD3<Float>(0, 1, 0)
        }
    }
}
