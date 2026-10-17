import Foundation
import Observation
import RealityKit
import simd

enum SliceAxis: String, CaseIterable, Identifiable {
    case x = "X", y = "Y", z = "Z"
    var id: String { rawValue }
    var vector: SIMD3<Float> {
        switch self {
        case .x: return SIMD3<Float>(1, 0, 0)
        case .y: return SIMD3<Float>(0, 1, 0)
        case .z: return SIMD3<Float>(0, 0, 1)
        }
    }
    var index: Int {
        switch self {
        case .x: return 0
        case .y: return 1
        case .z: return 2
        }
    }
}

enum ExplodeScope: String, CaseIterable, Identifiable {
    case model = "Whole model"
    case selection = "Selection"
    var id: String { rawValue }
}

/// Owns the loaded product structure and the RealityKit scene, and applies every view operation:
/// selection, visibility, scale, exploded view and slicing.
@MainActor
@Observable
final class ViewerModel {
    // MARK: Observable state (drives SwiftUI)

    private(set) var root: ProductNode?
    private(set) var fileName = ""
    private(set) var summary = ""
    var statusMessage: String?
    private(set) var isLoading = false

    var selectedID: UUID? {
        didSet { if selectedID != oldValue { refreshHighlight(); if explodeScope == .selection { applyExplode() } } }
    }

    var userScale: Float = 1 { didSet { applyTransform() } }
    var yaw: Float = 0 { didSet { applyTransform() } }

    var explodeAmount: Float = 0
    var explodeScope: ExplodeScope = .model

    var sliceEnabled = false
    var sliceAxis: SliceAxis = .y
    var sliceOffset: Float = 0.5      // 0...1 across the model's extent on `sliceAxis`
    var sliceFlip = false
    var showSlicePlane = true

    // MARK: Scene (not observed)

    /// Add this to a `RealityView`. User scale and rotation are applied here.
    @ObservationIgnored let contentRoot = Entity()
    @ObservationIgnored private let normalizer = Entity()
    @ObservationIgnored private let slicePlane = ModelEntity()
    @ObservationIgnored private var runtimes: [UUID: NodeRuntime] = [:]
    @ObservationIgnored private var worldTransforms: [UUID: simd_float4x4] = [:]
    @ObservationIgnored private var bounds: [UUID: AABB] = [:]
    @ObservationIgnored private var modelBounds = AABB()
    @ObservationIgnored private(set) var fitScale: Float = 1
    @ObservationIgnored private var highlighted: Set<UUID> = []
    @ObservationIgnored private var sliceTask: Task<Void, Never>?
    @ObservationIgnored private var sliceGeneration = 0

    init() {
        contentRoot.addChild(normalizer)
    }

    var hasModel: Bool { root != nil }
    var selectedNode: ProductNode? { root?.subtree.first { $0.id == selectedID } }

    /// Scale at which the model has the size it was authored at (glTF units are metres).
    var realSizeScale: Float { fitScale > 0 ? 1 / fitScale : 1 }

    // MARK: - Loading

    func load(url: URL) async {
        isLoading = true
        statusMessage = nil
        defer { isLoading = false }

        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }

        do {
            let data = try Data(contentsOf: url)
            try await ingest(data: data, name: url.deletingPathExtension().lastPathComponent)
        } catch {
            statusMessage = error.localizedDescription
        }
    }

    func loadBundledSample() async {
        guard let url = Bundle.main.url(forResource: "GearboxAssembly", withExtension: "glb") else {
            statusMessage = "The sample model is missing from the app bundle."
            return
        }
        await load(url: url)
    }

    private func ingest(data: Data, name: String) async throws {
        let loaded = try await Task.detached(priority: .userInitiated) {
            try GLBLoader.load(data: data, name: name)
        }.value
        try install(loaded)
        fileName = name
        summary = "\(loaded.partCount) parts · \(loaded.triangleCount.formatted()) triangles"
        if !loaded.warnings.isEmpty {
            statusMessage = Array(Set(loaded.warnings)).sorted().joined(separator: "\n")
        }
    }

    private func install(_ loaded: LoadedModel) throws {
        // Reset previous model.
        sliceTask?.cancel()
        normalizer.children.removeAll()
        runtimes = [:]
        worldTransforms = [:]
        bounds = [:]
        highlighted = []
        selectedID = nil
        explodeAmount = 0
        explodeScope = .model
        sliceEnabled = false
        sliceOffset = 0.5
        sliceFlip = false
        userScale = 1
        yaw = 0

        // Index the tree: world transforms and model-space bounds per node.
        modelBounds = indexTree(loaded.root, parentWorld: matrix_identity_float4x4)
        let maxDim = Swift.max(modelBounds.size.x, Swift.max(modelBounds.size.y, modelBounds.size.z))
        fitScale = maxDim > 0 ? 0.6 / maxDim : 1
        normalizer.scale = SIMD3<Float>(repeating: fitScale)
        normalizer.position = -modelBounds.center * fitScale

        runtimes = try SceneBuilder.build(loaded: loaded, under: normalizer)
        buildSlicePlane()
        root = loaded.root
        applyVisibility()
    }

    @discardableResult
    private func indexTree(_ node: ProductNode, parentWorld: simd_float4x4) -> AABB {
        let world = parentWorld * node.localTransform
        worldTransforms[node.id] = world
        var b = AABB()
        for primitive in node.primitives {
            for p in primitive.positions { b.expand(world.transformPoint(p)) }
        }
        for child in node.children {
            b.formUnion(indexTree(child, parentWorld: world))
        }
        bounds[node.id] = b
        return b
    }

    // MARK: - Selection

    /// Selects the part that owns `entity` (or one of its ancestors).
    func select(entity: Entity) {
        var current: Entity? = entity
        while let e = current {
            if let part = e.components[PartComponent.self] {
                selectedID = part.nodeID
                return
            }
            current = e.parent
        }
    }

    func selectParent() {
        if let parent = selectedNode?.parent { selectedID = parent.id }
    }

    private func refreshHighlight() {
        for id in highlighted {
            runtimes[id]?.primitives.forEach { $0.entity.model?.materials = $0.baseMaterials }
        }
        highlighted = []
        guard let node = selectedNode, node.id != root?.id else { return }
        for n in node.subtree {
            guard let runtime = runtimes[n.id] else { continue }
            for p in runtime.primitives {
                p.entity.model?.materials = p.baseMaterials.map(SceneBuilder.highlighted)
            }
            highlighted.insert(n.id)
        }
    }

    // MARK: - Visibility ("remove parts")

    /// Removes the selected node and everything below it from view.
    func removeSelectedSubtree() {
        guard let node = selectedNode else { return }
        node.isHidden = true
        applyVisibility()
        scheduleSlice()
    }

    func restoreSelectedSubtree() {
        guard let node = selectedNode else { return }
        for n in node.subtree {
            n.isHidden = false
            n.isOwnGeometryHidden = false
        }
        // A hidden ancestor would keep the subtree invisible.
        for a in node.ancestors { a.isHidden = false }
        applyVisibility()
        scheduleSlice()
    }

    /// Hides everything except the selected subtree (and keeps its ancestors so it stays in place).
    func isolateSelectedSubtree() {
        guard let root, let node = selectedNode else { return }
        let keep = Set(node.subtree.map(\.id))
        let path = Set(node.ancestors.map(\.id))
        for n in root.subtree {
            n.isHidden = !(keep.contains(n.id) || path.contains(n.id))
            n.isOwnGeometryHidden = path.contains(n.id)
        }
        applyVisibility()
        scheduleSlice()
    }

    func removeAllParts() {
        root?.subtree.forEach { $0.isHidden = true }
        applyVisibility()
    }

    func restoreAllParts() {
        root?.subtree.forEach {
            $0.isHidden = false
            $0.isOwnGeometryHidden = false
        }
        applyVisibility()
        scheduleSlice()
    }

    func toggleHidden(_ node: ProductNode) {
        node.isHidden.toggle()
        if !node.isHidden { node.isOwnGeometryHidden = false }
        applyVisibility()
        scheduleSlice()
    }

    func applyVisibility() {
        for runtime in runtimes.values {
            runtime.entity.isEnabled = !runtime.node.isHidden
            for p in runtime.primitives {
                p.entity.isEnabled = !p.clippedAway && !runtime.node.isOwnGeometryHidden
            }
        }
    }

    // MARK: - Scale and rotation

    func applyTransform() {
        let s = Swift.min(Swift.max(userScale, 0.05), 20)
        contentRoot.scale = SIMD3<Float>(repeating: s)
        contentRoot.orientation = simd_quatf(angle: yaw, axis: SIMD3<Float>(0, 1, 0))
    }

    // MARK: - Exploded view

    func applyExplode() {
        guard let root else { return }
        for runtime in runtimes.values {
            runtime.entity.position = runtime.baseTranslation
        }
        guard explodeAmount > 0.001 else { return }

        let scope = (explodeScope == .selection ? selectedNode : nil) ?? root
        let deltas = ExplodeLayout.offsets(scope: scope, bounds: bounds, amount: explodeAmount)
        for (id, delta) in deltas {
            guard
                let runtime = runtimes[id],
                let parent = runtime.node.parent,
                let parentWorld = worldTransforms[parent.id]
            else { continue }
            let linear = parentWorld.linear
            guard abs(linear.determinant) > 1e-9 else { continue }
            // `delta` is in model space; the entity's position lives in its parent's local space.
            runtime.entity.position = runtime.baseTranslation + linear.inverse * delta
        }
    }

    func resetExplode() {
        explodeAmount = 0
        applyExplode()
    }

    // MARK: - Slicing

    private func buildSlicePlane() {
        slicePlane.model = ModelComponent(
            mesh: .generateBox(size: 1),
            materials: [{
                var m = UnlitMaterial(color: .systemTeal)
                m.blending = .transparent(opacity: .init(floatLiteral: 0.22))
                return m
            }()]
        )
        slicePlane.isEnabled = false
        normalizer.addChild(slicePlane)
    }

    private func currentPlane() -> MeshClipper.Plane {
        let axis = sliceAxis.index
        var point = modelBounds.center
        point[axis] = modelBounds.min[axis] + (modelBounds.max[axis] - modelBounds.min[axis]) * sliceOffset
        let normal = sliceAxis.vector * (sliceFlip ? -1 : 1)
        return MeshClipper.Plane(point: point, normal: normal)
    }

    private func updateSlicePlane() {
        slicePlane.isEnabled = sliceEnabled && showSlicePlane && hasModel
        guard slicePlane.isEnabled else { return }
        let axis = sliceAxis.index
        var size = modelBounds.size * 1.05
        let thickness = Swift.max(size.x, Swift.max(size.y, size.z)) * 0.004
        size[axis] = thickness
        var center = modelBounds.center
        center[axis] = modelBounds.min[axis] + (modelBounds.max[axis] - modelBounds.min[axis]) * sliceOffset
        slicePlane.scale = size
        slicePlane.position = center
    }

    /// Debounced; slider drags call this many times per second.
    func scheduleSlice() {
        sliceTask?.cancel()
        sliceTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(120))
            guard !Task.isCancelled else { return }
            await self?.runSlice()
        }
    }

    private func runSlice() async {
        updateSlicePlane()
        sliceGeneration += 1
        let generation = sliceGeneration

        guard sliceEnabled else {
            restoreFullMeshes()
            return
        }

        struct Job: @unchecked Sendable {
            let nodeID: UUID
            let index: Int
            let data: PrimitiveData
            let world: simd_float4x4
        }
        var jobs: [Job] = []
        for (id, runtime) in runtimes where !runtime.node.isHidden {
            guard let world = worldTransforms[id] else { continue }
            for (i, p) in runtime.primitives.enumerated() {
                jobs.append(Job(nodeID: id, index: i, data: p.data, world: world))
            }
        }
        let plane = currentPlane()
        let work = jobs   // an immutable copy: detached closures may not capture a `var`

        let results = await Task.detached(priority: .userInitiated) {
            work.map { job in
                (job.nodeID, job.index, MeshClipper.clip(job.data, localToModel: job.world, plane: plane))
            }
        }.value

        guard generation == sliceGeneration, !Task.isCancelled else { return }

        for (id, index, clipped) in results {
            guard let primitive = runtimes[id]?.primitives[safe: index] else { continue }
            if let clipped, let mesh = try? SceneBuilder.makeMesh(clipped) {
                primitive.entity.model?.mesh = mesh
                primitive.clippedAway = false
            } else {
                primitive.clippedAway = true
            }
        }
        applyVisibility()
    }

    private func restoreFullMeshes() {
        for runtime in runtimes.values {
            for p in runtime.primitives {
                p.entity.model?.mesh = p.originalMesh
                p.clippedAway = false
            }
        }
        applyVisibility()
    }
}
