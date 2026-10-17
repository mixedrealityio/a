import XCTest
import simd
@testable import GLBViewer

final class GLBLoaderTests: XCTestCase {
    private func sampleData() throws -> Data {
        // Tests/GLBViewerTests/<this file> -> repo root -> Samples
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try Data(contentsOf: root.appendingPathComponent("Samples/GearboxAssembly.glb"))
    }

    func testLoadsProductStructure() throws {
        let model = try GLBLoader.load(data: sampleData(), name: "Gearbox")
        let names = model.root.subtree.map(\.name)
        XCTAssertEqual(model.partCount, 11)   // 4 housing + 3 shaft assembly + 4 bolts
        for expected in ["Housing", "Shaft Assembly", "Fasteners", "Bolt 4", "Gear A"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
        XCTAssertEqual(model.triangleCount, 12 + 12 + 12 + 12 + 128 + 192 + 192 + 4 * 64)
    }

    func testRejectsNonGLB() {
        XCTAssertThrowsError(try GLBLoader.load(data: Data("not a glb file at all".utf8), name: "x"))
    }
}

final class MeshClipperTests: XCTestCase {
    private func unitCube() -> PrimitiveData {
        let p: [SIMD3<Float>] = [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ]
        let idx: [UInt32] = [
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
            2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 3, 0, 4, 3, 4, 7,
        ]
        return PrimitiveData(
            positions: p,
            normals: GLBLoader.smoothNormals(positions: p, indices: idx),
            uvs: [], indices: idx, materialIndex: nil
        )
    }

    func testKeepsOnlyTheRequestedSide() throws {
        let plane = MeshClipper.Plane(point: .zero, normal: [1, 0, 0])
        let clipped = try XCTUnwrap(MeshClipper.clip(unitCube(), localToModel: matrix_identity_float4x4, plane: plane))
        for i in Set(clipped.indices) {
            XCTAssertGreaterThanOrEqual(clipped.positions[Int(i)].x, -1e-5)
        }
        XCTAssertTrue(clipped.indices.count % 3 == 0)
    }

    func testRespectsNodeTransform() throws {
        // Cube moved +10 in X: a model-space plane at x = 10 must still cut it in half.
        var m = matrix_identity_float4x4
        m.columns.3.x = 10
        let plane = MeshClipper.Plane(point: [10, 0, 0], normal: [1, 0, 0])
        let clipped = try XCTUnwrap(MeshClipper.clip(unitCube(), localToModel: m, plane: plane))
        for i in Set(clipped.indices) {
            XCTAssertGreaterThanOrEqual(clipped.positions[Int(i)].x, -1e-5)   // local space
        }
    }

    func testReturnsNilWhenEverythingIsCutAway() {
        let plane = MeshClipper.Plane(point: [5, 0, 0], normal: [1, 0, 0])
        XCTAssertNil(MeshClipper.clip(unitCube(), localToModel: matrix_identity_float4x4, plane: plane))
    }
}

final class ExplodeLayoutTests: XCTestCase {
    func testChildrenMoveAwayFromTheParentCentre() {
        let root = ProductNode(name: "root")
        let left = ProductNode(name: "left")
        let right = ProductNode(name: "right")
        root.addChild(left)
        root.addChild(right)

        var bounds: [UUID: AABB] = [:]
        func box(_ lo: SIMD3<Float>, _ hi: SIMD3<Float>) -> AABB { var b = AABB(); b.expand(lo); b.expand(hi); return b }
        bounds[left.id] = box([-3, 0, 0], [-1, 1, 1])
        bounds[right.id] = box([1, 0, 0], [3, 1, 1])
        bounds[root.id] = box([-3, 0, 0], [3, 1, 1])

        let offsets = ExplodeLayout.offsets(scope: root, bounds: bounds, amount: 1)
        XCTAssertLessThan(offsets[left.id]!.x, 0)
        XCTAssertGreaterThan(offsets[right.id]!.x, 0)
        XCTAssertTrue(ExplodeLayout.offsets(scope: root, bounds: bounds, amount: 0).isEmpty)
    }
}
