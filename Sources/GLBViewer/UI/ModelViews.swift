import RealityKit
import SwiftUI

/// Shared interaction for the model: tap to select, pinch to enlarge, drag to turn.
private struct ModelGestures: ViewModifier {
    @Environment(ViewerModel.self) private var model
    @State private var scaleAtStart: Float?
    @State private var yawAtStart: Float?

    func body(content: Content) -> some View {
        content
            .gesture(
                SpatialTapGesture()
                    .targetedToAnyEntity()
                    .onEnded { value in model.select(entity: value.entity) }
            )
            .gesture(
                MagnifyGesture()
                    .targetedToAnyEntity()
                    .onChanged { value in
                        if scaleAtStart == nil { scaleAtStart = model.userScale }
                        model.userScale = (scaleAtStart ?? 1) * Float(value.magnification)
                    }
                    .onEnded { _ in scaleAtStart = nil }
            )
            .gesture(
                DragGesture()
                    .targetedToAnyEntity()
                    .onChanged { value in
                        if yawAtStart == nil { yawAtStart = model.yaw }
                        model.yaw = (yawAtStart ?? 0) + Float(value.translation.width) * 0.004
                    }
                    .onEnded { _ in yawAtStart = nil }
            )
    }
}

/// The model in a volumetric window.
struct ModelVolumeView: View {
    @Environment(ViewerModel.self) private var model

    var body: some View {
        RealityView { content in
            model.contentRoot.position = .zero
            content.add(model.contentRoot)
        }
        .modifier(ModelGestures())
    }
}

/// The model at real-world scale in the room.
struct ImmersiveModelView: View {
    @Environment(ViewerModel.self) private var model

    var body: some View {
        RealityView { content in
            // About 1.5 m in front of the user, with its centre at eye height.
            model.contentRoot.position = SIMD3<Float>(0, 1.2, -1.5)
            content.add(model.contentRoot)
        }
        .modifier(ModelGestures())
    }
}
