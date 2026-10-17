import RealityKit
import SwiftUI

@main
struct GLBViewerApp: App {
    @State private var model = ViewerModel()

    init() {
        PartComponent.registerComponent()
    }

    var body: some SwiftUI.Scene {
        // Control panel: file loading, product structure, view operations.
        WindowGroup(id: "controls") {
            ControlPanelView()
                .environment(model)
        }
        .defaultSize(width: 560, height: 860)

        // The model itself, as a volume you can walk around.
        WindowGroup(id: "model") {
            ModelVolumeView()
                .environment(model)
        }
        .windowStyle(.volumetric)
        .defaultSize(width: 1.0, height: 1.0, depth: 1.0, in: .meters)

        // Real-size view in your room.
        ImmersiveSpace(id: "immersive") {
            ImmersiveModelView()
                .environment(model)
        }
        .immersionStyle(selection: .constant(.mixed), in: .mixed)
    }
}
