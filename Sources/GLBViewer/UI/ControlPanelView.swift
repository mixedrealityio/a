import SwiftUI
import UniformTypeIdentifiers

struct ControlPanelView: View {
    @Environment(ViewerModel.self) private var model
    @Environment(\.openWindow) private var openWindow
    @Environment(\.dismissWindow) private var dismissWindow
    @Environment(\.openImmersiveSpace) private var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) private var dismissImmersiveSpace

    @State private var showImporter = false
    @State private var inRoom = false

    private static let glbType = UTType(filenameExtension: "glb") ?? .data

    var body: some View {
        @Bindable var model = model

        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header
                if model.hasModel {
                    selectionActions
                    viewSection(model: $model)
                    explodeSection(model: $model)
                    sliceSection(model: $model)
                }
                structureSection
            }
            .padding(24)
        }
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [Self.glbType, .data]) { result in
            guard case .success(let url) = result else { return }
            Task {
                await model.load(url: url)
                if model.hasModel { openModelWindow() }
            }
        }
        .alert("Notice", isPresented: Binding(
            get: { model.statusMessage != nil },
            set: { if !$0 { model.statusMessage = nil } }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.statusMessage ?? "")
        }
    }

    // MARK: Sections

    private var header: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("GLB Viewer").font(.largeTitle.bold())
            HStack {
                Button("Open GLB…", systemImage: "folder") { showImporter = true }
                Button("Load Sample", systemImage: "shippingbox") {
                    Task {
                        await model.loadBundledSample()
                        if model.hasModel { openModelWindow() }
                    }
                }
                if model.isLoading { ProgressView().padding(.leading, 8) }
            }
            if model.hasModel {
                Text("\(model.fileName) — \(model.summary)")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var selectionActions: some View {
        GroupBox("Selected subtree") {
            VStack(alignment: .leading, spacing: 12) {
                Text(model.selectedNode.map { "\($0.name) — \($0.subtree.count) item(s)" } ?? "Tap a part in the model or pick a row below.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                let none = model.selectedNode == nil
                HStack {
                    Button("Remove", systemImage: "eye.slash") { model.removeSelectedSubtree() }
                    Button("Restore", systemImage: "eye") { model.restoreSelectedSubtree() }
                    Button("Isolate", systemImage: "scope") { model.isolateSelectedSubtree() }
                    Button("Parent", systemImage: "arrow.up") { model.selectParent() }
                }
                .disabled(none)
                HStack {
                    Button("Remove all parts", systemImage: "trash", role: .destructive) { model.removeAllParts() }
                    Button("Restore all", systemImage: "arrow.counterclockwise") { model.restoreAllParts() }
                }
            }
        }
    }

    private func viewSection(model: Bindable<ViewerModel>) -> some View {
        GroupBox("Size") {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Slider(value: model.userScale, in: 0.2...8)
                    Text(String(format: "%.1f×", self.model.userScale)).monospacedDigit().frame(width: 56)
                }
                HStack {
                    Button("Fit") { self.model.userScale = 1 }
                    Button("Real size") { self.model.userScale = self.model.realSizeScale }
                    Spacer()
                    Button(inRoom ? "Back to window" : "In my room", systemImage: inRoom ? "macwindow" : "view.3d") {
                        Task { await toggleRoom() }
                    }
                }
                Text("Pinch the model to enlarge it, drag sideways to turn it.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
        }
    }

    private func explodeSection(model: Bindable<ViewerModel>) -> some View {
        GroupBox("Exploded view") {
            VStack(alignment: .leading, spacing: 12) {
                Slider(value: model.explodeAmount, in: 0...1)
                    .onChange(of: self.model.explodeAmount) { self.model.applyExplode() }
                Picker("Explode", selection: model.explodeScope) {
                    ForEach(ExplodeScope.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .onChange(of: self.model.explodeScope) { self.model.applyExplode() }
                Button("Assemble", systemImage: "arrow.down.right.and.arrow.up.left") { self.model.resetExplode() }
            }
        }
    }

    private func sliceSection(model: Bindable<ViewerModel>) -> some View {
        GroupBox("Slice") {
            VStack(alignment: .leading, spacing: 12) {
                Toggle("Cut the model", isOn: model.sliceEnabled)
                    .onChange(of: self.model.sliceEnabled) { self.model.scheduleSlice() }
                if self.model.sliceEnabled {
                    Picker("Axis", selection: model.sliceAxis) {
                        ForEach(SliceAxis.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: self.model.sliceAxis) { self.model.scheduleSlice() }
                    Slider(value: model.sliceOffset, in: 0...1)
                        .onChange(of: self.model.sliceOffset) { self.model.scheduleSlice() }
                    Toggle("Keep the other side", isOn: model.sliceFlip)
                        .onChange(of: self.model.sliceFlip) { self.model.scheduleSlice() }
                    Toggle("Show cutting plane", isOn: model.showSlicePlane)
                        .onChange(of: self.model.showSlicePlane) { self.model.scheduleSlice() }
                }
            }
        }
    }

    private var structureSection: some View {
        GroupBox("Product structure") {
            ProductTreeView()
                .frame(height: 320)
        }
    }

    // MARK: Windows

    private func openModelWindow() {
        if !inRoom { openWindow(id: "model") }
    }

    private func toggleRoom() async {
        if inRoom {
            await dismissImmersiveSpace()
            inRoom = false
            openWindow(id: "model")
        } else {
            dismissWindow(id: "model")
            if case .opened = await openImmersiveSpace(id: "immersive") {
                inRoom = true
            } else {
                openWindow(id: "model")
            }
        }
    }
}
