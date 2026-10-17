import SwiftUI

/// The product structure as an expandable tree. Selecting a row selects its whole subtree.
struct ProductTreeView: View {
    @Environment(ViewerModel.self) private var model

    var body: some View {
        @Bindable var model = model

        if let root = model.root {
            List([root], children: \.childNodes, selection: $model.selectedID) { node in
                ProductRow(node: node)
            }
            .listStyle(.plain)
        } else {
            ContentUnavailableView("No model", systemImage: "cube", description: Text("Open a .glb file to see its product structure."))
        }
    }
}

private struct ProductRow: View {
    @Environment(ViewerModel.self) private var model
    let node: ProductNode

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: node.isPart ? "cube" : (node.childNodes == nil ? "cube.transparent" : "square.stack.3d.up"))
                .foregroundStyle(.secondary)
            Text(node.name)
                .lineLimit(1)
                .opacity(node.isHidden ? 0.4 : 1)
            Spacer()
            Button {
                model.toggleHidden(node)
            } label: {
                Image(systemName: node.isHidden ? "eye.slash" : "eye")
            }
            .buttonStyle(.borderless)
            .accessibilityLabel(node.isHidden ? "Show \(node.name)" : "Remove \(node.name)")
        }
    }
}
