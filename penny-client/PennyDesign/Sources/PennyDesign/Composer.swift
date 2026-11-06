import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

/// The composer: a leading glass `+` that opens the attach menu (Photos,
/// Camera, Files), the glass field at the same height on one centre line,
/// a pending photo's thumbnail above the text line when one is attached
/// (with a small glass xmark in its corner), and the blue send arrow only
/// when there's a draft — text or a photo. The conversation-state tint,
/// placeholder and reply chip live with the caller; the field only renders
/// what it is handed.
public struct Composer: View {
    @Binding private var text: String
    private let placeholder: String
    private let tint: Color?
    private let pendingImage: UIImage?
    private let canSend: Bool
    private let isFocused: FocusState<Bool>.Binding
    private let onSend: () -> Void
    private let onImagePicked: (UIImage) -> Void
    private let onRemovePendingImage: () -> Void

    @State private var photoPickerItem: PhotosPickerItem?
    @State private var isShowingPhotoPicker = false
    @State private var isShowingCamera = false
    @State private var isShowingFileImporter = false
    @State private var rowHeight: CGFloat = 0

    public init(
        text: Binding<String>,
        placeholder: String = "Message",
        tint: Color? = nil,
        pendingImage: UIImage? = nil,
        canSend: Bool = true,
        isFocused: FocusState<Bool>.Binding,
        onSend: @escaping () -> Void = {},
        onImagePicked: @escaping (UIImage) -> Void = { _ in },
        onRemovePendingImage: @escaping () -> Void = {}
    ) {
        _text = text
        self.placeholder = placeholder
        self.tint = tint
        self.pendingImage = pendingImage
        self.canSend = canSend
        self.isFocused = isFocused
        self.onSend = onSend
        self.onImagePicked = onImagePicked
        self.onRemovePendingImage = onRemovePendingImage
    }

    /// The send arrow appears when there's something to send — text or a
    /// photo — and only fires when the connection allows sending.
    private var hasDraft: Bool {
        !text.isEmpty || pendingImage != nil
    }

    public var body: some View {
        HStack(spacing: 8) {
            attachMenu
            field
        }
        .padding(.horizontal, Tokens.margin)
        // Measured between the paddings so this is the content row itself —
        // the taller of the `+` button and the glass field.
        .onGeometryChange(for: CGFloat.self, of: { $0.size.height }, action: { rowHeight = $0 })
        .padding(.vertical, Tokens.composerVerticalPadding)
        .preference(key: ComposerRowHeightKey.self, value: rowHeight)
        .photosPicker(isPresented: $isShowingPhotoPicker, selection: $photoPickerItem, matching: .images)
        .onChange(of: photoPickerItem) { _, item in
            loadPhotoPickerItem(item)
        }
        .fullScreenCover(isPresented: $isShowingCamera) {
            CameraPicker(onShot: onImagePicked)
                .ignoresSafeArea()
        }
        .fileImporter(isPresented: $isShowingFileImporter, allowedContentTypes: [.image]) { result in
            loadImage(from: result)
        }
    }

    /// The glass `+`: a `Menu` styled as the glass circle, so the attach
    /// menu opens anchored to it. Pinned to the token height — the field
    /// beside it uses the same token, so the pair always matches.
    private var attachMenu: some View {
        Menu {
            Button {
                isShowingPhotoPicker = true
            } label: {
                Label("Photos", systemImage: "photo.on.rectangle")
            }
            Button {
                isShowingCamera = true
            } label: {
                Label("Camera", systemImage: "camera")
            }
            Button {
                isShowingFileImporter = true
            } label: {
                Label("Files", systemImage: "folder")
            }
        } label: {
            Image(systemName: "plus")
                .fontWeight(.semibold)
                .frame(width: Tokens.composerAttachGlyphFrame, height: Tokens.composerAttachGlyphFrame)
        }
        .menuStyle(.button)
        .buttonStyle(.glass)
        .buttonBorderShape(.circle)
        // Pin the glass circle's layout box to the token height — the field
        // beside it uses the same token, so the pair always matches.
        .frame(width: Tokens.composerControlHeight, height: Tokens.composerControlHeight)
        .accessibilityLabel("Attach a photo or file")
    }

    private var field: some View {
        VStack(alignment: .leading, spacing: Tokens.composerPendingInset) {
            if let pendingImage {
                pendingThumbnail(pendingImage)
            }

            HStack(spacing: 4) {
                TextField(placeholder, text: $text, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...5)
                    .submitLabel(.send)
                    .focused(isFocused)
                    .onSubmit(onSend)

                if hasDraft {
                    Button(action: onSend) {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: Tokens.composerSendGlyphSize))
                            .foregroundStyle(.blue)
                    }
                    .buttonStyle(.plain)
                    .disabled(!canSend)
                    .accessibilityLabel("Send")
                }
            }
            // Single-line state: the field matches the `+` button's height
            // (placeholder, text and send arrow all centre in it). With a
            // pending photo the field grows to fit the thumbnail above the
            // text line instead.
            .frame(height: pendingImage == nil ? Tokens.composerControlHeight : nil)
            .padding(.leading, Tokens.composerFieldLeadingInset)
            .padding(.trailing, Tokens.composerFieldTrailingInset)
            .padding(.bottom, pendingImage == nil ? 0 : Tokens.composerPendingInset)
        }
        .glassEffect(fieldGlass, in: fieldShape)
        .animation(.easeInOut(duration: 0.2), value: hasDraft)
    }

    private var fieldGlass: Glass {
        guard let tint else { return .regular }
        return .regular.tint(tint)
    }

    private var fieldShape: AnyShape {
        pendingImage == nil ? AnyShape(.capsule) : AnyShape(.rect(cornerRadius: Tokens.composerFieldCorner))
    }

    /// A pending photo: 56pt wide, cropped to the attach mock's 16:10 box
    /// and rounded, with the small glass xmark at its top-trailing corner,
    /// nudged slightly outside it — the `chat-pending` screen's mark.
    private func pendingThumbnail(_ image: UIImage) -> some View {
        Image(uiImage: image)
            .resizable()
            .scaledToFill()
            .frame(
                width: Tokens.composerPendingThumbWidth,
                height: Tokens.composerPendingThumbHeight
            )
            .clipShape(RoundedRectangle(cornerRadius: Tokens.composerPendingThumbCorner, style: .continuous))
            .overlay(alignment: .topTrailing) {
                Button {
                    onRemovePendingImage()
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: Tokens.composerPendingXGlyph, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .frame(
                            width: Tokens.composerPendingXFrame,
                            height: Tokens.composerPendingXFrame
                        )
                }
                .buttonStyle(.glass)
                .buttonBorderShape(.circle)
                .offset(x: Tokens.composerPendingXInset, y: -Tokens.composerPendingXInset)
                .accessibilityLabel("Remove photo")
            }
            .padding(.top, Tokens.composerPendingInset)
            .padding(.leading, Tokens.composerPendingInset)
    }

    private func loadPhotoPickerItem(_ item: PhotosPickerItem?) {
        guard let item else { return }
        photoPickerItem = nil
        Task {
            guard let data = try? await item.loadTransferable(type: Data.self),
                  let image = UIImage(data: data) else { return }
            onImagePicked(image)
        }
    }

    private func loadImage(from result: Result<URL, any Error>) {
        guard case .success(let url) = result else { return }
        let grantedAccess = url.startAccessingSecurityScopedResource()
        defer {
            if grantedAccess {
                url.stopAccessingSecurityScopedResource()
            }
        }
        if let data = try? Data(contentsOf: url), let image = UIImage(data: data) {
            onImagePicked(image)
        }
    }
}

/// The composer content row's measured height — the taller of the `+`
/// button and the glass field. ChatCanvas derives the dock's float from it
/// (row height + the composer's vertical padding + T.dockComposerGap).
public struct ComposerRowHeightKey: PreferenceKey {
    public static let defaultValue: CGFloat = 0
    public static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}
