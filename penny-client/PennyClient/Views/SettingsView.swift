import SwiftUI
import PennyServices

struct SettingsView: View {
    @State private var viewModel: SettingsViewModel
    @State private var selectedChannelTypes = Set(ChannelType.allCases.map(\.rawValue))
    @State private var includeHistoryAttachments = true
    @State private var isShowingDeleteMessagesConfirmation = false
    @FocusState private var focusedConnectionField: ConnectionField?

    init(client: PennyWebSocketClient) {
        _viewModel = State(initialValue: SettingsViewModel(client: client, prefs: .shared))
    }

    var body: some View {
        settingsForm
    }

    private var settingsForm: some View {
        Form {
            if let commitHash = AppBuildInfo.current.commitHash {
                Section("Build") {
                    LabeledContent("Commit", value: commitHash)
                }
            }

            Section("Status") {
                LabeledContent("Connection", value: viewModel.client.statusText)
                LabeledContent("Pending", value: "\(viewModel.client.pendingCount)")
            }

            Section("Connection") {
                TextField("WebSocket URL", text: $viewModel.webSocketURL)
                    .textInputAutocapitalization(.never)
                    .keyboardType(.URL)
                    .autocorrectionDisabled()
                    .focused($focusedConnectionField, equals: .webSocketURL)

                LabeledContent("APNs Host", value: viewModel.apnsHost)

                TextField("Username", text: $viewModel.username)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .focused($focusedConnectionField, equals: .username)

                SecureField("Password", text: $viewModel.password)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .focused($focusedConnectionField, equals: .password)

                LabeledContent("Test Notification") {
                    Button("Send") {
                        viewModel.sendTestNotification()
                    }
                    .buttonStyle(.borderless)
                    .disabled(!viewModel.canSendTestNotification)
                }
            }
            // No Save button (#3169) — this section saves as you go.
            // Submitting from the keyboard (the return key) saves directly;
            // `focusedConnectionField` catches the other case, a field
            // losing focus with no Return (a tap elsewhere, or switching to
            // another field). `saveConnection()` itself is the no-op guard
            // against double-saving the same edit (it only reconnects when
            // a value actually changed), so both triggers can fire safely.
            .onSubmit {
                viewModel.saveConnection()
            }
            .onChange(of: focusedConnectionField) { previousField, _ in
                if previousField != nil {
                    viewModel.saveConnection()
                }
            }

            ImageAttachmentSettingsSection(viewModel: viewModel)

            Section("History") {
                ForEach(ChannelType.allCases) { channel in
                    Toggle(channel.title, isOn: Binding(
                        get: { selectedChannelTypes.contains(channel.rawValue) },
                        set: { isSelected in
                            if isSelected {
                                selectedChannelTypes.insert(channel.rawValue)
                            } else {
                                selectedChannelTypes.remove(channel.rawValue)
                            }
                        }
                    ))
                    .disabled(viewModel.client.historySyncing)
                }

                Toggle("Include attachments", isOn: $includeHistoryAttachments)
                    .disabled(viewModel.client.historySyncing)

                LabeledContent("History sync", value: viewModel.client.historyProgressText)

                HStack {
                    Button {
                        viewModel.startHistorySync(
                            channelTypes: Array(selectedChannelTypes).sorted(),
                            includeAttachments: includeHistoryAttachments
                        )
                    } label: {
                        Label(
                            viewModel.client.historySyncing ? "Syncing History" : "Sync History",
                            systemImage: viewModel.client.historySyncing
                                ? "arrow.triangle.2.circlepath"
                                : "clock.arrow.circlepath"
                        )
                    }
                    .disabled(selectedChannelTypes.isEmpty || viewModel.client.historySyncing)
                    Spacer()
                    Text(viewModel.client.historyStatus)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            #if DEBUG
            Section("Developer") {
                Button(role: .destructive) {
                    isShowingDeleteMessagesConfirmation = true
                } label: {
                    Label("Delete All Local Messages", systemImage: "trash")
                }
            }
            #endif

            if let prompt = viewModel.permissionPrompt {
                Section("Permission Request") {
                    LabeledContent("Domain", value: prompt.domain)
                    Text(prompt.url)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    HStack {
                        Button {
                            viewModel.decidePermissionPrompt(allowed: true)
                        } label: {
                            Label("Allow", systemImage: "checkmark")
                        }

                        Spacer()

                        Button(role: .destructive) {
                            viewModel.decidePermissionPrompt(allowed: false)
                        } label: {
                            Label("Block", systemImage: "xmark")
                        }
                    }
                    .buttonStyle(.borderless)
                }
            }

            Section("Runtime Config") {
                if viewModel.runtimeConfigParams.isEmpty {
                    ContentUnavailableView("No Config", systemImage: "slider.horizontal.3")
                } else {
                    ForEach(viewModel.runtimeConfigParams) { param in
                        RuntimeConfigRow(param: param, viewModel: viewModel)
                    }
                }
            }

            Section("Domain Permissions") {
                HStack {
                    TextField("Domain", text: $viewModel.domainDraft)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    Picker("Permission", selection: $viewModel.domainPermission) {
                        Text("Allow").tag(DomainPermission.allowed)
                        Text("Block").tag(DomainPermission.blocked)
                    }
                    .labelsHidden()
                }

                Button {
                    viewModel.submitDomainPermission()
                } label: {
                    Label("Save Domain", systemImage: "plus")
                }
                .disabled(!viewModel.canSubmitDomain)

                ForEach(viewModel.domainPermissions) { entry in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(entry.domain)
                            Text(entry.permission.title)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button(role: .destructive) {
                            viewModel.deleteDomain(entry)
                        } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.borderless)
                        .accessibilityLabel("Delete domain")
                    }
                }
            }
        }
        .navigationTitle("Settings")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            viewModel.refresh()
        }
        #if DEBUG
        .confirmationDialog(
            "Delete all local messages?",
            isPresented: $isShowingDeleteMessagesConfirmation,
            titleVisibility: .visible
        ) {
            Button("Delete All Messages", role: .destructive) {
                viewModel.deleteAllMessages()
            }
        } message: {
            Text("This removes every message stored on this device. It does not delete messages from the server.")
        }
        #endif
    }
}

/// Which Connection field last held focus (#3169) — losing focus on any of
/// them, with no Return pressed, is a save trigger alongside `.onSubmit`.
private enum ConnectionField: Hashable {
    case webSocketURL
    case username
    case password
}

struct AppBuildInfo {
    private static let commitHashKey = "PennyBuildCommitHash"

    let commitHash: String?

    static var current: AppBuildInfo {
        AppBuildInfo(infoDictionary: Bundle.main.infoDictionary)
    }

    init(infoDictionary: [String: Any]?) {
        commitHash = Self.trimmedNilIfEmpty(infoDictionary?[Self.commitHashKey] as? String)
    }

    private static func trimmedNilIfEmpty(_ value: String?) -> String? {
        guard let value else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}

private struct RuntimeConfigRow: View {
    let param: RuntimeConfigParam
    let viewModel: SettingsViewModel
    @FocusState private var isValueFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(param.key)
                .font(.headline)
            Text(param.description)
                .font(.caption)
                .foregroundStyle(.secondary)
            // No checkmark Save button (#3169) — submitting from the
            // keyboard (the return key) saves directly, and losing focus
            // with no Return (a tap elsewhere) saves the same way the
            // Connection section's fields do.
            TextField("Value", text: valueBinding)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .focused($isValueFocused)
                .onSubmit {
                    viewModel.saveConfigValue(for: param)
                }
                .onChange(of: isValueFocused) { wasFocused, isFocusedNow in
                    if wasFocused, !isFocusedNow {
                        viewModel.saveConfigValue(for: param)
                    }
                }
        }
        .padding(.vertical, 4)
    }

    private var valueBinding: Binding<String> {
        Binding {
            viewModel.configValue(for: param)
        } set: { newValue in
            viewModel.setConfigValue(newValue, for: param)
        }
    }
}

private extension DomainPermission {
    var title: String {
        switch self {
        case .allowed:
            return "Allowed"
        case .blocked:
            return "Blocked"
        }
    }
}

/// A message's source channel, by its wire `channel_type` (#3143) — shared
/// between the history-sync toggles below and `ChatMessageView`'s
/// cross-device "via" caption, so both read the one typed mapping rather
/// than each keeping its own channel-name string literals.
enum ChannelType: String, CaseIterable, Identifiable {
    case ios
    case signal
    case discord
    case browser

    var id: Self { self }

    var title: String {
        switch self {
        case .ios:
            return "iOS"
        case .signal:
            return "Signal"
        case .discord:
            return "Discord"
        case .browser:
            return "Browser"
        }
    }
}

#Preview {
    SettingsView(client: PennyWebSocketClient())
}
