import SwiftUI

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var viewModel: SettingsViewModel

    init(client: PennyWebSocketClient) {
        _viewModel = State(initialValue: SettingsViewModel(client: client, prefs: .shared))
    }

    var body: some View {
        NavigationStack {
            Form {
                if let commitHash = AppBuildInfo.current.commitHash {
                    Section("Build") {
                        LabeledContent("Commit", value: commitHash)
                    }
                }

                Section("Penny") {
                    NavigationLink {
                        SchedulesView(client: viewModel.client)
                    } label: {
                        Label("Schedules", systemImage: "calendar")
                    }

                    NavigationLink {
                        InsightsView(client: viewModel.client)
                    } label: {
                        Label("Insights", systemImage: "chart.bar.doc.horizontal")
                    }

                    NavigationLink {
                        MemoryManagementView(client: viewModel.client)
                    } label: {
                        Label("Memory Management", systemImage: "tray.full")
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

                    LabeledContent("APNs Host", value: viewModel.apnsHost)

                    TextField("Username", text: $viewModel.username)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    SecureField("Password", text: $viewModel.password)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }

                Section("Features") {
                    Toggle("1-2-3 layout", isOn: $viewModel.isMessageLayoutSwitcherEnabled)
                }

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
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .topBarTrailing) {
                    Button("Save") {
                        save()
                    }
                    .disabled(!viewModel.canSaveConnection)
                }
            }
            .task {
                viewModel.refresh()
            }
        }
    }

    private func save() {
        viewModel.saveConnection()
        dismiss()
    }
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

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(param.key)
                .font(.headline)
            Text(param.description)
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack {
                TextField("Value", text: valueBinding)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                Button {
                    viewModel.saveConfigValue(for: param)
                } label: {
                    Image(systemName: "checkmark.circle")
                }
                .accessibilityLabel("Save config value")
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

#Preview {
    SettingsView(client: PennyWebSocketClient())
}
