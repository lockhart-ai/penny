import SwiftUI

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    let client: PennyWebSocketClient

    @State private var webSocketURL: String
    @State private var username: String
    @State private var password: String

    init(client: PennyWebSocketClient) {
        self.client = client
        let prefs = Prefs.shared
        _webSocketURL = State(initialValue: prefs.webSocketURL ?? "")
        _username = State(initialValue: prefs.username ?? "")
        _password = State(initialValue: prefs.password ?? "")
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Status") {
                    LabeledContent("Connection", value: client.statusText)
                    LabeledContent("Pending", value: "\(client.pendingCount)")
                }

                Section("Connection") {
                    TextField("WebSocket URL", text: $webSocketURL)
                        .textInputAutocapitalization(.never)
                        .keyboardType(.URL)
                        .autocorrectionDisabled()

                    TextField("Username", text: $username)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    SecureField("Password", text: $password)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
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
                    .disabled(!canSave)
                }
            }
        }
    }

    private var canSave: Bool {
        !webSocketURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func save() {
        let prefs = Prefs.shared
        prefs.webSocketURL = webSocketURL.trimmingCharacters(in: .whitespacesAndNewlines)
        prefs.username = username.trimmingCharacters(in: .whitespacesAndNewlines)
        prefs.password = password
        client.reconnect()
        dismiss()
    }
}

#Preview {
    SettingsView(client: PennyWebSocketClient())
}
