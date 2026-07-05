import Foundation
import Security

/// Storage for small, sensitive strings (credentials, device identity).
///
/// This mirrors the injectable `UserDefaults` seam used by `Prefs`: production uses
/// `SystemKeychain`; tests substitute an in-memory implementation so unit tests never
/// touch the real system keychain.
protocol KeychainStore {
    func string(account: String) -> String?
    func set(_ value: String?, account: String)
}

/// Reads and writes generic-password items in the system keychain.
struct SystemKeychain: KeychainStore {
    static let defaultService = "PennyClient"

    private let service: String

    init(service: String = SystemKeychain.defaultService) {
        self.service = service
    }

    func string(account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess, let data = item as? Data else { return nil }

        return String(data: data, encoding: .utf8)
    }

    func set(_ value: String?, account: String) {
        guard let value else {
            delete(account: account)
            return
        }
        save(value, account: account)
    }

    private func save(_ value: String, account: String) {
        let data = Data(value.utf8)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        let attributes: [String: Any] = [kSecValueData as String: data]

        let status = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        if status == errSecItemNotFound {
            var addQuery = query
            addQuery[kSecValueData as String] = data
            SecItemAdd(addQuery as CFDictionary, nil)
        }
    }

    private func delete(account: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        SecItemDelete(query as CFDictionary)
    }
}
