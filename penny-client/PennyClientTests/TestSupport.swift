import Foundation
@testable import PennyClient

final class EmptyBundleMarker {}

final class InMemoryKeychain: KeychainStore {
    private var storage: [String: String] = [:]

    func string(account: String) -> String? {
        storage[account]
    }

    func set(_ value: String?, account: String) {
        storage[account] = value
    }
}

func makeUserDefaults() -> UserDefaults {
    let suiteName = "PennyClientTests.\(UUID().uuidString)"
    let userDefaults = UserDefaults(suiteName: suiteName)!
    userDefaults.removePersistentDomain(forName: suiteName)
    return userDefaults
}

@MainActor
func configuredPrefs(
    url: String? = "wss://example.test/penny/",
    username: String = "alice",
    password: String = "secret"
) -> Prefs {
    let prefs = Prefs(userDefaults: makeUserDefaults(), keychain: InMemoryKeychain(), bundle: Bundle(for: EmptyBundleMarker.self))
    prefs.webSocketURL = url
    prefs.username = username
    prefs.password = password
    return prefs
}

func configuredDatabase() -> DatabaseService {
    let database = DatabaseService()
    database.setupForTesting()
    return database
}
