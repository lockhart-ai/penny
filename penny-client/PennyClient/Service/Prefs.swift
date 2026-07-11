import Foundation

final class Prefs {
    static let shared = Prefs()

    private let userDefaults: UserDefaults
    private let keychain: KeychainStore
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private let bundledSecrets: SecretsPlist?

    init(userDefaults: UserDefaults = .standard, keychain: KeychainStore = SystemKeychain(), bundle: Bundle = .main) {
        self.userDefaults = userDefaults
        self.keychain = keychain
        self.bundledSecrets = SecretsPlist.load(from: bundle)
    }

    func string(forKey key: Key) -> String? {
        userDefaults.string(forKey: key.rawValue)
    }

    func set(_ value: String?, forKey key: Key) {
        userDefaults.set(value, forKey: key.rawValue)
    }

    func bool(forKey key: Key) -> Bool {
        userDefaults.bool(forKey: key.rawValue)
    }

    func set(_ value: Bool, forKey key: Key) {
        userDefaults.set(value, forKey: key.rawValue)
    }

    func integer(forKey key: Key) -> Int {
        userDefaults.integer(forKey: key.rawValue)
    }

    func set(_ value: Int, forKey key: Key) {
        userDefaults.set(value, forKey: key.rawValue)
    }

    func double(forKey key: Key) -> Double {
        userDefaults.double(forKey: key.rawValue)
    }

    func set(_ value: Double, forKey key: Key) {
        userDefaults.set(value, forKey: key.rawValue)
    }

    func data(forKey key: Key) -> Data? {
        userDefaults.data(forKey: key.rawValue)
    }

    func set(_ value: Data?, forKey key: Key) {
        userDefaults.set(value, forKey: key.rawValue)
    }

    func value<T: Decodable>(_ type: T.Type, forKey key: Key) -> T? {
        guard let data = data(forKey: key) else { return nil }
        return try? decoder.decode(type, from: data)
    }

    func set<T: Encodable>(_ value: T?, forKey key: Key) {
        guard let value else {
            removeValue(forKey: key)
            return
        }

        guard let data = try? encoder.encode(value) else { return }
        set(data, forKey: key)
    }

    func removeValue(forKey key: Key) {
        userDefaults.removeObject(forKey: key.rawValue)
    }

    /// Reads a sensitive value from the keychain, migrating any legacy plaintext
    /// value left in `UserDefaults` by an earlier app version on first read.
    func secureString(forKey key: Key) -> String? {
        if let legacy = userDefaults.string(forKey: key.rawValue) {
            keychain.set(legacy, account: key.rawValue)
            userDefaults.removeObject(forKey: key.rawValue)
            return legacy
        }
        return keychain.string(account: key.rawValue)
    }

    func setSecureString(_ value: String?, forKey key: Key) {
        keychain.set(value, account: key.rawValue)
        // Drop any legacy plaintext copy so it can't shadow the keychain value.
        userDefaults.removeObject(forKey: key.rawValue)
    }
}

extension Prefs {
    var webSocketURL: String? {
        get { string(forKey: .webSocketURL) ?? bundledSecrets?.webSocketURL }
        set { set(newValue, forKey: .webSocketURL) }
    }

    var username: String? {
        get { secureString(forKey: .username) ?? bundledSecrets?.username }
        set { setSecureString(newValue, forKey: .username) }
    }

    var password: String? {
        get { secureString(forKey: .password) ?? bundledSecrets?.password }
        set { setSecureString(newValue, forKey: .password) }
    }

    var isMessageLayoutSwitcherEnabled: Bool {
        get { bool(forKey: .isMessageLayoutSwitcherEnabled) }
        set { set(newValue, forKey: .isMessageLayoutSwitcherEnabled) }
    }

    struct Key: RawRepresentable, Hashable, ExpressibleByStringLiteral {
        let rawValue: String

        init(rawValue: String) {
            self.rawValue = rawValue
        }

        init(_ rawValue: String) {
            self.rawValue = rawValue
        }

        init(stringLiteral value: String) {
            self.rawValue = value
        }
    }
}

extension Prefs.Key {
    static let webSocketURL = Self("connection.webSocketURL")
    static let username = Self("connection.username")
    static let password = Self("connection.password")
    static let isMessageLayoutSwitcherEnabled = Self("features.messageLayoutSwitcherEnabled")
    static let historySyncState = Self("history.syncState")
}

private struct SecretsPlist: Decodable {
    let webSocketURL: String
    let username: String
    let password: String

    static func load(from bundle: Bundle) -> Self? {
        guard let url = bundle.url(forResource: "Secrets", withExtension: "plist") else {
            return nil
        }

        do {
            let data = try Data(contentsOf: url)
            return try PropertyListDecoder().decode(Self.self, from: data)
        } catch {
            return nil
        }
    }
}
