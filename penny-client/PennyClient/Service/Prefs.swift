import Foundation

final class Prefs {
    static let shared = Prefs()

    private let userDefaults: UserDefaults
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private let bundledSecrets: SecretsPlist?

    init(userDefaults: UserDefaults = .standard, bundle: Bundle = .main) {
        self.userDefaults = userDefaults
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
}

extension Prefs {
    var webSocketURL: String? {
        get { string(forKey: .webSocketURL) ?? bundledSecrets?.webSocketURL }
        set { set(newValue, forKey: .webSocketURL) }
    }

    var username: String? {
        get { string(forKey: .username) ?? bundledSecrets?.username }
        set { set(newValue, forKey: .username) }
    }

    var password: String? {
        get { string(forKey: .password) ?? bundledSecrets?.password }
        set { set(newValue, forKey: .password) }
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
