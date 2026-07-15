import Foundation
import Testing
@testable import PennyClient

@Suite(.serialized)
struct PrefsTests {
    @Test func storesTypedValuesInUserDefaults() {
        let userDefaults = makeUserDefaults()
        let prefs = Prefs(userDefaults: userDefaults)

        prefs.set("wss://example.test/penny/", forKey: .webSocketURL)
        prefs.set(true, forKey: "feature.enabled")
        prefs.set(42, forKey: "answer")
        prefs.set(3.14, forKey: "pi")

        #expect(prefs.webSocketURL == "wss://example.test/penny/")
        #expect(prefs.bool(forKey: "feature.enabled"))
        #expect(prefs.integer(forKey: "answer") == 42)
        #expect(prefs.double(forKey: "pi") == 3.14)
    }

    @Test func storesCredentialsInKeychainNotUserDefaults() {
        let userDefaults = makeUserDefaults()
        let keychain = InMemoryKeychain()
        let prefs = Prefs(userDefaults: userDefaults, keychain: keychain, bundle: Bundle(for: EmptyBundleMarker.self))

        prefs.username = "robert"
        prefs.password = "hunter2"

        #expect(prefs.username == "robert")
        #expect(prefs.password == "hunter2")
        #expect(keychain.string(account: Prefs.Key.username.rawValue) == "robert")
        #expect(keychain.string(account: Prefs.Key.password.rawValue) == "hunter2")
        #expect(userDefaults.string(forKey: Prefs.Key.username.rawValue) == nil)
        #expect(userDefaults.string(forKey: Prefs.Key.password.rawValue) == nil)
    }

    @Test func migratesLegacyCredentialsFromUserDefaultsToKeychain() {
        let userDefaults = makeUserDefaults()
        userDefaults.set("robert", forKey: Prefs.Key.username.rawValue)
        userDefaults.set("hunter2", forKey: Prefs.Key.password.rawValue)
        let keychain = InMemoryKeychain()
        let prefs = Prefs(userDefaults: userDefaults, keychain: keychain, bundle: Bundle(for: EmptyBundleMarker.self))

        #expect(prefs.username == "robert")
        #expect(prefs.password == "hunter2")
        #expect(keychain.string(account: Prefs.Key.username.rawValue) == "robert")
        #expect(keychain.string(account: Prefs.Key.password.rawValue) == "hunter2")
        #expect(userDefaults.string(forKey: Prefs.Key.username.rawValue) == nil)
        #expect(userDefaults.string(forKey: Prefs.Key.password.rawValue) == nil)
    }

    @Test func storesCodableValuesInUserDefaults() {
        let userDefaults = makeUserDefaults()
        let prefs = Prefs(userDefaults: userDefaults)
        let value = SamplePreference(name: "Penny", count: 3)

        prefs.set(value, forKey: "sample")

        #expect(prefs.value(SamplePreference.self, forKey: "sample") == value)
    }

    @Test func returnsNilConnectionValuesWhenNoDefaultsOrSecretsExist() {
        let userDefaults = makeUserDefaults()
        let prefs = Prefs(userDefaults: userDefaults, keychain: InMemoryKeychain(), bundle: Bundle(for: EmptyBundleMarker.self))

        #expect(prefs.webSocketURL == nil)
        #expect(prefs.username == nil)
        #expect(prefs.password == nil)
    }

}

private struct SamplePreference: Codable, Equatable {
    let name: String
    let count: Int
}
