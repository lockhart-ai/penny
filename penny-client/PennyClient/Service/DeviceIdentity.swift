import Foundation

enum DeviceIdentity {
    private static let keychain = SystemKeychain()
    private static let deviceIDAccount = "device_id"
    private static let deviceSecretAccount = "device_secret"

    static func stableDeviceID() -> String {
        stableUUID(account: deviceIDAccount)
    }

    static func deviceSecret() -> String {
        stableUUID(account: deviceSecretAccount)
    }

    private static func stableUUID(account: String) -> String {
        if let existingValue = keychain.string(account: account) {
            return existingValue
        }

        let newValue = UUID().uuidString
        keychain.set(newValue, account: account)
        return newValue
    }
}
