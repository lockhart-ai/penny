import Foundation
import os

protocol LogService: Sendable {
    func trace(_ message: @autoclosure () -> String, privacy: LogPrivacy)
    func debug(_ message: @autoclosure () -> String, privacy: LogPrivacy)
    func info(_ message: @autoclosure () -> String, privacy: LogPrivacy)
    func notice(_ message: @autoclosure () -> String, privacy: LogPrivacy)
    func warning(_ message: @autoclosure () -> String, privacy: LogPrivacy)
    func error(_ message: @autoclosure () -> String, privacy: LogPrivacy)
    func fault(_ message: @autoclosure () -> String, privacy: LogPrivacy)
    func critical(_ message: @autoclosure () -> String, privacy: LogPrivacy)
}

extension LogService {
    func trace(_ message: @autoclosure () -> String) {
        trace(message(), privacy: .private)
    }

    func debug(_ message: @autoclosure () -> String) {
        debug(message(), privacy: .private)
    }

    func info(_ message: @autoclosure () -> String) {
        info(message(), privacy: .private)
    }

    func notice(_ message: @autoclosure () -> String) {
        notice(message(), privacy: .private)
    }

    func warning(_ message: @autoclosure () -> String) {
        warning(message(), privacy: .private)
    }

    func error(_ message: @autoclosure () -> String) {
        error(message(), privacy: .private)
    }

    func fault(_ message: @autoclosure () -> String) {
        fault(message(), privacy: .private)
    }

    func critical(_ message: @autoclosure () -> String) {
        critical(message(), privacy: .private)
    }
}

enum LogPrivacy: Sendable {
    case `private`
    case `public`
}

enum LogCategory: String, Sendable {
    case app
    case database
    case notifications
    case pennyService
    case webSocket
}

struct OSLogService: LogService {
    private let logger: Logger

    init(subsystem: String = "PennyClient", category: LogCategory) {
        self.logger = Logger(subsystem: subsystem, category: category.rawValue)
    }

    func trace(_ message: @autoclosure () -> String, privacy: LogPrivacy) {
        log(message(), privacy: privacy) { message, privacy in
            switch privacy {
            case .private:
                logger.trace("\(message, privacy: .private)")
            case .public:
                logger.trace("\(message, privacy: .public)")
            }
        }
    }

    func debug(_ message: @autoclosure () -> String, privacy: LogPrivacy) {
        log(message(), privacy: privacy) { message, privacy in
            switch privacy {
            case .private:
                logger.debug("\(message, privacy: .private)")
            case .public:
                logger.debug("\(message, privacy: .public)")
            }
        }
    }

    func info(_ message: @autoclosure () -> String, privacy: LogPrivacy) {
        log(message(), privacy: privacy) { message, privacy in
            switch privacy {
            case .private:
                logger.info("\(message, privacy: .private)")
            case .public:
                logger.info("\(message, privacy: .public)")
            }
        }
    }

    func notice(_ message: @autoclosure () -> String, privacy: LogPrivacy) {
        log(message(), privacy: privacy) { message, privacy in
            switch privacy {
            case .private:
                logger.notice("\(message, privacy: .private)")
            case .public:
                logger.notice("\(message, privacy: .public)")
            }
        }
    }

    func warning(_ message: @autoclosure () -> String, privacy: LogPrivacy) {
        log(message(), privacy: privacy) { message, privacy in
            switch privacy {
            case .private:
                logger.warning("\(message, privacy: .private)")
            case .public:
                logger.warning("\(message, privacy: .public)")
            }
        }
    }

    func error(_ message: @autoclosure () -> String, privacy: LogPrivacy) {
        log(message(), privacy: privacy) { message, privacy in
            switch privacy {
            case .private:
                logger.error("\(message, privacy: .private)")
            case .public:
                logger.error("\(message, privacy: .public)")
            }
        }
    }

    func fault(_ message: @autoclosure () -> String, privacy: LogPrivacy) {
        log(message(), privacy: privacy) { message, privacy in
            switch privacy {
            case .private:
                logger.fault("\(message, privacy: .private)")
            case .public:
                logger.fault("\(message, privacy: .public)")
            }
        }
    }

    func critical(_ message: @autoclosure () -> String, privacy: LogPrivacy) {
        log(message(), privacy: privacy) { message, privacy in
            switch privacy {
            case .private:
                logger.critical("\(message, privacy: .private)")
            case .public:
                logger.critical("\(message, privacy: .public)")
            }
        }
    }

    private func log(
        _ message: String,
        privacy: LogPrivacy,
        write: (_ formattedMessage: String, _ privacy: LogPrivacy) -> Void
    ) {
        write("\(Self.utcTimestamp()) \(message)", privacy)
    }

    private static func utcTimestamp() -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        return formatter.string(from: Date())
    }
}
