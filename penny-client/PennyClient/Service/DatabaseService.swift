import Foundation
import SQLite
import SQLPropertyMacros

final class DatabaseService {
    static let shared = DatabaseService()

    private var databaseConnection: Connection!
    private var isSetup = false

    func setupForTesting() {
        connectForTesting()
        createTables()
        isSetup = true
    }

    func setup() {
        guard !isSetup else { return }
        connect()
        createTables()
        isSetup = true
    }

    fileprivate func connect() {
        let path = NSSearchPathForDirectoriesInDomains(
            .documentDirectory, .userDomainMask, true
        ).first!

        do {
            databaseConnection = try Connection("\(path)/db.sqlite3")
        } catch {
            fatalError(error.localizedDescription)
        }
    }

    fileprivate func connectForTesting() {
        do {
            databaseConnection = try Connection()
        } catch {
            fatalError(error.localizedDescription)
        }
    }

    fileprivate func createTables() {
        do {
            try MessageModel.createTable(database: databaseConnection)
            try migrateDatabase()
        } catch {
            fatalError(error.localizedDescription)
        }
    }

    fileprivate func migrateDatabase() throws {
        let currentVersion = databaseConnection.userVersion ?? 0

        if currentVersion < 1 {
            databaseConnection.userVersion = 1
        }

        if currentVersion < 2 {
            if try !MessageModel.columnExists(database: databaseConnection, name: "image_attachment_data_urls") {
                try databaseConnection.run(MessageModel.table().addColumn(MessageModel.imageAttachmentDataURLsExp, defaultValue: "[]"))
            }
            databaseConnection.userVersion = 2
        }
    }
}

extension DatabaseService {
    func loadMessages() -> [MessageModel] {
        setup()

        do {
            return try MessageModel.load(database: databaseConnection)
        } catch {
            print(error)
            return []
        }
    }

    @MainActor
    func loadMessagePage(_ request: MessagePageRequest) -> MessagePage {
        setup()

        do {
            let page = try MessageModel.loadPage(database: databaseConnection, request: request)
            return MessagePage(
                messages: page.models.map(ChatMessage.init(model:)),
                nextCursor: page.nextCursor,
                hasMore: page.hasMore
            )
        } catch {
            print(error)
            return MessagePage(messages: [], nextCursor: nil, hasMore: false)
        }
    }

    func minimumMessageID() -> Int? {
        setup()

        do {
            return try MessageModel.minimumID(database: databaseConnection)
        } catch {
            print(error)
            return nil
        }
    }

    func containsMessage(serverID: Int) -> Bool {
        setup()

        do {
            return try MessageModel.contains(database: databaseConnection, serverID: serverID)
        } catch {
            print(error)
            return false
        }
    }

    func save(message: MessageModel) {
        setup()

        do {
            try message.save(database: databaseConnection)
        } catch {
            print(error)
        }
    }
}

struct MessageModel: Codable, Identifiable, Hashable {
    init(message: ChatMessage) {
        id = message.id
        serverID = message.serverID
        createdAt = message.createdAt
        content = message.content
        sourceHint = message.sourceHint
        imageAttachmentDataURLs = message.imageAttachmentDataURLs
        isOutgoing = message.isOutgoing
    }

    init(id: Int, serverID: Int?, createdAt: Date, content: String, sourceHint: String?, imageAttachmentDataURLs: [String], isOutgoing: Bool) {
        self.id = id
        self.serverID = serverID
        self.createdAt = createdAt
        self.content = content
        self.sourceHint = sourceHint
        self.imageAttachmentDataURLs = imageAttachmentDataURLs
        self.isOutgoing = isOutgoing
    }

    @SqlProperty
    var id: Int
    @SqlProperty
    var serverID: Int?
    @SqlProperty
    var createdAt: Date
    @SqlProperty
    var content: String
    @SqlProperty
    var sourceHint: String?
    var imageAttachmentDataURLs: [String]
    fileprivate static var imageAttachmentDataURLsExp: SQLite.Expression<String> {
        Expression<String>("image_attachment_data_urls")
    }
    @SqlProperty
    var isOutgoing: Bool

    fileprivate static func table() -> Table {
        Table("messages")
    }

    fileprivate static func createTable(database: Connection) throws {
        try database.run(
            table().create(ifNotExists: true) { tableBuilder in
                tableBuilder.column(idExp, primaryKey: true)
                tableBuilder.column(serverIDExp, unique: true)
                tableBuilder.column(createdAtExp)
                tableBuilder.column(contentExp)
                tableBuilder.column(sourceHintExp)
                tableBuilder.column(imageAttachmentDataURLsExp, defaultValue: "[]")
                tableBuilder.column(isOutgoingExp)
            }
        )
    }

    fileprivate func save(database: Connection) throws {
        try database.run(
            MessageModel.table().insert(or: .replace,
                MessageModel.idExp <- id,
                MessageModel.serverIDExp <- serverID,
                MessageModel.createdAtExp <- createdAt,
                MessageModel.contentExp <- content,
                MessageModel.sourceHintExp <- sourceHint,
                MessageModel.imageAttachmentDataURLsExp <- MessageModel.encodedImageAttachmentDataURLs(imageAttachmentDataURLs),
                MessageModel.isOutgoingExp <- isOutgoing
            )
        )
    }

    fileprivate static func load(database: Connection) throws -> [MessageModel] {
        var result = [MessageModel]()
        for entry in try database.prepare(table().order(createdAtExp.asc)) {
            result.append(message(from: entry))
        }
        return result
    }

    @MainActor
    fileprivate static func loadPage(database: Connection, request: MessagePageRequest) throws -> (models: [MessageModel], nextCursor: MessagePageCursor?, hasMore: Bool) {
        let limit = max(1, request.limit)
        var query = filteredTable(for: request.filter)

        if let before = request.before {
            query = query.filter(createdAtExp < before.createdAt || (createdAtExp == before.createdAt && idExp < before.id))
        }

        query = query
            .order(createdAtExp.desc, idExp.desc)
            .limit(limit + 1)

        let fetched = try database.prepare(query).map(message(from:))
        let hasMore = fetched.count > limit
        let models = fetched.prefix(limit).reversed()
        let nextCursor = models.first.map { MessagePageCursor(createdAt: $0.createdAt, id: $0.id) }
        return (Array(models), nextCursor, hasMore)
    }

    fileprivate static func minimumID(database: Connection) throws -> Int? {
        for entry in try database.prepare(table().select(idExp).order(idExp.asc).limit(1)) {
            return entry[idExp]
        }
        return nil
    }

    fileprivate static func contains(database: Connection, serverID: Int) throws -> Bool {
        try database.scalar(table().filter(serverIDExp == serverID).count) > 0
    }

    private static func filteredTable(for filter: MessagePageFilter) -> Table {
        switch filter {
        case .all:
            return table()
        case .penny:
            return table().filter(sourceHintExp == "Penny" || sourceHintExp == "Startup" || sourceHintExp == "Test Push")
        case .schedule:
            return table().filter(sourceHintExp == "Schedule")
        case .chat:
            return table().filter(isOutgoingExp || sourceHintExp == "Chat")
        case .notifier:
            return table().filter(sourceHintExp == "Notifier")
        case .collector:
            return table().filter(sourceHintExp.like("Collector: %"))
        }
    }

    private static func message(from entry: Row) -> MessageModel {
        MessageModel(
            id: entry[idExp],
            serverID: entry[serverIDExp],
            createdAt: entry[createdAtExp],
            content: entry[contentExp],
            sourceHint: entry[sourceHintExp],
            imageAttachmentDataURLs: decodedImageAttachmentDataURLs(entry[imageAttachmentDataURLsExp]),
            isOutgoing: entry[isOutgoingExp]
        )
    }

    fileprivate static func encodedImageAttachmentDataURLs(_ dataURLs: [String]) -> String {
        guard let data = try? JSONEncoder().encode(dataURLs),
              let encoded = String(data: data, encoding: .utf8) else {
            return "[]"
        }
        return encoded
    }

    fileprivate static func decodedImageAttachmentDataURLs(_ encoded: String) -> [String] {
        guard let data = encoded.data(using: .utf8),
              let dataURLs = try? JSONDecoder().decode([String].self, from: data) else {
            return []
        }
        return dataURLs
    }

    fileprivate static func columnExists(database: Connection, name: String) throws -> Bool {
        try database.schema.columnDefinitions(table: "messages").contains { column in
            column.name == name
        }
    }
}
