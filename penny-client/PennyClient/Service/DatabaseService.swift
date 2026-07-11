import Foundation
import SQLite
import SQLPropertyMacros

final class DatabaseService {
    static let shared = DatabaseService()

    private let logger: any LogService
    private var databaseConnection: Connection!
    private var isSetup = false
    private let historySaveQueue = DispatchQueue(
        label: "com.penny.history-save",
        qos: .userInitiated
    )

    init(logger: any LogService = OSLogService(category: .database)) {
        self.logger = logger
    }

    private func measureQuery<T>(_ name: String, _ query: () throws -> T) rethrows -> T {
        let startDate = Date()
        defer {
            let elapsedMS = Int(Date().timeIntervalSince(startDate) * 1_000)
            logger.debug("database query completed (name=\(name), elapsed_ms=\(elapsedMS))", privacy: .public)
        }
        return try query()
    }

    private func measureWrite<T>(_ name: String, _ write: () throws -> T) rethrows -> T {
        let startDate = Date()
        defer {
            let elapsedMS = Int(Date().timeIntervalSince(startDate) * 1_000)
            logger.debug("database write completed (name=\(name), elapsed_ms=\(elapsedMS))", privacy: .public)
        }
        return try write()
    }

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

        if currentVersion < 3 {
            if try !MessageModel.columnExists(database: databaseConnection, name: "channel_type") {
                try databaseConnection.run(MessageModel.table().addColumn(MessageModel.channelTypeExp))
            }
            if try !MessageModel.columnExists(database: databaseConnection, name: "device_label") {
                try databaseConnection.run(MessageModel.table().addColumn(MessageModel.deviceLabelExp))
            }
            if try !MessageModel.columnExists(database: databaseConnection, name: "device_identifier") {
                try databaseConnection.run(MessageModel.table().addColumn(MessageModel.deviceIdentifierExp))
            }
            if try !MessageModel.columnExists(database: databaseConnection, name: "parent_id") {
                try databaseConnection.run(MessageModel.table().addColumn(MessageModel.parentIDExp))
            }
            databaseConnection.userVersion = 3
        }

        if currentVersion < 4 {
            if try !MessageModel.columnExists(database: databaseConnection, name: "embedding") {
                try databaseConnection.run(MessageModel.table().addColumn(MessageModel.embeddingExp))
            }
            databaseConnection.userVersion = 4
        }

        if currentVersion < 5 {
            try MessageModel.createIndexes(database: databaseConnection)
            databaseConnection.userVersion = 5
        }

        if currentVersion < 6 {
            try MessageModel.createEmbeddedRowsIndex(database: databaseConnection)
            databaseConnection.userVersion = 6
        }
    }
}

extension DatabaseService {
    func loadMessages() -> [MessageModel] {
        setup()

        do {
            return try measureQuery("load_messages") {
                try MessageModel.load(database: databaseConnection)
            }
        } catch {
            logger.error("Database operation failed: \(error.localizedDescription)", privacy: .public)
            return []
        }
    }

    func loadMessagesInBackground() async -> [MessageModel] {
        await withCheckedContinuation { continuation in
            historySaveQueue.async { [self] in
                continuation.resume(returning: loadMessages())
            }
        }
    }

    func loadEmbeddedMessagesInBackground(filter: MessagePageFilter) async -> [MessageModel] {
        await withCheckedContinuation { continuation in
            historySaveQueue.async { [self] in
                continuation.resume(returning: loadEmbeddedMessages(filter: filter))
            }
        }
    }

    func loadEmbeddedMessages(filter: MessagePageFilter) -> [MessageModel] {
        setup()

        do {
            return try measureQuery("load_embedded_messages") {
                try MessageModel.loadEmbedded(database: databaseConnection, filter: filter)
            }
        } catch {
            logger.error("Database operation failed: \(error.localizedDescription)", privacy: .public)
            return []
        }
    }

    @MainActor
    func loadMessagePage(_ request: MessagePageRequest) -> MessagePage {
        setup()

        do {
            let page = try measureQuery("load_message_page") {
                try MessageModel.loadPage(database: databaseConnection, request: request)
            }
            return MessagePage(
                messages: page.models.map(ChatMessage.init(model:)),
                nextCursor: page.nextCursor,
                hasMore: page.hasMore
            )
        } catch {
            logger.error("Database operation failed: \(error.localizedDescription)", privacy: .public)
            return MessagePage(messages: [], nextCursor: nil, hasMore: false)
        }
    }

    func minimumMessageID() -> Int? {
        setup()

        do {
            return try measureQuery("minimum_message_id") {
                try MessageModel.minimumID(database: databaseConnection)
            }
        } catch {
            logger.error("Database operation failed: \(error.localizedDescription)", privacy: .public)
            return nil
        }
    }

    func containsMessage(serverID: Int) -> Bool {
        setup()

        do {
            return try measureQuery("contains_message") {
                try MessageModel.contains(database: databaseConnection, serverID: serverID)
            }
        } catch {
            logger.error("Database operation failed: \(error.localizedDescription)", privacy: .public)
            return false
        }
    }

    @discardableResult
    func reconcileLegacyMessage(outboxID: Int, canonicalID: Int) -> Bool {
        setup()

        do {
            return try measureWrite("reconcile_legacy_message") {
                try MessageModel.reconcileLegacyMessage(
                    database: databaseConnection,
                    outboxID: outboxID,
                    canonicalID: canonicalID
                )
            }
        } catch {
            logger.error("Database operation failed: \(error.localizedDescription)", privacy: .public)
            return false
        }
    }

    @MainActor
    @discardableResult
    func reconcileLocalMessage(content: String, createdAt: Date, canonicalID: Int) -> Int? {
        setup()

        do {
            return try measureWrite("reconcile_local_message") {
                try MessageModel.reconcileLocalMessage(
                    database: databaseConnection,
                    content: content,
                    createdAt: createdAt,
                    canonicalID: canonicalID
                )
            }
        } catch {
            logger.error("Database operation failed: \(error.localizedDescription)", privacy: .public)
            return nil
        }
    }

    func save(message: MessageModel) {
        setup()

        do {
            var message = message
            if message.embedding == nil, let serverID = message.serverID {
                message.embedding = try measureQuery("message_embedding") {
                    try MessageModel.embedding(
                        database: databaseConnection,
                        serverID: serverID
                    )
                }
            }
            try measureWrite("save_message") {
                try message.save(database: databaseConnection)
            }
        } catch {
            logger.error("Database operation failed: \(error.localizedDescription)", privacy: .public)
        }
    }

    /// Persist a history page in one transaction away from the main actor.
    /// Keeping the whole page in one transaction avoids a UI stall caused by
    /// opening and committing SQLite work once per message.
    func saveMessagesInBackground(
        _ messages: [MessageModel],
        preserveAttachments: Bool
    ) async -> Int {
        await withCheckedContinuation { continuation in
            historySaveQueue.async { [self] in
                continuation.resume(
                    returning: saveMessages(messages, preserveAttachments: preserveAttachments)
                )
            }
        }
    }

    @discardableResult
    func saveMessages(_ messages: [MessageModel], preserveAttachments: Bool) -> Int {
        guard !messages.isEmpty else { return 0 }
        setup()

        do {
            var messages = messages
            try measureWrite("save_messages") {
                try databaseConnection.transaction {
                    for index in messages.indices {
                        if messages[index].embedding == nil, let serverID = messages[index].serverID {
                            messages[index].embedding = try measureQuery("message_embedding") {
                                try MessageModel.embedding(
                                    database: databaseConnection,
                                    serverID: serverID
                                )
                            }
                        }
                        if preserveAttachments, let serverID = messages[index].serverID {
                            let existing = try measureQuery("message_attachments") {
                                try MessageModel.attachments(
                                    database: databaseConnection,
                                    serverID: serverID
                                )
                            }
                            if let existing {
                                messages[index].imageAttachmentDataURLs = existing
                            }
                        }
                        try messages[index].save(database: databaseConnection)
                    }
                }
            }
            return messages.count
        } catch {
            logger.error("Database operation failed: \(error.localizedDescription)", privacy: .public)
            return 0
        }
    }

    func deleteAllMessages() {
        setup()

        do {
            _ = try measureWrite("delete_all_messages") {
                try databaseConnection.run(MessageModel.table().delete())
            }
        } catch {
            logger.error("Database operation failed: \(error.localizedDescription)", privacy: .public)
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
        channelType = message.channelType
        deviceLabel = message.deviceLabel
        deviceIdentifier = message.deviceIdentifier
        parentID = message.parentID
        imageAttachmentDataURLs = message.imageAttachmentDataURLs
        embedding = message.embedding
        isOutgoing = message.isOutgoing
    }

    init(
        id: Int,
        serverID: Int?,
        createdAt: Date,
        content: String,
        sourceHint: String?,
        channelType: String? = nil,
        deviceLabel: String? = nil,
        deviceIdentifier: String? = nil,
        parentID: Int? = nil,
        imageAttachmentDataURLs: [String],
        isOutgoing: Bool,
        embedding: Data? = nil
    ) {
        self.id = id
        self.serverID = serverID
        self.createdAt = createdAt
        self.content = content
        self.sourceHint = sourceHint
        self.channelType = channelType
        self.deviceLabel = deviceLabel
        self.deviceIdentifier = deviceIdentifier
        self.parentID = parentID
        self.imageAttachmentDataURLs = imageAttachmentDataURLs
        self.embedding = embedding
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
    @SqlProperty
    var channelType: String?
    @SqlProperty
    var deviceLabel: String?
    @SqlProperty
    var deviceIdentifier: String?
    @SqlProperty
    var parentID: Int?
    var imageAttachmentDataURLs: [String]
    fileprivate static var imageAttachmentDataURLsExp: SQLite.Expression<String> {
        Expression<String>("image_attachment_data_urls")
    }
    var embedding: Data?
    fileprivate static var embeddingExp: SQLite.Expression<Data?> {
        Expression<Data?>("embedding")
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
                tableBuilder.column(embeddingExp)
                tableBuilder.column(isOutgoingExp)
            }
        )
    }

    fileprivate static func createIndexes(database: Connection) throws {
        try database.run(table().createIndex(createdAtExp, idExp, ifNotExists: true))
        try database.run(table().createIndex(sourceHintExp, createdAtExp, idExp, ifNotExists: true))
        try database.run(table().createIndex(serverIDExp, isOutgoingExp, contentExp, createdAtExp, idExp, ifNotExists: true))
    }

    fileprivate static func createEmbeddedRowsIndex(database: Connection) throws {
        try database.run("""
            CREATE INDEX IF NOT EXISTS index_messages_embedded_created_at_id
            ON messages(created_at, id)
            WHERE embedding IS NOT NULL
            """)
    }

    fileprivate func save(database: Connection) throws {
        try database.run(
            MessageModel.table().insert(or: .replace,
                MessageModel.idExp <- id,
                MessageModel.serverIDExp <- serverID,
                MessageModel.createdAtExp <- createdAt,
                MessageModel.contentExp <- content,
                MessageModel.sourceHintExp <- sourceHint,
                MessageModel.channelTypeExp <- channelType,
                MessageModel.deviceLabelExp <- deviceLabel,
                MessageModel.deviceIdentifierExp <- deviceIdentifier,
                MessageModel.parentIDExp <- parentID,
                MessageModel.imageAttachmentDataURLsExp <- MessageModel.encodedImageAttachmentDataURLs(imageAttachmentDataURLs),
                MessageModel.embeddingExp <- embedding,
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

    fileprivate static func loadEmbedded(database: Connection, filter: MessagePageFilter) throws -> [MessageModel] {
        let query = filteredTable(for: filter)
            .filter(embeddingExp != nil)
            .order(createdAtExp.asc)
        return try database.prepare(query).map { row in
            message(from: row)
        }
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

    fileprivate static func reconcileLegacyMessage(
        database: Connection,
        outboxID: Int,
        canonicalID: Int
    ) throws -> Bool {
        guard outboxID != canonicalID,
              let legacy = try database.pluck(table().filter(serverIDExp == outboxID)) else {
            return false
        }

        let legacyID = legacy[idExp]
        if let canonical = try database.pluck(table().filter(serverIDExp == canonicalID)) {
            guard canonical[idExp] != legacyID else { return false }
            try database.run(table().filter(idExp == legacyID).delete())
            return true
        }

        try database.run(
            table()
                .filter(idExp == legacyID)
                .update(idExp <- canonicalID, serverIDExp <- canonicalID)
        )
        return true
    }

    @MainActor
    fileprivate static func reconcileLocalMessage(
        database: Connection,
        content: String,
        createdAt: Date,
        canonicalID: Int
    ) throws -> Int? {
        let candidates = try database.prepare(
            table()
                .filter(serverIDExp == nil && idExp < 0 && isOutgoingExp && contentExp == content)
        ).map(message(from:))
        guard let local = candidates.min(by: {
            abs($0.createdAt.timeIntervalSince(createdAt)) < abs($1.createdAt.timeIntervalSince(createdAt))
        }) else {
            return nil
        }

        if let canonical = try database.pluck(table().filter(serverIDExp == canonicalID)) {
            guard canonical[idExp] != local.id else { return nil }
            try database.run(table().filter(idExp == local.id).delete())
        } else {
            try database.run(
                table()
                    .filter(idExp == local.id)
                    .update(idExp <- canonicalID, serverIDExp <- canonicalID)
            )
        }
        return local.id
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
            channelType: entry[channelTypeExp],
            deviceLabel: entry[deviceLabelExp],
            deviceIdentifier: entry[deviceIdentifierExp],
            parentID: entry[parentIDExp],
            imageAttachmentDataURLs: decodedImageAttachmentDataURLs(entry[imageAttachmentDataURLsExp]),
            isOutgoing: entry[isOutgoingExp],
            embedding: entry[embeddingExp]
        )
    }

    fileprivate static func embedding(database: Connection, serverID: Int) throws -> Data? {
        try database.pluck(table().filter(serverIDExp == serverID))?[embeddingExp]
    }

    fileprivate static func attachments(database: Connection, serverID: Int) throws -> [String]? {
        guard let row = try database.pluck(table().filter(serverIDExp == serverID)) else {
            return nil
        }
        return decodedImageAttachmentDataURLs(row[imageAttachmentDataURLsExp])
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
