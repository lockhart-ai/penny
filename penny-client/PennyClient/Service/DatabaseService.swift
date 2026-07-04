import Foundation
import SQLite
import SQLPropertyMacros

final class DatabaseService {
    static let shared = DatabaseService()

    private var db: Connection!
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
            db = try Connection("\(path)/db.sqlite3")
        } catch {
            fatalError(error.localizedDescription)
        }
    }

    fileprivate func connectForTesting() {
        do {
            db = try Connection()
        } catch {
            fatalError(error.localizedDescription)
        }
    }

    fileprivate func createTables() {
        do {
            try MessageModel.createTable(db: db)
            try migrateDatabase()
        } catch {
            fatalError(error.localizedDescription)
        }
    }

    fileprivate func migrateDatabase() throws {
        let currentVersion = db.userVersion ?? 0

        if currentVersion < 1 {
            db.userVersion = 1
        }

        if currentVersion < 2 {
            if try !MessageModel.columnExists(db: db, name: "image_attachment_data_urls") {
                try db.run(MessageModel.table().addColumn(MessageModel.imageAttachmentDataURLsExp, defaultValue: "[]"))
            }
            db.userVersion = 2
        }
    }
}

extension DatabaseService {
    func loadMessages() -> [MessageModel] {
        setup()

        do {
            return try MessageModel.load(db: db)
        } catch {
            print(error)
            return []
        }
    }

    func save(message: MessageModel) {
        setup()

        do {
            try message.save(db: db)
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

    fileprivate static func createTable(db: Connection) throws {
        try db.run(
            table().create(ifNotExists: true) { t in
                t.column(idExp, primaryKey: true)
                t.column(serverIDExp, unique: true)
                t.column(createdAtExp)
                t.column(contentExp)
                t.column(sourceHintExp)
                t.column(imageAttachmentDataURLsExp, defaultValue: "[]")
                t.column(isOutgoingExp)
            }
        )
    }

    fileprivate func save(db: Connection) throws {
        try db.run(
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

    fileprivate static func load(db: Connection) throws -> [MessageModel] {
        var result = [MessageModel]()
        for entry in try db.prepare(table().order(createdAtExp.asc)) {
            result.append(
                MessageModel(
                    id: entry[idExp],
                    serverID: entry[serverIDExp],
                    createdAt: entry[createdAtExp],
                    content: entry[contentExp],
                    sourceHint: entry[sourceHintExp],
                    imageAttachmentDataURLs: decodedImageAttachmentDataURLs(entry[imageAttachmentDataURLsExp]),
                    isOutgoing: entry[isOutgoingExp]
                )
            )
        }
        return result
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

    fileprivate static func columnExists(db: Connection, name: String) throws -> Bool {
        try db.schema.columnDefinitions(table: "messages").contains { column in
            column.name == name
        }
    }
}
