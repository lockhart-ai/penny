import Foundation
import Testing
@testable import PennyClient

@Suite(.serialized)
@MainActor
struct DatabaseServiceTests {
    @Test func savesAndLoadsMessagesIncludingAttachments() {
        let database = DatabaseService()
        database.setupForTesting()
        let createdAt = Date(timeIntervalSince1970: 1_783_128_000)
        let model = MessageModel(
            id: 42,
            serverID: 42,
            createdAt: createdAt,
            content: "Hello **Penny**",
            sourceHint: "Chat",
            imageAttachmentDataURLs: ["data:image/png;base64,aGVsbG8="],
            isOutgoing: false
        )

        database.save(message: model)

        let loaded = database.loadMessages()
        #expect(loaded.count == 1)
        #expect(loaded.first?.id == 42)
        #expect(loaded.first?.serverID == 42)
        #expect(loaded.first?.createdAt == createdAt)
        #expect(loaded.first?.content == "Hello **Penny**")
        #expect(loaded.first?.sourceHint == "Chat")
        #expect(loaded.first?.imageAttachmentDataURLs == ["data:image/png;base64,aGVsbG8="])
        #expect(loaded.first?.isOutgoing == false)
    }

    @Test func persistsEmbeddingsAndPreservesExistingVectorWhenSyncPayloadOmitsIt() {
        let database = DatabaseService()
        database.setupForTesting()
        let embedding = Data([0, 0, 128, 63])
        database.save(message: MessageModel(
            id: 42,
            serverID: 42,
            createdAt: Date(timeIntervalSince1970: 1),
            content: "Original",
            sourceHint: "Chat",
            imageAttachmentDataURLs: [],
            isOutgoing: false,
            embedding: embedding
        ))

        database.save(message: MessageModel(
            id: 42,
            serverID: 42,
            createdAt: Date(timeIntervalSince1970: 1),
            content: "History refresh",
            sourceHint: "Chat",
            imageAttachmentDataURLs: [],
            isOutgoing: false,
            embedding: nil
        ))

        #expect(database.loadMessages().first?.content == "History refresh")
        #expect(database.loadMessages().first?.embedding == embedding)
    }

    @Test func backgroundHistorySavePreservesAttachmentsWhenPageOmitsThem() async {
        let database = DatabaseService()
        database.setupForTesting()
        let createdAt = Date(timeIntervalSince1970: 1)
        database.save(message: MessageModel(
            id: 7,
            serverID: 7,
            createdAt: createdAt,
            content: "Original",
            sourceHint: "Chat",
            imageAttachmentDataURLs: ["data:image/png;base64,aGVsbG8="],
            isOutgoing: false
        ))

        let saved = await database.saveMessagesInBackground(
            [MessageModel(
                id: 7,
                serverID: 7,
                createdAt: createdAt,
                content: "Updated",
                sourceHint: "Chat",
                imageAttachmentDataURLs: [],
                isOutgoing: false
            )],
            preserveAttachments: true
        )

        #expect(saved == 1)
        #expect(database.loadMessages().first?.content == "Updated")
        #expect(database.loadMessages().first?.imageAttachmentDataURLs == ["data:image/png;base64,aGVsbG8="])
    }

    @Test func loadsMessagesInCreationOrder() {
        let database = DatabaseService()
        database.setupForTesting()
        database.save(message: MessageModel(id: 2, serverID: 2, createdAt: Date(timeIntervalSince1970: 20), content: "Second", sourceHint: nil, imageAttachmentDataURLs: [], isOutgoing: false))
        database.save(message: MessageModel(id: 1, serverID: 1, createdAt: Date(timeIntervalSince1970: 10), content: "First", sourceHint: nil, imageAttachmentDataURLs: [], isOutgoing: false))

        let loaded = database.loadMessages()

        #expect(loaded.map(\.content) == ["First", "Second"])
    }

    @Test func deletesAllMessagesWithoutRemovingTheDatabase() {
        let database = DatabaseService()
        database.setupForTesting()
        database.save(message: makeMessage(id: 1, content: "First"))
        database.save(message: makeMessage(id: 2, content: "Second"))

        database.deleteAllMessages()

        #expect(database.loadMessages().isEmpty)
        database.save(message: makeMessage(id: 3, content: "Still usable"))
        #expect(database.loadMessages().map(\.content) == ["Still usable"])
    }

    @Test func latestMessagePageReturnsNewestMessagesInDisplayOrder() {
        let database = DatabaseService()
        database.setupForTesting()
        saveNumberedMessages(in: database, ids: 1...5)

        let page = database.loadMessagePage(MessagePageRequest(limit: 2, filter: .all))

        #expect(page.messages.map(\.id) == [4, 5])
        #expect(page.nextCursor == MessagePageCursor(createdAt: Date(timeIntervalSince1970: 4), id: 4))
        #expect(page.hasMore)
    }

    @Test func olderMessagePageUsesCursorWithoutOverlap() {
        let database = DatabaseService()
        database.setupForTesting()
        saveNumberedMessages(in: database, ids: 1...5)
        let latestPage = database.loadMessagePage(MessagePageRequest(limit: 2, filter: .all))

        let olderPage = database.loadMessagePage(MessagePageRequest(limit: 2, before: latestPage.nextCursor, filter: .all))
        let oldestPage = database.loadMessagePage(MessagePageRequest(limit: 2, before: olderPage.nextCursor, filter: .all))

        #expect(olderPage.messages.map(\.id) == [2, 3])
        #expect(olderPage.hasMore)
        #expect(oldestPage.messages.map(\.id) == [1])
        #expect(oldestPage.hasMore == false)
    }

    @Test func embeddedMessageLoadOnlyReturnsEmbeddedRowsAndAppliesFilters() {
        let database = DatabaseService()
        database.setupForTesting()
        database.save(message: makeMessage(id: 1, content: "Penny", sourceHint: "Penny", embedding: floatData([1, 0])))
        database.save(message: makeMessage(id: 2, content: "Schedule", sourceHint: "Schedule", embedding: floatData([0, 1])))
        database.save(message: makeMessage(id: 3, content: "Unembedded", sourceHint: "Schedule"))

        #expect(database.loadEmbeddedMessages(filter: .all).map(\.id) == [1, 2])
        #expect(database.loadEmbeddedMessages(filter: .schedule).map(\.id) == [2])
    }

    @Test func messagePagesApplySourceFilters() {
        let database = DatabaseService()
        database.setupForTesting()
        database.save(message: makeMessage(id: 1, content: "Penny", sourceHint: "Penny"))
        database.save(message: makeMessage(id: 2, content: "Startup", sourceHint: "Startup"))
        database.save(message: makeMessage(id: 3, content: "Schedule", sourceHint: "Schedule"))
        database.save(message: makeMessage(id: 4, content: "Chat", sourceHint: "Chat"))
        database.save(message: makeMessage(id: 5, content: "Test Push", sourceHint: "Test Push"))
        database.save(message: makeMessage(id: 6, content: "Notifier", sourceHint: "Notifier"))
        database.save(message: makeMessage(id: 7, content: "Collector", sourceHint: "Collector: flight-deals"))
        database.save(message: makeMessage(id: 8, serverID: nil, content: "Outgoing", isOutgoing: true))

        #expect(database.loadMessagePage(MessagePageRequest(limit: 20, filter: .penny)).messages.map(\.id) == [1, 2, 5])
        #expect(database.loadMessagePage(MessagePageRequest(limit: 20, filter: .schedule)).messages.map(\.id) == [3])
        #expect(database.loadMessagePage(MessagePageRequest(limit: 20, filter: .chat)).messages.map(\.id) == [4, 8])
        #expect(database.loadMessagePage(MessagePageRequest(limit: 20, filter: .notifier)).messages.map(\.id) == [6])
        #expect(database.loadMessagePage(MessagePageRequest(limit: 20, filter: .collector)).messages.map(\.id) == [7])
    }

    @Test func messageIdentityHelpersUsePersistedRows() {
        let database = DatabaseService()
        database.setupForTesting()
        database.save(message: makeMessage(id: -3, serverID: nil, content: "Local", isOutgoing: true))
        database.save(message: makeMessage(id: 10, serverID: 10, content: "Remote"))

        #expect(database.minimumMessageID() == -3)
        #expect(database.containsMessage(serverID: 10))
        #expect(database.containsMessage(serverID: 11) == false)
    }

    @Test func reconcilesLegacyOutboxServerIDToCanonicalMessageID() {
        let database = DatabaseService()
        database.setupForTesting()
        database.save(message: makeMessage(id: 17, serverID: 17, content: "Legacy"))

        #expect(database.reconcileLegacyMessage(outboxID: 17, canonicalID: 42))
        let loaded = database.loadMessages()
        #expect(loaded.count == 1)
        #expect(loaded.first?.id == 42)
        #expect(loaded.first?.serverID == 42)
    }

    @Test func removesLegacyRowWhenCanonicalMessageAlreadyExists() {
        let database = DatabaseService()
        database.setupForTesting()
        database.save(message: makeMessage(id: 17, serverID: 17, content: "Legacy"))
        database.save(message: makeMessage(id: 42, serverID: 42, content: "Canonical"))

        #expect(database.reconcileLegacyMessage(outboxID: 17, canonicalID: 42))
        let loaded = database.loadMessages()
        #expect(loaded.map(\.content) == ["Canonical"])
    }

    @Test func reconcilesLocalMessageWithoutScanningUnrelatedRows() {
        let database = DatabaseService()
        database.setupForTesting()
        saveNumberedMessages(in: database, ids: 1...200)
        database.save(message: makeMessage(id: -2, serverID: nil, content: "Different", isOutgoing: true))
        database.save(message: makeMessage(id: -3, serverID: nil, content: "Echo", isOutgoing: true))
        database.save(message: makeMessage(id: -4, serverID: nil, content: "Echo", isOutgoing: true))

        let localID = database.reconcileLocalMessage(
            content: "Echo",
            createdAt: Date(timeIntervalSince1970: -4),
            canonicalID: 400
        )

        #expect(localID == -4)
        let loaded = database.loadMessages()
        #expect(loaded.contains { $0.id == 400 && $0.serverID == 400 && $0.content == "Echo" })
        #expect(loaded.contains { $0.id == -3 && $0.serverID == nil && $0.content == "Echo" })
    }
}

private func saveNumberedMessages(in database: DatabaseService, ids: ClosedRange<Int>) {
    for id in ids {
        database.save(message: makeMessage(id: id, content: "Message \(id)"))
    }
}

private func makeMessage(
    id: Int,
    serverID: Int? = nil,
    content: String,
    sourceHint: String? = nil,
    isOutgoing: Bool = false,
    embedding: Data? = nil
) -> MessageModel {
    MessageModel(
        id: id,
        serverID: serverID ?? (isOutgoing ? nil : id),
        createdAt: Date(timeIntervalSince1970: TimeInterval(id)),
        content: content,
        sourceHint: sourceHint,
        imageAttachmentDataURLs: [],
        isOutgoing: isOutgoing,
        embedding: embedding
    )
}

private func floatData(_ values: [Float]) -> Data {
    var values = values
    return Data(bytes: &values, count: values.count * MemoryLayout<Float>.size)
}
