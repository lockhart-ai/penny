import Foundation
import Testing
@testable import PennyClient

@Suite(.serialized)
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

    @Test func loadsMessagesInCreationOrder() {
        let database = DatabaseService()
        database.setupForTesting()
        database.save(message: MessageModel(id: 2, serverID: 2, createdAt: Date(timeIntervalSince1970: 20), content: "Second", sourceHint: nil, imageAttachmentDataURLs: [], isOutgoing: false))
        database.save(message: MessageModel(id: 1, serverID: 1, createdAt: Date(timeIntervalSince1970: 10), content: "First", sourceHint: nil, imageAttachmentDataURLs: [], isOutgoing: false))

        let loaded = database.loadMessages()

        #expect(loaded.map(\.content) == ["First", "Second"])
    }
}
