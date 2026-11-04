import Foundation
import Testing
@testable import PennyServices

struct ChatMessageTests {
    @Test func serverRunLinkIsOptionalAndSurvivesMessageConversion() throws {
        for runID in [nil, "fixture-run"] as [String?] {
            var payload: [String: JSONValue] = [
                "id": .number(2), "message_id": .number(42), "created_at": .string("2026-01-01T00:00:00Z"),
                "content": .string("Synthetic response")
            ]
            if let runID { payload["run_id"] = .string(runID) }
            let server = try JSONDecoder().decode(ServerChatMessage.self, from: JSONEncoder().encode(payload))
            let message = ChatMessage.remote(server)
            #expect(message.id == 42)
            #expect(message.runID == runID)
            #expect(MessageModel(message: message).runID == runID)
        }
    }

    @Test func displayTimestampOmitsDateForMessagesSentToday() {
        let referenceDate = testDate(year: 2026, month: 7, day: 12, hour: 16, minute: 30)
        let createdAt = testDate(year: 2026, month: 7, day: 12, hour: 9, minute: 15)
        let message = makeChatMessage(createdAt: createdAt)

        #expect(message.displayTimestamp(relativeTo: referenceDate, calendar: testCalendar) == createdAt.formatted(date: .omitted, time: .shortened))
    }

    @Test func displayTimestampIncludesDateForMessagesNotSentToday() {
        let referenceDate = testDate(year: 2026, month: 7, day: 12, hour: 16, minute: 30)
        let createdAt = testDate(year: 2026, month: 7, day: 11, hour: 21, minute: 45)
        let message = makeChatMessage(createdAt: createdAt)

        #expect(message.displayTimestamp(relativeTo: referenceDate, calendar: testCalendar) == createdAt.formatted(date: .abbreviated, time: .shortened))
    }

    private var testCalendar: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = testTimeZone
        return calendar
    }

    private var testTimeZone: TimeZone {
        TimeZone(secondsFromGMT: 0) ?? TimeZone.current
    }

    private func testDate(year: Int, month: Int, day: Int, hour: Int, minute: Int) -> Date {
        let components = DateComponents(
            calendar: testCalendar,
            timeZone: testTimeZone,
            year: year,
            month: month,
            day: day,
            hour: hour,
            minute: minute
        )

        guard let date = components.date else {
            Issue.record("Expected valid test date components")
            return Date(timeIntervalSince1970: 0)
        }

        return date
    }

    private func makeChatMessage(createdAt: Date) -> ChatMessage {
        ChatMessage(
            id: 1,
            serverID: 1,
            createdAt: createdAt,
            content: "Hello",
            sourceHint: nil,
            imageAttachments: [],
            isOutgoing: false
        )
    }
}
