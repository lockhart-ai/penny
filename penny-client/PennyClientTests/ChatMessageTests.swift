import Foundation
import Testing
@testable import PennyClient

struct ChatMessageTests {
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
