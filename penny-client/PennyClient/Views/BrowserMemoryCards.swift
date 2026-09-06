import SwiftUI
import PennyServices

struct BrowserMemorySummary: View {
    let memory: MemoryRecord

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(memory.name, systemImage: memory.type == .collection ? "tray.full" : "doc.text")
                .font(.headline)
            Text(memory.description).font(.subheadline).foregroundStyle(.secondary)
            ViewThatFits(in: .horizontal) {
                HStack { metadata }
                VStack(alignment: .leading) { metadata }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            if let collected = memory.lastCollectedAt {
                Text("Last collected \(BrowserDisplay.date(collected))").font(.caption)
            } else if memory.extractionPrompt != nil {
                Text("Never collected").font(.caption)
            }
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder private var metadata: some View {
        Text(memory.type.rawValue.capitalized)
        Text("\(memory.entryCount) entries")
        if memory.archived { Text("Archived") }
        Label(memory.notificationsEnabled ? "Notifications on" : "Notifications off",
              systemImage: memory.notificationsEnabled ? "bell" : "bell.slash")
    }
}

struct BrowserEntryCard: View {
    let entry: MemoryEntryRecord
    let isLog: Bool
    @Binding var expanded: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let key = entry.key { Text(key).font(.headline) }
            Text(BrowserDisplay.date(entry.createdAt)).font(isLog ? .subheadline : .caption).foregroundStyle(.secondary)
            if isLog { Text(entry.author).font(.caption).foregroundStyle(.secondary) }
            Text(verbatim: entry.content)
                .lineLimit(BrowserDisplay.isLong(entry.content) && !expanded ? 20 : nil)
                .textSelection(.enabled)
            if BrowserDisplay.isLong(entry.content) {
                Button(expanded ? "Show less" : "Show more") { expanded.toggle() }
                    .buttonStyle(.borderless)
            }
        }
        .padding(.vertical, 4)
    }
}

struct BrowserMemoryConfig: View {
    let memory: MemoryRecord

    var body: some View {
        Section("Memory") {
            Text(memory.description).textSelection(.enabled)
            LabeledContent("Type", value: memory.type.rawValue.capitalized)
            LabeledContent("Status", value: memory.archived ? "Archived" : "Active")
            LabeledContent("Notifications", value: memory.notificationsEnabled ? "On" : "Off")
            LabeledContent("Entries", value: String(memory.entryCount))
            LabeledContent("Last collected", value: memory.lastCollectedAt.map(BrowserDisplay.date) ?? "Never")
        }
        Section("Schedule") {
            Text(verbatim: memory.schedule ?? "No automatic schedule")
                .font(.system(.body, design: .monospaced)).textSelection(.enabled)
        }
        Section("Extraction prompt") {
            Text(verbatim: memory.extractionPrompt ?? "No extraction routine")
                .textSelection(.enabled)
        }
    }
}

@MainActor
enum BrowserDisplay {
    static func date(_ value: String) -> String {
        guard let date = DateParser.parse(value) else { return value }
        return date.formatted(date: .abbreviated, time: .shortened)
    }

    static func isLong(_ content: String) -> Bool {
        content.count > 600 || content.components(separatedBy: "\n").count > 20
    }

    static func json<Value: Encodable>(_ value: Value) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        do {
            guard let text = String(data: try encoder.encode(value), encoding: .utf8) else { return "Unable to decode JSON text." }
            return text
        } catch {
            return "Unable to represent this value as JSON: \(error.localizedDescription)"
        }
    }

    static func text(_ value: JSONValue) -> String {
        if case .string(let text) = value { return text }
        return json(value)
    }
}
