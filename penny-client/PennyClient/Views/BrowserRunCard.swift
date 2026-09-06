import SwiftUI
import UIKit
import PennyServices

struct BrowserRunCard: View {
    let run: PromptLogRun
    let model: DataBrowserViewModel

    private var expansion: Binding<Bool> { model.expansion("run:\(run.runID)") }
    private var showingRecord: Binding<Bool> {
        Binding(get: { model.recordRuns.contains(run.runID) }, set: { value in
            if value { model.recordRuns.insert(run.runID) } else { model.recordRuns.remove(run.runID) }
        })
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            DisclosureGroup(isExpanded: expansion) {
                if expansion.wrappedValue { expandedBody }
            } label: { summary }
            BrowserCopyButton(label: "Copy run") { BrowserDisplay.json(run) }
        }
        .padding(.vertical, 6)
    }

    private var summary: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(run.agentName).font(.headline)
            if let target = run.runTarget { Label(target, systemImage: "scope").font(.subheadline) }
            Text(BrowserDisplay.date(run.startedAt)).font(.caption).foregroundStyle(.secondary)
            if let outcome = run.runOutcome {
                Text(outcome.rawValue.replacingOccurrences(of: "_", with: " ").capitalized)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(outcome == .failed || outcome == .incomplete ? .orange : .primary)
            }
            if let reason = run.runReason, !reason.isEmpty { Text(reason).font(.subheadline) }
            ForEach(run.health.flags, id: \.self) { flag in
                Label(flag.rawValue.replacingOccurrences(of: "_", with: " ").capitalized, systemImage: "flag")
                    .font(.caption).foregroundStyle(.orange)
            }
            Text("\(run.promptCount) prompts · \(run.totalInputTokens) in / \(run.totalOutputTokens) out · \(Double(run.totalDurationMS) / 1000, specifier: "%.1f") s")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private var expandedBody: some View {
        VStack(alignment: .leading, spacing: 12) {
            if !run.record.isEmpty {
                Picker("Run content", selection: showingRecord) {
                    Text("Prompts").tag(false)
                    Text("Record").tag(true)
                }
                .pickerStyle(.segmented)
            }
            if showingRecord.wrappedValue && !run.record.isEmpty {
                Text(verbatim: run.record).textSelection(.enabled)
            } else {
                ForEach(Array(run.prompts.enumerated()), id: \.element.id) { index, prompt in
                    BrowserPromptCard(prompt: prompt, step: index + 1, runID: run.runID, model: model)
                }
            }
        }
        .padding(.top, 8)
    }
}

private struct BrowserPromptCard: View {
    let prompt: PromptLogEntry
    let step: Int
    let runID: String
    let model: DataBrowserViewModel

    private var identifier: String { "prompt:\(runID):\(prompt.id)" }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            DisclosureGroup(isExpanded: model.expansion(identifier)) {
                if model.expanded.contains(identifier) { turns }
            } label: {
                VStack(alignment: .leading, spacing: 4) {
                    Label("Step \(step) · \(prompt.model)", systemImage: prompt.hasTools ? "wrench" : "text.bubble")
                    Text("\(prompt.inputTokens) in / \(prompt.outputTokens) out · \(Double(prompt.durationMS) / 1000, specifier: "%.1f") s")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            BrowserCopyButton(label: "Copy prompt") { BrowserDisplay.json(prompt) }
        }
        .padding(.vertical, 4)
    }

    private var turns: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(prompt.messages.enumerated()), id: \.offset) { index, message in
                BrowserTurnCard(title: role(message), content: messageBody(message),
                                identifier: "\(identifier):turn:\(index)", model: model) {
                    BrowserDisplay.json(JSONValue.object(["promptlog_id": .number(Double(prompt.id)), "turn": message]))
                }
            }
            if !prompt.thinking.isEmpty {
                BrowserTurnCard(title: "Thinking", content: prompt.thinking, identifier: "\(identifier):thinking", model: model) {
                    BrowserDisplay.json(JSONValue.object(["promptlog_id": .number(Double(prompt.id)), "thinking": .string(prompt.thinking)]))
                }
            }
            BrowserTurnCard(title: "Response", content: BrowserDisplay.text(prompt.response), identifier: "\(identifier):response", model: model) {
                BrowserDisplay.json(JSONValue.object(["promptlog_id": .number(Double(prompt.id)), "response": prompt.response]))
            }
        }
        .padding(.top, 8)
    }

    private func role(_ value: JSONValue) -> String {
        guard case .object(let fields) = value, case .string(let role) = fields["role"] else { return "Unknown role" }
        return role.capitalized
    }

    private func messageBody(_ value: JSONValue) -> String {
        guard case .object(let fields) = value else { return BrowserDisplay.text(value) }
        // Tool metadata is meaningful: show the whole object when it has fields beyond role/content.
        guard Set(fields.keys).isSubset(of: ["role", "content"]), let content = fields["content"] else { return BrowserDisplay.json(value) }
        return BrowserDisplay.text(content)
    }
}

private struct BrowserTurnCard: View {
    let title: String
    let content: String
    let identifier: String
    let model: DataBrowserViewModel
    let copy: () -> String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            DisclosureGroup(isExpanded: model.expansion(identifier)) {
                if model.expanded.contains(identifier) {
                    Text(verbatim: content).textSelection(.enabled)
                }
            } label: {
                VStack(alignment: .leading) {
                    Text(title).font(.subheadline.weight(.semibold))
                    if !model.expanded.contains(identifier) {
                        Text(verbatim: content).lineLimit(1).font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            BrowserCopyButton(label: "Copy \(title.lowercased())", value: copy)
        }
    }
}

private struct BrowserCopyButton: View {
    let label: String
    let value: () -> String
    @State private var copied = false

    var body: some View {
        Button {
            UIPasteboard.general.string = value()
            copied = true
        } label: {
            Label(copied ? "Copied" : label, systemImage: copied ? "checkmark" : "doc.on.doc")
                .font(.caption)
        }
        .buttonStyle(.borderless)
        .accessibilityLabel(label)
    }
}
