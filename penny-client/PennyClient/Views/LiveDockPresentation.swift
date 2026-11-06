import Foundation
import PennyDesign
import PennyServices

/// The live dock's (#3042) per-tool icon and label. A handful of known
/// tools get bespoke phrasing — ported as-is from the old typing row's
/// status line (#2965) — but a job's tool sequence is arbitrary (CLAUDE.md
/// "a job's tool sequence is arbitrary"): a plugin can add a tool this
/// switch has never heard of, so the default branch is never "not
/// covered" — it humanises the tool's own name and, when one is given,
/// its first string argument, the same way an unrecognised `browse` query
/// would read.
struct LiveDockToolPresentation {
    let icon: String
    let label: String
}

func liveDockPresentation(for tool: AgentProgressToolItem) -> LiveDockToolPresentation {
    switch tool.name {
    case "browse":
        return liveDockBrowsePresentation(tool)
    case "search_emails":
        return LiveDockToolPresentation(icon: "envelope", label: "Searched emails")
    case "read_emails":
        return LiveDockToolPresentation(icon: "envelope.open", label: "Read emails")
    case "list_emails":
        return LiveDockToolPresentation(icon: "envelope", label: "Listed emails")
    case "list_folders":
        return LiveDockToolPresentation(icon: "folder", label: "Listed email folders")
    case "draft_email":
        return LiveDockToolPresentation(icon: "square.and.pencil", label: "Drafted an email")
    case "generate_image":
        return LiveDockToolPresentation(icon: "photo", label: "Generated an image")
    default:
        return liveDockGenericPresentation(tool)
    }
}

private func liveDockBrowsePresentation(_ tool: AgentProgressToolItem) -> LiveDockToolPresentation {
    guard case .array(let queries)? = tool.arguments["queries"],
          let query = queries.compactMap(liveDockStringValue).first else {
        return LiveDockToolPresentation(icon: "magnifyingglass", label: "Looked up")
    }
    if let host = formattedURLHost(from: query) {
        return LiveDockToolPresentation(icon: "doc.text", label: "Read \(host)")
    }
    return LiveDockToolPresentation(icon: "magnifyingglass", label: "Searched \"\(query)\"")
}

/// The generic default for a tool this switch has no bespoke case for: its
/// own name, humanised (`check_availability` → "Check availability"), plus
/// its first string argument — keys sorted so the choice is deterministic,
/// since `arguments` is an unordered dictionary — shortened to a host when
/// that argument is a URL, the same rule `browse` uses.
private func liveDockGenericPresentation(_ tool: AgentProgressToolItem) -> LiveDockToolPresentation {
    let name = liveDockHumanisedToolName(tool.name)
    guard let value = liveDockFirstStringArgument(tool.arguments) else {
        return LiveDockToolPresentation(icon: "wrench.and.screwdriver", label: name)
    }
    let formattedValue = formattedURLHost(from: value) ?? value
    return LiveDockToolPresentation(icon: "wrench.and.screwdriver", label: "\(name) · \(formattedValue)")
}

private func liveDockHumanisedToolName(_ name: String) -> String {
    let words = name.split(separator: "_").map(String.init)
    guard let first = words.first else { return name }
    return ([first.capitalized] + words.dropFirst()).joined(separator: " ")
}

/// The first string-shaped argument, by sorted key — `arguments` is an
/// unordered `[String: AgentProgressValue]`, so "first" has to mean
/// something deterministic. An array argument (like `browse`'s `queries`)
/// counts via its own first string element.
private func liveDockFirstStringArgument(_ arguments: [String: AgentProgressValue]) -> String? {
    for key in arguments.keys.sorted() {
        switch arguments[key] {
        case .string(let value)?:
            return value
        case .array(let values)?:
            if let first = values.compactMap(liveDockStringValue).first { return first }
        default:
            continue
        }
    }
    return nil
}

private func liveDockStringValue(_ value: AgentProgressValue) -> String? {
    if case .string(let string) = value { return string }
    return nil
}

private func formattedURLHost(from string: String) -> String? {
    guard let components = URLComponents(string: string), let host = components.host else { return nil }
    return host.hasPrefix("www.") ? String(host.dropFirst(4)) : host
}

extension AgentProgressRunItem {
    /// The dock's chronological stream (#3042): each step's thinking chunk
    /// (if the server sent one) followed by its tool rows, oldest step
    /// first. A tool is "running" — the animated ellipsis, not the blue
    /// check — exactly when it belongs to the run's LAST step: structural
    /// position, not a per-tool flag the server has never sent, since
    /// nothing in the wire protocol marks one tool call "finished" on its
    /// own (CLAUDE.md "structural state over model judgment").
    var liveDockItems: [LiveDockItem] {
        let lastStepNumber = steps.last?.number
        return steps.flatMap { step -> [LiveDockItem] in
            var items: [LiveDockItem] = []
            if let thinking = step.thinking {
                items.append(LiveDockItem(id: "\(id)-\(step.number)-thinking", kind: .thinking(thinking)))
            }
            let running = step.number == lastStepNumber
            items.append(contentsOf: step.tools.enumerated().map { toolIndex, tool in
                let presentation = liveDockPresentation(for: tool)
                return LiveDockItem(
                    id: "\(id)-\(step.number)-tool-\(toolIndex)",
                    kind: .tool(symbol: presentation.icon, label: presentation.label, running: running)
                )
            })
            return items
        }
    }
}
