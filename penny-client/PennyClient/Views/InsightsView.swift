import SwiftUI

struct InsightsView: View {
    @State private var viewModel: InsightsViewModel

    init(client: PennyWebSocketClient) {
        _viewModel = State(initialValue: InsightsViewModel(client: client))
    }

    var body: some View {
        Form {
            Section("Filters") {
                TextField("Agent", text: $viewModel.selectedAgentName)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                TextField("Search", text: $viewModel.query)
                Toggle("Flagged only", isOn: $viewModel.flaggedOnly)
                Button {
                    viewModel.refresh()
                } label: {
                    Label("Apply", systemImage: "line.3.horizontal.decrease.circle")
                }
            }

            Section("Summary") {
                LabeledContent("Runs", value: "\(viewModel.runs.count)")
                LabeledContent("Prompts", value: "\(viewModel.totalPromptCount)")
                LabeledContent("Needs Attention", value: "\(viewModel.failedRunCount)")
            }

            Section("Runs") {
                if viewModel.runs.isEmpty {
                    ContentUnavailableView("No Runs", systemImage: "chart.bar.doc.horizontal")
                } else {
                    ForEach(viewModel.runs) { run in
                        PromptRunRow(run: run)
                    }

                    if viewModel.hasMore {
                        Button {
                            viewModel.loadMore()
                        } label: {
                            Label("Load More", systemImage: "ellipsis.circle")
                        }
                    }
                }
            }
        }
        .navigationTitle("Insights")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    viewModel.refresh()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .accessibilityLabel("Refresh insights")
            }
        }
        .task {
            viewModel.refresh()
        }
    }
}

private struct PromptRunRow: View {
    let run: PromptLogRun

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text(run.agentName)
                    .font(.headline)
                Spacer()
                if let runOutcome = run.runOutcome {
                    Text(runOutcomeLabel(runOutcome))
                        .font(.caption)
                        .foregroundStyle(outcomeColor(runOutcome))
                }
            }

            if let runTarget = run.runTarget, !runTarget.isEmpty {
                Label(runTarget, systemImage: "scope")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            Text(run.record.isEmpty ? run.runReason ?? "No run record" : run.record)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .lineLimit(3)

            HStack {
                Label("\(run.promptCount)", systemImage: "text.bubble")
                Label("\(run.totalInputTokens + run.totalOutputTokens)", systemImage: "number")
                if !run.health.flags.isEmpty {
                    Label("\(run.health.flags.count)", systemImage: "flag")
                        .foregroundStyle(.orange)
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }

    private func runOutcomeLabel(_ outcome: RunOutcome) -> String {
        switch outcome {
        case .failed:
            return "Failed"
        case .noWork:
            return "No Work"
        case .worked:
            return "Worked"
        case .incomplete:
            return "Incomplete"
        case .cancelled:
            return "Cancelled"
        }
    }

    private func outcomeColor(_ outcome: RunOutcome) -> Color {
        switch outcome {
        case .failed, .incomplete:
            return .orange
        case .worked:
            return .green
        case .noWork, .cancelled:
            return .secondary
        }
    }
}

#Preview {
    NavigationStack {
        InsightsView(client: PennyWebSocketClient())
    }
}
