import SwiftUI

struct SchedulesView: View {
    @State private var viewModel: SchedulesViewModel

    init(client: PennyWebSocketClient) {
        _viewModel = State(initialValue: SchedulesViewModel(client: client))
    }

    var body: some View {
        Form {
            if let errorText = viewModel.errorText {
                Section {
                    Label(errorText, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                }
            }

            Section("Add Schedule") {
                TextField("Natural language command", text: $viewModel.draftCommand, axis: .vertical)
                    .lineLimit(1...4)
                Button {
                    viewModel.addSchedule()
                } label: {
                    Label("Add", systemImage: "plus")
                }
                .disabled(!viewModel.canAddSchedule)
            }

            Section("Schedules") {
                if viewModel.schedules.isEmpty {
                    ContentUnavailableView("No Schedules", systemImage: "calendar.badge.clock")
                } else {
                    ForEach(viewModel.schedules) { schedule in
                        ScheduleEditorRow(schedule: schedule, viewModel: viewModel)
                    }
                }
            }
        }
        .navigationTitle("Schedules")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    viewModel.refresh()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .accessibilityLabel("Refresh schedules")
            }
        }
        .task {
            viewModel.refresh()
        }
    }
}

private struct ScheduleEditorRow: View {
    let schedule: ScheduleItem
    let viewModel: SchedulesViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text(schedule.timingDescription)
                    .font(.headline)
                Spacer()
                Text(schedule.cronExpression)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            TextField("Prompt", text: promptBinding, axis: .vertical)
                .lineLimit(2...6)

            HStack {
                Button {
                    viewModel.save(schedule: schedule)
                } label: {
                    Label("Save", systemImage: "checkmark")
                }

                Spacer()

                Button(role: .destructive) {
                    viewModel.delete(schedule: schedule)
                } label: {
                    Label("Delete", systemImage: "trash")
                }
            }
            .buttonStyle(.borderless)
        }
        .padding(.vertical, 4)
    }

    private var promptBinding: Binding<String> {
        Binding {
            viewModel.promptText(for: schedule)
        } set: { newValue in
            viewModel.setPromptText(newValue, for: schedule)
        }
    }
}

#Preview {
    NavigationStack {
        SchedulesView(client: PennyWebSocketClient())
    }
}
