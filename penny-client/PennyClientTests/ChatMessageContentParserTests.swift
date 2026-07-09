import Foundation
import Testing
@testable import PennyClient

struct ChatMessageContentParserTests {
    @Test func parsesMarkdownTableBetweenTextBlocks() {
        let content = """
        Before
        | Name | Count | Status |
        | --- | ---: | :---: |
        | Apples | **3** | Fresh |
        | Pears | 12 | Ripe |
        After
        """

        let blocks = ChatMessageContentParser.parse(content)

        #expect(blocks.count == 3)

        guard case .text = blocks[0].kind else {
            Issue.record("Expected leading text block")
            return
        }

        guard case .table(let table) = blocks[1].kind else {
            Issue.record("Expected table block")
            return
        }

        #expect(table.headers == ["Name", "Count", "Status"])
        #expect(table.rows == [["Apples", "**3**", "Fresh"], ["Pears", "12", "Ripe"]])
        #expect(isLeading(table.alignment(for: 0)))
        #expect(isTrailing(table.alignment(for: 1)))
        #expect(isCenter(table.alignment(for: 2)))

        guard case .text = blocks[2].kind else {
            Issue.record("Expected trailing text block")
            return
        }
    }

    @Test func keepsPipesInsideItalicTableCells() {
        let content = """
        | Item | Description |
        | --- | --- |
        | A | *left | right* |
        | B | _up | down_ |
        """

        let blocks = ChatMessageContentParser.parse(content)

        #expect(blocks.count == 1)
        guard case .table(let table) = blocks[0].kind else {
            Issue.record("Expected table block")
            return
        }

        #expect(table.rows == [["A", "*left | right*"], ["B", "_up | down_"]])
    }

    @Test func keepsEscapedPipesInsideTableCells() {
        let content = """
        | Item | Description |
        | --- | --- |
        | A | left \\| right |
        """

        let blocks = ChatMessageContentParser.parse(content)

        #expect(blocks.count == 1)
        guard case .table(let table) = blocks[0].kind else {
            Issue.record("Expected table block")
            return
        }

        #expect(table.rows == [["A", "left | right"]])
    }

    @Test func keepsIndentedContinuationInsidePreviousTableCell() {
        let content = """
        | Meme | Caption snippet |
        |------|-----------------|
        | **#15** | "Meme contrasting dogs looking guilty with cats staring intensely... " -
          *highlighting that cats just *look* like tiny statues* |
        | **#57** | "Orange cat wearing a pizza slice on its head" |
        """

        let blocks = ChatMessageContentParser.parse(content)

        #expect(blocks.count == 1)
        guard case .table(let table) = blocks[0].kind else {
            Issue.record("Expected table block")
            return
        }

        #expect(table.rows.count == 2)
        #expect(table.rows[0] == [
            "**#15**",
            "\"Meme contrasting dogs looking guilty with cats staring intensely... \" -\n*highlighting that cats just *look* like tiny statues*"
        ])
        #expect(table.rows[1] == ["**#57**", "\"Orange cat wearing a pizza slice on its head\""])
    }

    @Test func preservesLineBreaksInMarkdownTextBlocks() {
        let text = ChatMessageContentParser.markdownText(from: "first\n*second*\nthird")

        #expect(String(text.characters) == "first\nsecond\nthird")
    }

    @Test func leavesSeparatorWithoutBodyRowsAsText() {
        let content = """
        | Name | Count |
        | --- | ---: |
        """

        let blocks = ChatMessageContentParser.parse(content)

        #expect(blocks.count == 1)
        guard case .text = blocks[0].kind else {
            Issue.record("Expected text block")
            return
        }
    }

    @Test func rendersRawLinksAsHostWithLinkSymbol() {
        let text = ChatMessageContentParser.markdownText(from: "Open https://example.com/path?item=1 and www.apple.com/support.")

        #expect(String(text.characters) == "Open example.com and www.apple.com.")
        #expect(linkTargets(in: text) == [
            URL(string: "https://example.com/path?item=1"),
            URL(string: "https://www.apple.com/support")
        ])
    }

    @Test func preservesExistingMarkdownLinkText() {
        let text = ChatMessageContentParser.markdownText(from: "Read [the docs](https://example.com/docs).")

        #expect(String(text.characters) == "Read the docs.")
        #expect(linkTargets(in: text) == [URL(string: "https://example.com/docs")])
    }

    @Test func compactsMarkdownLinksWhoseLabelsAreURLs() {
        let text = ChatMessageContentParser.markdownText(from: "[www.remote-rocketship.com](https://www.remote-rocketship.com/ca/company/sherpallc/)")

        #expect(String(text.characters) == "www.remote-rocketship.com")
        #expect(linkTargets(in: text) == [URL(string: "https://www.remote-rocketship.com/ca/company/sherpallc/")])
    }

    @Test func compactsAngleBracketAutolinks() {
        let text = ChatMessageContentParser.markdownText(from: "<https://formula1.com>")

        #expect(String(text.characters) == "formula1.com")
        #expect(linkTargets(in: text) == [URL(string: "https://formula1.com")])
    }

    private func linkTargets(in text: AttributedString) -> [URL] {
        text.runs.compactMap(\.link)
    }

    private func isLeading(_ alignment: ChatMarkdownTable.ColumnAlignment) -> Bool {
        if case .leading = alignment {
            return true
        }
        return false
    }

    private func isCenter(_ alignment: ChatMarkdownTable.ColumnAlignment) -> Bool {
        if case .center = alignment {
            return true
        }
        return false
    }

    private func isTrailing(_ alignment: ChatMarkdownTable.ColumnAlignment) -> Bool {
        if case .trailing = alignment {
            return true
        }
        return false
    }
}
