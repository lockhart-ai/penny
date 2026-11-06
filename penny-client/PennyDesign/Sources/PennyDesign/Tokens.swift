import SwiftUI

/// The UIFont for a SwiftUI text style at a weight: the preferred font for
/// that text style, with the weight trait set to the weight the text uses.
/// Every reading of the real font metrics — `capHeight` below and the
/// `Tokens.pageIconSize` derivation in Tokens — goes through this.
func uiFont(_ style: Font.TextStyle, _ weight: Font.Weight) -> UIFont {
    let uiStyle: UIFont.TextStyle = switch style {
    case .largeTitle: .largeTitle
    case .title: .title1
    case .title2: .title2
    case .title3: .title3
    case .headline: .headline
    case .subheadline: .subheadline
    case .body: .body
    case .callout: .callout
    case .footnote: .footnote
    case .caption: .caption1
    case .caption2: .caption2
    @unknown default: .body
    }
    let uiWeight: UIFont.Weight = switch weight {
    case .ultraLight: .ultraLight
    case .thin: .thin
    case .light: .light
    case .regular: .regular
    case .medium: .medium
    case .semibold: .semibold
    case .bold: .bold
    case .heavy: .heavy
    case .black: .black
    default: .regular
    }
    let base = UIFont.preferredFont(forTextStyle: uiStyle)
    return UIFont(descriptor: base.fontDescriptor.addingAttributes([
        .traits: [UIFontDescriptor.TraitKey.weight: uiWeight]
    ]), size: base.pointSize)
}

/// Cap height of a SwiftUI text style at a weight, from the real font
/// metrics: the UIFont for that text style, with the weight trait set to
/// the weight the text uses, reading `.capHeight`. This is the number the
/// icon-alignment rule needs — the distance from the first line's baseline
/// up to the top of its capital letters.
func capHeight(_ style: Font.TextStyle, _ weight: Font.Weight) -> CGFloat {
    uiFont(style, weight).capHeight
}

/// Every spacing and size constant used anywhere. No magic numbers in views.
public enum Tokens {
    public static let margin: CGFloat = 16              // screen edge margin
    public static let messageGap: CGFloat = 12          // between two messages
    public static let bubbleMetaGap: CGFloat = 4        // bubble → metadata line
    public static let dockComposerGap: CGFloat = 8      // dock bottom → composer top
    public static let dockChromeGap: CGFloat = 8        // dock top → chrome row bottom (.top placement)
    public static let topFadeBleed: CGFloat = 36        // chrome row bottom → top fade fully clear
    public static let chromeTopGap: CGFloat = 2         // safe-area top → nav row top (measured off chat.png)
    public static let lastMessageComposerGap: CGFloat = 12
    public static let capsuleContentGap: CGFloat = 12   // capsule bottom → first content
    public static let iconColumn: CGFloat = 38          // dock/stream icon column
    public static let tile: CGFloat = 29                // Settings-app icon tile
    public static let cardCorner: CGFloat = 16
    public static let bubbleCorner: CGFloat = 20
    public static let menuCorner: CGFloat = 16
    public static let menuRowHeight: CGFloat = 44
    /// The one height every glass control in the nav row shares — the tab
    /// capsule, the search field capsule and every glass circle — Apple's
    /// minimum tap target. They share one centre line; the capsule and
    /// field read it as content + the glass's 4pt inset each side, the
    /// circles as their full side.
    public static let navControlHeight: CGFloat = 44
    /// The one glyph size every glass circle's symbol uses — the nav
    /// circles' size, shared with every other glass circle in the app.
    public static let navGlyphSize: CGFloat = 19.5
    public static let imageBubbleWidth: CGFloat = 270   // 16:10 flair image bubble
    public static let drawingImageWidth: CGFloat = 260  // 4:5 drawing bubble
    public static let userPhotoBubbleWidth: CGFloat = 260 // the user's photo bubble
    /// Cap on an image band's height: an extreme aspect crops (centre) at
    /// this height instead of towering — a 4:5 drawing (325pt at its fixed
    /// width) fits uncropped beneath it.
    public static let bubbleImageMaxHeight: CGFloat = 360
    /// The gap each side of a bubble before the row's margin — how far a
    /// bubble keeps from the opposite screen edge.
    public static let bubbleSideGap: CGFloat = 40
    /// A text-only bubble's padding on all sides.
    public static let bubbleTextPadding: CGFloat = 10
    /// A text-only bubble's gap between its quote block and its text.
    public static let bubbleQuoteGap: CGFloat = 8
    /// A bubble-with-image's text padding: the text sits below the image
    /// band with normal padding (horizontal 14, top 8, bottom 10).
    public static let bubbleTextPaddingH: CGFloat = 14
    public static let bubbleTextTop: CGFloat = 8
    public static let bubbleTextBottom: CGFloat = 10
    /// Penny's bubble's soft shadow and 0.5pt hairline.
    public static let bubbleShadowOpacity: Double = 0.06
    public static let bubbleShadowRadius: CGFloat = 8
    public static let bubbleShadowY: CGFloat = 2
    public static let bubbleHairline: CGFloat = 0.5
    /// The quote block's own grid: the bar's width, corner and gap to the
    /// text, the author→quoted-line gap, and the block's padding and corner.
    public static let quoteBarWidth: CGFloat = 3
    public static let quoteBarCorner: CGFloat = 1.5
    public static let quoteBarTextGap: CGFloat = 8
    public static let quoteLineGap: CGFloat = 1
    public static let quoteBlockPadding: CGFloat = 8
    public static let quoteBlockCorner: CGFloat = 10
    public static let dockScreenCapFraction: CGFloat = 0.4
    /// The composer's control height, shared by the `+` button and the
    /// message field's capsule so the row reads as one matched pair.
    public static let composerControlHeight: CGFloat = 47
    /// The composer's own vertical padding around its content row (the
    /// `+` button and the glass field), so the canvas can derive the dock's
    /// float from the row's measured height.
    public static let composerVerticalPadding: CGFloat = 8
    /// Measured: the composer view is ~74pt tall and its glass field starts
    /// ~18pt below the view's layout frame top — the composer's vertical
    /// padding (8) plus the glass effect's own padding around the field
    /// (the field fills the row at `composerControlHeight`). The canvas
    /// margins below derive their 12pt / 8pt gaps from this.
    public static let composerGlassInset: CGFloat = 18
    /// The glass field's inner paddings and corner: the text line sits this
    /// far inside the capsule's leading/trailing edges, and a field holding
    /// a pending photo rounds at `composerFieldCorner` instead of the capsule.
    public static let composerFieldLeadingInset: CGFloat = 14
    public static let composerFieldTrailingInset: CGFloat = 10
    public static let composerFieldCorner: CGFloat = 22
    /// The send arrow's glyph size inside the field, and the `+` attach
    /// button's glyph box inside its glass circle.
    public static let composerSendGlyphSize: CGFloat = 22
    public static let composerAttachGlyphFrame: CGFloat = 32
    /// A pending photo's thumbnail inside the field: `composerPendingThumbWidth`
    /// wide, cropped to the attach mock's 16:10 box, rounded, with the glass
    /// xmark as a `composerPendingXFrame` circle riding its leading edge,
    /// inset `composerPendingXInset` from it.
    public static let composerPendingThumbWidth: CGFloat = 56
    public static let composerPendingThumbHeight: CGFloat = 35
    public static let composerPendingThumbCorner: CGFloat = 8
    public static let composerPendingInset: CGFloat = 10
    public static let composerPendingXFrame: CGFloat = 28
    public static let composerPendingXGlyph: CGFloat = 11
    public static let composerPendingXInset: CGFloat = 4
    /// The bottom fade's dissolve ramp above the composer — the gap the
    /// chat's content keeps below its last message, so the last message's
    /// edge sits exactly where the fade clears.
    public static let composerFadeBleed: CGFloat = lastMessageComposerGap
    /// The five metric columns' fixed widths, so values line up card to card.
    /// Sized to fit the run card's text column (card interior minus tile,
    /// spacing and chevron — measured at 284pt) with no overflow: the
    /// strip is 282pt wide.
    public static let metricColumns: [CGFloat] = [44, 76, 36, 68, 46]
    public static let metricSpacing: CGFloat = 3
    // MARK: The header grid — one derived column set for every page

    /// A white card row's inner leading/trailing padding (the card sits at
    /// the screen margin; its tile starts this far in).
    public static let cardInnerPadding: CGFloat = 12
    /// The gap between a leading icon tile and its text — card rows and
    /// the page header alike.
    public static let iconTextGap: CGFloat = 12
    /// A card row's vertical padding — every ListRow carries it, so the
    /// hairline dividers between a card's rows span the card.
    public static let rowVerticalPadding: CGFloat = 11
    /// The gap between a ListRow's title line, its subtitle and its caption.
    public static let rowLineGap: CGFloat = 3
    /// The rows' text column: where a card row's text starts on screen
    /// (screen margin + card inner padding + tile + icon→text gap).
    /// The page header's title and byline start exactly here.
    public static let rowTextX = margin + cardInnerPadding + tile + iconTextGap
    /// The page header icon's fixed side: the span from the one-line
    /// title's cap top down to the byline's baseline, computed from the
    /// font metrics — the title's cap height, plus its descent below its
    /// baseline, plus the title→byline line gap, plus the byline's ascent.
    /// Titles are one line (`PageTitle` line-limits title and byline), so
    /// this is the same on every page and is never measured from a wrapped
    /// layout.
    public static let pageIconSize: CGFloat = {
        let title = uiFont(pageTitleStyle, pageTitleWeight)
        let byline = uiFont(pageBylineStyle, .regular)
        // Distance from a line box's top to its baseline: the font's
        // ascender plus its half of the leading.
        let ascentInBox = { (font: UIFont) in font.ascender + font.leading / 2 }
        let drop = (title.lineHeight - ascentInBox(title))   // title's descent below its baseline
            + pageTitleGap                                   // the title→byline line gap
            + ascentInBox(byline)                            // byline's ascent to its baseline
        return capHeight(pageTitleStyle, pageTitleWeight) + drop
    }()
    /// The header icon's left edge: the tile ends `iconTextGap` before the
    /// text column. The section headers start on this same column, so the
    /// header icon and the section headers form one column.
    public static let pageIconX = rowTextX - iconTextGap - pageIconSize
    /// What PageTitle adds inside the caller's screen-margin horizontal
    /// padding so its tile lands on `pageIconX`.
    public static let pageTitleLeadingInset = pageIconX - margin
    /// The section headers' column — the header icon's left edge — and the
    /// inset a section header adds inside the screen-margin padding.
    public static let sectionHeaderX = pageIconX
    public static let sectionHeaderInset = sectionHeaderX - margin

    // MARK: Page-title text styles

    /// Every non-chat page's title style. PageTitle reads it for both the
    /// title text and the tile's cap-height alignment, so they cannot drift.
    public static let pageTitleStyle: Font.TextStyle = .title2
    public static let pageTitleWeight: Font.Weight = .bold
    /// The byline under the title, and the second line the header icon's
    /// fixed side spans down to.
    public static let pageBylineStyle: Font.TextStyle = .subheadline
    /// The gap between the title line and the byline line.
    public static let pageTitleGap: CGFloat = 2
}

// MARK: - Semantic colour

/// Run types: Chat = purple, Collector = blue, History = orange.
public enum RunKind: String, CaseIterable, Identifiable {
    case chat, collector, history

    public var id: Self { self }

    public var color: Color {
        switch self {
        case .chat: return .purple
        case .collector: return .blue
        case .history: return .orange
        }
    }

    public var symbol: String {
        switch self {
        case .chat: return "bubble.left.fill"
        case .collector: return "tray.full.fill"
        case .history: return "clock.arrow.circlepath"
        }
    }

    public var name: String {
        switch self {
        case .chat: return "Chat"
        case .collector: return "Collector"
        case .history: return "History"
        }
    }
}

/// Memory surfaces: one icon per kind, never per item — Penny never sets
/// icons on anything. Collections all carry the collector's tray (the
/// collectors fill them); logs, entries and the archived row get their own.
public enum MemoryKind: String, CaseIterable, Identifiable {
    case collection, log, entry, archived

    public var id: Self { self }

    public var color: Color {
        switch self {
        case .collection: return .blue
        case .log, .archived: return .gray
        case .entry: return .teal
        }
    }

    public var symbol: String {
        switch self {
        case .collection: return "tray.full.fill"
        case .log: return "text.alignleft"
        case .entry: return "doc.text.fill"
        case .archived: return "archivebox.fill"
        }
    }
}

/// Run statuses: WORKED green, NOTHING NEW gray, FAILED red, FLAGGED orange,
/// RUNNING blue (live).
public enum RunStatus: String, CaseIterable {
    case worked = "WORKED"
    case nothingNew = "NOTHING NEW"
    case failed = "FAILED"
    case flagged = "FLAGGED"
    case running = "RUNNING"

    public var color: Color {
        switch self {
        case .worked: return .green
        case .nothingNew: return .gray
        case .failed: return .red
        case .flagged: return .orange
        case .running: return .blue
        }
    }

    public var symbol: String? {
        switch self {
        case .flagged: return "exclamationmark.triangle.fill"
        default: return nil
        }
    }
}

/// Turn roles: SYSTEM gray, USER blue, THINKING purple, TOOL CALL and
/// RESPONSE green, RESULT teal.
public enum TurnRole: String, CaseIterable {
    case system = "SYSTEM"
    case user = "USER"
    case thinking = "THINKING"
    case toolCall = "TOOL CALL"
    case response = "RESPONSE"
    case result = "RESULT"

    public var color: Color {
        switch self {
        case .system: return .gray
        case .user: return .blue
        case .thinking: return .purple
        case .toolCall, .response: return .green
        case .result: return .teal
        }
    }

    public var symbol: String {
        switch self {
        case .system: return "gearshape.fill"
        case .user: return "person.fill"
        case .thinking: return "bubble.left.fill"
        case .toolCall: return "wrench.and.screwdriver.fill"
        case .response: return "text.bubble.fill"
        case .result: return "doc.text.fill"
        }
    }
}

/// The "Penny is working" rainbow — the only colour accent outside the
/// semantic palette.
public let pennyRainbow = AngularGradient(
    colors: [.blue, .purple, .pink, .orange, .blue],
    center: .center
)
