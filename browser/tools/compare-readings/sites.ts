/**
 * The pages the harness visits.
 *
 * Twenty mainstream news front pages — US, UK, Canada, international — chosen
 * for being the class of page #1942 opened on: JS-rendered, consent-walled,
 * heavily templated indexes of other pages, published by organisations big
 * enough that their front page is a fair test of a general reader.  A front page
 * or a section front, never a story, because the index reading is the thing on
 * trial.  Nothing here is anyone's browsing history: it is the list you would
 * write from memory of "the news".
 *
 * Plus ONE article control.  The PR's claim has two halves — index pages read
 * better AND article pages are untouched — and a list of twenty indexes can only
 * evidence the first.  Wikipedia because it is a real article page that will
 * still be at that URL in a year, unlike any news story.
 *
 * Bump LIST_VERSION when the list changes, so two result files are comparable
 * or visibly are not.
 */

export const LIST_VERSION = "2026-08-24.1";

export type SiteKind = "index" | "article-control";

export interface Site {
  url: string;
  kind: SiteKind;
}

const index = (url: string): Site => ({ url, kind: "index" });

export const SITES: Site[] = [
  index("https://www.bbc.com/news"),
  index("https://www.cnn.com/"),
  index("https://www.nytimes.com/"),
  index("https://www.theguardian.com/international"),
  index("https://www.reuters.com/"),
  index("https://apnews.com/"),
  index("https://www.cbc.ca/news"),
  index("https://www.nbcnews.com/"),
  index("https://abcnews.go.com/"),
  index("https://www.cbsnews.com/"),
  index("https://www.washingtonpost.com/"),
  index("https://www.foxnews.com/"),
  index("https://www.bloomberg.com/"),
  index("https://www.aljazeera.com/"),
  index("https://www.npr.org/"),
  index("https://www.politico.com/"),
  index("https://www.axios.com/"),
  index("https://www.ctvnews.ca/"),
  index("https://globalnews.ca/"),
  index("https://news.sky.com/"),
  { url: "https://en.wikipedia.org/wiki/Journalism", kind: "article-control" },
];
