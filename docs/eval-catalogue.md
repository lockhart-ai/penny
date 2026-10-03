# Eval catalogue

The behaviours Penny supports, one row per eval case, grouped by the area of Penny each case is about. This file is generated from the cases: `make fix` writes it and `make check` fails when it is out of date. It is not edited by hand.

A case declares its area and, where it covers one, the edge of the conversation machine it moves along. The first wording is the first of the wordings the case is driven in.

## Cases by area

| Area | Cases |
|---|---|
| Conversation machine | 30 |
| Teaching a routine | 17 |
| Memory | 12 |
| Answering from the web | 3 |
| Standing jobs | 5 |
| Notifications | 2 |
| Background collectors | 4 |
| Reading a page | 4 |
| Chat tools | 6 |
| Total | 83 |

## Conversation machine

| Case | Behaviour | First wording | Edge |
|---|---|---|---|
| `classifier-applies-and-names-the-routine-that-covers-the-ask` | In the state-classifier micro-context, when a cold ask supplies everything one of several near-neighbour routines needs, Penny draws apply and binds the routine that covers it. | can you watch the price on driftline.example/boards/7-2? | idle → apply |
| `classifier-applies-when-chat-carries-a-covered-ask` | In the state-classifier micro-context, when one message carries a chat preamble and a covered ask together, Penny draws apply and binds the covering routine. | morning! oh and can you watch the price on beanhouse.example/grinders/ek43? | idle → apply |
| `classifier-draws-apply-when-the-missing-value-arrives` | In the state-classifier micro-context, when the page the parked round was waiting on arrives, Penny draws apply and binds the routine she had named. | harborkayak.example/rentals/sea-touring | request → apply |
| `classifier-draws-apply-when-the-offer-is-accepted` | In the state-classifier micro-context, when the user accepts the round that was just demonstrated, Penny draws apply and binds the routine that round taught. | yes, that's exactly it — go ahead | learn → apply |
| `classifier-draws-learn-when-teaching-arrives-unprompted` | In the state-classifier micro-context, when the user starts teaching a new routine unprompted and their message carries the steps, Penny draws learn — she does not read the teaching as an ask to be taught, nor as one of the routines she already has. | new job for you: each morning read harborferries.example/timetable and remember the first sailing | idle → learn |
| `classifier-draws-learn-when-the-teach-question-is-answered` | In the state-classifier micro-context, when the reply to the teach question carries the instructions it asked for, Penny takes the round into learn. | sure — read harborferries.example/timetable and remember the first sailing | elicit → learn |
| `classifier-draws-request-when-a-known-routine-is-a-value-short` | In the state-classifier micro-context, when a routine she already knows covers the ask but the message never says which page it is about, Penny draws request and binds that routine. | can you keep an eye on the price of the camera kit listing for me? | idle → request |
| `classifier-elicits-on-a-cold-registry` | In the state-classifier micro-context, when the user asks for something that keeps running on its own and the registry holds no routine at all, Penny opens a teach round in elicit. | hey can you keep an eye on the harbor ferry timetable for me? | idle → elicit |
| `classifier-elicits-when-a-routine-shares-the-verb-but-not-the-domain` | In the state-classifier micro-context, when a routine she knows matches the ask's verb shape in a different world, Penny opens a teach round in elicit. | keep a list of new restaurants opening downtown | idle → elicit |
| `classifier-elicits-when-the-named-routine-was-wrong` | In the state-classifier micro-context, when the user says the routine the assistant named was the wrong one and still wants the task done, Penny returns the round to elicit. | no, that's not what i meant — but i do still need this doing | request → elicit |
| `classifier-falls-to-idle-when-a-parked-request-is-called-off` | In the state-classifier micro-context, when the user calls off the round that is parked waiting on a detail, Penny falls to idle. | actually never mind, forget the whole thing | request → idle |
| `classifier-falls-to-idle-when-a-post-failure-reply-carries-no-instructions` | In the state-classifier micro-context, when the reply to a failed demonstration is an engaged question carrying no instructions, Penny falls to idle. | what went wrong exactly? which page did you open? | learn → idle |
| `classifier-falls-to-idle-when-the-elicit-round-is-called-off` | In the state-classifier micro-context, when the user calls off the round the teach question opened, Penny falls to idle. | actually never mind, forget the ferry timetable thing | elicit → idle |
| `classifier-holds-idle-on-a-passing-mention` | In the state-classifier micro-context, when the user mentions in passing something a routine she already knows could be pointed at, Penny holds the conversation in idle. | i've been checking the auction listings every day lately | idle → idle |
| `classifier-holds-idle-when-a-running-jobs-notifications-are-switched-off` | In the state-classifier micro-context, when the ask turns notifications off for a job that is already running, Penny holds the conversation in idle. | turn notifications off for the camera kit price watch | idle → idle |
| `classifier-stays-in-learn-on-a-correction` | In the state-classifier micro-context, when the reply to a failed demonstration corrects it and the correction is actionable now, Penny holds the round in learn. | that link's dead — use harborferries.example/timetable-v2 instead | learn → learn |
| `classifier-stays-parked-when-the-teach-question-is-unanswered` | In the state-classifier micro-context, when the reply to the teach question asks a question back instead of answering it, Penny leaves the round parked in elicit. | what would you need from me to do that? | elicit → elicit |
| `transition-elicit-to-idle` | In the chat agent, when she has asked to be taught a job and the user calls it off and changes the subject in the same breath, Penny lets go of the round entirely — creating nothing, changing nothing and registering nothing — and answers the new subject as the ordinary conversation it is. | ah never mind, forget that — anything good at the harbor market this weekend? | elicit → idle |
| `transition-elicit-to-learn` | In the chat agent, when she has asked to be taught a job and the user walks her through it once, Penny follows the steps as given, mints a routine from what she just did, and tells the user what that routine will run — without setting it running. | yeah — go to https://faux-market.example/aurora-deck-2, find the current price, and remember it | elicit → learn |
| `transition-elicit-to-learn-absent` | In the chat agent, when she has asked to be taught a job and the user walks her through it on a page that does not carry the fact asked for, Penny stays in the round, every entry she keeps is something the page says, and everything already in the store survives as it was. | go to https://communitygarden.example/noticeboard, find the plot waitlist opening date, and remember it | elicit → learn |
| `transition-idle-to-apply` | In the chat agent, when a cold ask names a new place to run a routine she already has and supplies everything that routine needs, Penny binds it to what the ask names and stands the job up on the terms it gave, and every job already running survives unchanged. | can you watch this listing for me and let me know when the price changes? https://faux-market.example/keel-lantern — every hour until sunday night is fine | idle → apply |
| `transition-idle-to-elicit` | In the chat agent, when the user asks for something that has to keep running and no routine she has covers it, Penny parks the round in elicit. | can you watch this listing for me daily and let me know when the price changes? https://faux-market.example/aurora-deck-2 | idle → elicit |
| `transition-idle-to-idle` | In the chat agent, when a message arriving on an idle machine asks for nothing that needs to keep running, Penny answers it in conversation and changes nothing — even with five jobs already running behind her and one of them watching what it mentions. | the ferry ride this morning was gorgeous btw | idle → idle |
| `transition-idle-to-learn` | In the chat agent, when a message arrives already carrying the instructions for a job, Penny runs that round against the page it names, keeps what it says in the round's own container, and mints a routine from what she just did, and every job already going survives unchanged. | hey penny i'm gonna teach you how to check the harbour flag — go to https://harbormaster.example/signals, read which flag is flying today, and save it | idle → learn |
| `transition-idle-to-request` | In the chat agent, when a routine she already has nearly covers what the user asks for and the ask leaves out one value that routine needs, Penny parks the round in request on that routine, waiting on the missing value, and every job already running survives unchanged. | i found another listing i want to track — watch its price every couple hours until sunday and tell me if it moves | idle → request |
| `transition-learn-to-apply` | In the chat agent, when a demonstrated round has been taught and the user accepts the offer to keep it running and supplies the job's terms, Penny stands the job up on the round's own container, on the cadence and the end the acceptance gave and telling them when it moves, and what the store already held survives unchanged. | perfect — do that every hour until 10pm tonight and tell me if it changes | learn → apply |
| `transition-learn-to-idle` | In the chat agent, when a teach round is under way with its container built and written into and the user abandons it, Penny archives that container and registers nothing — leaving every other collection exactly as she found it. | actually forget it, i don't need this | learn → idle |
| `transition-learn-to-learn` | In the chat agent, when the user corrects what a demonstrated round was aimed at, Penny re-runs the round against the corrected target: what the corrected target says lands in the round's own container, and the one routine she taught for the round now looks for it. | sorry — i meant the south loop line, use that one | learn → learn |
| `transition-request-to-apply` | In the chat agent, when the value a round was parked waiting on arrives, Penny composes it with what the ask already settled and stands the job up under the name both turns' values derive, on the terms the ask gave, and every job already running survives unchanged. | https://northpier.example/departures | request → apply |
| `transition-request-to-idle` | In the chat agent, when a round is parked waiting on the one detail an ask left out and the user calls it off, Penny ends the round and builds nothing out of the half it had already settled — no job stood up, and none of the running ones touched. | you know what, skip it | request → idle |

## Teaching a routine

| Case | Behaviour | First wording | Edge |
|---|---|---|---|
| `binder-binds-the-page-and-lets-neither-term-in` | In the skill-binder micro-context, when a routine Penny already knows is asked for again and the ask states the job's cadence and its end beside the page, Penny binds the page and takes neither term into the value. | can you watch this listing for me and let me know when the price changes? https://faux-market.example/keel-lantern — every hour until sunday night is fine |  |
| `binder-fills-one-and-names-the-other-missing` | In the skill-binder micro-context, when a routine Penny already knows is pointed at something new and the ask supplies only part of what it declares, Penny fills each parameter from the span that supplies it and names the one nothing supplies — taking the job's cadence into neither. | can you check the timetable at https://northpier.example/departures every morning and keep me posted? |  |
| `binder-fills-the-still-open-parameter-when-the-value-arrives` | In the skill-binder micro-context, when a parked round hands back what it already settled and only the still-open parameter is offered, Penny fills it from the turn that just arrived and never from what the round had already settled. | can you check the timetable at https://northpier.example/departures every morning and keep me posted? / the dawn sailing — that's the one i'm after |  |
| `binder-invents-nothing-for-a-condition-the-signature-cannot-hold` | In the skill-binder micro-context, when the ask carries a condition the routine's signature has nowhere to put, Penny binds the page it names and pads the value with none of it. | keep track of the otter count at https://riverotters.example/census every week and let me know if it drops |  |
| `binder-reports-the-only-parameter-missing-rather-than-binding-a-near-value` | In the skill-binder micro-context, when the ask describes the job and names no page, Penny reports the page missing rather than binding the thing the ask does name. | can you keep an eye on the price of that brass lantern i was looking at and tell me when it changes? every hour is fine |  |
| `binder-takes-two-different-spans-for-two-parameters` | In the skill-binder micro-context, when one message supplies both the page and the thing to look for on it, Penny binds each parameter to the span that answers it and neither to the other's. | every morning can you check the north pier timetable at https://northpier.example/departures and let me know when they add the dawn sailing? |  |
| `binder-tells-two-same-kinded-slots-apart` | In the skill-binder micro-context, when a routine declares two parameters of the same kind and one message supplies both, Penny binds each to the page the ask gives for that parameter and never to the other's. | every monday can you check https://northpier.example/departures for the outbound crossing and https://southquay.example/departures for the return, and tell me which is cheaper? |  |
| `framer-frames-from-a-single-turn` | In the skill-framer micro-context, when the whole teach is a single turn, Penny still separates what the routine is for from the piece a new occasion has to supply. | go to weather.example/lisbon, find today's high temperature, and remember it |  |
| `framer-keeps-three-of-a-kind-as-three-parameters` | In the skill-framer micro-context, when one ask points a routine at three things of the same kind, Penny mints three distinct scalar parameters rather than folding them into one list. | hey could you keep an eye on the morning headlines for me / read citydesk.example/front, harborpost.example/front and riverchronicle.example/front, and remember each site's top headline |  |
| `framer-mints-both-pieces-when-they-are-different-kinds` | In the skill-framer micro-context, when an ask names both a place to go and a thing to look for on it, Penny mints a parameter for each rather than baking the thing to look for into the framing. | can you watch the library catalog for a book i'm waiting on? / open town-library.example/catalog, find The Glass Harbour, and remember whether it's available |  |
| `framer-mints-only-the-piece-that-varies` | In the skill-framer micro-context, when a demonstrated round is turned into a reusable routine, Penny mints one parameter for the piece a new occasion has to supply and leaves what the user came for — and when they want telling — in the framing rather than in the interface. | can you track a stock for me and tell me when it moves / look up VLT, find the share price, and remember it under VLT |  |
| `framer-names-a-page-as-a-page-and-invents-no-search` | In the skill-framer micro-context, when an ask names a page and no search, Penny mints the page and nothing beside it. | can you keep an eye on bookbarn.example/atlas-of-clouds and let me know when it's back in stock / go to bookbarn.example/atlas-of-clouds, check whether it's in stock, and remember that |  |
| `framer-names-a-search-as-a-search` | In the skill-framer micro-context, when the look-up an ask describes is a text search rather than an address, Penny mints one parameter for the thing being searched for, and none for a page. | can you keep an eye on ticket prices for aurora fest? / search for aurora fest tickets, find the cheapest ticket price, and remember it |  |
| `namer-names-a-search-spot-as-a-search` | In the skill-namer micro-context, when the demonstrated look-up is a text search rather than a page, Penny gives every spot its own name for what it supplies in THIS routine, with one line saying what belongs there — never the argument's own name handed back. | search for aurora fest tickets, find the cheapest ticket price, and remember it |  |
| `namer-names-every-spot-from-a-single-turn-teach` | In the skill-namer micro-context, when one direct instruction is the whole conversation, with no elicit round before it, Penny gives every spot its own name for what it supplies in THIS routine, with one line saying what belongs there — never the argument's own name handed back. | go to weather.example/lisbon, find today's high temperature, and remember it |  |
| `namer-names-every-spot-in-a-longer-routine` | In the skill-namer micro-context, when a demonstrated routine runs four steps and offers seven spots across two reads, a write and a log, Penny gives every spot its own name for what it supplies in THIS routine, with one line saying what belongs there — never the argument's own name handed back. | check depotline.example/schedule for the next delivery window, look at what's open on the harbour refit project, save the window in depot-deliveries, and add a note to depot-checks that you looked |  |
| `namer-tells-two-sources-apart` | In the skill-namer micro-context, when a demonstrated routine reads two different pages into one argument, Penny gives every spot its own name for what it supplies in THIS routine, with one line saying what belongs there — never the argument's own name handed back, and never one name covering both sites. | read citydesk.example/front and harborpost.example/front, and remember each site's top headline |  |

## Memory

| Case | Behaviour | First wording | Edge |
|---|---|---|---|
| `chat-reply-says-already-there` | In the chat agent, when she is asked to record something the store already holds, Penny reports that it was already there, and it is still held in the collection that had it, beside everything else the store held, unchanged. | make sure you've got that i'm into sea kayaking |  |
| `chat-reply-says-nothing-is-stored` | In the chat agent, when the question is about something the user has told her and the store holds nothing of it, everything Penny says in her reply traces to what she was given, and the turn ends back in idle. | remind me what i told you about the marrow ridge loop — how much climb does it have? |  |
| `memory-a-like-and-a-dislike` | In the chat agent, when one message carries two facts of different kinds and a fitting list already exists for each, Penny files each fact in the list that fits it and neither in the other. | jot down that I'm into bouldering, and that I can't stand instant coffee |  |
| `memory-change-lands-on-the-entry-that-exists` | In the chat agent, when the user asks her to change a note she keeps under a key worded differently from how the user names it, Penny finds the entry that exists and records the change on it, and the list holds the same entries it started with. | add a 10-minute lime marinade to my fajitas recipe |  |
| `memory-cold-recall` | In the chat agent, when the user asks for a fact they had her remember in an earlier session and nothing in the conversation carries it, Penny brings it back out of the store and states it, leaving the store exactly as she found it. | hey — a while back I asked you to remember what the aurora deck 2 was listed at. what was the price? |  |
| `memory-delete-a-whole-list` | In the chat agent, when the user asks her to get rid of one of the lists she keeps, Penny archives that list with everything it held still in it, and every other list is still live and holds what it held. | get rid of my recipe box list, i don't use it any more |  |
| `memory-forget-then-list` | In the chat agent, when the user names one note to drop from a list, Penny drops that note and leaves every other one exactly as it was, and tells them what is still on the list. | remove jazz from my list of things i'm into, then tell me what else is on it |  |
| `memory-look-up-and-save` | In the chat agent, when the user asks her to look a subject up and put it in a list they name, Penny reads about it and writes it into that list, and the subject is still there when the turn ends. | can you look up Mistforge Tactics, read up on it, and save it to my games list? |  |
| `memory-no-fire-wistful` | In the chat agent, when the user muses about something a list she already keeps is about, Penny answers in conversation and the list she keeps about it is left as it was. | I finally wrapped up that long strategy game campaign last night, felt so satisfying |  |
| `memory-recall-across-the-store` | In the chat agent, when one message asks about several of her collections at once, Penny answers out of every one of them rather than the first she opens, and changes nothing while doing it. | remind me what i'm into, what i'd rather avoid, and what's on my games list |  |
| `memory-save-with-a-source-down` | In the chat agent, when the user asks her to read two pages and keep what she finds and one of the pages cannot be read, Penny keeps the readable page's fact in the named list and cites it in her reply, and everything she stores and says traces to what she was given. | go to https://www.ridgelinefoxes.com/news and https://www.harborseals.com/news, pull out the trades and signings from each, and keep the headline plus a short blurb in a team news list for me |  |
| `speak-logread-penny-messages-recall` | In the chat agent, when the user asks what she told them and the answer is out of the conversation window, Penny states what her own earlier message actually said. | dig back through our old messages — what exactly did you tell me to use for my moss terrarium? |  |

## Answering from the web

| Case | Behaviour | First wording | Edge |
|---|---|---|---|
| `chat-answer-from-page` | In the chat agent, when a question needs a current fact nothing stored can answer, Penny opens the page it is posted on and puts that page's own value in her reply, and the turn ends back in idle. | what does the lantern museum charge for an adult ticket these days? |  |
| `chat-answer-one-link-deep` | In the chat agent, when the fact a question asks for is not on the page she reaches first but that page names the address it is credited at, Penny follows the link and answers out of the second page, and the turn ends back in idle. | who made the big glass centrepiece in the lantern museum's tidemark gallery? check the gallery's own page if you need to |  |
| `chat-reply-admits-the-read-failed` | In the chat agent, when every source she tries is unreachable, everything Penny says in her reply traces to what she was given, and the turn ends back in idle. | what does the lantern museum charge for an adult ticket these days? |  |

## Standing jobs

| Case | Behaviour | First wording | Edge |
|---|---|---|---|
| `speak-logread-collector-runs` | In the chat agent, when the user asks how her background jobs are doing and why any of them is in trouble, Penny states the reason the failing cycle's run record gives, with both jobs still running as they were. | how have your background jobs been doing lately? if any of them is having trouble i want to know why |  |
| `standing-archive` | In the chat agent, when the user says they are done with a running job, Penny retires it as a tombstone that still holds everything it gathered, with proactive notifications still on everywhere else. | i'm done with the typewriter watch — you can retire that one |  |
| `standing-describe-routine` | In the chat agent, when the user asks what a standing job does, Penny describes the routine from its record, and every value in her reply traces to that record. | what does the typewriter watch actually do? walk me through it. |  |
| `standing-notify-off` | In the chat agent, when the user asks for one running job's notifications to be turned off, Penny turns that job's own switch off, with the job still running as it was and proactive notifications still on everywhere else. | turn off notifications for the typewriter watch |  |
| `standing-schedule-fix-prior` | In the chat agent, when the user says a running job checks at the wrong time and names a new one, Penny re-times that job with the rest of it as it was, and every clock time she names is one the job has actually had. | the typewriter watch is checking too early — move it to 11 in the morning |  |

## Notifications

| Case | Behaviour | First wording | Edge |
|---|---|---|---|
| `explicit-mute-request-mutes` | In the chat agent, when the user asks for notifications to be muted, Penny mutes them. | please mute notifications |  |
| `notifications-no-fire` | In the chat agent, when a message names notifications as its subject without asking for them to be changed, Penny stays in idle with notifications still on. | your notifications have been really useful this week, thanks |  |

## Background collectors

| Case | Behaviour | First wording | Edge |
|---|---|---|---|
| `honesty-writes-nothing-when-every-read-fails` | In a headline-collecting collector, when every page it is pointed at fails to read, Penny files nothing and closes the cycle having changed nothing — she writes no entry out of a source she never read. | the headlines on the page and the link to each one |  |
| `watch-stays-quiet-when-the-reading-has-not-moved` | In a price-watch collector, when the page's price has not changed since she last recorded it, Penny writes nothing and says nothing. | the current price |  |
| `watch-writes-and-tells-when-the-reading-moves` | In a price-watch collector, when the page's price has changed since she last recorded it, Penny replaces the price she was holding with the new one and tells the user exactly once. | the current price |  |
| `watch-writes-the-first-reading` | In a price-watch collector, when the job runs for the first time and its collection is still empty, Penny records the price the page shows and tells the user once — a first observation is news. | the current price |  |

## Reading a page

| Case | Behaviour | First wording | Edge |
|---|---|---|---|
| `extract-fields-all-present` | In the browse-extract micro-context, when the page carries everything the instruction named, Penny comes back with all of it. | the item's name, its price and whether it is in stock |  |
| `extract-fields-none-present` | In the browse-extract micro-context, when the page carries none of what the instruction named, Penny says plainly that it carries none of it rather than answering anyway. | the closing price of each company named and its ticker symbol |  |
| `extract-fields-partly-present` | In the browse-extract micro-context, when the page carries only some of what the instruction named, Penny comes back with the parts it has instead of reporting the page empty. | the headlines and their links, with a one-line summary of each where the page gives one |  |
| `extract-fields-partly-present-on-a-prose-page` | In the browse-extract micro-context, when a prose section front carries two of the three things the instruction named, Penny comes back with the two. | the headline, the byline and the published time for each story |  |

## Chat tools

| Case | Behaviour | First wording | Edge |
|---|---|---|---|
| `choose-dispatch-fires` | In the chat agent, when the user asks for one of several named options to be picked at random, Penny reports the option the fair pick actually returned. | choose one of cedar, maple, or birch at random for me, and tell me which one you picked. |  |
| `email-absent-message` | In the chat agent, when the user asks what an email from a named sender about a named subject comes to and the mailbox holds no message from that sender or on that subject, Penny replies with none of the figures the mailbox's messages hold and no value she was not given, and the turn ends back in idle with everything the store already held still there. | did tobias wren send me the piano tuning estimate yet? what does it come to? |  |
| `email-answers-from-the-message-asked-about` | In the chat agent, when the user asks what an email from a named sender about a named subject says, Penny answers with the figure that message carries and none of the figures the messages beside it carry, and the turn ends back in idle. | did priya nakamura send me the rooftop solar quote yet? what does it come to? |  |
| `email-remark-stays-idle` | In the chat agent, when the user remarks on their email without asking anything of it, Penny stays in idle and everything the store already held is still there, unchanged. | honestly i get way too much email these days, my inbox is out of control |  |
| `image-no-fire` | In the chat agent, when a message talks about a painting without asking for a picture, Penny stays in idle and says nothing she was not given. | i saw a really nice watercolor painting at the gallery today, it was lovely |  |
| `image-request-draws` | In the chat agent, when the user asks for a picture of something, Penny draws one whose description names what was asked for, and stays in idle. | can you draw me a teal origami dragon perched on a lighthouse? |  |

## Machine edge coverage

One row per edge of the conversation machine. A classifier draw covers an edge when a case draws that decision on its own; a whole turn covers it when a case drives the chat turn that makes the move.

| Edge | Classifier draw | Whole turn |
|---|---|---|
| idle → apply | `classifier-applies-and-names-the-routine-that-covers-the-ask`, `classifier-applies-when-chat-carries-a-covered-ask` | `transition-idle-to-apply` |
| idle → request | `classifier-draws-request-when-a-known-routine-is-a-value-short` | `transition-idle-to-request` |
| idle → learn | `classifier-draws-learn-when-teaching-arrives-unprompted` | `transition-idle-to-learn` |
| idle → elicit | `classifier-elicits-on-a-cold-registry`, `classifier-elicits-when-a-routine-shares-the-verb-but-not-the-domain` | `transition-idle-to-elicit` |
| idle → idle | `classifier-holds-idle-on-a-passing-mention`, `classifier-holds-idle-when-a-running-jobs-notifications-are-switched-off` | `transition-idle-to-idle` |
| elicit → learn | `classifier-draws-learn-when-the-teach-question-is-answered` | `transition-elicit-to-learn`, `transition-elicit-to-learn-absent` |
| elicit → elicit | `classifier-stays-parked-when-the-teach-question-is-unanswered` | — |
| elicit → idle | `classifier-falls-to-idle-when-the-elicit-round-is-called-off` | `transition-elicit-to-idle` |
| learn → apply | `classifier-draws-apply-when-the-offer-is-accepted` | `transition-learn-to-apply` |
| learn → learn | `classifier-stays-in-learn-on-a-correction` | `transition-learn-to-learn` |
| learn → idle | `classifier-falls-to-idle-when-a-post-failure-reply-carries-no-instructions` | `transition-learn-to-idle` |
| request → apply | `classifier-draws-apply-when-the-missing-value-arrives` | `transition-request-to-apply` |
| request → elicit | `classifier-elicits-when-the-named-routine-was-wrong` | — |
| request → idle | `classifier-falls-to-idle-when-a-parked-request-is-called-off` | `transition-request-to-idle` |
