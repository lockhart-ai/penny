"""User-facing response strings for Penny.

All textual responses that Penny sends to users are defined here.
Parameterized strings use .format() style templates.
"""


class PennyResponse:
    """All user-facing response strings, organized by feature area."""

    # ── Agent Errors ─────────────────────────────────────────────────────────

    # The one fixed message a chat turn can still answer with (#2370, docs/principles.md
    # §2 item 9): the model runner itself could not be reached for this turn's last
    # draw.  Every other terminal shape — the step limit, a spent reroll budget, an
    # empty answer, nothing learned, nothing to configure — hands the model that fact
    # and lets it write the reply (see FACTS_* below and Agent._narrate_or_account).
    AGENT_MODEL_ERROR = "Sorry, I had trouble reaching the AI model. Please try again."

    # ── Narrated-or-accounted terminal facts (#2370) ────────────────────────────
    #
    # Plain statements of a run's own state — never an order, and in SECOND PERSON,
    # since the model reading these is Penny — appended as the last user turn before
    # Agent._narrate_or_account's one tools-off draw, so the model can tell the user
    # honestly what happened.  Reused verbatim as the deterministic account's own
    # opening line (via `account_of`) when even that draw is unusable; FACTS_NO_REPLY_AT_SEND
    # is the one exception, read only by the account (never handed to a draw), so it
    # stays third person.
    FACTS_REROLLS_EXHAUSTED = "Your last {attempts} answers this turn came back unusable."
    FACTS_BUDGET_EXCEEDED = "Your last answer ran past its thinking budget before it could finish."
    FACTS_MAX_STEPS = (
        "The step limit of {steps} tool steps was reached before an answer was written."
    )
    FACTS_EMPTY_WITH_TOOLS = "Your last answer came back empty, after {count} tool calls."
    FACTS_EMPTY_NO_TOOLS = "Your last answer came back empty."
    FACTS_NO_REPLY_AT_SEND = "This turn didn't produce a reply to send."

    # The channel's own backstop facts (#2371): built when ``handle_message`` catches an
    # exception that escaped everything inside it — a bug, not a model that couldn't be
    # reached (that's AGENT_MODEL_ERROR above, decided inside the agent). Third person,
    # like FACTS_NO_REPLY_AT_SEND, for the same reason: read only by ``account_of``,
    # never handed to a model draw.
    FACTS_TURN_FAILED_BEFORE_REPLY = "This turn hit an unexpected error before a reply was written."
    FACTS_TURN_FAILED_AFTER_REPLY = (
        "This turn hit an unexpected error after a reply was written, so it may not have "
        "reached you."
    )

    # The deterministic account's own shape (`account_of` in agents/models.py): the
    # facts above, plus a plain list of what was tried, built from the record.
    ACCOUNT_NO_TOOL_CALLS = "I made no tool calls this turn."
    ACCOUNT_TRIED_PREFIX = "What I tried: "
    ACCOUNT_FAILED_SUFFIX = " (failed)"

    # ── Validation ───────────────────────────────────────────────────────────

    # Appended, as its own plain line, to a reply where the hallucinated-URL
    # validator (#2721) dropped a link still unsourced after its one retry — a
    # dangling handle citation or a raw URL absent from the source material.
    # The link never reaches the user silently (visible degradation over silent
    # success, docs/principles.md): this names the gap in Penny's own voice,
    # since it rides inside her finished message rather than a fact handed to
    # a draw.
    LINK_DROPPED_NOTICE = (
        "I left out a link there — I couldn't trace it back to a page I actually read."
    )

    # ── Channel ──────────────────────────────────────────────────────────────

    DELIVERY_FAILURE = "Sorry, I had trouble delivering that message. Please try again."
    THREADING_NOT_SUPPORTED_COMMANDS = "Commands can't be used in threads."
    UNKNOWN_COMMAND = "Unknown command: /{command_name}. Use /commands to see available commands."
    COMMAND_ERROR = "Failed to run command: {error}"

    # ── Config ───────────────────────────────────────────────────────────────

    CONFIG_HEADER = "**Runtime Configuration**"
    CONFIG_GROUP_HEADER = "**{group}**"
    CONFIG_FOOTER = "Use `/config <key> <value>` to change a setting."
    CONFIG_UNKNOWN_PARAM = (
        "Unknown config parameter: {key}\nUse /config to see all available parameters."
    )
    CONFIG_PARAM_DISPLAY = "• **{key}**: {value} ({description})"
    CONFIG_INVALID_VALUE = "Invalid value for {key}: {error}"
    CONFIG_UPDATED = "Ok, updated {key} to {value}"

    # ── Profile ──────────────────────────────────────────────────────────────

    PROFILE_NO_PROFILE = (
        "You don't have a profile yet! Set it up with:\n"
        "`/profile <name> <location> <date of birth>`\n\n"
        "For example: `/profile sam denver march 5 1990` \U0001f4dd"
    )
    PROFILE_REQUIRED = (
        "Hey! I need to collect some basic info about you before we can chat. "
        "Please run `/profile <name> <location> <date of birth>` "
        "to set up your profile.\n\n"
        "For example: `/profile sam denver march 5 1990` \U0001f4dd"
    )
    PROFILE_HEADER = "**Your Profile**"
    PROFILE_NAME = "**Name**: {name}"
    PROFILE_LOCATION = "**Location**: {location}"
    PROFILE_TIMEZONE = "**Timezone**: {timezone}"
    PROFILE_DOB = "**Date of Birth**: {dob}"

    PROFILE_CREATE_PARSE_ERROR = (
        "I couldn't understand that. Please provide your name, location, "
        "and date of birth.\n\n"
        "Example: `/profile sam denver march 5 1990`"
    )
    PROFILE_DATE_PARSE_ERROR = (
        "I couldn't parse '{date}' as a date. Try something like 'january 10 1995' \U0001f4c5"
    )
    PROFILE_TIMEZONE_ERROR = (
        "I couldn't find a timezone for '{location}'. Can you be more specific? \U0001f5fa\ufe0f"
    )
    PROFILE_CREATED = "Got it! Your profile is set up. Welcome, {name}! \U0001f389"

    PROFILE_UPDATE_PARSE_ERROR = (
        "I couldn't understand that. Please provide name and/or location.\n\n"
        "Example: `/profile sam denver`"
    )
    PROFILE_UPDATE_NAME = "name to **{name}**"
    PROFILE_UPDATE_LOCATION = "location to **{location}** ({timezone})"
    PROFILE_UPDATED = "Ok, I updated your {changes}! \u2705"
    PROFILE_UNCHANGED = "Your profile is unchanged \U0001f937"

    # ── Onboarding ───────────────────────────────────────────────────────────

    ONBOARDING_INTERESTS_PROMPT = (
        "Now tell me some things you're interested in so I can start "
        "looking for interesting stuff for you!"
    )

    # ── Commands Index ───────────────────────────────────────────────────────

    COMMANDS_HEADER = "**Available Commands**"
    COMMANDS_UNKNOWN = "Unknown command: /{name}. Use /commands to see available commands."
    COMMANDS_HELP_HEADER = "**Command: /{name}**"

    # ── Search ───────────────────────────────────────────────────────────────

    NO_RESULTS_TEXT = "No results found"
