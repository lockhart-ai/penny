"""User-facing response strings for Penny.

All textual responses that Penny sends to users are defined here.
Parameterized strings use .format() style templates.
"""


class PennyResponse:
    """All user-facing response strings, organized by feature area."""

    # ── General ──────────────────────────────────────────────────────────────

    FALLBACK_RESPONSE = "Sorry, I couldn't generate a response."
    RESTART_FALLBACK = "I just restarted!"

    # ── Agent Errors ─────────────────────────────────────────────────────────

    AGENT_MODEL_ERROR = "Sorry, I had trouble reaching the AI model. Please try again."
    AGENT_EMPTY_RESPONSE = "Sorry, the model generated an empty response."
    AGENT_MAX_STEPS = "Sorry, I couldn't complete that request within the allowed steps."
    AGENT_TOOLS_UNAVAILABLE = "Sorry, I wasn't able to get results right now ({tools})."

    # ── Channel ──────────────────────────────────────────────────────────────

    DELIVERY_FAILURE = "Sorry, I had trouble delivering that message. Please try again."
    THREADING_NOT_SUPPORTED_COMMANDS = "Commands can't be used in threads."
    UNKNOWN_COMMAND = "Unknown command: /{command_name}. Use /commands to see available commands."
    COMMAND_ERROR = "Failed to run command: {error}"

    # ── Vision ───────────────────────────────────────────────────────────────

    VISION_NOT_CONFIGURED_MESSAGE = (
        "I can see you sent an image but I don't have vision configured right now."
    )
    VISION_IMAGE_CONTEXT = "User said '{user_text}' and included an image of: {caption}"
    VISION_IMAGE_ONLY_CONTEXT = "User sent an image of: {caption}"

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

    # ── Schedule ─────────────────────────────────────────────────────────────

    SCHEDULE_NO_TASKS = "You don't have any scheduled tasks yet \U0001f4c5"
    SCHEDULE_NEED_TIMEZONE = (
        "I need to know your timezone first. Send me your location or tell me your city \U0001f4cd"
    )
    SCHEDULE_PARSE_ERROR = (
        "Sorry, I couldn't understand that schedule format. "
        "Try something like: /schedule daily 9am what's the news?"
    )
    SCHEDULE_INVALID_CRON = (
        "Sorry, I couldn't figure out the timing. "
        "Try something like: /schedule daily 9am what's the news?"
    )
    SCHEDULE_DELETED_NO_REMAINING = "No more scheduled tasks."
    SCHEDULE_STILL_SCHEDULED = "**Still scheduled:**"
    SCHEDULE_INVALID_NUMBER = "Invalid schedule number: {number}"
    SCHEDULE_NO_SCHEDULE_WITH_NUMBER = "No schedule with number {number}"
    SCHEDULE_DELETED_PREFIX = "Deleted '{timing} {prompt}' \u2705"
    SCHEDULE_ADDED = "Added {timing}: {prompt} \u2705"

    # ── Email ────────────────────────────────────────────────────────────────

    EMAIL_NO_QUERY_TEXT = "Please ask a question about your email. Usage: /email <question>"
    EMAIL_ERROR = "Failed to search email: {error}"

    # ── Zoho ─────────────────────────────────────────────────────────────────

    ZOHO_NO_QUERY_TEXT = "Please ask a question about your Zoho email. Usage: /zoho <question>"
    ZOHO_ERROR = "Failed to search Zoho email: {error}"

    # ── Draw ─────────────────────────────────────────────────────────────────

    DRAW_USAGE = "Please describe what you want to draw. Usage: /draw <prompt>"
    DRAW_ERROR = "Failed to generate image: {error}"

    # ── Bug ──────────────────────────────────────────────────────────────────

    BUG_USAGE = "Please provide a bug description. Usage: /bug <description>"
    BUG_FILED = "Bug filed! {issue_url}"
    BUG_ERROR = "Failed to create issue: {error}"

    # ── Feature ──────────────────────────────────────────────────────────────

    FEATURE_USAGE = "Please provide a feature description. Usage: /feature <description>"
    FEATURE_FILED = "Feature request filed! {issue_url}"
    FEATURE_ERROR = "Failed to create issue: {error}"

    # ── Commands Index ───────────────────────────────────────────────────────

    COMMANDS_HEADER = "**Available Commands**"
    COMMANDS_UNKNOWN = "Unknown command: /{name}. Use /commands to see available commands."
    COMMANDS_HELP_HEADER = "**Command: /{name}**"

    # ── Mute ──────────────────────────────────────────────────────────────────

    MUTE_ENABLED = "Notifications muted. Use /unmute when you want them back."
    MUTE_ALREADY = "Notifications are already muted."
    UNMUTE_ENABLED = "Notifications unmuted."
    UNMUTE_ALREADY = "Notifications aren't muted."

    # ── Preferences ───────────────────────────────────────────────────────

    PREF_NO_LIKES = "You don't have any likes yet."
    PREF_NO_DISLIKES = "You don't have any dislikes yet."
    PREF_LIKES_HEADER = "**Your Likes**"
    PREF_DISLIKES_HEADER = "**Your Dislikes**"
    PREF_INVALID_NUMBER = "Invalid preference number: {number}"
    PREF_NO_PREF_WITH_NUMBER = "No preference with number {number}"
    PREF_ADDED = "Added '{content}' to your {valence}s."
    PREF_DELETED = "Removed '{content}' from your {valence}s."
    PREF_STILL_REMAINING = "**Remaining:**"
    PREF_DELETED_NO_REMAINING = "No more {valence}s."

    # ── Search ───────────────────────────────────────────────────────────────

    NO_RESULTS_TEXT = "No results found"
