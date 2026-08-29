"""The chat agent's contracts, grouped by the state the turn LANDS in.

The chat system prompt is composed as head + ``STATE_INSTRUCTIONS[state]`` + tail
(``penny.conversation_machine.conversation_prompt``), where ``state`` is the turn's
landed machine state.  So the state a turn lands in *is* the microcontext under
test, and it is what these subdirectories are keyed to — never the state the turn
started from, which belongs to the classifier's own draw.
"""
