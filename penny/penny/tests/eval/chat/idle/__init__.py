"""Chat in IDLE — ordinary conversation, and everything handled inside the turn.

Idle is the state that owns answers, recall, forgetting and changing what is already
stored, the tool dispatches that fire from natural language, and the recoveries from
a bad draw.  It is also where a round ENDS: a task dropped, abandoned or answered
leaves the machine here, and those turns run the idle instruction like any other.
"""
