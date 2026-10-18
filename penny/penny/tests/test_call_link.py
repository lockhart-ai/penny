"""CallLink is a pure value type — its constructors are unit-tested directly
(no DB, no model), the same way CODEOWNERS parsing or config loading are."""

from penny.llm.call_link import CallLink


def test_top_carries_the_given_run_id_with_no_parent_and_no_position():
    link = CallLink.top("run-1")
    assert link.run_id == "run-1"
    assert link.parent_run_id is None
    assert link.parent_tool_call_id is None
    assert link.tool_call_id is None


def test_new_top_mints_a_run_id_when_the_caller_has_none():
    a, b = CallLink.new_top(), CallLink.new_top()
    assert a.run_id and b.run_id
    assert a.run_id != b.run_id


def test_for_tool_call_sets_position_and_leaves_identity_unchanged():
    top = CallLink.top("run-1")
    dispatched = top.for_tool_call("call-1")
    assert dispatched.run_id == "run-1"
    assert dispatched.parent_run_id is None
    assert dispatched.parent_tool_call_id is None
    assert dispatched.tool_call_id == "call-1"


def test_a_draw_made_straight_from_a_dispatched_tool_call_links_to_that_call():
    """The browse-extractor shape: a draw made from INSIDE one dispatched tool
    call carries that call's id as its PARENT tool call."""
    dispatched = CallLink.top("run-1").for_tool_call("call-browse-1")
    draw = dispatched.child()
    assert draw.run_id != "run-1"
    assert draw.parent_run_id == "run-1"
    assert draw.parent_tool_call_id == "call-browse-1"
    assert draw.tool_call_id is None


def test_a_draw_made_inside_another_draw_has_no_parent_tool_call():
    """The bug the run_id/tool_call_id split rules out: a draw nested inside
    ANOTHER draw (never itself inside a tool call) must NOT inherit the outer
    draw's own parent tool call — it was made from inside a RUN (the outer
    draw), not from inside that tool call directly, so its own
    parent_tool_call_id is None even though the outer draw's was set."""
    dispatched = CallLink.top("run-1").for_tool_call("call-browse-1")
    outer_draw = dispatched.child()
    assert outer_draw.parent_tool_call_id == "call-browse-1"

    inner_draw = outer_draw.child()
    assert inner_draw.parent_run_id == outer_draw.run_id
    assert inner_draw.parent_tool_call_id is None


def test_nesting_two_draws_deep_outside_any_tool_call_stays_none_throughout():
    top = CallLink.top("run-1")
    first = top.child()
    second = first.child()
    assert first.parent_tool_call_id is None
    assert second.parent_tool_call_id is None
    assert second.parent_run_id == first.run_id
