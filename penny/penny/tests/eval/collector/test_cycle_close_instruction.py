"""A collector cycle, holding a lot of prior research, closes through the ``done`` tool
rather than narrating ``done(notify=…)`` out as text (#2806).

**Built from a real prod cycle, genericised.** A research job that had been running
hourly for a day — one topic, a growing pile of its own prior findings, no notification
sent yet — kept drawing a reply whose entire content was the literal call syntax from its
own close instruction (``Finish with done(notify=…)``) instead of actually calling the
tool. The runtime discards that draw and re-draws with a note saying what was wrong
(#2797); the retry usually recovers, but at a cost of one extra, slow draw per discard,
and the production job hit it sixteen times in a single day. Every real topic, entry and
name from that cycle is swapped here for a different real one of the same shape — a
hobbyist research job on a DIY espresso machine PID retrofit — so nothing private
survives; the SHAPE that is believed to trigger the bug (a long job instruction, and a
holdings dump of over a dozen prior findings) is kept.

**One case, one world, five wordings of the job's own intro sentence.** The five numbered
research points are the FACTS and stay constant across every arm — only the sentence that
introduces them varies, the same axis ``test_collector_honesty.py`` varies its one
instruction phrase through. The holdings, the one page the cycle can browse this run, and
the container are all the same world for every arm.

**Claims are facts about the end state, never about HOW the model got there.** This case
claims the cycle filed at least one entry and closed — a closed run is one that reached a
``done()`` call, discarded draws and retries included, so a sample that discards the
literal-text draft and recovers on the retry still counts as closed. Whether THIS cycle's
draw was itself discarded as text-instead-of-a-call is not asserted — asserting that the
model refrained from the bug would be asserting a route. It is read instead, after the
run, directly off each sample's own persisted ``promptlog.discard_reason`` (the structural
mark the runtime itself stamps, #2702) — the diagnostic evidence behind this case's PR,
kept out of the claims and the committed assertions themselves.

Report-only (``min_pass_rate=None``) — a failure-driven case, read fresh rather than against
an existing floor. Every url, product and research topic here is real and swapped
one-for-one with the prod cycle's own (same kind, different thing) — the repo is public.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.database.memory import EntryInput
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    Seeder,
    collection_entries,
    seeded_run_id,
)
from penny.tests.eval.drivers.collector_cycles import CollectorCyclesEval, CycleArm
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.fixtures import CannedPage
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "collector-close-instruction"

# The author on a seeded registry row — a fixture's own hand, never a real agent's.
_SEED_AUTHOR = "eval-seed"
_SEED_RUN = seeded_run_id("close-instruction-entry-condition")

# The one job every arm runs — one container, one topic, matching the prod cycle's shape
# (a long-running hourly research job with a sizeable pile of its own prior findings and no
# notification sent yet).
_CONTAINER = "home_espresso_pid_rd"
_CONTAINER_DESCRIPTION = (
    "Research for a DIY espresso machine PID temperature controller retrofit project."
)
_SCHEDULE = "FREQ=HOURLY"

# The one page this cycle can browse — a real-shaped technical forum thread, on a real
# domain, with invented thread content (the same pattern the suite's other reading cases use
# for a page that answers a real-world question).
_SOURCE = "https://forum.arduino.cc/t/zero-cross-ssr-switching-for-pid-boiler-control/655321"

_PAGE_TEXT = (
    f"Title: Zero-cross SSR switching for PID boiler control | Arduino Forum\n{_SOURCE}\n\n"
    "Started a thread on driving a 1500W espresso boiler heater through a solid-state "
    "relay (SSR) under PID control, since naive time-proportioning causes visible flicker "
    "and EMI on the mains line.\n\n"
    "#### Zero-cross switching\n"
    "An SSR with a zero-cross trigger only turns on at the AC waveform's zero crossing, "
    "so the heater element never sees a hard mid-cycle switch. This is the right choice "
    "for a resistive load like a boiler element — a random-turn-on SSR is cheaper but "
    "throws a much bigger EMI spike and is harder to filter.\n\n"
    "#### PID period vs. mains cycle\n"
    "A time-proportioning output needs its period to be a whole number of AC half-cycles "
    "(10ms at 50Hz, ~8.3ms at 60Hz) or the duty cycle drifts against what the SSR can "
    "actually deliver. A 1-2 second PID output period, quantised to the mains cycle, is "
    "the usual compromise between temperature ripple and switching wear.\n\n"
    "#### Safety replies in this thread\n"
    "Several people pointed out an SSR can fail SHORTED (stuck on), so the PID output "
    "alone is not a safety mechanism — a separate thermal fuse or a hardware watchdog "
    "that cuts the heater's mains feed independently of the control loop is what actually "
    "limits a runaway boiler."
)

_PAGE = CannedPage(match=_SOURCE.lower(), text=_PAGE_TEXT)

# The five numbered research points — the FACTS, constant across every arm.  Real crates,
# real hardware, real kernel feature, matching the prod cycle's own shape (one focus on
# control-loop math, one on RT timing, one on the language/library doing the loop, one on
# switching/sampling jitter, one on the add-on board).
_FOCUS_LIST = (
    "1. PID tuning methods (Ziegler-Nichols, Cohen-Coon) for stable boiler temperature.\n"
    "2. Setting up a Real-Time (RT) kernel on the Raspberry Pi for deterministic "
    "control-loop timing.\n"
    "3. Using the Rust programming language for the core control loop, focusing on "
    "libraries like `rppal`.\n"
    "4. Managing solid-state relay (SSR) switching and thermocouple sampling for minimal "
    "temperature jitter.\n"
    "5. Hardware considerations like the Sequent Microsystems HAT for industrial-grade I/O."
)

# A concrete, reachable source for point 4's own topic — read it before adding anything new,
# so every arm's cycle has one real page it can actually fetch rather than an open-ended
# search the mock browser cannot answer deterministically.
_SOURCE_POINTER = f"A relevant thread worth reading first: {_SOURCE}"

# Five wordings of ONE intro sentence — the arm axis, the same one
# ``test_collector_honesty.py`` varies through a single instruction phrase.  Same job, same
# five points, same destination; only the sentence introducing them changes.
INTROS = (
    "Research for a DIY espresso machine PID temperature controller retrofit, using a "
    "Raspberry Pi and a Sequent Microsystems industrial I/O HAT. Focus on:",
    "This job researches a DIY PID retrofit for an espresso machine's boiler, built "
    "around a Raspberry Pi and a Sequent Microsystems HAT. Focus areas:",
    "Research project: converting an espresso machine's boiler to PID control with a "
    "Raspberry Pi and a Sequent Microsystems HAT. Look into:",
    "Investigate a DIY PID temperature-control retrofit for an espresso machine, using a "
    "Raspberry Pi paired with a Sequent Microsystems HAT. Points to cover:",
    "Dig into a DIY espresso boiler PID retrofit project built on a Raspberry Pi and a "
    "Sequent Microsystems HAT, covering:",
)

# The holdings this job already has when the cycle starts — fifteen prior findings, the same
# order of magnitude as the prod cycle's own pile, so the context this case drives a cycle
# against is the same SIZE the bug was seen at, not just the same topic.  Real crates, real
# parts, real kernel features; invented specifics (benchmark numbers, part choices) the way
# every other fixture page in this suite invents its specifics.
_HOLDINGS: tuple[EntryInput, ...] = (
    EntryInput(
        key="pid_tuning_methods",
        content=(
            "### PID Tuning for a Boiler\n"
            "A home boiler is a slow, high-mass thermal system, so a P-only loop "
            "oscillates badly.\n"
            "* **Ziegler-Nichols**: drive the heater fully on/off (relay method) to find "
            "the critical gain and period, then derive Kp/Ki/Kd from the standard table — "
            "aggressive but a reasonable start.\n"
            "* **Cohen-Coon**: fits the boiler's open-loop step response (near "
            "first-order-plus-dead-time) better, usually less overshoot for this mass.\n"
            "* Nudge Kd up slightly afterwards to damp the dip right after a shot pulls "
            "cold water through the boiler."
        ),
    ),
    EntryInput(
        key="rust_rppal_gpio",
        content=(
            "### Rust GPIO/SPI on the Pi: `rppal`\n"
            "`rppal` is the standard crate for talking to the Pi's GPIO, SPI, I2C and PWM "
            "peripherals from Rust without a C wrapper.\n"
            "* Use `rppal::gpio::Gpio` for the SSR's digital drive line.\n"
            "* `rppal::spi` and `rppal::i2c` cover a thermocouple amplifier board that "
            "exposes either bus.\n"
            "* No root needed on recent Raspberry Pi OS as long as the invoking user is "
            "in the `gpio`/`i2c`/`spi` groups."
        ),
    ),
    EntryInput(
        key="ssr_zero_cross_switching",
        content=(
            "### SSR Switching: Zero-Cross vs. Random-Turn-On\n"
            "A zero-cross SSR only switches at the AC waveform's zero crossing, so the "
            "boiler element never takes a hard mid-cycle hit — far less EMI than a "
            "random-turn-on SSR.\n"
            "* Time-proportioning output period should be a whole number of AC "
            "half-cycles (10ms @ 50Hz, ~8.3ms @ 60Hz) or the duty cycle drifts against "
            "what the SSR can deliver.\n"
            "* An SSR can fail SHORTED — a thermal fuse or a watchdog-driven relay on "
            "the mains feed, independent of the PID loop, is the actual safety limit."
        ),
    ),
    EntryInput(
        key="thermocouple_filtering",
        content=(
            "### Filtering a Thermocouple Reading\n"
            "A type-K thermocouple on a boiler is noisy — mains hum and SSR switching "
            "both couple into a few millivolts of signal.\n"
            "* A simple moving average over the last N samples removes most of the "
            "switching noise at the cost of some lag.\n"
            "* A 1-D Kalman filter (constant-velocity model) tracks a slowly-ramping "
            "boiler temperature with less lag than a moving average of comparable "
            "noise rejection."
        ),
    ),
    EntryInput(
        key="pi_rt_kernel_setup",
        content=(
            "### A Real-Time Kernel for the Control Loop\n"
            "`PREEMPT_RT` gives the control-loop thread deterministic scheduling instead "
            "of best-effort.\n"
            "* The `linux-raspi-realtime` package on recent Raspberry Pi OS / Ubuntu "
            "builds ships `PREEMPT_RT` already patched in.\n"
            "* Pin the control thread to an isolated core with `isolcpus=` and raise its "
            "priority with `chrt -f`, so the UI/logging tasks on the other cores can't "
            "starve it."
        ),
    ),
    EntryInput(
        key="sequent_microsystems_hat_specs",
        content=(
            "### Sequent Microsystems Industrial I/O HAT\n"
            "A Sequent Microsystems HAT gives the Pi opto-isolated digital I/O and "
            "0-10V analog channels — the isolation matters for a line-powered SSR "
            "sharing a board with low-voltage logic.\n"
            "* Stackable — several HATs can share the Pi's 40-pin header at distinct "
            "I2C addresses.\n"
            "* Ships a Python and a C library; a Rust driver has to talk to it over "
            "raw I2C via `rppal::i2c`."
        ),
    ),
    EntryInput(
        key="pi5_thermal_management",
        content=(
            "### Cooling a Pi 5 in an Enclosed Case\n"
            "A Pi 5 running a tight control loop plus a small display runs hotter than "
            "a Pi 4, and a sealed project enclosure traps that heat.\n"
            "* The official Raspberry Pi Active Cooler (PWM blower) is the simplest fit "
            "for a closed case; a passive heatsink-only case risks throttling under "
            "sustained load.\n"
            "* PWM fan control lets the fan stay quiet at idle and spin up only under "
            "load."
        ),
    ),
    EntryInput(
        key="embedded_gui_oled_encoder",
        content=(
            "### A Small UI: OLED + Rotary Encoder\n"
            "A setpoint display and a rotary encoder for adjusting it should be a "
            "LOW-priority task, never inside the control loop's own thread.\n"
            "* `embedded-graphics` draws to common SPI/I2C OLED controllers from Rust "
            "with no heap allocation.\n"
            "* Debounce the encoder's quadrature pulses in its own interrupt handler and "
            "hand the control loop only the resulting setpoint value, via a lock-free "
            "single-producer/single-consumer queue."
        ),
    ),
    EntryInput(
        key="wireless_telemetry_ble",
        content=(
            "### Wireless Temperature Telemetry over BLE\n"
            "A phone-readable live boiler-temperature graph is a nice-to-have, built as "
            "a GATT service advertising the current reading and setpoint.\n"
            "* A shorter BLE connection interval (7.5-15ms) keeps the graph responsive; "
            "the default (~100ms on many stacks) reads as laggy for a fast-changing "
            "boiler temperature during a shot.\n"
            "* Keep this on its own low-priority task — same rule as the OLED UI."
        ),
    ),
    EntryInput(
        key="sd_card_logging_strategy",
        content=(
            "### Logging Temperature History Without Blocking the Loop\n"
            "SD card writes are high-latency and unpredictable — never call them from "
            "the control loop's own thread.\n"
            "* Write temperature samples into a RAM ring buffer from the control loop; "
            "a separate low-priority thread drains the buffer to disk in batches.\n"
            "* `ext4` handles the small, frequent metadata updates of a running log "
            "file more gracefully than `FAT32` on a Pi."
        ),
    ),
    EntryInput(
        key="power_isolation_ssr_noise",
        content=(
            "### Keeping SSR Switching Noise Off the Thermocouple Signal\n"
            "The SSR's drive line and the thermocouple amplifier's analog input should "
            "not share a return path.\n"
            "* Star-ground the amplifier's analog ground separately from the SSR "
            "drive/mains-adjacent wiring.\n"
            "* An opto-isolated drive line (as the Sequent Microsystems HAT already "
            "provides) keeps the SSR's switching transient off the Pi's own 3.3V rail "
            "entirely."
        ),
    ),
    EntryInput(
        key="rust_embedded_workflow",
        content=(
            "### Cross-Compiling and Flashing for This Project\n"
            "The control loop itself runs on the Pi's own Linux userspace (not a bare "
            "`no_std` target), so the workflow is a normal `cargo build --target "
            "aarch64-unknown-linux-gnu`, cross-compiled from a desktop.\n"
            "* `cross` wraps the right Docker toolchain for this target without a "
            "hand-rolled sysroot.\n"
            "* `probe-rs` is only relevant if a SEPARATE microcontroller (e.g. a "
            "watchdog board) is added later — nothing here needs it yet."
        ),
    ),
    EntryInput(
        key="cascade_pid_feedforward",
        content=(
            "### Cascade PID and Feedforward for the Brew Dip\n"
            "Pulling a shot draws cold water through the boiler and causes a sharp "
            "temperature dip a single PID loop reacts to only after the fact.\n"
            "* A feedforward term, triggered the instant the pump/solenoid activates, "
            "can pre-empt the dip rather than merely correcting it.\n"
            "* A cascade arrangement (an outer loop on the sensed boiler temperature, "
            "an inner loop on the SSR's duty cycle) is the more general version of the "
            "same idea."
        ),
    ),
    EntryInput(
        key="safety_watchdog_interlocks",
        content=(
            "### A Hardware Watchdog Independent of the PID Loop\n"
            "The control software failing (a hung loop, a crashed process) must not "
            "leave the heater on.\n"
            "* A hardware watchdog timer that the control loop must re-trigger every "
            "cycle, wired to cut the SSR's mains feed via a separate relay on timeout, "
            "survives a software hang the PID code itself cannot detect.\n"
            "* This interlock sits BETWEEN the SSR and the mains, never implemented "
            "purely in software on the same process that could hang."
        ),
    ),
    EntryInput(
        key="ads1115_thermocouple_adc",
        content=(
            "### Reading the Thermocouple: MAX31855 + ADS1115\n"
            "A type-K thermocouple needs cold-junction compensation, which the "
            "MAX31855 amplifier chip does on-board and exposes over SPI as a ready "
            "temperature reading.\n"
            "* Where a plain analog amplifier is used instead, the ADS1115 (I2C, "
            "16-bit) is a common choice for reading its output on the Pi.\n"
            "* Either path keeps the ADC/amplifier physically close to the thermocouple "
            "to minimise the run of low-level analog wiring exposed to SSR noise."
        ),
    ),
)


def _prompt(intro: str) -> str:
    """The job's free-text instructions (#2455 naive baseline) — what a job's own
    ``extraction_prompt`` carries verbatim, with no skill/program rendering seam.

    ONE shape for every arm: the five numbered points are the facts and never move; the
    intro sentence is the whole arm axis, the same way ``test_collector_honesty.py``
    varies its one instruction phrase."""
    return f"{intro}\n{_FOCUS_LIST}\n{_SOURCE_POINTER}"


def _world() -> World:
    """The ground every arm of this case runs against: the one page its single cycle can
    browse.

    ONE world for all five arms — what varies here is the intro sentence, and the page
    that answers the job's own research points is the same one every time."""
    return World(name="espresso PID retrofit research", pages=(_PAGE,), keeps=(), excludes=())


def _seeder(intro: str) -> Seeder:
    """The world this case's cycle starts in: a job already configured with its prompt and
    its FIFTEEN prior findings — then every claim that world makes, asserted out loud."""

    def seed(db: Database) -> None:
        db.memories.create_collection(
            _CONTAINER,
            _CONTAINER_DESCRIPTION,
            extraction_prompt=_prompt(intro),
            schedule=_SCHEDULE,
            notify=False,
        )
        memory = db.memory(_CONTAINER)
        assert memory is not None, f"the job's container {_CONTAINER!r} must exist to seed"
        memory.write_rule_decided(list(_HOLDINGS), author=_SEED_AUTHOR, run_id=_SEED_RUN)
        _assert_the_world(db, intro)

    return seed


def _assert_the_world(db: Database, intro: str) -> None:
    """Everything the seeder is responsible for, asserted where it fails loudly."""
    row = db.memories.get(_CONTAINER)
    assert row is not None, f"the job's container {_CONTAINER!r} must exist"
    assert not row.archived, "the job must be live when the cycle runs"
    assert not row.notify, (
        "this case claims what the cycle WROTE and whether it closed — a notifying job "
        "would also enter the notify micro-context on a correct cycle as readily as on "
        "one that mishandled the close, which is not what this case measures"
    )
    prompt = row.extraction_prompt
    assert prompt is not None, (
        f"the job's container {_CONTAINER!r} must carry a prompt — a container with none "
        "leaves its cycles nothing to run"
    )
    assert intro in prompt, (
        f"this arm's intro must reach the model as {intro!r} — the intro sentence is the "
        f"whole arm axis.  Prompt: {prompt!r}"
    )
    held = collection_entries(db, _CONTAINER)
    assert len(held) == len(_HOLDINGS), (
        f"the container must hold exactly the {len(_HOLDINGS)} seeded findings when the "
        f"cycle starts — it holds {len(held)}"
    )


def _arm(intro: str) -> CycleArm:
    """One arm of the one case: this case's one page, under this arm's intro wording."""
    return CycleArm(text=intro, seed=_seeder(intro), pages=[_PAGE], world=_world())


def _arms() -> list[CycleArm]:
    """This case's five arms — five wordings of the job's intro sentence, over one world."""
    return [_arm(intro) for intro in INTROS]


_CASE_ID = "cycle-closes-through-done-not-narrated-as-text"

# The one sentence this case exists to check, in the fixed form: "In <the locus>, when <X>,
# Penny <does Y>."  The case id is a filename; this is the contract.
_BEHAVIOUR = (
    "In a background research collector carrying a sizeable pile of its own prior "
    "findings, when a cycle has done what it means to do this run, Penny closes it "
    "through the done tool rather than narrating done(notify=…) out as text."
)


def _wrote_at_least_one_entry(sample: SampleObservation, _world: World) -> Answer:
    """The cycle filed at least one fresh finding — in this collection or, for the same
    reason ``test_collector_honesty.py``'s claim reads every write, in any other."""
    filed = [f"{entry.collection}/{entry.key}" for entry in sample.entries]
    return bool(filed), f"filed {len(filed)} entry(ies): {filed}"


def _cycle_closed(sample: SampleObservation, _world: World) -> Answer:
    """The cycle reached a real ``done()`` close — a run record stamped ``run_outcome`` —
    whatever it took to get there: a discarded text-written draft and a recovered retry are
    exactly as closed as a cycle that called the tool on its first draw.  HOW it got there is
    read afterwards, off each sample's own promptlog, never claimed here."""
    return sample.run_outcome is not None, f"run_outcome={sample.run_outcome!r}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_cycle_closes_through_the_tool_not_as_narrated_text(
    collector_cycles_eval: CollectorCyclesEval, model: str
) -> None:
    """A research job with fifteen prior findings runs one more cycle — it should file what
    it found this run and close through the done tool, the same as a cycle with no prior
    findings at all."""
    cohort = await collector_cycles_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        area=Area.BACKGROUND_COLLECTORS,
        model=model,
        collection=_CONTAINER,
        arms=_arms(),
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only — a failure-driven case, read fresh (#2806)
        family=_FAMILY,
    )
    # STORE
    cohort.claim(
        "state: the cycle filed at least one entry", _wrote_at_least_one_entry, SpecCategory.STORE
    )
    cohort.claim("state: the cycle closed", _cycle_closed, SpecCategory.STORE)

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()

    # REPLY_SPREAD is not measured: the job does not notify, so every sample's reply is
    # empty and a spread over no pair prints a number where there is no measurement.
    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED)
