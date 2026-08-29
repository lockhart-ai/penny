"""Shared eval machinery — the parts every case reuses, in one place.

Nothing here is a behavioural contract.  These are the run's moving parts: the
artifact store and its checkpoint, the cohort/report/assemble pipeline that turns
samples into a readable run, the roster and endpoint preflights, the worlds and
fixtures a case declares its ground with, and the seeders that write a preceding
history the way production wrote it.  Their own unit tests live beside them.
"""
