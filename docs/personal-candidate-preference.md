<!-- SPDX-License-Identifier: LGPL-2.1-or-later -->
# Personal candidate preference

`EditorOptions::personal_candidate_preference` defaults to **false**. Enable it
through `Editor::set_editor_options`. The standard Chewing engine then uses
confirmed manual choices for reversible preedit preferences, candidate-window
ranking, and D512 personal bonuses. Simple, fuzzy, and external converters do
not use or learn this history. The caller owns the Windows setting and its
default; no C configuration key is added here.

## Confirm client insertion

Selecting a candidate does not learn it. The selected reading tokens and output
must survive editing and reach a commit. The caller must then confirm that the
exact commit text reached the client. Automatic composition, automatic pins,
preview, cancellation, changed output, missing clients, and rejected insertion
do not learn. Keyboard Tab at the end of preedit uses the existing sentence
revolution; only changed words in its final selected path are manual choices.

For a synchronous insertion, call
`editor.confirm_personal_candidate_commit(inserted_unchanged)` before `ack()` or
another key. Duplicate confirmation has no effect. For deferred edit sessions,
detach the receipt before clearing output or processing another key:

```rust,ignore
let text = editor.display_commit().to_owned();
let receipt = editor.take_personal_candidate_commit();
editor.ack();
// Keep text and receipt until the client insertion has completed.
// On the editor's thread, confirm actual insertion of this exact text:
if let Some(receipt) = receipt {
    receipt.confirm(inserted_unchanged);
}
```

`PersonalCandidateCommit` is owned and single-use. Dropping it rejects the
insertion. Normal editor clear/commit does not revoke an extracted full receipt;
switching personal preference off revokes the old learning epoch. A receipt
for a partial overflow choice learns only after all fragments are confirmed
and its remaining reading tokens survive. A failed fragment cancels the whole
choice. Recency follows manual selection order even when confirmations arrive
in reverse order within an editor. Clearing an uncommitted tail cancels it.
The current automatic overflow path emits complete intervals, so it preserves
a selected word whose suffix has no standalone dictionary entry.

## Candidate order and reversible preedit

Within each complete reading group, latest comes first, then descending counts;
ties retain the incoming order. Groups retain their slots and absent candidates
are never inserted into a window. Frequency sorting, when enabled, runs first.
Personal window sorting applies **only to multi-syllable words**. Single-character
slots retain their incoming order, matching the Mac setting that protects ETen
DOS single-character order. This does not install an ETen sequence dataset.
There is no dependency on `sort_candidates_by_frequency`, no added fixed-order
switch, and no added ETen switch. An already open window keeps its displayed
numeric-key snapshot across history updates and switch changes.

Preedit preferences apply to words and single characters. Longer learned words
win overlapping positions; ties prefer the earlier position. A single character
cannot split an existing multi-character word. Explicit live selections and
surviving manual deletion fragments are protected. Only valid dictionary entries
for the exact complete reading are eligible. Preferences constrain a temporary
composition copy, retain the original language-model, history, and user-dictionary
costs, and never become manual learning events. This fork disables rare-reading
penalties while keeping those readings available as candidates. Turning the switch
off withdraws them without committing or deleting text, history, or manual choices.

After choosing and confirming `重` for `ㄔㄨㄥˊ`, the next preedit is `重`.
The candidate window retains its original single-character order, including
`蟲` first when that is the original order.

## Storage

`candidate-ranking.json` resides beside `user_dict.csv` and `history_dict.bin`
under `SearchPath::user_file_path` (currently the user directory's `v4` folder).
`Editor::new` uses its supplied versioned `user_datadir`; an editor without one
keeps an in-memory store. The schema matches Mac `Snapshot` version 1:

```json
{"version":1,"groups":[{"keyArray":["ㄕˋ","ㄕˋ"],"latest":"世事","counts":{"世事":1}}]}
```

First tone has no mark. Counts do not decay and saturate at `u64::MAX`. Unknown
versions, malformed groups, duplicate reading groups, invalid counts, and read
failures make the entire file read-only for that store's lifetime. In-memory
learning continues; saving never replaces the unreadable file. Logs and Debug
output for the new store/receipts do not contain readings, words, or file data.

The store retains only its path until personal preference is first enabled.
An editor that never enables it neither reads nor writes preference files.
First enable starts one background worker for that store. Initial loading is
asynchronous; choices confirmed during loading remain visible and are merged
with the loaded counts. Turning the switch off prevents new learning and keeps
already confirmed pending events eligible for persistence.

Confirmation updates memory and queues persistence. Refresh requests also go to
the worker. Receipts stay on the editor's thread. The worker takes owned confirmed
increments from the pending queue and receives commands through the channel.
It coalesces requests and owns disk state. Reads,
JSON parsing and serialization, locking, writes, sync and rename all happen
outside the UI state mutex. The mutex protects only in-memory deltas and the
snapshot pointer; candidate sorting and whole-snapshot comparison run outside
it. A save publishes its snapshot and retires only that batch's increments, so
choices confirmed during an in-flight write remain visible exactly once.

`Editor::flush()` is a lifecycle barrier that waits for queued preference work;
Editor drop also waits. Neither the key path nor receipt confirmation calls
this barrier. The last store owner requests final persistence and joins the
worker. Flush and shutdown may wait for another writer's file lock. Normal
background saves use a stable `candidate-ranking.json.lock` sidecar with
nonblocking `File::try_lock`; contention retains events for a later save or
flush. Writers reread and merge under the file lock, write a unique
same-directory temporary file, sync it, and rename it over the snapshot.
I/O failure retains the old file and pending increments; an unreadable snapshot
remains read-only for the store's lifetime. New files use mode 0600 on Unix.
No changes means no file write. New composition requests refresh; visibility
of another writer's changes is asynchronous. Across independent processes,
latest follows serialized save order; selection-order sequence numbers are
local to an editor.

The caller never waits for preference I/O during keys or confirmation. This
is not a hard scheduling-latency or power-loss durability guarantee: lifecycle
flush can wait on storage and the containing directory is not separately synced.
See the Rust contracts for
[channels](https://doc.rust-lang.org/std/sync/mpsc/index.html),
[locking](https://doc.rust-lang.org/std/fs/struct.File.html#method.try_lock),
[rename](https://doc.rust-lang.org/std/fs/fn.rename.html), and SQLite's discussion
of [atomic commit and filesystem assumptions](https://www.sqlite.org/atomiccommit.html).

## D512 and caller integration

`D512Options::personal_bonus` and `suggestion_personal_bonus` are independent,
finite nonnegative numbers, both defaulting to **2.0**. Automatic uses the first;
Suggestion uses the second. Each exact latest-word interval contributes one
`personal_hit`; a single-character interval overlapping an original native
multi-character word contributes none. Matching text at a different reading
or inside another interval is not a hit.

The temporary personal-preference path is the default presented to D512, even
when it lies outside the original ten native paths. Fusion uses
`raw + weight * model_logp + personal_hits * bonus`. D512 may replace that default
only after the configured strict margin. Personal preferences are prepared
before reranking and never reapplied over its winner. Suggestion preserves the
personal default until acceptance. A scoring failure retains that default.
Accepted suggestion words are manual choices and still require client insertion
confirmation before learning.

`d512_calibration()` supplies each candidate's `personal_hits` and the active
record's `bonus`; the caller maps them to `personalHits` and `bonus` in its local
calibration JSON. When the editor switch is off, both effective values are zero
and saved records remain intact. Reading calibration does not write personal
history. Direct `ChewingEngine` use without an owning Editor has no personal
store. The caller continues to own calibration IDs, phases, final commit text,
conflict reporting, and opt-in file I/O; see [D512 integration](d512.md).
