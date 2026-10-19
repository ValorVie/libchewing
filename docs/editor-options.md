<!-- SPDX-License-Identifier: LGPL-2.1-or-later -->
# Editor input habits

These options are available through the Rust `EditorOptions` API. Callers use
`Editor::set_editor_options` to change them at runtime.

## Cursor movement

`cursor_wrap: bool` defaults to **true**, preserving upstream behavior.
Left at the beginning moves to the end; Right at the end moves to the beginning.
When false, left/right movement stops at the respective boundary. This also
applies to automatic cursor movement after candidate selection, with either
value of `phrase_choice_rearward`.

```rust,ignore
editor.set_editor_options(|options| options.cursor_wrap = false);
```

## Ctrl punctuation

`ctrl_symbol_new_phonetic: bool` defaults to **false**, preserving upstream
behavior. When enabled, Ctrl with no Shift, Alt, or Super uses these six mappings:

- `,` → `，`
- `.` → `。`
- `;` → `；`
- `'` → `、`
- `[` → `【`
- `]` → `】`

With a nonempty composition and no unfinished syllable, the symbol is inserted
at the cursor through the existing punctuation path. It stays in preedit and
uses the mapped fullwidth symbol regardless of `character_form`.

With an unfinished syllable, the editor returns `EditorKeyBehavior::Bell`:
the key is consumed, the syllable and composition stay unchanged, and nothing
is inserted. With both buffers empty, it returns `EditorKeyBehavior::Ignore`
so the caller can pass the shortcut to the application. In candidate selection,
Ctrl retains its existing Bell behavior. Other Ctrl combinations, including
Ctrl+/, Ctrl+digits, and Ctrl+Shift, keep their existing behavior.

Events must carry the original punctuation keysym and the Control state,
not an ASCII control character. CapsLock and NumLock do not count as extra
chord modifiers.

```rust,ignore
editor.set_editor_options(|options| options.ctrl_symbol_new_phonetic = true);
```

Both fields are Rust API options; no C configuration keys are added.


## Personal candidate preference

`personal_candidate_preference: bool` defaults to **false**. See
[personal candidate preference](personal-candidate-preference.md) for confirmed
commit learning, single-character window protection, storage, and D512 bonuses.
