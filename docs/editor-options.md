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

This is a Rust API option; no C configuration key is added.
