# Audapolis File Format Version 3

An audapolis document of file version 3 (named „audapolis document“ in the following text) consists of three things:
- Content
- Metadata
- Sources

## Content

The *content* of an audapolis document consists of a list of `DocumentItem`s.
The *content* list MAY NOT be empty.

Semantically the content is a number of sections.
At the moment, they can only be a "paragraph", meaning a list of `ParagraphItem`s and a speaker.
Every section MUST end with a `DocumentItem` of type `paragraph_end`.

### `DocumentItem`s

There are 5 types of document items:
- `paragraph_start`
- `paragraph_end`
- `text`
- `non_text`
- `artificial_silence`

Every document item MUST have an uuid in the `uuid` property in addition the properties described below.

#### `paragraph_start`

A `paragraph_start` marks the beginning of a paragraph.
It also sets the current speaker and language.
The first item of every paragraph MUST be a `paragraph_start`.
It MAY NOT occur anywhere else in the content.

It has two additional properties:
- speaker
- language

The speaker MUST be a string.
It MAY be the empty string when the speaker is unset or overlapping speech cannot be attributed to one speaker. The original speaker set is preserved in transcription provenance; display labels MUST NOT become invented speaker identities.
It MAY NOT consist of only white space (as defined in [ECMAScript Section `TrimString`](https://tc39.es/ecma262/#sec-trimstring)).
It MAY otherwise contain the entire range of unicode characters.

The `language` MUST be a string or null.
It MUST be null or an ietf language code.
It MAY NOT be another value.

#### `paragraph_end`

A `paragraph_end` marks the end of a paragraph and thus the end of a section.
It MAY ONLY occur at the end of a paragraph.
It has no additional properties.


#### `text`

A `text` document item describes a bit of media and its associated text
(mostly a representation of what was spoken within this media snippet).

A `text` has the following additional properties:
1. `source`
2. `sourceStart`
3. `length`
4. `text`
5. `conf`

`source` contains the id of the media file the content is from.
A media document with this id MUST be `sources` of this document.

`sourceStart` marks the start of this snippet within the source in seconds.
It MUST be zero or greater.

`length` contains the length of the snippet in the source file in seconds.
It MUST be greater than zero.

`sourceStart` + `length` MAY NOT be larger than the length of the source file.

`text` contains text associated with the snippet.
It MUST be a string.
It MAY NOT be the empty string.
It MAY NOT consist of only white space (as defined in [ECMAScript Section `TrimString`](https://tc39.es/ecma262/#sec-trimstring)).
It MAY contain the entire range of unicode characters.
Applications MAY choose to ignore or not render certain white space characters, e.g. newlines.

`conf` contains the confidence that the text matches the snippet.
It MUST be null when confidence is unavailable, or a value in the inclusive range
of 0 and 1, where 0 marks the lowest possible confidence and 1 marks the highest
possible confidence. A provider log probability MUST NOT be treated as a
calibrated confidence score. Existing numeric values remain supported.

#### `non_text`

A `non_text` document item describes a bit of media without associated text, e.g. silence, music or noise.

It has the same additional properties as `text` except for `text` and `conf`.

#### `artificial_silence`

An `artificial_silence` document item describes a snippet that has no source media.
It has neither associated audio nor video.
During playback, audapolis plays silence and shows the "No Video"-Icon.
During export, audapolis exports silence and a black video.

It has the following additional properties:
- `length`

`length` contains the length of the snippet in seconds.
It MUST be greater than zero.

### Conclusion

The document items describe above allow us to divide the `content` into sections.
A section is a `paragraph_start` followed by zero or more `text`, `non_text` or `aritifical_silence` item, finished by a `paragraph_end`.

The first item of a document is always a `paragraph_start`.

## Metadata

In addition to the content, some metadata can be stored with the document:
- `display_video`
- `display_speaker_names`

`display_video` MUST be a boolean value and describes whether the videos SHOULD be shown when opening this document.

`display_speaker_names` MUST be a boolean values and describes whether the speaker names SHOULD be shown when opening this document.

## Sources

A source is a media file in an format that is supported by the version of electron/chrome used by audapolis.
It SHOULD be in one of the "safe formats".

Safe formats are formats that are known to provide accurate seeking and playback in electron.
At the time of writing those are:
- mp4
- wav

## On-Disk-Format

An audapolis document can be written to disk in a zip format.
The zip file SHOULD be saved with the extension ".audapolis".
It MUST contain the content and metadata as a file named `document.json` containing the JSON-representation of an object
with the following keys:
- `content` - a list of document items as described in section "Content"
- `metadata` - an object with the keys and values as described in section "Metadata"
- `version` - the number 3

If the document contains sources, a folder named `sources` MUST be present in the zip file.
In this folder, each source media file MUST be stored under its id without a file extension.

## Appendix A: Macro Items

The content can also be represented as `MacroItem`s.

This format is never stored on disk, but can be generated on the fly from the Content of a document.
Helper functions for this can be found in the audapolis source code.

At the moment, these can only be a `paragraph`.

Every macro item has a uuid in addition to the properties described below.

#### `paragraph`

A macro item of type paragraph has two additional properties:
1. `speaker`
2. `content`

The speaker MUST be a string.
The speaker MAY be the empty string.
The empty string represents that no speaker is set for this paragraph.

The content is a list of `ParagraphItem`s.
`ParagraphItem`s are `DocumentItem` of types `text`, `non_text`, and `artificial_silence`.
The content list MAY be empty, representing a paragraph with no content.

### Optional transcription provenance

This fork adds optional `metadata.transcriptions`, an array of immutable import
records. Each record has `id`, `version: 1`, `provider`, `model`, `language_code`,
`duration`, request options, `raw_response`, the source hash, `regions`, and optional
`annotations`. Credentials are never stored here.

A region stores `sourceStart`, `length`, original provider `word_indices`,
`boundary_status` (`unverified`, `overlap`, or `needs_alignment`), and optional
`speaker_ids`. These describe the original source, not current item UUIDs.
Any edited item intersecting an uncertain source region inherits its warning.
Copy/paste merges import records by ID; correction and undo retain the original
source regions. Repeated paste does not duplicate the full provider response.

An annotation stores `type` (`audio_event` or `unaligned_word`), `text`, `start`,
`end`, `word_indices`, and nullable `speaker_id`. Annotations never create or
extend editable audio intervals. Untimed words remain visible in transcription
notes; text and subtitle exports include only timed content. Overlapping speech
remains one coarse audio group, with its complete original speaker set retained.

Old `metadata.transcription.items` UUID maps are migrated to source regions when
opening a project. Migration preserves surviving references; it cannot recover
warnings already lost by edits made in an older build. Replay the saved raw
response to regenerate a clean import when needed.

Records remain after editing or deletion and can include the full original
transcript. Files without transcription metadata continue to load. Missing
confidence, unknown gaps and unresolved annotations do not qualify automatic cuts.
