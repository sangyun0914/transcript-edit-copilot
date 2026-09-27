import { defaultEditorState, EditorState } from '../state/editor/types';
import { paste } from '../state/editor/edit';
import { finishTranscriptCorrection } from '../state/editor/transcript_correction';
import { alignmentWarning, importTranscript, transcriptionWarnings } from './transcription';
import { selectionDocument } from '../state/editor/selectors';
import { deserializeDocument, serializeDocument } from './document';
import editorReducer from '../state/editor';
import { assertSome } from '../util';
import { ActionCreators } from 'redux-undo';

function importedState(source = 'source-1'): EditorState {
  const state: EditorState = JSON.parse(JSON.stringify(defaultEditorState));
  const imported = importTranscript(
    [
      {
        speaker: 'speaker_0',
        content: [
          {
            type: 'word',
            word: 'before',
            sourceStart: 0,
            length: 1,
            conf: null,
            asr: { word_indices: [0], boundary_status: 'unverified' },
          },
          {
            type: 'word',
            word: 'uncertain',
            sourceStart: 1,
            length: 1,
            conf: null,
            asr: { word_indices: [1, 2], boundary_status: 'needs_alignment' },
          },
        ],
      },
    ],
    source,
    'kor',
    {
      version: 1,
      provider: 'elevenlabs',
      model: 'scribe_v2',
      language_code: 'kor',
      duration: 2,
      request: {},
      raw_response: { text: 'synthetic' },
    }
  );
  state.document = {
    ...imported,
    sources: { [source]: { objectUrl: 'test', fileContents: new ArrayBuffer(0) } },
  };
  state.cursor.current = 'user';
  return state;
}

function warnedWords(state: EditorState | null) {
  if (!state) throw new Error('Missing editor');
  const warnings = transcriptionWarnings(state.document.metadata.transcriptions);
  return state.document.content.filter(
    (item) => item.type === 'text' && alignmentWarning(item, warnings)
  );
}

test('paste retains warnings after regenerating IDs and deduplicates the import record', () => {
  const state = importedState();
  state.selection = { startIndex: 2, length: 1, headPosition: 'left' };
  const payload = selectionDocument(state);
  state.selection = null;
  state.cursor.userIndex = 3;
  assertSome(paste.reducers);
  assertSome(paste.reducers.fulfilled);
  paste.reducers.fulfilled(state, payload);
  const words = warnedWords(state);
  expect(words).toHaveLength(2);
  expect(words[0].uuid).not.toBe(words[1].uuid);
  expect(state.document.metadata.transcriptions).toHaveLength(1);
});

test('cross-document paste, save and reopen preserve both sources and their warnings', async () => {
  const from = importedState('source-2');
  from.selection = { startIndex: 2, length: 1, headPosition: 'left' };
  const into = importedState();
  into.cursor.userIndex = 3;
  assertSome(paste.reducers);
  assertSome(paste.reducers.fulfilled);
  paste.reducers.fulfilled(into, selectionDocument(from));
  expect(warnedWords(into)).toHaveLength(2);
  expect(into.document.metadata.transcriptions).toHaveLength(2);
  global.URL.createObjectURL = jest.fn(() => 'restored');
  into.document = await deserializeDocument(
    await serializeDocument(into.document).generateAsync({ type: 'nodebuffer' })
  );
  expect(warnedWords(into)).toHaveLength(2);
  expect(Object.keys(into.document.sources)).toEqual(['source-1', 'source-2']);
});

test('paste into an empty document preserves provenance', () => {
  const from = importedState();
  from.selection = { startIndex: 2, length: 1, headPosition: 'left' };
  const into: EditorState = JSON.parse(JSON.stringify(defaultEditorState));
  assertSome(paste.reducers);
  assertSome(paste.reducers.fulfilled);
  paste.reducers.fulfilled(into, selectionDocument(from));
  expect(warnedWords(into)).toHaveLength(1);
  expect(into.document.metadata.transcriptions).toHaveLength(1);
});

test('correction inherits uncertainty from any selected interval and undo restores the original', () => {
  const state = importedState();
  state.selection = { startIndex: 1, length: 2, headPosition: 'left' };
  state.transcriptCorrectionState = 'corrected phrase';
  let history: ReturnType<typeof editorReducer> = {
    ...editorReducer(undefined, { type: 'init' }),
    present: state,
    _latestUnfiltered: state,
  };
  history = editorReducer(history, finishTranscriptCorrection());
  expect(warnedWords(history.present)).toEqual([
    expect.objectContaining({ text: 'corrected phrase', length: 2 }),
  ]);
  history = editorReducer(history, ActionCreators.undo());
  expect(history.present?.document).toEqual(state.document);
  expect(warnedWords(history.present)).toEqual([expect.objectContaining({ text: 'uncertain' })]);
});

test('legacy UUID provenance migrates to source regions when reopened', async () => {
  const state = importedState();
  const record = state.document.metadata.transcriptions?.[0];
  assertSome(record);
  const { id: _id, regions, ...metadata } = record;
  state.document.metadata.transcription = {
    ...metadata,
    items: {
      [state.document.content[1].uuid]: regions[0],
      [state.document.content[2].uuid]: regions[1],
    },
  };
  delete state.document.metadata.transcriptions;
  global.URL.createObjectURL = jest.fn(() => 'restored');
  state.document = await deserializeDocument(
    await serializeDocument(state.document).generateAsync({ type: 'nodebuffer' })
  );
  expect(warnedWords(state)).toHaveLength(1);
  expect(state.document.metadata.transcription).toBeUndefined();
  expect(state.document.metadata.transcriptions?.[0].raw_response).toEqual(record.raw_response);
});
