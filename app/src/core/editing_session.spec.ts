import { produce } from 'immer';
import _ from 'lodash';
import fs from 'fs';
import path from 'path';
import { ActionCreators } from 'redux-undo';
import editorReducer from '../state/editor';
import { defaultEditorState, EditorState } from '../state/editor/types';
import {
  deleteSelection,
  deleteSomething,
  insertParagraphEnd,
  trimSelectedGap,
} from '../state/editor/edit';
import { setSelection } from '../state/editor/selection';
import {
  finishTranscriptCorrection,
  setTranscriptCorrectionText,
} from '../state/editor/transcript_correction';
import { contentToVtt } from './webvtt';
import { setUserIndex } from '../state/editor/play';
import { addUuids } from '../util/test_helper';
import {
  replaySession,
  SessionEvent,
  SessionJournal,
  sessionDocument,
  sessionHash,
} from './editing_session';

function initial(): EditorState {
  const state = _.cloneDeep(defaultEditorState);
  state.document.content = addUuids([
    { type: 'paragraph_start', speaker: 'Speaker', language: 'ko' },
    { type: 'text', text: '안녕', source: 's', sourceStart: 0.12, length: 0.88, conf: 0.8 },
    { type: 'non_text', source: 's', sourceStart: 1, length: 3 },
    { type: 'text', text: '하세요', source: 's', sourceStart: 4, length: 0.6, conf: 0.9 },
    { type: 'paragraph_end' },
  ]);
  state.document.sources = { s: { fileContents: new ArrayBuffer(8), objectUrl: 'blob:test' } };
  state.cursor.userIndex = 1;
  return state;
}

const pythonFixtures = fs
  .readFileSync(path.resolve(__dirname, '../../../ml/fixtures/execution-v2.jsonl'), 'utf8')
  .trim()
  .split('\n')
  .map((line) => JSON.parse(line));

test.each(pythonFixtures)(
  'Python reference agrees with real editor operations and undo: $id',
  (row) => {
    let state = editorReducer(undefined, { type: '@@init' });
    const before = initial();
    before.document.content = row.before.content;
    state = { ...state, present: before };
    const events: SessionEvent[] = [];
    const journal = new SessionJournal(before, { collection_mode: 'synthetic' }, (e) =>
      events.push(e)
    );
    const dispatch = (action: any) => {
      const previous = state.present as EditorState;
      state = editorReducer(state, action);
      journal.observe(previous, state.present as EditorState, action.type);
    };
    const edit = row.prediction;
    const targets = (edit.target || []).map((id: number) => row.mapping[id].index);
    if (['correct_text', 'delete_span', 'trim_silence', 'remove_silence'].includes(edit.action)) {
      dispatch(
        setSelection({ startIndex: targets[0], length: targets.length, headPosition: 'right' })
      );
      if (edit.action === 'correct_text') {
        dispatch(setTranscriptCorrectionText(edit.text));
        dispatch(finishTranscriptCorrection());
      } else if (edit.action === 'trim_silence') {
        dispatch(
          trimSelectedGap({ uuid: before.document.content[targets[0]].uuid, keepMs: edit.keep_ms })
        );
      } else dispatch(deleteSelection());
    } else if (edit.action === 'split_paragraph') {
      dispatch(setUserIndex(targets[0]));
      dispatch(insertParagraphEnd());
    } else if (edit.action === 'merge_paragraph') {
      dispatch(setSelection({ startIndex: targets[0] - 2, length: 2, headPosition: 'right' }));
      dispatch(deleteSelection());
    }
    const after = state.present as EditorState;
    expect(after.document.content).toStrictEqualExceptUuids(row.after.content);
    expect(after.document.content).toBeValidDocumentContent();
    expect(contentToVtt(after.document.content, false, false, null).toString()).toEqual(
      contentToVtt(row.after.content, false, false, null).toString()
    );
    expect(replaySession(row.events)).toEqual(row.after);
    if (edit.action !== 'no_action') {
      dispatch(ActionCreators.undo());
      expect((state.present as EditorState).document.content).toEqual(before.document.content);
      dispatch(ActionCreators.redo());
      expect((state.present as EditorState).document.content).toEqual(after.document.content);
    }
    journal.finish('fixture_complete');
    expect(events[0]).toMatchObject({ schema_version: 2, actor: 'synthetic' });
    expect(
      events.filter((e) => e.kind === 'transaction').every((e) => e.actor === 'synthetic')
    ).toBe(true);
    expect(replaySession(events)).toEqual(sessionDocument((state.present as EditorState).document));
  }
);

test('replays the same persisted UI fixture as the Python verifier', () => {
  const events = fs
    .readFileSync(path.resolve(__dirname, '../../../ml/fixtures/editing-session.jsonl'), 'utf8')
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line));
  const document = replaySession(events);
  expect(document.content.filter((item) => item.type === 'paragraph_start')).toHaveLength(2);
  expect(
    document.content.reduce((total, item) => total + ('length' in item ? item.length : 0), 0)
  ).toBeCloseTo(2.725);
});

test('records semantic changes and replays actual reducer deletion, captions, undo and redo', () => {
  const events: SessionEvent[] = [];
  let state = editorReducer(undefined, { type: '@@init' });
  state = { ...state, present: initial() };
  const journal = new SessionJournal(state.present!, { collection_mode: 'reference_guided' }, (e) =>
    events.push(e)
  );
  const dispatch = (action: any) => {
    const before = state.present!;
    state = editorReducer(state, action);
    journal.observe(before, state.present!, action.type);
  };
  dispatch(deleteSomething('right'));
  expect(journal.edits).toBe(0);
  expect(events.some((e) => e.kind === 'context' && e.after.selection?.length === 1)).toBe(true);
  dispatch(deleteSomething('right'));
  dispatch(ActionCreators.undo());
  dispatch(ActionCreators.redo());
  dispatch(setUserIndex(2));
  dispatch(insertParagraphEnd());
  dispatch(setUserIndex(2));
  dispatch(deleteSomething('right'));
  journal.finish('test_complete');
  const transactions = events.filter((e) => e.kind === 'transaction');
  expect(transactions.map((e) => e.origin)).toEqual(['human', 'undo', 'redo', 'human', 'human']);
  expect(transactions.map((e) => e.operation)).toEqual([
    'delete_span',
    'other_transaction',
    'delete_span',
    'split_paragraph',
    'merge_paragraph',
  ]);
  const persisted = JSON.parse(JSON.stringify(events));
  expect(replaySession(persisted)).toEqual(sessionDocument(state.present!.document));
  expect(events[0].document.sources).toBeUndefined();
  expect(JSON.stringify(events)).not.toContain('blob:test');
});

test('multiword corrections and arbitrary large transactions are retained, display changes are omitted', () => {
  const events: SessionEvent[] = [];
  const before = initial();
  const journal = new SessionJournal(before, {}, (e) => events.push(e));
  const display = produce(before, (s) => {
    s.document.metadata.display_video = !s.document.metadata.display_video;
  });
  journal.observe(before, display, 'display');
  expect(journal.edits).toBe(0);
  const after = produce(display, (s) => {
    s.document.content.splice(1, 3, {
      ...s.document.content[1],
      type: 'text',
      text: '안녕하세요 여러분',
      source: 's',
      sourceStart: 0.12,
      length: 4.48,
      conf: 0.8,
    });
  });
  journal.observe(display, after, 'transcript_correction/finishTranscriptCorrection');
  expect(events.find((e) => e.kind === 'transaction')?.operation).toBe('correct_text');
  expect(replaySession(events)).toEqual(sessionDocument(after.document));
});

test('automatic playback ticks do not flood the log and paused contexts remain candidates, not no-action labels', () => {
  const events: SessionEvent[] = [];
  const before = initial();
  before.playing = true;
  const journal = new SessionJournal(before, {}, (e) => events.push(e));
  expect(events[1].eligible).toBe(false);
  const ticking = produce(before, (s) => {
    s.cursor.playerTime = 2.36;
  });
  journal.observe(before, ticking, 'editor/setTimePlayer');
  expect(events).toHaveLength(2);
  const paused = produce(ticking, (s) => {
    s.playing = false;
  });
  journal.observe(ticking, paused, 'editor/setPlay');
  expect(events[events.length - 1]).toMatchObject({
    kind: 'opportunity',
    eligible: true,
    prediction_shown: false,
  });
  expect(events[events.length - 1].expectedOutput).toBeUndefined();
});

test('corrupted, reordered and post-end records fail replay', () => {
  const events: SessionEvent[] = [];
  const before = initial();
  const journal = new SessionJournal(before, {}, (e) => events.push(e));
  const after = produce(before, (s) => {
    s.document.content.splice(1, 1);
  });
  journal.observe(before, after, 'editor/deleteSelection');
  journal.finish('test_complete');
  for (const corrupt of [
    (e: SessionEvent[]) => {
      e[0].document.content[1].text = 'changed';
    },
    (e: SessionEvent[]) => {
      e[2].patch.index = -1;
    },
    (e: SessionEvent[]) => {
      e[2].patch.remove[0].sourceStart = 99;
    },
    (e: SessionEvent[]) => {
      e[2].seq = 99;
    },
    (e: SessionEvent[]) => {
      e.push({ kind: 'context', seq: e.length });
    },
  ]) {
    const altered = _.cloneDeep(events);
    corrupt(altered);
    expect(() => replaySession(altered)).toThrow();
  }
  expect(sessionHash({ value: -0 })).toBe(sessionHash({ value: 0 }));
  expect(() => sessionHash({ value: NaN })).toThrow();
});

test('manual gap trim retains exact edge audio and rejects stale, playing and invalid requests', () => {
  const before = initial();
  before.selection = { startIndex: 2, length: 1, headPosition: 'right' };
  const gap = before.document.content[2];
  for (const keepMs of [0, -1, 3000, NaN, 1.5]) {
    const state = _.cloneDeep(before);
    trimSelectedGap.reducer(state, { uuid: gap.uuid, keepMs });
    expect(state).toEqual(before);
  }
  const stale = _.cloneDeep(before);
  trimSelectedGap.reducer(stale, { uuid: 'stale', keepMs: 725 });
  expect(stale).toEqual(before);
  const after = _.cloneDeep(before);
  trimSelectedGap.reducer(after, { uuid: gap.uuid, keepMs: 725 });
  expect(after.document.content.slice(2, 4)).toMatchObject([
    { source: 's', sourceStart: 1, length: 0.3625 },
    { source: 's', sourceStart: 3.6375, length: 0.3625 },
  ]);
  expect(after.document.sources).toEqual(before.document.sources);
  before.playing = true;
  const playing = _.cloneDeep(before);
  trimSelectedGap.reducer(playing, { uuid: gap.uuid, keepMs: 725 });
  expect(playing).toEqual(before);
});
