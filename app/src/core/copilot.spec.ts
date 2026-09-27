import fs from 'fs';
import path from 'path';
import _ from 'lodash';
import { ActionCreators } from 'redux-undo';
import editorReducer from '../state/editor';
import { defaultEditorState, EditorState } from '../state/editor/types';
import { acceptCopilotTransaction } from '../state/editor/copilot';
import { trimSelectedGap } from '../state/editor/edit';
import {
  CopilotPrediction,
  CopilotResponse,
  copilotPreviewText,
  inferManualPrediction,
  isTimed,
  validCopilotResult,
} from './copilot';
import {
  replaySession,
  SessionDocument,
  SessionEvent,
  SessionJournal,
  sessionContext,
  sessionDocument,
  sessionHash,
} from './editing_session';

const fixtures = fs
  .readFileSync(path.resolve(__dirname, '../../../ml/fixtures/execution-v2.jsonl'), 'utf8')
  .trim()
  .split('\n')
  .map((line) => JSON.parse(line));

function responseFor(
  before: SessionDocument,
  after: SessionDocument,
  prediction: CopilotPrediction
): CopilotResponse {
  const timed = before.content.filter(isTimed);
  return {
    request: { request_id: 'r', document_id: 'd', revision: 1, interaction_epoch: 2 },
    status: prediction.action === 'no_action' ? 'no_action' : 'proposal',
    proposal_id: prediction.action === 'no_action' ? null : 'p',
    prediction,
    target_uuids: (prediction.target || []).map((alias) => timed[alias]?.uuid),
    document_hash: sessionHash(before),
    result_document: after,
    result_hash: sessionHash(after),
    context: {},
    model: {},
    timings: {},
  };
}

function fixture(id: string) {
  const row = _.cloneDeep(fixtures.find((row) => row.id === id));
  return { ...row, response: responseFor(row.before, row.after, row.prediction) };
}

test.each(fixtures)('all-seven reference operation validates and stays undoable: $id', (row) => {
  const response = responseFor(row.before, row.after, row.prediction);
  const original = _.cloneDeep(defaultEditorState);
  original.document.content = row.before.content;
  original.document.metadata = { ...original.document.metadata, ...row.before.metadata };
  original.document.sources = { s: { fileContents: new ArrayBuffer(8), objectUrl: 'blob:s' } };
  let state = editorReducer(undefined, { type: '@@INIT' });
  state = { ...state, present: original, _latestUnfiltered: original };
  const events: SessionEvent[] = [];
  const journal = new SessionJournal(original, { collection_mode: 'synthetic' }, (event) =>
    events.push(event)
  );
  const dispatch = (action: any) => {
    const before = state.present as EditorState;
    state = editorReducer(state, action);
    journal.observe(before, state.present as EditorState, action.type);
  };
  const edits = row.prediction.action !== 'no_action';
  expect(validCopilotResult(row.before, response)).toBe(edits);
  expect(inferManualPrediction(row.before, row.after)).toEqual(edits ? row.prediction : null);
  dispatch(
    acceptCopilotTransaction({
      proposal: response,
      context_hash: sessionHash(sessionContext(original)),
    })
  );
  expect(sessionDocument(state.present!.document)).toEqual(row.after);
  expect(state.present!.document.sources).toBe(original.document.sources);
  expect(state.past).toHaveLength(edits ? 1 : 0);
  if (edits) {
    expect(events.find((event) => event.kind === 'transaction')?.operation).toBe(
      row.prediction.action
    );
    dispatch(ActionCreators.undo());
    expect(sessionDocument(state.present!.document)).toEqual(row.before);
    dispatch(ActionCreators.redo());
    expect(sessionDocument(state.present!.document)).toEqual(row.after);
  }
  journal.finish('test');
  expect(replaySession(events)).toEqual(row.after);
});

test('target aliases must match UUIDs even if the proposed source edit itself is valid', () => {
  const { before, response } = fixture('delete-with-gap');
  for (const targets of [
    [1, 2, 3],
    [0, 0, 2],
    [2, 1, 0],
    [-1, 1, 2],
    [0.5, 1, 2],
  ]) {
    expect(
      validCopilotResult(before, {
        ...response,
        prediction: { ...response.prediction, target: targets },
      })
    ).toBe(false);
  }
});

test('cuts reject discontinuous sources, hidden paragraph removal and unrelated mutations', () => {
  for (const change of [
    (before: SessionDocument) => {
      (before.content[2] as any).source = 'other';
    },
    (before: SessionDocument) => {
      (before.content[2] as any).sourceStart += 0.1;
    },
    (before: SessionDocument) => {
      before.content[2] = { type: 'artificial_silence', uuid: before.content[2].uuid, length: 1 };
    },
  ]) {
    const { before, after, prediction } = fixture('delete-with-gap');
    change(before);
    expect(validCopilotResult(before, responseFor(before, after, prediction))).toBe(false);
    expect(inferManualPrediction(before, after)).toBeNull();
  }
  const { before, response } = fixture('delete-with-gap');
  const unrelated = _.cloneDeep(response);
  (unrelated.result_document!.content[1] as any).text = 'unrelated';
  unrelated.result_hash = sessionHash(unrelated.result_document);
  expect(validCopilotResult(before, unrelated)).toBe(false);
  const changedMetadata = _.cloneDeep(response);
  changedMetadata.result_document!.metadata = { extra: true };
  changedMetadata.result_hash = sessionHash(changedMetadata.result_document);
  expect(validCopilotResult(before, changedMetadata)).toBe(false);
  const { before: paragraphs } = fixture('merge-caption');
  const after = {
    ...paragraphs,
    content: [paragraphs.content[0], paragraphs.content[paragraphs.content.length - 1]],
  };
  expect(
    validCopilotResult(
      paragraphs,
      responseFor(paragraphs, after, { action: 'delete_span', target: [0, 1] })
    )
  ).toBe(false);
  expect(inferManualPrediction(paragraphs, after)).toBeNull();
});

test('trim preserves both exact outside edges, uses a fresh right UUID, and never becomes zero removal', () => {
  const { before, response } = fixture('trim-pause');
  expect(response.prediction.keep_ms).toBe(725);
  for (const keep_ms of [0, -1, 0.5, 2000, 5000, NaN, Infinity]) {
    expect(
      validCopilotResult(before, { ...response, prediction: { ...response.prediction, keep_ms } })
    ).toBe(false);
  }
  const first = before.content.findIndex((item: any) => item.uuid === response.target_uuids[0]);
  for (const change of [
    (result: SessionDocument) => {
      (result.content[first] as any).sourceStart += 0.01;
    },
    (result: SessionDocument) => {
      (result.content[first + 1] as any).sourceStart -= 0.01;
    },
    (result: SessionDocument) => {
      (result.content[first] as any).length += 0.01;
      (result.content[first + 1] as any).length -= 0.01;
    },
    (result: SessionDocument) => {
      (result.content[first + 1] as any).source = 'other';
    },
    (result: SessionDocument) => {
      result.content[first + 1].uuid = result.content[first].uuid;
    },
    (result: SessionDocument) => {
      result.content.splice(first + 1, 1);
    },
  ]) {
    const changed = _.cloneDeep(response);
    change(changed.result_document!);
    changed.result_hash = sessionHash(changed.result_document);
    expect(validCopilotResult(before, changed)).toBe(false);
    expect(inferManualPrediction(before, changed.result_document!)).toBeNull();
  }
  const removed = fixture('remove-pause');
  expect(
    validCopilotResult(removed.before, {
      ...removed.response,
      prediction: { ...removed.prediction, action: 'trim_silence', keep_ms: 0 },
    })
  ).toBe(false);
  expect(
    validCopilotResult(before, {
      ...response,
      prediction: { ...response.prediction, action: 'remove_silence' },
    })
  ).toBe(false);
});

test('real manual trim is recognized at arbitrary positive millisecond durations', () => {
  const { before } = fixture('trim-pause');
  for (const keepMs of [1, 333, 725]) {
    const editor = _.cloneDeep(defaultEditorState);
    editor.document.content = before.content;
    editor.document.metadata = { ...editor.document.metadata, ...before.metadata };
    editor.selection = { startIndex: 2, length: 1, headPosition: 'right' };
    let state = editorReducer(undefined, { type: '@@INIT' });
    state = { ...state, present: editor, _latestUnfiltered: editor };
    state = editorReducer(state, trimSelectedGap({ uuid: before.content[2].uuid, keepMs }));
    expect(inferManualPrediction(before, sessionDocument(state.present!.document))).toEqual({
      action: 'trim_silence',
      target: [1],
      keep_ms: keepMs,
    });
  }
});

test('audio previews distinguish deleting speech, shortening a pause, and removing it', () => {
  expect(copilotPreviewText({ action: 'delete_span', target: [0] })).toContain('source audio');
  expect(copilotPreviewText({ action: 'trim_silence', target: [1], keep_ms: 725 })).toBe(
    'Shorten this pause to 725 ms (keep both edges)'
  );
  expect(copilotPreviewText({ action: 'remove_silence', target: [1] })).toBe(
    'Remove this entire pause'
  );
  expect(new Set(fixtures.map((row) => row.prediction.action)).size).toBe(7);
  for (const row of fixtures) expect(copilotPreviewText(row.prediction)).toBeTruthy();
});
