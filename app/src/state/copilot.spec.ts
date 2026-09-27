import { configureStore } from '@reduxjs/toolkit';
import { ActionCreators } from 'redux-undo';
import _ from 'lodash';
import copilot, {
  acceptCopilot,
  copilotEligible,
  copilotKey,
  createCopilotMiddleware,
  currentCopilotProposal,
  dismissCopilot,
  localCopilotServer,
  markCopilotShown,
  requestCopilot,
  setCopilotBlocked,
  setCopilotEnabled,
  setCopilotFeedbackOrigin,
} from './copilot';
import editorReducer from './editor';
import { defaultEditorState, EditorState } from './editor/types';
import { setUserIndex } from './editor/play';
import { setText } from './editor/edit';
import { acceptCopilotTransaction } from './editor/copilot';
import { finishTranscriptCorrection } from './editor/transcript_correction';
import { Page, NavState } from './nav';
import type { ServerState } from './server';
import * as api from '../server_api/copilot';
import { CopilotResponse, inferManualPrediction, validCopilotResult } from '../core/copilot';
import {
  replaySession,
  SessionEvent,
  SessionJournal,
  sessionContext,
  sessionDocument,
  sessionHash,
} from '../core/editing_session';

jest.mock('../server_api/copilot');
const local = { hostname: 'http://localhost', port: 8000, token: 'test', name: 'local server' };

function initial(): EditorState {
  const state = _.cloneDeep(defaultEditorState);
  state.document.content = [
    { type: 'paragraph_start', uuid: 'p1', language: 'en', speaker: 'S' },
    { type: 'text', uuid: 'w1', text: 'helo', source: 's', sourceStart: 0, length: 1, conf: 1 },
    { type: 'non_text', uuid: 'g1', source: 's', sourceStart: 1, length: 0.2 },
    { type: 'text', uuid: 'w2', text: 'world', source: 's', sourceStart: 1.2, length: 1, conf: 1 },
    { type: 'paragraph_end', uuid: 'p2' },
  ];
  state.document.sources = { s: { fileContents: new ArrayBuffer(1), objectUrl: 'blob:s' } };
  state.cursor.userIndex = 1;
  return state;
}

function proposal(body: api.PredictRequest): CopilotResponse {
  const after = _.cloneDeep(body.document);
  if (after.content[1].type === 'text') after.content[1].text = 'hello';
  return {
    request: body.request,
    status: 'proposal',
    proposal_id: body.request.request_id,
    prediction: { action: 'correct_text', target: [0], text: 'hello' },
    target_uuids: ['w1'],
    document_hash: sessionHash(body.document),
    result_document: after,
    result_hash: sessionHash(after),
    context: {},
    model: {},
    timings: {},
  };
}

function setup(editor = initial()) {
  const history = editorReducer(undefined, { type: '@@INIT' });
  const store = configureStore({
    reducer: {
      editor: editorReducer,
      copilot,
      server: (state: ServerState = { servers: [local], selectedServer: 0 }): ServerState => state,
      nav: (state: NavState = { page: Page.Editor }): NavState => state,
    },
    preloadedState: {
      editor: { ...history, present: editor, _latestUnfiltered: editor },
      server: { servers: [local], selectedServer: 0 },
      nav: { page: Page.Editor },
      copilot: copilot(undefined, { type: '@@INIT' }),
    },
    middleware: (getDefault) =>
      getDefault({ serializableCheck: false }).concat(createCopilotMiddleware()),
  });
  store.dispatch(setCopilotEnabled(true));
  return store;
}

async function flush() {
  for (let i = 0; i < 20; i++) await Promise.resolve();
}

beforeEach(() => {
  jest.useFakeTimers();
  jest.clearAllMocks();
  (api.copilotHealth as jest.Mock).mockResolvedValue({ status: 'ready' });
  (api.cancelCopilot as jest.Mock).mockResolvedValue(undefined);
  (api.feedbackCopilot as jest.Mock).mockResolvedValue(undefined);
  (api.predictCopilot as jest.Mock).mockImplementation(async (_server, body) => proposal(body));
});
afterEach(() => {
  jest.clearAllTimers();
  jest.useRealTimers();
});

test('a source-preserving correction spanning a gap validates, while audio or unrelated edits fail', () => {
  const before = sessionDocument(initial().document);
  const response = proposal({
    document: before,
    request: { request_id: 'test' },
  } as api.PredictRequest);
  response.prediction = { action: 'correct_text', target: [0, 1, 2], text: 'hello world' };
  response.target_uuids = ['w1', 'g1', 'w2'];
  const first = before.content[1];
  if (first.type !== 'text') throw Error('fixture');
  response.result_document = {
    ...before,
    content: [before.content[0], { ...first, text: 'hello world', length: 2.2 }, before.content[4]],
  };
  response.result_hash = sessionHash(response.result_document);
  expect(validCopilotResult(before, response)).toBe(true);
  expect(inferManualPrediction(before, response.result_document)).toEqual(response.prediction);
  const changed = _.cloneDeep(response);
  (changed.result_document?.content[1] as any).sourceStart = 0.1;
  changed.result_hash = sessionHash(changed.result_document);
  expect(validCopilotResult(before, changed)).toBe(false);
  response.result_document.content[0] = {
    ...response.result_document.content[0],
    speaker: 'Other',
  } as any;
  response.result_hash = sessionHash(response.result_document);
  expect(validCopilotResult(before, response)).toBe(false);
});

test('shown, acceptance and one undo retain actual consecutive transitions outside editor history', async () => {
  const store = setup(),
    original = sessionDocument(store.getState().editor.present!.document);
  const recorded: SessionEvent[] = [];
  const beforeEditor = store.getState().editor.present!;
  const journal = new SessionJournal(beforeEditor, { collection_mode: 'unguided' }, (event) =>
    recorded.push(event)
  );
  await store.dispatch(requestCopilot('explicit'));
  const response = store.getState().copilot.proposal!.response;
  expect(store.getState().copilot.measurements.map((item) => item.event)).toEqual([
    'trigger',
    'request',
    'response',
  ]);
  expect(store.getState().editor.past).toHaveLength(0);
  store.dispatch(markCopilotShown(response.proposal_id!));
  store.dispatch(acceptCopilot());
  const acceptedEditor = store.getState().editor.present!;
  journal.observe(beforeEditor, acceptedEditor, acceptCopilotTransaction.type);
  expect(recorded.find((event) => event.kind === 'transaction')?.operation).toBe('correct_text');
  expect(sessionDocument(store.getState().editor.present!.document)).toEqual(
    response.result_document
  );
  expect(store.getState().editor.past).toHaveLength(1);
  store.dispatch(ActionCreators.undo());
  journal.observe(acceptedEditor, store.getState().editor.present!, '@@redux-undo/UNDO');
  journal.finish('test');
  expect(replaySession(recorded)).toEqual(original);
  expect(sessionDocument(store.getState().editor.present!.document)).toEqual(original);
  expect(store.getState().copilot.proposal).toBeNull();
  expect(store.getState().copilot.history).toHaveLength(2);
  const [commit, undo] = store.getState().copilot.history;
  expect(undo.undo_of).toBe(commit.transaction_id);
  expect(undo.before).toEqual(commit.after);
  expect(undo.after).toEqual(commit.before);
  await flush();
  const events = (api.feedbackCopilot as jest.Mock).mock.calls.map((call) => call[1]);
  expect(events.map((event) => event.outcome)).toEqual(['shown', 'accepted', 'undone']);
  expect(events.map((event) => event.sequence)).toEqual([2, 3, 4]);
  expect(events[1].transition).toEqual(commit);
  expect(events[2].transition).toEqual(undo);
  const measured = store.getState().copilot.measurements;
  expect(measured.map((item) => item.event)).toEqual(
    expect.arrayContaining(['visible_complete_preview', 'accepted', 'undone'])
  );
  expect(measured.every((item) => item.since_trigger_ms >= 0)).toBe(true);
  expect(JSON.stringify(measured)).not.toContain('hello');
});

test('cursor changes cancel requests and reject late responses even after returning to the same document', async () => {
  let resolve: (value: CopilotResponse) => void = () => {};
  let body: api.PredictRequest;
  (api.predictCopilot as jest.Mock).mockImplementation((_server, value) => {
    body = value;
    return new Promise((done) => {
      resolve = done;
    });
  });
  const store = setup();
  const pending = store.dispatch(requestCopilot('explicit'));
  await flush();
  store.dispatch(setUserIndex(3));
  store.dispatch(setUserIndex(1));
  expect((api.predictCopilot as jest.Mock).mock.calls[0][2].aborted).toBe(true);
  resolve(proposal(body!));
  await pending;
  await flush();
  expect(store.getState().copilot.proposal).toBeNull();
  expect(api.cancelCopilot).toHaveBeenCalledTimes(1);
  expect((api.feedbackCopilot as jest.Mock).mock.calls.map((call) => call[1].outcome)).toEqual([
    'invalidated',
  ]);
  expect(store.getState().editor.past).toHaveLength(0);
  expect(store.getState().copilot.measurements.map((item) => item.event)).toEqual(
    expect.arrayContaining(['cancel_requested', 'cancel_http_ack', 'stale_discard'])
  );
});

test('timings include automatic debounce and delayed visible preview; automation is labelled', async () => {
  let now = 0;
  const clock = jest.spyOn(performance, 'now').mockImplementation(() => now);
  try {
    const store = setup();
    store.dispatch(setCopilotFeedbackOrigin('ai_computer_use'));
    store.dispatch(setText({ absoluteIndex: 1, text: 'helo again' }));
    now = 650;
    jest.advanceTimersByTime(650);
    await flush();
    const response = store.getState().copilot.proposal!.response;
    expect(
      store.getState().copilot.measurements.find((m) => m.event === 'request')?.since_trigger_ms
    ).toBe(650);
    expect(
      store.getState().copilot.measurements.some((m) => m.event === 'visible_complete_preview')
    ).toBe(false);
    now = 800;
    store.dispatch(markCopilotShown(response.proposal_id!));
    const shown = store
      .getState()
      .copilot.measurements.find((m) => m.event === 'visible_complete_preview');
    expect(shown?.since_trigger_ms).toBe(800);
    expect(shown?.since_request_ms).toBe(150);
    await flush();
    expect((api.feedbackCopilot as jest.Mock).mock.calls[0][1].origin).toBe('ai_computer_use');
    expect(store.getState().copilot.history[0].origin).toBe('ai_computer_use');
  } finally {
    clock.mockRestore();
  }
});

test('no-action and failed requests retain latency outcomes without a fake visible preview', async () => {
  (api.predictCopilot as jest.Mock).mockImplementationOnce(async (_server, body) => ({
    ...proposal(body),
    status: 'no_action',
    proposal_id: null,
    prediction: { action: 'no_action' },
    result_document: null,
    result_hash: null,
    target_uuids: [],
  }));
  const store = setup();
  await store.dispatch(requestCopilot('explicit'));
  expect(store.getState().copilot.measurements.find((m) => m.event === 'response')?.outcome).toBe(
    'no_action'
  );
  (api.predictCopilot as jest.Mock).mockRejectedValueOnce(new Error('deadline'));
  await store.dispatch(requestCopilot('explicit'));
  const events = store.getState().copilot.measurements.map((m) => m.event);
  expect(events).toContain('error');
  expect(events).not.toContain('visible_complete_preview');
});

test('a stale acceptance cannot add an undo frame and dismissal never edits the document', async () => {
  const store = setup();
  await store.dispatch(requestCopilot('explicit'));
  const response = store.getState().copilot.proposal!.response;
  const context_hash = sessionHash(sessionContext(store.getState().editor.present!));
  store.dispatch(markCopilotShown(response.proposal_id!));
  store.dispatch(dismissCopilot());
  store.dispatch(acceptCopilotTransaction({ proposal: response, context_hash }));
  expect(store.getState().editor.past).toHaveLength(0);
  expect(sessionHash(sessionDocument(store.getState().editor.present!.document))).toBe(
    response.document_hash
  );
});

test('inline corrections debounce only after text focus is released, with real edit history', async () => {
  const editor = initial();
  editor.selection = { startIndex: 1, length: 1, headPosition: 'right' };
  editor.transcriptCorrectionState = 'Hello';
  const store = setup(editor);
  store.dispatch(setCopilotBlocked(true));
  store.dispatch(finishTranscriptCorrection());
  jest.advanceTimersByTime(700);
  await flush();
  expect(api.predictCopilot).not.toHaveBeenCalled();
  store.dispatch(setCopilotBlocked(false));
  jest.advanceTimersByTime(650);
  await flush();
  expect(api.predictCopilot).toHaveBeenCalledTimes(1);
  const request = (api.predictCopilot as jest.Mock).mock.calls[0][1];
  expect(request.trigger).toBe('automatic');
  expect(request.history).toHaveLength(1);
  expect(request.history[0].prediction).toEqual({
    action: 'correct_text',
    target: [0],
    text: 'Hello',
  });
});

test('failed acceptance feedback blocks further inference until a new explicit session', async () => {
  const store = setup();
  await store.dispatch(requestCopilot('explicit'));
  store.dispatch(markCopilotShown(store.getState().copilot.proposal!.response.proposal_id!));
  (api.feedbackCopilot as jest.Mock).mockImplementation(async (_server, event) => {
    if (event.outcome === 'accepted') throw Error('offline');
  });
  store.dispatch(acceptCopilot());
  await flush();
  expect(store.getState().copilot.feedbackError).toBe('offline');
  await store.dispatch(requestCopilot('explicit'));
  expect(api.predictCopilot).toHaveBeenCalledTimes(1);
  expect(store.getState().editor.past).toHaveLength(1);
  store.dispatch(setCopilotEnabled(false));
  store.dispatch(setCopilotEnabled(true));
  expect(store.getState().copilot.feedbackError).toBeNull();
});

test('copilot never uses a remote ASR configuration, and busy editor states are ineligible', () => {
  const remote = { ...local, hostname: 'https://asr.example.com' };
  expect(localCopilotServer({ server: { servers: [remote], selectedServer: 0 } })).toBeUndefined();
  expect(localCopilotServer({ server: { servers: [remote, local], selectedServer: 0 } })).toEqual(
    local
  );
  const store = setup();
  expect(copilotEligible(store.getState())).toBe(true);
  for (const patch of [
    { playing: true },
    { transcriptCorrectionState: 'edit' },
    { exportPopup: 'document' as const },
    { filterPopup: true },
  ]) {
    expect(
      copilotEligible({
        ...store.getState(),
        editor: { present: { ...store.getState().editor.present!, ...patch } },
      })
    ).toBe(false);
  }
});

test('keyboard ownership requires a visible preview, transcript focus, and an ordinary key', async () => {
  const store = setup();
  await store.dispatch(requestCopilot('explicit'));
  const response = store.getState().copilot.proposal!.response;
  store.dispatch(markCopilotShown(response.proposal_id!));
  document.body.innerHTML =
    '<div id="scroll-container"><div id="document" tabindex="0"><span data-testid="copilot-preview"></span><input /></div></div>';
  const rect = { top: 10, bottom: 100, left: 10, right: 100, width: 90, height: 90 } as DOMRect;
  document.getElementById('scroll-container')!.getBoundingClientRect = () => rect;
  const preview = document.querySelector('[data-testid="copilot-preview"]')!;
  preview.getBoundingClientRect = () => rect;
  const event = {
    key: 'Tab',
    repeat: false,
    shiftKey: false,
    altKey: false,
    ctrlKey: false,
    metaKey: false,
    target: document.getElementById('document'),
  };
  expect(copilotKey(event, store.getState())).toBe('accept');
  for (const patch of [
    { repeat: true },
    { shiftKey: true },
    { isComposing: true },
    { keyCode: 229 },
    { target: document.querySelector('input') },
  ])
    expect(copilotKey({ ...event, ...patch }, store.getState())).toBeNull();
  preview.getBoundingClientRect = () => ({ ...rect, top: 200, bottom: 250 });
  expect(copilotKey(event, store.getState())).toBeNull();
  expect(currentCopilotProposal(store.getState())).not.toBeNull();
});

test('unsupported manual edits rotate the feedback session instead of sending a broken chain', async () => {
  const store = setup(),
    beforeId = store.getState().copilot.documentId;
  store.dispatch(setText({ absoluteIndex: 1, text: 'hello' }));
  const afterKnownId = store.getState().copilot.documentId;
  expect(afterKnownId).toBe(beforeId);
  expect(store.getState().copilot.history).toHaveLength(1);
  store.dispatch({ type: 'editor/renameSpeaker', payload: { oldName: 'S', newName: 'Other' } });
  expect(store.getState().copilot.documentId).not.toBe(beforeId);
  expect(store.getState().copilot.history).toHaveLength(0);
});

test('manual word edits keep low-confidence source history and leave no-ops unchanged', async () => {
  const editor = initial();
  const item = editor.document.content[1];
  if (item.type !== 'text') throw Error('fixture');
  item.conf = 0.9;
  const store = setup(editor);
  const sessionId = store.getState().copilot.documentId;
  const before = sessionDocument(store.getState().editor.present!.document);

  store.dispatch(setText({ absoluteIndex: 1, text: 'helo' }));
  expect(sessionDocument(store.getState().editor.present!.document)).toEqual(before);
  expect(store.getState().copilot.documentId).toBe(sessionId);
  expect(store.getState().copilot.history).toHaveLength(0);
  jest.advanceTimersByTime(650);
  await flush();
  expect(api.predictCopilot).not.toHaveBeenCalled();

  store.dispatch(setText({ absoluteIndex: 1, text: 'hello' }));
  const after = sessionDocument(store.getState().editor.present!.document);
  const expected = _.cloneDeep(before);
  const corrected = expected.content[1];
  if (corrected.type !== 'text') throw Error('fixture');
  corrected.text = 'hello';
  corrected.conf = 1;
  expect(after).toEqual(expected);
  expect(store.getState().copilot.documentId).toBe(sessionId);
  expect(store.getState().copilot.history).toHaveLength(1);
  expect(store.getState().copilot.history[0]).toMatchObject({
    before,
    after,
    prediction: { action: 'correct_text', target: [0], text: 'hello' },
    undo_of: null,
  });
  jest.advanceTimersByTime(650);
  await flush();
  expect(api.predictCopilot).toHaveBeenCalledTimes(1);
  const request = (api.predictCopilot as jest.Mock).mock.calls[0][1];
  expect(request.history).toEqual(store.getState().copilot.history);
  expect(request.document).toEqual(after);
});
