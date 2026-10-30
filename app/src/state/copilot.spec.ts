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
  refreshCopilotHealth,
  retryCopilotFeedback,
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
import serverReducer, { receiveLocalServerInfo } from './server';
import * as api from '../server_api/copilot';
import * as copilotCore from '../core/copilot';
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

function setup(editor = initial(), healthStatus: string | null = 'ready') {
  const history = editorReducer(undefined, { type: '@@INIT' });
  const store = configureStore({
    reducer: {
      editor: editorReducer,
      copilot,
      server: (
        state: ServerState = { servers: [local], selectedServer: 0 },
        action: { type: string; payload?: ServerState }
      ): ServerState =>
        action.type === 'test/server' ? action.payload! : serverReducer(state, action),
      nav: (state: NavState = { page: Page.Editor }): NavState => state,
    },
    preloadedState: {
      editor: { ...history, present: editor, _latestUnfiltered: editor },
      server: { servers: [local], selectedServer: 0 },
      nav: { page: Page.Editor },
      copilot: {
        ...copilot(undefined, { type: '@@INIT' }),
        documentId: 'test-document',
        healthStatus,
      },
    },
    middleware: (getDefault) =>
      getDefault({ serializableCheck: false }).concat(createCopilotMiddleware()),
  });
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
  expect(new Set(events.map((event) => event.event_id)).size).toBe(3);
  expect(events.every((event) => event.request.request_id === response.request.request_id)).toBe(
    true
  );
  expect(events.map((event) => event.revision)).toEqual([0, 1, 2]);
  expect(events.every((event) => Number.isFinite(Date.parse(event.observed_at)))).toBe(true);
  expect(store.getState().copilot.events.every((event) => event.persistence === 'saved')).toBe(
    true
  );
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

test('metrics and feedback reuse validation, but a changed document is checked again', async () => {
  const validate = jest.spyOn(copilotCore, 'validCopilotResult');
  try {
    const store = setup();
    await store.dispatch(requestCopilot('explicit'));
    const current = currentCopilotProposal(store.getState())!;
    expect(validate).toHaveBeenCalledTimes(1);
    store.dispatch(markCopilotShown(current.response.proposal_id!));
    await flush();
    expect(currentCopilotProposal(store.getState())?.shown).toBe(true);
    expect(validate).toHaveBeenCalledTimes(1);
    const changed = _.cloneDeep(store.getState());
    const item = changed.editor.present!.document.content[1];
    if (item.type !== 'text') throw new Error('fixture');
    item.text = 'a different document';
    expect(currentCopilotProposal(changed)).toBeNull();
    expect(validate).toHaveBeenCalledTimes(2);
  } finally {
    validate.mockRestore();
  }
});

test('only new committed edits are sent after acknowledgement, with one full retry after server reset', async () => {
  const received: api.PredictRequest['history'] = [];
  (api.predictCopilot as jest.Mock).mockImplementation(
    async (_server, body: api.PredictRequest) => {
      if (body.history_base === null) received.splice(0);
      received.push(...body.history);
      const last = received[received.length - 1];
      return {
        ...proposal(body),
        history_tip: {
          count: received.length,
          transaction_id: last?.transaction_id ?? null,
          sequence: last?.sequence ?? 0,
        },
      };
    }
  );
  const store = setup();
  store.dispatch(setText({ absoluteIndex: 1, text: 'helo one' }));
  await store.dispatch(requestCopilot('explicit'));
  const first = (api.predictCopilot as jest.Mock).mock.calls[0][1] as api.PredictRequest;
  expect(first.history).toHaveLength(1);
  expect(first.history_base).toBeNull();
  store.dispatch(setText({ absoluteIndex: 1, text: 'helo two' }));
  await store.dispatch(requestCopilot('explicit'));
  const second = (api.predictCopilot as jest.Mock).mock.calls[1][1] as api.PredictRequest;
  expect(second.history).toHaveLength(1);
  expect(second.history_base).toBe(first.history[0].transaction_id);
  expect(second.history[0]).toEqual(store.getState().copilot.history[1]);
  store.dispatch(setUserIndex(3));
  (api.predictCopilot as jest.Mock).mockRejectedValueOnce(
    Object.assign(new Error('The local session was reset.'), { code: 'history_resync_required' })
  );
  await store.dispatch(requestCopilot('explicit'));
  const attempted = (api.predictCopilot as jest.Mock).mock.calls[2][1] as api.PredictRequest;
  const retried = (api.predictCopilot as jest.Mock).mock.calls[3][1] as api.PredictRequest;
  expect(attempted.history).toEqual([]);
  expect(attempted.history_base).toBe(second.history[0].transaction_id);
  expect(retried.history).toEqual(store.getState().copilot.history);
  expect(retried.history_base).toBeNull();
  expect(retried.request.request_id).not.toBe(attempted.request.request_id);
  const requestMeasurements = store
    .getState()
    .copilot.measurements.filter((m) => m.event === 'request');
  expect(requestMeasurements[3].trace_id).toBe(requestMeasurements[2].trace_id);
  const requests = [first, second, attempted, retried];
  requests.forEach((body) => {
    expect(body.request_transport).toBe('sequenced-v1');
    expect(body.history_transport).toBe('incremental');
    expect(body.request.request_id).toMatch(/^[0-9a-f-]{36}:\d+$/);
    expect(body.request.request_id.split(':')[1]).toBe(String(body.sequence));
  });
  expect(requests.map((body) => body.sequence)).toEqual(
    [...new Set(requests.map((body) => body.sequence))].sort((a, b) => a - b)
  );
  expect(store.getState().copilot.status).toBe('preview');
  expect(store.getState().copilot.error).toBeNull();
});

test('acknowledged sessions retain the 32nd edit and its acceptance undo association', async () => {
  const store = setup();
  const received: api.PredictRequest['history'] = [];
  (api.predictCopilot as jest.Mock).mockImplementation(
    async (_server, body: api.PredictRequest) => {
      received.push(...body.history);
      const last = received[received.length - 1];
      return {
        ...proposal(body),
        history_tip: {
          count: received.length,
          transaction_id: last.transaction_id,
          sequence: last.sequence,
        },
      };
    }
  );
  for (let i = 0; i < 31; i++) {
    store.dispatch(setText({ absoluteIndex: 1, text: `manual ${i}` }));
    await store.dispatch(requestCopilot('explicit'));
  }
  const documentId = store.getState().copilot.documentId;
  const proposalId = store.getState().copilot.proposal?.response.proposal_id;
  if (!proposalId) throw new Error('Missing engineering proposal.');
  store.dispatch(markCopilotShown(proposalId));
  store.dispatch(acceptCopilot());
  await flush();
  expect(store.getState().copilot.documentId).toBe(documentId);
  expect(store.getState().copilot.history).toHaveLength(32);
  const accepted = store.getState().copilot.history[31];
  expect(store.getState().copilot.accepted[accepted.transaction_id].undone).toBe(false);
  store.dispatch(ActionCreators.undo());
  await flush();
  expect(store.getState().copilot.history).toHaveLength(33);
  expect(store.getState().copilot.accepted[accepted.transaction_id].undone).toBe(true);
  expect((api.feedbackCopilot as jest.Mock).mock.calls.slice(-1)[0][1].outcome).toBe('undone');
});

test('a cancelled worker that remains busy retries the unchanged cursor automatically', async () => {
  let finishOld: (response: CopilotResponse) => void = () => {};
  (api.predictCopilot as jest.Mock)
    .mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finishOld = resolve;
        })
    )
    .mockRejectedValueOnce(Object.assign(new Error('Worker still stopping.'), { code: 'busy' }));
  const store = setup();
  store.dispatch(setUserIndex(3));
  jest.advanceTimersByTime(150);
  await flush();
  const first = (api.predictCopilot as jest.Mock).mock.calls[0][1];
  store.dispatch(setUserIndex(1));
  await flush();
  expect(api.cancelCopilot).toHaveBeenCalledTimes(1);
  jest.advanceTimersByTime(150);
  await flush();
  expect(api.predictCopilot).toHaveBeenCalledTimes(2);
  jest.advanceTimersByTime(250);
  await flush();
  expect(api.predictCopilot).toHaveBeenCalledTimes(3);
  const calls = (api.predictCopilot as jest.Mock).mock.calls.map((call) => call[1]);
  expect(calls[2].sequence).toBeGreaterThan(calls[1].sequence);
  expect(calls[2].request.request_id).not.toBe(calls[1].request.request_id);
  const measurements = store.getState().copilot.measurements.filter((m) => m.event === 'request');
  expect(measurements[2].trace_id).toBe(measurements[1].trace_id);
  finishOld(proposal(first));
  await flush();
  expect(store.getState().copilot.proposal?.response.request).toEqual(calls[2].request);
  expect(store.getState().copilot.error).toBeNull();
});

test('busy backoff is bounded and blocking the editor cancels its pending retry', async () => {
  (api.predictCopilot as jest.Mock).mockRejectedValue(
    Object.assign(new Error('Worker busy.'), { code: 'busy' })
  );
  const store = setup();
  store.dispatch(setUserIndex(3));
  jest.advanceTimersByTime(150);
  await flush();
  let calls = 1;
  for (const delay of [250, 500, 1000, 2000, 2000]) {
    jest.advanceTimersByTime(delay - 1);
    await flush();
    expect(api.predictCopilot).toHaveBeenCalledTimes(calls);
    jest.advanceTimersByTime(1);
    await flush();
    expect(api.predictCopilot).toHaveBeenCalledTimes(++calls);
  }
  const originalTrace = store
    .getState()
    .copilot.measurements.find((m) => m.event === 'request')?.trace_id;
  expect(originalTrace).toBeDefined();
  store.dispatch(setCopilotBlocked(true));
  jest.advanceTimersByTime(10000);
  await flush();
  expect(api.predictCopilot).toHaveBeenCalledTimes(calls);
  store.dispatch(setCopilotBlocked(false));
  jest.advanceTimersByTime(150);
  await flush();
  expect(api.predictCopilot).toHaveBeenCalledTimes(calls + 1);
  const latestTrace = store
    .getState()
    .copilot.measurements.filter((m) => m.event === 'request')
    .slice(-1)[0].trace_id;
  expect(latestTrace).not.toBe(originalTrace);
});

test('automatic latency includes synchronous manual-edit preparation before debounce', async () => {
  let now = 100;
  const originalInfer = copilotCore.inspectManualEdit;
  const clock = jest.spyOn(performance, 'now').mockImplementation(() => now);
  const infer = jest.spyOn(copilotCore, 'inspectManualEdit').mockImplementation((before, after) => {
    now += 40;
    return originalInfer(before, after);
  });
  try {
    const store = setup();
    store.dispatch(setText({ absoluteIndex: 1, text: 'edited manually' }));
    now = 290;
    jest.advanceTimersByTime(150);
    await flush();
    const request = store.getState().copilot.measurements.find((m) => m.event === 'request');
    expect(request?.since_trigger_ms).toBe(190);
    now = 320;
    const proposalId = store.getState().copilot.proposal?.response.proposal_id;
    if (!proposalId) throw new Error('Missing engineering proposal.');
    store.dispatch(markCopilotShown(proposalId));
    const shown = store
      .getState()
      .copilot.measurements.find((m) => m.event === 'visible_complete_preview');
    expect(shown?.since_trigger_ms).toBe(220);
  } finally {
    infer.mockRestore();
    clock.mockRestore();
  }
});

test('timings include automatic debounce and delayed visible preview; automation is labelled', async () => {
  let now = 0;
  const clock = jest.spyOn(performance, 'now').mockImplementation(() => now);
  try {
    const store = setup();
    store.dispatch(setCopilotFeedbackOrigin('ai_computer_use'));
    store.dispatch(setText({ absoluteIndex: 1, text: 'helo again' }));
    now = 150;
    jest.advanceTimersByTime(150);
    await flush();
    const response = store.getState().copilot.proposal!.response;
    expect(
      store.getState().copilot.measurements.find((m) => m.event === 'request')?.since_trigger_ms
    ).toBe(150);
    expect(
      store.getState().copilot.measurements.some((m) => m.event === 'visible_complete_preview')
    ).toBe(false);
    now = 300;
    store.dispatch(markCopilotShown(response.proposal_id!));
    const shown = store
      .getState()
      .copilot.measurements.find((m) => m.event === 'visible_complete_preview');
    expect(shown?.since_trigger_ms).toBe(300);
    expect(shown?.since_request_ms).toBe(150);
    await flush();
    expect((api.predictCopilot as jest.Mock).mock.calls[0][1].origin).toBe('ai_computer_use');
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
  jest.advanceTimersByTime(150);
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
  expect(
    store.getState().copilot.events.find((event) => event.outcome === 'accepted')?.persistence
  ).toBe('pending');
  await flush();
  expect(store.getState().copilot.feedbackError).toBe('offline');
  expect(
    store.getState().copilot.events.find((event) => event.outcome === 'accepted')?.persistence
  ).toBe('failed');
  await store.dispatch(requestCopilot('explicit'));
  expect(api.predictCopilot).toHaveBeenCalledTimes(1);
  expect(store.getState().editor.past).toHaveLength(1);
  store.dispatch(setCopilotEnabled(false));
  store.dispatch(setCopilotEnabled(true));
  expect(store.getState().copilot.feedbackError).toBeNull();
});

test('feedback can be retried with the original event without discarding the session or edit history', async () => {
  const store = setup();
  await store.dispatch(requestCopilot('explicit'));
  store.dispatch(markCopilotShown(store.getState().copilot.proposal!.response.proposal_id!));
  await flush();
  (api.feedbackCopilot as jest.Mock).mockRejectedValueOnce(new Error('offline'));
  store.dispatch(acceptCopilot());
  await flush();
  const failed = (api.feedbackCopilot as jest.Mock).mock.calls[1][1];
  expect(store.getState().copilot.feedbackError).toBe('offline');
  await store.dispatch(retryCopilotFeedback());
  expect((api.feedbackCopilot as jest.Mock).mock.calls[2][1]).toEqual(failed);
  expect(store.getState().copilot.feedbackError).toBeNull();
  expect(store.getState().copilot.documentId).toBe('test-document');
  expect(store.getState().copilot.history).toHaveLength(1);
  expect(store.getState().editor.past).toHaveLength(1);
});

test('feedback retry preserves chronology when shown fails before the preview is invalidated', async () => {
  const store = setup();
  await store.dispatch(requestCopilot('explicit'));
  (api.feedbackCopilot as jest.Mock).mockRejectedValueOnce(new Error('offline'));
  store.dispatch(markCopilotShown(store.getState().copilot.proposal!.response.proposal_id!));
  store.dispatch(setUserIndex(3));
  await flush();
  expect(api.feedbackCopilot).toHaveBeenCalledTimes(1);
  expect(store.getState().copilot.events.map((event) => event.persistence)).toEqual([
    'failed',
    'failed',
  ]);
  await store.dispatch(retryCopilotFeedback());
  expect((api.feedbackCopilot as jest.Mock).mock.calls.map((call) => call[1].sequence)).toEqual([
    2, 2, 3,
  ]);
  expect((api.feedbackCopilot as jest.Mock).mock.calls.map((call) => call[1].outcome)).toEqual([
    'shown',
    'shown',
    'invalidated',
  ]);
  expect(store.getState().copilot.feedbackError).toBeNull();
  expect(store.getState().copilot.events.every((event) => event.persistence === 'saved')).toBe(
    true
  );
});

test('opening a document automatically waits for the local model, then requests once without a toggle', async () => {
  (api.copilotHealth as jest.Mock).mockResolvedValueOnce({ status: 'starting' });
  const store = setup(initial(), null);
  expect(store.getState().copilot.enabled).toBe(true);
  store.dispatch({ type: 'editor/openDocumentFromMemory/fulfilled', payload: initial().document });
  await flush();
  jest.advanceTimersByTime(1000);
  await flush();
  expect(api.predictCopilot).not.toHaveBeenCalled();
  await store.dispatch(refreshCopilotHealth());
  jest.advanceTimersByTime(149);
  await flush();
  expect(api.predictCopilot).not.toHaveBeenCalled();
  jest.advanceTimersByTime(1);
  await flush();
  expect(api.predictCopilot).toHaveBeenCalledTimes(1);
  expect((api.predictCopilot as jest.Mock).mock.calls[0][1].trigger).toBe('automatic');
  expect((api.predictCopilot as jest.Mock).mock.calls[0][1].request.document_id).not.toBe('');
  await store.dispatch(refreshCopilotHealth());
  store.dispatch({ type: 'test/server', payload: { servers: [{ ...local }], selectedServer: 0 } });
  jest.advanceTimersByTime(1000);
  await flush();
  expect(api.predictCopilot).toHaveBeenCalledTimes(1);
});

test('cold gateway failure becomes unavailable, and recovery automatically requests once', async () => {
  const store = setup(initial(), null);
  store.dispatch({ type: 'test/server', payload: { servers: [], selectedServer: 0 } });
  store.dispatch(receiveLocalServerInfo({ state: 'failed to start', port: null, token: null }));
  expect(store.getState().server.servers).toEqual([]);
  expect(store.getState().copilot.healthStatus).toBe('unavailable');
  jest.advanceTimersByTime(1000);
  await flush();
  expect(api.copilotHealth).not.toHaveBeenCalled();
  expect(api.predictCopilot).not.toHaveBeenCalled();
  store.dispatch(receiveLocalServerInfo({ state: 'starting', port: 45001, token: null }));
  expect(store.getState().copilot.healthStatus).toBe('starting');
  expect(store.getState().server.servers).toEqual([]);
  store.dispatch(receiveLocalServerInfo({ state: 'running', port: 45001, token: 'ready' }));
  await flush();
  jest.advanceTimersByTime(150);
  await flush();
  expect(store.getState().copilot.healthStatus).toBe('ready');
  expect(api.predictCopilot).toHaveBeenCalledTimes(1);
  expect((api.predictCopilot as jest.Mock).mock.calls[0][1].trigger).toBe('automatic');
});

test('a late healthy response cannot overwrite a stopped gateway lifecycle', async () => {
  let resolve: (value: { status: string }) => void = () => {};
  (api.copilotHealth as jest.Mock).mockImplementationOnce(
    () =>
      new Promise((done) => {
        resolve = done;
      })
  );
  const store = setup(initial(), null);
  const pending = store.dispatch(refreshCopilotHealth());
  store.dispatch(
    receiveLocalServerInfo({ state: 'stopped', port: local.port, token: local.token })
  );
  expect(store.getState().copilot.healthStatus).toBe('unavailable');
  resolve({ status: 'ready' });
  await pending;
  expect(store.getState().copilot.healthStatus).toBe('unavailable');
  jest.advanceTimersByTime(1000);
  await flush();
  expect(api.predictCopilot).not.toHaveBeenCalled();
});

test('cursor movement debounces automatically, cancels old inference and resumes after typing', async () => {
  let resolve: (value: CopilotResponse) => void = () => {};
  let body: api.PredictRequest;
  (api.predictCopilot as jest.Mock).mockImplementationOnce((_server, value) => {
    body = value;
    return new Promise((done) => {
      resolve = done;
    });
  });
  const store = setup();
  store.dispatch(setUserIndex(3));
  jest.advanceTimersByTime(150);
  await flush();
  expect(api.predictCopilot).toHaveBeenCalledTimes(1);
  store.dispatch(setUserIndex(2));
  expect((api.predictCopilot as jest.Mock).mock.calls[0][2].aborted).toBe(true);
  jest.advanceTimersByTime(75);
  store.dispatch(setUserIndex(1));
  store.dispatch(setCopilotBlocked(true));
  jest.advanceTimersByTime(1000);
  await flush();
  resolve(proposal(body!));
  await flush();
  expect(store.getState().copilot.proposal).toBeNull();
  expect(api.predictCopilot).toHaveBeenCalledTimes(1);
  store.dispatch(setCopilotBlocked(false));
  jest.advanceTimersByTime(150);
  await flush();
  expect(api.predictCopilot).toHaveBeenCalledTimes(2);
  expect((api.predictCopilot as jest.Mock).mock.calls[1][1].focus_uuid).toBe('w1');
});

test('a busy model defers automatic inference until ready without requiring another cursor movement', async () => {
  const store = setup(initial(), 'busy');
  store.dispatch(setUserIndex(3));
  jest.advanceTimersByTime(1000);
  await flush();
  expect(api.predictCopilot).not.toHaveBeenCalled();
  await store.dispatch(refreshCopilotHealth());
  jest.advanceTimersByTime(150);
  await flush();
  expect(api.predictCopilot).toHaveBeenCalledTimes(1);
  expect((api.predictCopilot as jest.Mock).mock.calls[0][1].focus_uuid).toBe('w2');
});

test('automatic inference rechecks readiness after its debounce expires', async () => {
  const store = setup();
  store.dispatch(setUserIndex(3));
  jest.advanceTimersByTime(149);
  (api.copilotHealth as jest.Mock).mockResolvedValueOnce({ status: 'busy' });
  await store.dispatch(refreshCopilotHealth());
  jest.advanceTimersByTime(1);
  await flush();
  expect(api.predictCopilot).not.toHaveBeenCalled();
  await store.dispatch(refreshCopilotHealth());
  jest.advanceTimersByTime(150);
  await flush();
  expect(api.predictCopilot).toHaveBeenCalledTimes(1);
});

test('no-action and dismissal do not repeatedly predict the same unchanged context', async () => {
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
  store.dispatch(setUserIndex(3));
  jest.advanceTimersByTime(150);
  await flush();
  store.dispatch(setCopilotBlocked(true));
  store.dispatch(setCopilotBlocked(false));
  await store.dispatch(refreshCopilotHealth());
  jest.advanceTimersByTime(2000);
  await flush();
  expect(api.predictCopilot).toHaveBeenCalledTimes(1);
  store.dispatch(setUserIndex(1));
  jest.advanceTimersByTime(150);
  await flush();
  store.dispatch(markCopilotShown(store.getState().copilot.proposal!.response.proposal_id!));
  store.dispatch(dismissCopilot());
  store.dispatch(setCopilotBlocked(true));
  store.dispatch(setCopilotBlocked(false));
  jest.advanceTimersByTime(2000);
  await flush();
  expect(api.predictCopilot).toHaveBeenCalledTimes(2);
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
  jest.advanceTimersByTime(150);
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
  jest.advanceTimersByTime(150);
  await flush();
  expect(api.predictCopilot).toHaveBeenCalledTimes(1);
  const request = (api.predictCopilot as jest.Mock).mock.calls[0][1];
  expect(request.history).toEqual(store.getState().copilot.history);
  expect(request.document).toEqual(after);
});
