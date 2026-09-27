import {
  createAction,
  createSelector,
  createSlice,
  Middleware,
  PayloadAction,
} from '@reduxjs/toolkit';
import { v4 as uuidv4 } from 'uuid';
import type { ServerConfig, ServerState } from './server';
import type { EditorState } from './editor/types';
import { Page, NavState } from './nav';
import { currentIndex } from './editor/selectors';
import { exposeReducersWindow } from './util';
import { acceptCopilotTransaction } from './editor/copilot';
import { sessionContext, sessionDocument, sessionHash } from '../core/editing_session';
import {
  CommittedTransition,
  CopilotFeedback,
  CopilotRequestIdentity,
  CopilotResponse,
  inferManualPrediction,
  isTimed,
  sameRequest,
  validCopilotResult,
} from '../core/copilot';
import * as api from '../server_api/copilot';

interface Proposal {
  response: CopilotResponse;
  shown: boolean;
  contextHash: string;
  expiresAt: number;
}

export interface CopilotMeasurement {
  event: string;
  trace_id: string;
  trigger: 'automatic' | 'explicit';
  at_ms: number;
  since_trigger_ms: number;
  since_request_ms?: number;
  request_id?: string;
  outcome?: string;
  timings?: Record<string, number | string>;
}

interface LatencyTrace {
  id: string;
  trigger: 'automatic' | 'explicit';
  started: number;
  requested?: number;
  request_id?: string;
}

export interface CopilotState {
  enabled: boolean;
  blocked: boolean;
  nativeDialog: boolean;
  documentId: string;
  revision: number;
  epoch: number;
  sequence: number;
  status: 'idle' | 'requesting' | 'preview' | 'no_action' | 'suppressed' | 'error';
  request: CopilotRequestIdentity | null;
  proposal: Proposal | null;
  history: CommittedTransition[];
  accepted: Record<string, { proposal: CopilotResponse; undone: boolean }>;
  events: CopilotFeedback[];
  error: string | null;
  feedbackError: string | null;
  health: string | null;
  measurements: CopilotMeasurement[];
  feedbackOrigin: CopilotFeedback['origin'];
}

type CopilotRoot = {
  copilot: CopilotState;
  editor: { present: EditorState | null };
  server: ServerState;
  nav: NavState;
};

const initialState: CopilotState = {
  enabled: false,
  blocked: false,
  nativeDialog: false,
  documentId: '',
  revision: 0,
  epoch: 0,
  sequence: 0,
  status: 'idle',
  request: null,
  proposal: null,
  history: [],
  accepted: {},
  events: [],
  error: null,
  feedbackError: null,
  health: null,
  measurements: [],
  feedbackOrigin: 'human',
};

const slice = createSlice({
  name: 'copilot',
  initialState,
  reducers: {
    measured: (state, action: PayloadAction<CopilotMeasurement>) => {
      state.measurements.push(action.payload);
      if (state.measurements.length > 512) state.measurements.shift();
    },
    clearCopilotMeasurements: (state) => {
      state.measurements = [];
    },
    setCopilotFeedbackOrigin: (state, action: PayloadAction<CopilotFeedback['origin']>) => {
      state.feedbackOrigin = action.payload;
    },
    setCopilotEnabled: (state, action: PayloadAction<boolean>) => {
      state.enabled = action.payload;
    },
    setCopilotBlocked: (state, action: PayloadAction<boolean>) => {
      state.blocked = action.payload;
    },
    nativeDialog: (state, action: PayloadAction<boolean>) => {
      state.nativeDialog = action.payload;
    },
    resetSession: (state, action: PayloadAction<string>) => {
      state.documentId = action.payload;
      state.revision = 0;
      state.epoch++;
      state.sequence = 0;
      state.history = [];
      state.accepted = {};
      state.events = [];
      state.request = null;
      state.proposal = null;
      state.status = 'idle';
      state.error = null;
      state.feedbackError = null;
    },
    observed: (
      state,
      action: PayloadAction<{ changed: boolean; transition?: CommittedTransition }>
    ) => {
      state.epoch++;
      if (action.payload.changed) state.revision++;
      if (action.payload.transition) {
        state.sequence = action.payload.transition.sequence;
        state.history.push(action.payload.transition);
      }
    },
    invalidated: (state) => {
      state.request = null;
      state.proposal = null;
      state.status = 'idle';
    },
    started: (
      state,
      action: PayloadAction<{ request: CopilotRequestIdentity; sequence: number }>
    ) => {
      state.request = action.payload.request;
      state.sequence = action.payload.sequence;
      state.proposal = null;
      state.status = 'requesting';
      state.error = null;
    },
    completed: (
      state,
      action: PayloadAction<{ response: CopilotResponse; contextHash: string; expiresAt: number }>
    ) => {
      if (!state.request || !sameRequest(state.request, action.payload.response.request)) return;
      state.status =
        action.payload.response.status === 'proposal' ? 'preview' : action.payload.response.status;
      state.proposal =
        action.payload.response.status === 'proposal' ? { ...action.payload, shown: false } : null;
    },
    recorded: (state, action: PayloadAction<CopilotFeedback>) => {
      if (state.documentId !== action.payload.document_id) return;
      state.sequence = Math.max(state.sequence, action.payload.sequence);
      state.events.push(action.payload);
      if (state.events.length > 100) state.events.shift();
      if (
        action.payload.outcome === 'shown' &&
        state.proposal?.response.proposal_id === action.payload.proposal_id
      )
        state.proposal.shown = true;
    },
    accepted: (state, action: PayloadAction<{ id: string; proposal: CopilotResponse }>) => {
      state.accepted[action.payload.id] = { proposal: action.payload.proposal, undone: false };
    },
    undone: (state, action: PayloadAction<string>) => {
      if (state.accepted[action.payload]) state.accepted[action.payload].undone = true;
    },
    failed: (state, action: PayloadAction<string>) => {
      state.error = action.payload;
      state.status = 'error';
      state.request = null;
      state.proposal = null;
    },
    feedbackFailed: (state, action: PayloadAction<{ documentId: string; error: string }>) => {
      if (state.documentId === action.payload.documentId)
        state.feedbackError = action.payload.error;
    },
    health: (state, action: PayloadAction<string>) => {
      state.health = action.payload;
    },
  },
});

export default slice.reducer;
export const {
  setCopilotEnabled,
  setCopilotBlocked,
  clearCopilotMeasurements,
  setCopilotFeedbackOrigin,
} = slice.actions;
export const requestCopilot = createAction<'automatic' | 'explicit'>('copilot/request');
export const acceptCopilot = createAction('copilot/accept');
export const dismissCopilot = createAction('copilot/dismiss');
export const markCopilotShown = createAction<string>('copilot/shown');
export const expireCopilot = createAction<string>('copilot/expire');
export const invalidateCopilot = createAction<string>('copilot/invalidate');
exposeReducersWindow({
  setCopilotEnabled,
  setCopilotBlocked,
  requestCopilot,
  acceptCopilot,
  dismissCopilot,
  markCopilotShown,
  clearCopilotMeasurements,
  setCopilotFeedbackOrigin,
});

export function localCopilotServer(state: Pick<CopilotRoot, 'server'>): ServerConfig | undefined {
  const local = (server: ServerConfig) => {
    try {
      return ['localhost', '127.0.0.1', '[::1]'].includes(new URL(server.hostname).hostname);
    } catch {
      return false;
    }
  };
  const selected = state.server.servers[state.server.selectedServer];
  return selected && local(selected) ? selected : state.server.servers.find(local);
}

export function copilotEligible(state: CopilotRoot): boolean {
  const editor = state.editor.present;
  return Boolean(
    state.copilot.enabled &&
      !state.copilot.blocked &&
      !state.copilot.nativeDialog &&
      !state.copilot.feedbackError &&
      state.nav.page === Page.Editor &&
      editor &&
      !editor.playing &&
      editor.transcriptCorrectionState === null &&
      editor.exportPopup === 'hidden' &&
      !editor.filterPopup &&
      editor.document.content.some(isTimed)
  );
}

function validateCurrentProposal(state: CopilotRoot): Proposal | null {
  const proposal = state.copilot.proposal,
    editor = state.editor.present;
  if (
    !proposal ||
    !editor ||
    !copilotEligible(state) ||
    !state.copilot.request ||
    !sameRequest(proposal.response.request, state.copilot.request) ||
    proposal.response.request.document_id !== state.copilot.documentId ||
    proposal.response.request.revision !== state.copilot.revision ||
    proposal.response.request.interaction_epoch !== state.copilot.epoch ||
    proposal.contextHash !== sessionHash(sessionContext(editor)) ||
    !validCopilotResult(sessionDocument(editor.document), proposal.response)
  )
    return null;
  return proposal;
}

const memoizedProposal = createSelector(
  (state: CopilotRoot) => state.copilot,
  (state: CopilotRoot) => state.editor.present,
  (state: CopilotRoot) => state.nav.page,
  (copilot, editor, page) =>
    validateCurrentProposal({
      copilot,
      editor: { present: editor },
      nav: { page },
      server: { servers: [], selectedServer: 0 },
    })
);

export function currentCopilotProposal(state: CopilotRoot): Proposal | null {
  const proposal = memoizedProposal(state);
  return proposal && Date.now() < proposal.expiresAt ? proposal : null;
}

export function copilotKey(
  event: {
    key: string;
    repeat: boolean;
    shiftKey: boolean;
    altKey: boolean;
    ctrlKey: boolean;
    metaKey: boolean;
    isComposing?: boolean;
    keyCode?: number;
    target: EventTarget | null;
  },
  state: CopilotRoot
): 'accept' | 'dismiss' | null {
  const target = event.target instanceof HTMLElement ? event.target : null;
  if (
    event.repeat ||
    event.shiftKey ||
    event.altKey ||
    event.ctrlKey ||
    event.metaKey ||
    event.isComposing ||
    event.keyCode === 229 ||
    !target?.closest('#document') ||
    target.closest('input, textarea, select, button, [contenteditable="true"], [role="dialog"]') ||
    !currentCopilotProposal(state)?.shown ||
    !copilotPreviewVisible()
  )
    return null;
  return event.key === 'Tab' ? 'accept' : event.key === 'Escape' ? 'dismiss' : null;
}

export function copilotPreviewVisible(): boolean {
  const preview = document.querySelector('[data-testid="copilot-preview"]');
  const container = document.getElementById('scroll-container');
  if (!preview || !container) return false;
  const rect = preview.getBoundingClientRect(),
    viewport = container.getBoundingClientRect();
  return (
    rect.width > 0 &&
    rect.height > 0 &&
    rect.bottom > Math.max(0, viewport.top) &&
    rect.top < Math.min(window.innerHeight, viewport.bottom) &&
    rect.right > Math.max(0, viewport.left) &&
    rect.left < Math.min(window.innerWidth, viewport.right)
  );
}

function editorContext(editor: EditorState | null): unknown {
  return editor
    ? {
        ...sessionContext(editor),
        filterPopup: editor.filterPopup,
        exportPopup: editor.exportPopup,
      }
    : null;
}

function latestActive(history: CommittedTransition[]): CommittedTransition | undefined {
  const active: CommittedTransition[] = [];
  for (const transition of history) transition.undo_of ? active.pop() : active.push(transition);
  return active[active.length - 1];
}

export const createCopilotMiddleware =
  (): Middleware<Record<string, never>, CopilotRoot> => (store) => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    let active: {
      controller: AbortController;
      server: ServerConfig;
      request: CopilotRequestIdentity;
      trace: LatencyTrace;
    } | null = null;
    let generation = 0,
      feedbackQueue = Promise.resolve();
    let pendingAutomatic = false;
    let pendingTrace: LatencyTrace | undefined;
    let lastCommitAt: number | undefined;
    const traces = new Map<string, LatencyTrace>();
    const measure = (
      event: string,
      trace: LatencyTrace | undefined,
      extra: Pick<CopilotMeasurement, 'outcome' | 'timings'> = {}
    ) => {
      if (!trace) return;
      const now = performance.now();
      store.dispatch(
        slice.actions.measured({
          event,
          trace_id: trace.id,
          trigger: trace.trigger,
          at_ms: now,
          since_trigger_ms: now - trace.started,
          ...(trace.requested !== undefined && { since_request_ms: now - trace.requested }),
          ...(trace.request_id && { request_id: trace.request_id }),
          ...extra,
        })
      );
    };
    const newTrace = (
      trigger: 'automatic' | 'explicit',
      started = performance.now()
    ): LatencyTrace => {
      const trace = { id: uuidv4(), trigger, started };
      measure('trigger', trace);
      return trace;
    };
    const sequences = new Map<string, number>();
    const cancel = () => {
      generation++;
      pendingAutomatic = false;
      if (timer) clearTimeout(timer);
      timer = undefined;
      measure('debounce_cancelled', pendingTrace);
      pendingTrace = undefined;
      if (active) {
        const trace = active.trace;
        measure('cancel_requested', trace);
        active.controller.abort();
        void api
          .cancelCopilot(active.server, active.request)
          .then(() => measure('cancel_http_ack', trace))
          .catch(() => measure('cancel_http_failed', trace));
        active = null;
      }
    };
    const feedback = (
      response: CopilotResponse,
      outcome: CopilotFeedback['outcome'],
      transition?: CommittedTransition,
      server = localCopilotServer(store.getState())
    ) => {
      if (!response.proposal_id || !server) return;
      measure(
        outcome === 'shown' ? 'visible_complete_preview' : outcome,
        traces.get(response.request.request_id)
      );
      const event: CopilotFeedback = {
        document_id: response.request.document_id,
        proposal_id: response.proposal_id,
        outcome,
        sequence:
          transition?.sequence ??
          Math.max(
            sequences.get(response.request.document_id) || 0,
            store.getState().copilot.documentId === response.request.document_id
              ? store.getState().copilot.sequence
              : 0
          ) + 1,
        origin: store.getState().copilot.feedbackOrigin,
        reason: 'unspecified',
        ...(transition && { transition }),
      };
      sequences.set(event.document_id, event.sequence);
      store.dispatch(slice.actions.recorded(event));
      feedbackQueue = feedbackQueue
        .then(() => api.feedbackCopilot(server, event))
        .catch((error) => {
          store.dispatch(
            slice.actions.feedbackFailed({
              documentId: event.document_id,
              error: String(error.message || error),
            })
          );
        });
    };
    const invalidate = (outcome: 'invalidated' | 'expired' = 'invalidated') => {
      const proposal = store.getState().copilot.proposal;
      if (proposal) feedback(proposal.response, proposal.shown ? outcome : 'invalidated');
      cancel();
      store.dispatch(slice.actions.invalidated());
    };
    const predict = async (trigger: 'automatic' | 'explicit', trace = newTrace(trigger)) => {
      invalidate();
      const ownGeneration = generation;
      await feedbackQueue;
      if (generation !== ownGeneration || !copilotEligible(store.getState())) {
        measure('trigger_suppressed', trace);
        return;
      }
      const state = store.getState(),
        editor = state.editor.present as EditorState,
        server = localCopilotServer(state);
      if (!server) {
        store.dispatch(slice.actions.failed('A local server is required for copilot.'));
        return;
      }
      const timed = editor.document.content.filter(isTimed),
        index = currentIndex(editor);
      const focus = editor.document.content.slice(index).find(isTimed) || timed[timed.length - 1];
      const request = {
        document_id: state.copilot.documentId,
        request_id: uuidv4(),
        revision: state.copilot.revision,
        interaction_epoch: state.copilot.epoch,
      };
      const sequence = state.copilot.sequence + 1;
      sequences.set(request.document_id, sequence);
      const controller = new AbortController();
      trace.requested = performance.now();
      trace.request_id = request.request_id;
      traces.set(request.request_id, trace);
      if (traces.size > 512) traces.delete(traces.keys().next().value);
      measure('request', trace);
      active = { controller, server, request, trace };
      store.dispatch(slice.actions.started({ request, sequence }));
      const timeout = setTimeout(() => {
        if (active?.request.request_id === request.request_id) {
          cancel();
          store.dispatch(slice.actions.failed('Local prediction deadline exceeded.'));
        }
      }, 31000);
      try {
        const response = await api.predictCopilot(
          server,
          {
            request,
            sequence,
            document: sessionDocument(editor.document),
            focus_uuid: focus.uuid,
            selection: editor.selection
              ? editor.document.content
                  .slice(
                    editor.selection.startIndex,
                    editor.selection.startIndex + editor.selection.length
                  )
                  .filter(isTimed)
                  .map((item) => item.uuid)
              : [],
            history: state.copilot.history,
            trigger,
            deadline_ms: 30000,
          },
          controller.signal
        );
        measure('response', trace, { outcome: response.status, timings: response.timings });
        const current = store.getState(),
          present = current.editor.present;
        if (
          generation !== ownGeneration ||
          !current.copilot.request ||
          !sameRequest(request, response.request) ||
          !sameRequest(request, current.copilot.request) ||
          !copilotEligible(current) ||
          !present ||
          response.document_hash !== sessionHash(sessionDocument(present.document))
        ) {
          measure('stale_discard', trace);
          if (response.status === 'proposal') feedback(response, 'invalidated', undefined, server);
          return;
        }
        if (
          response.status === 'proposal' &&
          !validCopilotResult(sessionDocument(present.document), response)
        ) {
          feedback(response, 'invalidated', undefined, server);
          throw new Error('Local prediction failed source validation.');
        }
        if (!['proposal', 'no_action', 'suppressed'].includes(response.status))
          throw new Error('Unknown local prediction response.');
        store.dispatch(
          slice.actions.completed({
            response,
            contextHash: sessionHash(sessionContext(present)),
            expiresAt: Date.now() + 30000,
          })
        );
      } catch (error) {
        measure(controller.signal.aborted ? 'request_settled_after_cancel' : 'error', trace);
        if (generation === ownGeneration && !controller.signal.aborted)
          store.dispatch(slice.actions.failed(String(error.message || error)));
      } finally {
        clearTimeout(timeout);
        if (active?.request.request_id === request.request_id) active = null;
      }
    };
    return (next) => (action) => {
      const before = store.getState(),
        oldEditor = before.editor.present;
      if (acceptCopilotTransaction.type === action.type) {
        const current = currentCopilotProposal(before);
        if (
          !current?.shown ||
          !sameRequest(current.response.request, action.payload.proposal.request)
        )
          return action;
      }
      const result = next(action);
      if (
        /^editor\/(saveDocument|openDocumentFromDisk)\/(pending|fulfilled|rejected)$/.test(
          action.type
        )
      ) {
        invalidate();
        store.dispatch(slice.actions.nativeDialog(action.type.endsWith('/pending')));
        store.dispatch(slice.actions.observed({ changed: false }));
      }
      if (requestCopilot.match(action)) return predict(action.payload);
      if (markCopilotShown.match(action)) {
        const proposal = currentCopilotProposal(store.getState());
        if (proposal && !proposal.shown && proposal.response.proposal_id === action.payload)
          feedback(proposal.response, 'shown');
        return result;
      }
      if (acceptCopilot.match(action)) {
        const proposal = currentCopilotProposal(store.getState());
        if (proposal?.shown)
          store.dispatch(
            acceptCopilotTransaction({
              proposal: proposal.response,
              context_hash: proposal.contextHash,
            })
          );
        return result;
      }
      if (dismissCopilot.match(action)) {
        const proposal = currentCopilotProposal(store.getState());
        if (proposal?.shown) {
          feedback(proposal.response, 'dismissed');
          cancel();
          store.dispatch(slice.actions.invalidated());
        }
        return result;
      }
      if (expireCopilot.match(action)) {
        if (store.getState().copilot.proposal?.response.proposal_id === action.payload)
          invalidate('expired');
        return result;
      }
      if (invalidateCopilot.match(action)) {
        if (store.getState().copilot.proposal?.response.proposal_id === action.payload)
          invalidate();
        return result;
      }
      if (setCopilotEnabled.match(action)) {
        if (before.copilot.enabled === action.payload) return result;
        invalidate();
        store.dispatch(slice.actions.resetSession(uuidv4()));
        const server = localCopilotServer(store.getState());
        if (action.payload && server)
          void api
            .copilotHealth(server)
            .then((health) => store.dispatch(slice.actions.health(health.message || health.status)))
            .catch((error) => store.dispatch(slice.actions.health(String(error.message || error))));
        return result;
      }
      if (setCopilotBlocked.match(action)) {
        if (before.copilot.blocked !== action.payload) {
          const resumeAutomatic = pendingAutomatic && !action.payload;
          invalidate();
          store.dispatch(slice.actions.observed({ changed: false }));
          if (resumeAutomatic && copilotEligible(store.getState())) {
            pendingAutomatic = true;
            pendingTrace = newTrace('automatic', lastCommitAt);
            timer = setTimeout(() => {
              pendingAutomatic = false;
              const trace = pendingTrace;
              pendingTrace = undefined;
              void predict('automatic', trace);
            }, 650);
          }
        }
        return result;
      }
      if (action.type.startsWith('copilot/')) return result;
      const after = store.getState(),
        newEditor = after.editor.present;
      const sessionChanged =
        [
          'editor/openDocumentFromDisk/fulfilled',
          'editor/openDocumentFromMemory/fulfilled',
          'editor/closeDocument/pending',
        ].includes(action.type) ||
        before.server !== after.server ||
        oldEditor?.document.sources !== newEditor?.document.sources;
      if (sessionChanged) {
        lastCommitAt = undefined;
        invalidate();
        store.dispatch(slice.actions.resetSession(uuidv4()));
        return result;
      }
      if (!oldEditor || !newEditor) return result;
      const oldDocument = sessionDocument(oldEditor.document),
        newDocument = sessionDocument(newEditor.document);
      const changed =
        oldEditor.document !== newEditor.document &&
        sessionHash(oldDocument) !== sessionHash(newDocument);
      const contextChanged =
        sessionHash(editorContext(oldEditor)) !== sessionHash(editorContext(newEditor)) ||
        before.nav.page !== after.nav.page;
      if (!changed && !contextChanged) return result;
      if (changed) lastCommitAt = performance.now();
      let transition: CommittedTransition | undefined;
      const proposal = before.copilot.proposal;
      let acceptedProposal: CopilotResponse | undefined;
      let undoneId: string | undefined;
      if (changed && before.copilot.enabled) {
        let prediction: CommittedTransition['prediction'] | null = null,
          undoOf: string | null = null;
        if (action.type === acceptCopilotTransaction.type && proposal) {
          prediction = proposal.response.prediction;
          acceptedProposal = proposal.response;
        } else if (action.type === '@@redux-undo/UNDO') {
          const original = latestActive(before.copilot.history);
          if (
            original &&
            sessionHash(original.after) === sessionHash(oldDocument) &&
            sessionHash(original.before) === sessionHash(newDocument)
          ) {
            prediction = original.prediction;
            undoOf = original.transaction_id;
            if (before.copilot.accepted[undoOf] && !before.copilot.accepted[undoOf].undone)
              undoneId = undoOf;
          }
        } else prediction = inferManualPrediction(oldDocument, newDocument);
        if (prediction)
          transition = {
            sequence: before.copilot.sequence + 1,
            origin: store.getState().copilot.feedbackOrigin,
            transaction_id: uuidv4(),
            before: oldDocument,
            after: newDocument,
            prediction,
            undo_of: undoOf,
          };
      }
      if (acceptedProposal && transition) {
        cancel();
        store.dispatch(slice.actions.invalidated());
        store.dispatch(slice.actions.observed({ changed, transition }));
        store.dispatch(
          slice.actions.accepted({ id: transition.transaction_id, proposal: acceptedProposal })
        );
        feedback(acceptedProposal, 'accepted', transition);
      } else {
        invalidate();
        if (transition) transition.sequence = store.getState().copilot.sequence + 1;
        store.dispatch(slice.actions.observed({ changed, transition }));
        if (undoneId && transition) {
          feedback(before.copilot.accepted[undoneId].proposal, 'undone', transition);
          store.dispatch(slice.actions.undone(undoneId));
        }
      }
      if (changed && (!transition || store.getState().copilot.history.length >= 32))
        store.dispatch(slice.actions.resetSession(uuidv4()));
      if (
        changed &&
        store.getState().copilot.enabled &&
        action.type !== '@@redux-undo/UNDO' &&
        action.type !== '@@redux-undo/REDO'
      ) {
        pendingAutomatic = true;
        if (copilotEligible(store.getState())) {
          pendingTrace = newTrace('automatic', lastCommitAt);
          timer = setTimeout(() => {
            pendingAutomatic = false;
            const trace = pendingTrace;
            pendingTrace = undefined;
            void predict('automatic', trace);
          }, 650);
        }
      }
      return result;
    };
  };

export const copilotMiddleware = createCopilotMiddleware();
