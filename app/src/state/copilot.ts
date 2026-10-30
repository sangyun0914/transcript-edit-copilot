import {
  createAction,
  createSelector,
  createSlice,
  Middleware,
  PayloadAction,
} from '@reduxjs/toolkit';
import { v4 as uuidv4 } from 'uuid';
import type { ServerConfig, ServerState } from './server';
import { localServerStatus } from './server';
import type { EditorState } from './editor/types';
import { Page, NavState } from './nav';
import { currentIndex } from './editor/selectors';
import { exposeReducersWindow } from './util';
import { acceptCopilotTransaction } from './editor/copilot';
import { sessionContext, sessionDocument, sessionHash } from '../core/editing_session';
import {
  CommittedTransition,
  CopilotFeedback,
  CopilotHistoryTip,
  CopilotRequestIdentity,
  CopilotResponse,
  inspectManualEdit,
  isTimed,
  sameRequest,
  validCopilotResult,
} from '../core/copilot';
import * as api from '../server_api/copilot';
import type { Document } from '../core/document';

const AUTOMATIC_DEBOUNCE_MS = 150;
const BUSY_RETRY_INITIAL_MS = 250;
const BUSY_RETRY_MAX_MS = 2000;

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
  events: (CopilotFeedback & { persistence: 'pending' | 'saved' | 'failed' })[];
  error: string | null;
  feedbackError: string | null;
  health: string | null;
  healthStatus: string | null;
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
  enabled: true,
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
  healthStatus: null,
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
      state.events.push({ ...action.payload, persistence: 'pending' });
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
    feedbackSettled: (
      state,
      action: PayloadAction<{ eventId: string; persistence: 'saved' | 'failed' }>
    ) => {
      const event = state.events.find((item) => item.event_id === action.payload.eventId);
      if (event) event.persistence = action.payload.persistence;
    },
    health: (state, action: PayloadAction<{ status: string | null; message?: string }>) => {
      state.health = action.payload.message || null;
      state.healthStatus = action.payload.status;
    },
    feedbackRecovered: (state) => {
      if (state.events.every((event) => event.persistence === 'saved')) state.feedbackError = null;
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
export const refreshCopilotHealth = createAction('copilot/refreshHealth');
export const retryCopilotFeedback = createAction('copilot/retryFeedback');
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

// Redux snapshots retain these references across metrics and feedback updates.
const validProposalResult = createSelector(
  (document: Document) => document.content,
  (document: Document) => document.metadata.transcription,
  (document: Document) => document.metadata.transcriptions,
  (_document: Document, response: CopilotResponse) => response,
  (content, transcription, transcriptions, response) =>
    validCopilotResult(
      {
        content,
        metadata: {
          ...(transcription && { transcription }),
          ...(transcriptions && { transcriptions }),
        },
      },
      response
    )
);

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
    !validProposalResult(editor.document, proposal.response)
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

function serverIdentity(state: CopilotRoot): string {
  const server = localCopilotServer(state);
  return server
    ? sessionHash([server.hostname, server.port, server.token, server.state ?? null])
    : '';
}

function automaticContext(state: CopilotRoot): string {
  return sessionHash({
    document: state.copilot.documentId,
    revision: state.copilot.revision,
    cursor: state.editor.present?.cursor,
    selection: state.editor.present?.selection,
    server: serverIdentity(state),
  });
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
    let lastAttempt: string | undefined;
    let healthRequest: Promise<void> | undefined;
    const failedFeedbackDocuments = new Set<string>();
    const traces = new Map<string, LatencyTrace>();
    let acknowledgedHistory: { documentId: string; tip: CopilotHistoryTip } | undefined;
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
    const cancel = (reason = 'context_changed') => {
      generation++;
      pendingAutomatic = false;
      if (timer) clearTimeout(timer);
      timer = undefined;
      measure('debounce_cancelled', pendingTrace);
      pendingTrace = undefined;
      if (active) {
        const trace = active.trace;
        measure('cancel_requested', trace, { outcome: reason });
        lastAttempt = undefined;
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
      const current = store.getState().copilot;
      const sameDocument = current.documentId === response.request.document_id;
      const event: CopilotFeedback = {
        event_id: uuidv4(),
        document_id: response.request.document_id,
        request: response.request,
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
        observed_at: new Date().toISOString(),
        revision: sameDocument ? current.revision : null,
        interaction_epoch: sameDocument ? current.epoch : null,
        ...(transition && { transition }),
      };
      sequences.set(event.document_id, event.sequence);
      store.dispatch(slice.actions.recorded(event));
      feedbackQueue = feedbackQueue
        .then(async () => {
          if (failedFeedbackDocuments.has(event.document_id)) {
            store.dispatch(
              slice.actions.feedbackSettled({ eventId: event.event_id, persistence: 'failed' })
            );
            return;
          }
          await api.feedbackCopilot(server, event);
          store.dispatch(
            slice.actions.feedbackSettled({ eventId: event.event_id, persistence: 'saved' })
          );
        })
        .catch((error) => {
          failedFeedbackDocuments.add(event.document_id);
          store.dispatch(
            slice.actions.feedbackSettled({ eventId: event.event_id, persistence: 'failed' })
          );
          store.dispatch(
            slice.actions.feedbackFailed({
              documentId: event.document_id,
              error: String(error.message || error),
            })
          );
        });
    };
    const invalidate = (
      outcome: 'invalidated' | 'expired' = 'invalidated',
      reason = 'context_changed'
    ) => {
      const proposal = store.getState().copilot.proposal;
      if (proposal) {
        measure('preview_invalidated', traces.get(proposal.response.request.request_id), {
          outcome: reason,
        });
        feedback(proposal.response, proposal.shown ? outcome : 'invalidated');
      }
      cancel(reason);
      store.dispatch(slice.actions.invalidated());
    };
    const predict = async (
      trigger: 'automatic' | 'explicit',
      trace = newTrace(trigger),
      resynced = false,
      busyRetries = 0
    ): Promise<void> => {
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
      if (
        trigger === 'automatic' &&
        (state.copilot.healthStatus !== 'ready' || (server.state && server.state !== 'running'))
      ) {
        pendingAutomatic = true;
        pendingTrace = trace;
        measure('trigger_suppressed', trace, { outcome: 'model_not_ready' });
        return;
      }
      lastAttempt = automaticContext(state);
      const timed = editor.document.content.filter(isTimed),
        index = currentIndex(editor);
      const focus = editor.document.content.slice(index).find(isTimed) || timed[timed.length - 1];
      const sequence = state.copilot.sequence + 1;
      const request = {
        document_id: state.copilot.documentId,
        request_id: `${uuidv4()}:${sequence}`,
        revision: state.copilot.revision,
        interaction_epoch: state.copilot.epoch,
      };
      sequences.set(request.document_id, sequence);
      const controller = new AbortController();
      trace = { ...trace, requested: performance.now(), request_id: request.request_id };
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
        const history = state.copilot.history;
        const acknowledged =
          acknowledgedHistory?.documentId === state.copilot.documentId
            ? acknowledgedHistory.tip
            : undefined;
        const base =
          acknowledged &&
          acknowledged.count > 0 &&
          history[acknowledged.count - 1]?.transaction_id === acknowledged.transaction_id
            ? acknowledged
            : undefined;
        const response = await api.predictCopilot(
          server,
          {
            request,
            request_transport: 'sequenced-v1',
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
            history: base ? history.slice(base.count) : history,
            history_transport: 'incremental',
            history_base: base?.transaction_id ?? null,
            trigger,
            deadline_ms: 30000,
            origin: state.copilot.feedbackOrigin,
            observed_at: new Date().toISOString(),
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
          (response.status !== 'proposal' &&
            response.document_hash !== sessionHash(sessionDocument(present.document)))
        ) {
          measure('stale_discard', trace);
          if (response.status === 'proposal') feedback(response, 'invalidated', undefined, server);
          return;
        }
        if (response.status === 'proposal' && !validProposalResult(present.document, response)) {
          feedback(response, 'invalidated', undefined, server);
          throw new Error('Local prediction failed source validation.');
        }
        if (!['proposal', 'no_action', 'suppressed'].includes(response.status))
          throw new Error('Unknown local prediction response.');
        if (response.history_tip) {
          const tip = response.history_tip,
            last = history[history.length - 1];
          if (
            tip.count !== history.length ||
            tip.transaction_id !== (last?.transaction_id ?? null) ||
            tip.sequence !== (last?.sequence ?? 0)
          )
            throw new Error('Local prediction returned an inconsistent history acknowledgement.');
          acknowledgedHistory = { documentId: state.copilot.documentId, tip };
        }
        store.dispatch(
          slice.actions.completed({
            response,
            contextHash: sessionHash(sessionContext(present)),
            expiresAt: Date.now() + 30000,
          })
        );
      } catch (error) {
        if (
          error.code === 'history_resync_required' &&
          !resynced &&
          generation === ownGeneration &&
          !controller.signal.aborted
        ) {
          acknowledgedHistory = undefined;
          measure('history_resync', trace);
          active = null;
          await predict(trigger, trace, true, busyRetries);
          return;
        }
        if (
          error.code === 'busy' &&
          generation === ownGeneration &&
          !controller.signal.aborted &&
          copilotEligible(store.getState())
        ) {
          const delay = Math.min(
            BUSY_RETRY_INITIAL_MS * 2 ** Math.min(busyRetries, 3),
            BUSY_RETRY_MAX_MS
          );
          measure('busy_retry', trace, { timings: { retry_delay_ms: delay } });
          lastAttempt = undefined;
          active = null;
          pendingAutomatic = trigger === 'automatic';
          pendingTrace = trace;
          timer = setTimeout(() => {
            timer = undefined;
            pendingAutomatic = false;
            pendingTrace = undefined;
            if (generation === ownGeneration && copilotEligible(store.getState()))
              void predict(trigger, trace, resynced, busyRetries + 1);
          }, delay);
          return;
        }
        measure(controller.signal.aborted ? 'request_settled_after_cancel' : 'error', trace);
        if (generation === ownGeneration && !controller.signal.aborted)
          store.dispatch(slice.actions.failed(String(error.message || error)));
      } finally {
        clearTimeout(timeout);
        if (active?.request.request_id === request.request_id) active = null;
      }
    };
    const scheduleAutomatic = (started?: number) => {
      const state = store.getState();
      if (!state.copilot.enabled || !state.editor.present || state.nav.page !== Page.Editor) return;
      if (lastAttempt === automaticContext(state)) return;
      pendingAutomatic = true;
      if (!copilotEligible(state)) return;
      const server = localCopilotServer(state);
      if (!server || (server.state && server.state !== 'running')) return;
      if (!pendingTrace) pendingTrace = newTrace('automatic', started);
      if (state.copilot.healthStatus !== 'ready') {
        if (state.copilot.healthStatus === null) void checkHealth();
        return;
      }
      if (timer) return;
      timer = setTimeout(() => {
        timer = undefined;
        pendingAutomatic = false;
        const trace = pendingTrace;
        pendingTrace = undefined;
        void predict('automatic', trace);
      }, AUTOMATIC_DEBOUNCE_MS);
    };
    const checkHealth = (): Promise<void> => {
      const state = store.getState(),
        server = localCopilotServer(state),
        identity = serverIdentity(state);
      if (!server || (server.state && server.state !== 'running')) {
        const unavailable = server
          ? !['not started', 'starting'].includes(server.state as string)
          : localServerStatus(state.server) === 'unavailable';
        store.dispatch(
          slice.actions.health({
            status: unavailable ? 'unavailable' : 'starting',
            message: unavailable
              ? 'The local server could not start or has stopped. Restart the app to try again.'
              : 'The local server is starting.',
          })
        );
        return Promise.resolve();
      }
      if (healthRequest) return healthRequest;
      healthRequest = api
        .copilotHealth(server)
        .then((health) => {
          if (serverIdentity(store.getState()) !== identity) return;
          store.dispatch(slice.actions.health(health));
          if (pendingAutomatic && health.status === 'ready') scheduleAutomatic();
        })
        .catch((error) => {
          if (serverIdentity(store.getState()) !== identity) return;
          store.dispatch(
            slice.actions.health({ status: 'unavailable', message: String(error.message || error) })
          );
        })
        .finally(() => {
          healthRequest = undefined;
        });
      return healthRequest;
    };
    return (next) => (action) => {
      const interactionStarted = performance.now(),
        before = store.getState(),
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
        if (store.getState().copilot.proposal) lastAttempt = undefined;
        invalidate('invalidated', action.type);
        store.dispatch(slice.actions.nativeDialog(action.type.endsWith('/pending')));
        store.dispatch(slice.actions.observed({ changed: false }));
        scheduleAutomatic(interactionStarted);
      }
      if (refreshCopilotHealth.match(action)) return checkHealth();
      if (retryCopilotFeedback.match(action)) {
        const state = store.getState(),
          server = localCopilotServer(state),
          documentId = state.copilot.documentId;
        if (!server || !state.copilot.feedbackError) return result;
        feedbackQueue = feedbackQueue.then(async () => {
          const failed = store
            .getState()
            .copilot.events.filter(
              (event) => event.document_id === documentId && event.persistence === 'failed'
            );
          for (const { persistence: _persistence, ...event } of failed) {
            try {
              await api.feedbackCopilot(server, event);
              store.dispatch(
                slice.actions.feedbackSettled({ eventId: event.event_id, persistence: 'saved' })
              );
            } catch (error) {
              store.dispatch(
                slice.actions.feedbackFailed({ documentId, error: String(error.message || error) })
              );
              return;
            }
          }
          failedFeedbackDocuments.delete(documentId);
          if (store.getState().copilot.documentId === documentId) {
            store.dispatch(slice.actions.feedbackRecovered());
            scheduleAutomatic(interactionStarted);
          }
        });
        return feedbackQueue;
      }
      if (requestCopilot.match(action))
        return predict(action.payload, newTrace(action.payload, interactionStarted));
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
          invalidate('invalidated', 'preview_not_visible');
        return result;
      }
      if (setCopilotEnabled.match(action)) {
        if (before.copilot.enabled === action.payload) return result;
        invalidate();
        store.dispatch(slice.actions.resetSession(uuidv4()));
        lastAttempt = undefined;
        if (action.payload) {
          void checkHealth();
          scheduleAutomatic(interactionStarted);
        }
        return result;
      }
      if (setCopilotBlocked.match(action)) {
        if (before.copilot.blocked !== action.payload) {
          if (before.copilot.proposal) lastAttempt = undefined;
          invalidate('invalidated', action.payload ? 'editor_blocked' : 'editor_resumed');
          store.dispatch(slice.actions.observed({ changed: false }));
          scheduleAutomatic(interactionStarted);
        }
        return result;
      }
      if (action.type.startsWith('copilot/')) return result;
      const after = store.getState(),
        newEditor = after.editor.present;
      const serverChanged = serverIdentity(before) !== serverIdentity(after);
      const sessionChanged =
        [
          'editor/openDocumentFromDisk/fulfilled',
          'editor/openDocumentFromMemory/fulfilled',
          'editor/closeDocument/pending',
        ].includes(action.type) ||
        serverChanged ||
        oldEditor?.document.sources !== newEditor?.document.sources;
      if (sessionChanged) {
        lastAttempt = undefined;
        invalidate('invalidated', action.type);
        store.dispatch(slice.actions.resetSession(uuidv4()));
        if (serverChanged) {
          store.dispatch(slice.actions.health({ status: null }));
          void checkHealth();
        }
        scheduleAutomatic(interactionStarted);
        return result;
      }
      if (before.server !== after.server) {
        void checkHealth();
        scheduleAutomatic(interactionStarted);
      }
      if (!oldEditor || !newEditor) return result;
      const oldDocument = sessionDocument(oldEditor.document),
        newDocument = sessionDocument(newEditor.document);
      const documentChanged = oldEditor.document !== newEditor.document;
      const manualEdit =
        documentChanged &&
        before.copilot.enabled &&
        action.type !== acceptCopilotTransaction.type &&
        action.type !== '@@redux-undo/UNDO'
          ? inspectManualEdit(oldDocument, newDocument)
          : null;
      const changed =
        documentChanged &&
        (manualEdit ? manualEdit.changed : sessionHash(oldDocument) !== sessionHash(newDocument));
      const contextChanged =
        sessionHash(editorContext(oldEditor)) !== sessionHash(editorContext(newEditor)) ||
        before.nav.page !== after.nav.page;
      if (!changed && !contextChanged) return result;
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
        } else prediction = manualEdit?.prediction ?? null;
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
        if (!copilotEligible(after) && proposal) lastAttempt = undefined;
        invalidate('invalidated', action.type);
        if (transition) transition.sequence = store.getState().copilot.sequence + 1;
        store.dispatch(slice.actions.observed({ changed, transition }));
        if (undoneId && transition) {
          feedback(before.copilot.accepted[undoneId].proposal, 'undone', transition);
          store.dispatch(slice.actions.undone(undoneId));
        }
      }
      if (changed && !transition) store.dispatch(slice.actions.resetSession(uuidv4()));
      scheduleAutomatic(interactionStarted);
      return result;
    };
  };

export const copilotMiddleware = createCopilotMiddleware();
