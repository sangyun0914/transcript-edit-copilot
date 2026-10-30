import type { ServerConfig } from '../state/server';
import type { SessionDocument } from '../core/editing_session';
import type {
  CommittedTransition,
  CopilotFeedback,
  CopilotRequestIdentity,
  CopilotResponse,
} from '../core/copilot';
import { fetchFromServer } from './index';

export interface PredictRequest {
  request: CopilotRequestIdentity;
  document: SessionDocument;
  focus_uuid: string;
  selection: string[];
  sequence: number;
  request_transport?: 'sequenced-v1';
  history: CommittedTransition[];
  history_transport?: 'incremental';
  history_base?: string | null;
  trigger: 'automatic' | 'explicit';
  deadline_ms: number;
  origin: CopilotFeedback['origin'];
  observed_at: string;
}

export function predictCopilot(
  server: ServerConfig,
  request: PredictRequest,
  signal: AbortSignal
): Promise<CopilotResponse> {
  return fetchFromServer(
    server,
    'POST',
    'copilot/predict',
    undefined,
    { json: request },
    signal
  ).then((response) => response.json());
}

export async function feedbackCopilot(server: ServerConfig, event: CopilotFeedback): Promise<void> {
  const response = await fetchFromServer(server, 'POST', 'copilot/feedback', undefined, {
    json: event,
  });
  const receipt = await response.json();
  if (
    receipt.recorded !== true ||
    receipt.persisted !== true ||
    receipt.document_id !== event.document_id ||
    receipt.event_id !== event.event_id ||
    receipt.proposal_id !== event.proposal_id ||
    receipt.outcome !== event.outcome ||
    !/^[a-f0-9]{64}$/.test(receipt.event_hash) ||
    !Number.isInteger(receipt.seq) ||
    receipt.seq < 0
  )
    throw new Error('Local copilot feedback was not durably confirmed.');
}

export function cancelCopilot(
  server: ServerConfig,
  request: CopilotRequestIdentity
): Promise<void> {
  return fetchFromServer(server, 'POST', 'copilot/cancel', undefined, { json: request }).then(
    () => {}
  );
}

export function copilotHealth(
  server: ServerConfig
): Promise<{ status: string; message?: string; model?: Record<string, unknown> }> {
  return fetchFromServer(server, 'GET', 'copilot/health').then((response) => response.json());
}
