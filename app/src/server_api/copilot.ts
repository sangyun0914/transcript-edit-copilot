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
  history: CommittedTransition[];
  trigger: 'automatic' | 'explicit';
  deadline_ms: number;
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

export function feedbackCopilot(server: ServerConfig, event: CopilotFeedback): Promise<void> {
  return fetchFromServer(server, 'POST', 'copilot/feedback', undefined, { json: event }).then(
    () => {}
  );
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
