import type { CopilotFeedback } from '../core/copilot';
import { feedbackCopilot } from './copilot';
import { fetchFromServer } from './index';

jest.mock('./index');

test('feedback only succeeds with a matching durable receipt, including after a lost acknowledgement', async () => {
  const server = { hostname: 'http://localhost', port: 8000, token: 'test', name: 'local' };
  const event: CopilotFeedback = {
    event_id: 'event',
    document_id: 'doc',
    request: { document_id: 'doc', request_id: 'request', revision: 0, interaction_epoch: 0 },
    proposal_id: 'proposal',
    outcome: 'shown',
    sequence: 1,
    origin: 'synthetic',
    reason: 'unspecified',
    observed_at: '2026-10-11T01:00:00.000Z',
    revision: 0,
    interaction_epoch: 0,
  };
  const receipt = {
    recorded: true,
    persisted: true,
    document_id: event.document_id,
    proposal_id: event.proposal_id,
    event_id: event.event_id,
    outcome: event.outcome,
    event_hash: 'a'.repeat(64),
    seq: 4,
  };
  const respond = (value: unknown) =>
    (fetchFromServer as jest.Mock).mockResolvedValue({ json: async () => value });
  respond({ recorded: true, proposal_id: event.proposal_id, outcome: event.outcome });
  await expect(feedbackCopilot(server, event)).rejects.toThrow('durably confirmed');
  respond({ ...receipt, event_id: 'another-event' });
  await expect(feedbackCopilot(server, event)).rejects.toThrow('durably confirmed');
  respond(receipt);
  await expect(feedbackCopilot(server, event)).resolves.toBeUndefined();
  expect(fetchFromServer).toHaveBeenLastCalledWith(server, 'POST', 'copilot/feedback', undefined, {
    json: event,
  });
});
