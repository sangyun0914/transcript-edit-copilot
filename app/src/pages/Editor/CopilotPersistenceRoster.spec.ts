import { summarizeUIObservations } from '../../../scripts/copilot_ui_results';

test('abstention, suppression, errors and unissued hydration failures never become zero-latency previews', () => {
  const absent = {
    request_issued: true,
    model_attempted: true,
    visible_preview: false,
    trigger_to_visible_ms: null,
    renderer_response_complete_ms: null,
  };
  const summary = summarizeUIObservations(6, [
    {
      ...absent,
      visible_preview: true,
      trigger_to_visible_ms: 120,
      renderer_response_complete_ms: 100,
    },
    { ...absent, renderer_response_complete_ms: 70 },
    { ...absent, renderer_response_complete_ms: 80 },
    { ...absent, issued_request_count: 2 },
    { ...absent, request_issued: false, model_attempted: false },
  ]);
  expect(summary).toMatchObject({
    planned: 6,
    finished_cases: 5,
    unfinished_cases: 1,
    issued: 5,
    model_attempts: 4,
    observed_previews: 1,
    trigger_to_visible: { n: 1, values: [120], p50: 120, p95: 120 },
    renderer_response_complete: { n: 3, values: [70, 80, 100], p50: 80, p95: 100 },
  });
  expect(summarizeUIObservations(1, [absent]).trigger_to_visible).toMatchObject({
    n: 0,
    p50: null,
    p95: null,
  });
});
