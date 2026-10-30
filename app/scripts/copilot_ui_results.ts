export interface UIObservation {
  request_issued: boolean;
  issued_request_count?: number;
  model_attempted: boolean;
  visible_preview: boolean;
  trigger_to_visible_ms: number | null;
  renderer_response_complete_ms: number | null;
}

function percentiles(values: number[]) {
  values.sort((a, b) => a - b);
  const percentile = (q: number) =>
    values.length ? values[Math.ceil(values.length * q) - 1] : null;
  return {
    n: values.length,
    values,
    p50: percentile(0.5),
    p95: percentile(0.95),
    method: 'nearest_rank',
  };
}

export function summarizeUIObservations(
  planned: number,
  results: UIObservation[]
): {
  planned: number;
  finished_cases: number;
  unfinished_cases: number;
  issued: number;
  model_attempts: number;
  observed_previews: number;
  trigger_to_visible: ReturnType<typeof percentiles>;
  renderer_response_complete: ReturnType<typeof percentiles>;
} {
  const issued = results.filter((r) => r.request_issued);
  const visible = issued.filter((r) => r.visible_preview);
  const valid = (value: number | null): value is number =>
    typeof value === 'number' && Number.isFinite(value) && value >= 0;
  return {
    planned,
    finished_cases: results.length,
    unfinished_cases: planned - results.length,
    issued: issued.reduce((count, row) => count + (row.issued_request_count ?? 1), 0),
    model_attempts: results.filter((r) => r.model_attempted).length,
    observed_previews: visible.length,
    trigger_to_visible: percentiles(visible.map((r) => r.trigger_to_visible_ms).filter(valid)),
    renderer_response_complete: percentiles(
      issued.map((r) => r.renderer_response_complete_ms).filter(valid)
    ),
  };
}
