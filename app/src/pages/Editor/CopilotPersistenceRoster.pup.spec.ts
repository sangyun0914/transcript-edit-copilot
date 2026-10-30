/** @jest-environment ./scripts/copilot_ui_environment.cjs */

import { spawn, ChildProcessWithoutNullStreams } from 'child_process';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { createInterface } from 'readline';
import { getPortPromise } from 'portfinder';
import { contentToVtt } from '../../core/webvtt';
import type { hydrateCopilotFixture } from '../../../scripts/copilot_ui_fixture';
import { summarizeUIObservations, UIObservation } from '../../../scripts/copilot_ui_results';

declare const copilotUIOutput: string;
declare const copilotUITemp: string;

jest.setTimeout(20 * 60 * 1000);

const mode = process.env.COPILOT_UI_MODE;
const actual = mode === 'roster-actual';
const enabled = actual || mode === 'roster-scripted';
const rosterPath = path.resolve(
  process.env.COPILOT_UI_ROSTER || '../artifacts/ui/passive-actions-v3/registered-roster-001'
);
const rosterSha =
  process.env.COPILOT_UI_ROSTER_SHA ||
  '55b4e22f2c27c12ce59baa7f3e21e973b5b3e4cf6f26124f225eacf1c9da9663';
const protocolPath = path.resolve(
  '../artifacts/ui/passive-actions-v3/registered-latency-protocol-001.json'
);
const protocolSha = 'fa603615e1ea49892d3f4dd9f920eec3667b90dff8672014b4fc442f0ef7edbb';
const token = 'isolated-synthetic-ui-test';
const results: (Record<string, any> & UIObservation)[] = [];
const network: Record<string, any>[] = [];
const pageErrors: string[] = [];
const pending = new Map<number, (result: any) => void>();
let bridge: ChildProcessWithoutNullStreams;
let port: number;
let counter = 0;
let identity: any;
let bridgeError = '';

const hash = (data: string | Buffer) => crypto.createHash('sha256').update(data).digest('hex');
const write = (name: string, value: unknown) =>
  fs.writeFileSync(path.join(copilotUIOutput, name), JSON.stringify(value, null, 2) + '\n');

function bindings(): any[] {
  const directory = path.join(copilotUIOutput, 'bridge');
  if (!fs.existsSync(directory)) return [];
  return fs
    .readdirSync(directory)
    .filter((name) => /^bindings-.*\.jsonl$/.test(name))
    .flatMap((name) =>
      fs
        .readFileSync(path.join(directory, name), 'utf8')
        .trim()
        .split('\n')
        .filter(Boolean)
        .map((line) => JSON.parse(line))
    );
}

function exchange(message: Record<string, unknown>): Promise<any> {
  return new Promise((resolve) => {
    const id = ++counter;
    pending.set(id, resolve);
    bridge.stdin.write(JSON.stringify({ id, ...message }) + '\n');
  });
}

const state = () =>
  page.evaluate(() => {
    const editor = window.store.getState().editor.present;
    if (!editor) throw new Error('No document.');
    const digest = (window as any).require('crypto');
    const metadata: Record<string, unknown> = {};
    if (editor.document.metadata.transcription)
      metadata.transcription = editor.document.metadata.transcription;
    if (editor.document.metadata.transcriptions)
      metadata.transcriptions = editor.document.metadata.transcriptions;
    return {
      document: { content: editor.document.content, metadata },
      sourceHashes: Object.fromEntries(
        Object.keys(editor.document.sources).map((id) => [
          id,
          digest
            .createHash('sha256')
            .update(new Uint8Array(editor.document.sources[id].fileContents))
            .digest('hex'),
        ])
      ),
      copilot: window.store.getState().copilot,
      timeOrigin: performance.timeOrigin,
      observedAt: performance.now(),
    };
  });

beforeAll(async () => {
  if (!enabled) return;
  if (actual && !process.env.COPILOT_UI_DEPLOYMENT)
    throw new Error('Select an explicit verified deployment before actual UI inference.');
  if (hash(fs.readFileSync(path.join(rosterPath, 'manifest.json'))) !== rosterSha)
    throw new Error('Registered roster manifest changed.');
  if (hash(fs.readFileSync(protocolPath)) !== protocolSha)
    throw new Error('Registered latency protocol changed.');
  port = await getPortPromise({ host: '127.0.0.1', port: 16000 });
  const args = [
    '-u',
    path.resolve('../server/tests/copilot_p6_bridge.py'),
    '--feedback-directory',
    path.join(copilotUITemp, 'feedback'),
    '--output',
    path.join(copilotUIOutput, 'bridge'),
    '--http-port',
    String(port),
    '--roster',
    rosterPath,
    '--roster-sha256',
    rosterSha,
  ];
  if (actual) args.push('--deployment', path.resolve(String(process.env.COPILOT_UI_DEPLOYMENT)));
  else
    args.push(
      '--native-fixture-tokenizer',
      path.resolve('../artifacts/model-preparation/gemma26-v1/.venv/bin/python')
    );
  if (process.env.COPILOT_UI_FORMAT_ROSTER)
    args.push(
      '--format-roster',
      path.resolve(process.env.COPILOT_UI_FORMAT_ROSTER),
      '--format-roster-sha256',
      String(process.env.COPILOT_UI_FORMAT_ROSTER_SHA)
    );
  bridge = spawn(path.resolve('../server/.venv/bin/python'), args);
  bridge.stderr.on('data', (data) => {
    bridgeError += data.toString();
    fs.appendFileSync(path.join(copilotUIOutput, 'bridge-stderr.log'), data);
  });
  const lines = createInterface({ input: bridge.stdout });
  lines.on('line', (line) => {
    const result = JSON.parse(line);
    pending.get(result.id)?.(result);
    pending.delete(result.id);
  });
  bridge.on('exit', (code) => {
    for (const resolve of pending.values())
      resolve({ status: 500, body: { detail: `Bridge exit ${code}: ${bridgeError}` } });
    pending.clear();
    lines.close();
  });
  identity = await exchange({ command: 'info' });
  write('identity.json', identity);
  expect(identity.status).toBe(200);
  if (actual) expect(identity.body.runtime_evidence_enabled).toBe(false);
  await page.waitForFunction(() => !!window.store);
  page.on('pageerror', (error) => pageErrors.push(error.message));
  const cdp = await page.target().createCDPSession();
  await cdp.send('Network.enable');
  const observed = new Set<string>();
  cdp.on('Network.requestWillBeSent', (event: any) => {
    if (new URL(event.request.url).pathname.replace(/\/$/, '') !== '/copilot/predict') return;
    observed.add(event.requestId);
    network.push({ kind: 'request', ...event });
  });
  for (const [name, kind] of [
    ['Network.responseReceived', 'response_headers'],
    ['Network.loadingFinished', 'body_complete'],
    ['Network.loadingFailed', 'network_failure'],
  ])
    cdp.on(name, (event: any) => {
      if (observed.has(event.requestId)) network.push({ kind, ...event });
    });
  await page.addScriptTag({
    type: 'module',
    content: `import { hydrateCopilotFixture, validateCopilotFixtureDocument } from ${JSON.stringify(
      '/@fs/' + path.resolve('scripts/copilot_ui_fixture.ts')
    )}; window.p6HydrateFixture = hydrateCopilotFixture; window.p6ValidateFixture = validateCopilotFixtureDocument;`,
  });
  await page.waitForFunction(() => typeof (window as any).p6HydrateFixture === 'function');
  await page.evaluate(
    ({ token, port }) => {
      localStorage.setItem('intro/editor', 'true');
      window.store.dispatch({
        type: 'server/setLocalServer',
        payload: {
          hostname: 'http://127.0.0.1',
          port,
          token,
          state: 'running',
        },
      });
      window.store.dispatch(window.reducers.setCopilotFeedbackOrigin('synthetic'));
    },
    { token, port }
  );
});

async function hydrate(fixture: any, media: any) {
  return page.evaluate(
    ({ fixture, media }) => {
      (window as any).p6ValidateFixture(fixture.document.content);
      window.store.dispatch(window.reducers.setCopilotEnabled(false));
      document.getElementById('copilot-roster-inspection')?.remove();
      const fs = (window as any).require('fs');
      const crypto = (window as any).require('crypto');
      const bytes = media.path ? fs.readFileSync(media.path) : new Uint8Array(0);
      const digest = crypto.createHash('sha256').update(bytes).digest('hex');
      if (media.path && digest !== media.sha256) throw new Error('Renderer source bytes differ.');
      const buffer = new Uint8Array(bytes).buffer;
      const sources: Record<string, any> = {};
      for (const id of media.source_ids)
        sources[id] = {
          fileContents: buffer,
          objectUrl: media.path ? URL.createObjectURL(new Blob([buffer])) : '',
        };
      const metadata = Object.assign(
        { display_video: false, display_speaker_names: true },
        fixture.document.metadata
      );
      return window.store
        .dispatch(
          window.reducers.openDocumentFromMemory({
            content: fixture.document.content,
            metadata,
            sources,
          })
        )
        .then(() => {
          const content = fixture.document.content;
          const focus = content.findIndex((item: any) => item.uuid === fixture.runtime.focus_uuid);
          if (focus < 0) throw new Error('Fixture focus is absent.');
          window.store.dispatch(window.reducers.setSelection(null));
          window.store.dispatch(window.reducers.setUserIndex(focus));
          if (fixture.runtime.selection.length) {
            const indices = fixture.runtime.selection.map((id: string) =>
              content.findIndex((item: any) => item.uuid === id)
            );
            const startIndex = Math.min.apply(null, indices),
              end = Math.max.apply(null, indices);
            window.store.dispatch(
              window.reducers.setSelection({
                startIndex,
                length: end - startIndex + 1,
                headPosition: focus === startIndex ? 'left' : 'right',
              })
            );
          }
          window.store.dispatch(window.reducers.setCopilotEnabled(true));
          window.store.dispatch(window.reducers.setCopilotBlocked(false));
          const hydrate: typeof hydrateCopilotFixture = (window as any).p6HydrateFixture;
          hydrate(fixture);
          window.store.dispatch(window.reducers.clearCopilotMeasurements());
          document.getElementById('document')?.focus();
          return { source_sha256: digest, bytes: buffer.byteLength };
        });
    },
    { fixture, media }
  );
}

async function exportSession(documentId: string, name: string) {
  await page.waitForFunction(() =>
    window.store.getState().copilot.events.every((e) => e.persistence !== 'pending')
  );
  const result = await page.evaluate(
    ({ documentId, port, token }) =>
      fetch(
        `http://127.0.0.1:${port}/copilot/feedback/export?document_id=${encodeURIComponent(
          documentId
        )}`,
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      ).then((response) =>
        response.text().then((text) => ({
          status: response.status,
          text,
          head: response.headers.get('X-Copilot-Evidence-Head'),
        }))
      ),
    { documentId, port, token }
  );
  fs.writeFileSync(path.join(copilotUIOutput, `${name}.jsonl`), result.text);
  return result;
}

(enabled ? test : test.skip)(
  'fixed roster preserves every outcome through actual renderer HTTP and registered hydration',
  async () => {
    const roster = fs
      .readFileSync(path.join(rosterPath, 'roster.jsonl'), 'utf8')
      .trim()
      .split('\n')
      .map((line) => JSON.parse(line));
    for (const entry of roster) {
      const name = `case-${String(entry.order).padStart(2, '0')}`;
      const documentId = `p6-${path.basename(copilotUIOutput)}-${entry.order}`;
      const result: Record<string, any> & UIObservation = {
        ...entry,
        started_at: new Date().toISOString(),
        actor: 'synthetic',
        mode,
        status: 'started',
        request_issued: false,
        model_attempted: false,
        visible_preview: false,
        trigger_to_visible_ms: null,
        renderer_response_complete_ms: null,
        model_quality: 'not_scored_by_ui_driver',
        real_model_action_coverage: [],
        cache_classification: 'unmeasured_without_actual_cache_evidence',
        intervention_policy: entry.order % 4 === 3 ? 'dismiss' : 'accept_then_one_undo',
      };
      results.push(result);
      try {
        const seeded = await exchange({
          command: 'seed_case',
          case_id: entry.case_id,
          document_id: documentId,
        });
        write(`${name}-seed.json`, seeded);
        if (seeded.status !== 200) throw new Error(seeded.body.detail);
        result.media = seeded.body.media;
        result.renderer_source = await hydrate(seeded.body.fixture, seeded.body.media);
        const before = await state();
        expect(before.document).toEqual(seeded.body.fixture.session_document);
        expect(before.copilot.history).toEqual(seeded.body.fixture.runtime.history);
        expect(before.copilot.sequence).toBe(seeded.body.fixture.renderer_sequence_before_request);
        write(`${name}-before.json`, before);
        await page.evaluate(() => {
          document.getElementById('document')?.focus();
          return window.store.dispatch(window.reducers.requestCopilot('explicit'));
        });
        const predicted = await state();
        write(`${name}-predicted.json`, predicted);
        result.request_issued = predicted.copilot.measurements.some((m) => m.event === 'request');
        result.status = predicted.copilot.status;
        result.model_error = predicted.copilot.error;
        const response = predicted.copilot.measurements.find((m) => m.event === 'response');
        result.renderer_response_complete_ms = response?.since_trigger_ms ?? null;
        result.runtime_timings = response?.timings ?? null;
        if (predicted.copilot.proposal) {
          await page.waitForFunction(() => window.store.getState().copilot.proposal?.shown);
          const shown = await state();
          const visible = shown.copilot.measurements.find(
            (m) => m.event === 'visible_complete_preview'
          );
          result.visible_preview = !!visible;
          result.trigger_to_visible_ms = visible?.since_trigger_ms ?? null;
          result.proposed_action = shown.copilot.proposal?.response.prediction.action;
          await page.screenshot({
            path: path.join(copilotUIOutput, `${name}-preview.png`),
            captureBeyondViewport: false,
          });
          if (result.intervention_policy === 'dismiss') {
            await page.keyboard.press('Escape');
            expect((await state()).document).toEqual(before.document);
          } else {
            const expected = shown.copilot.proposal?.response.result_document;
            await page.keyboard.press('Tab');
            await page.waitForFunction(() =>
              window.store.getState().copilot.events.some((e) => e.outcome === 'accepted')
            );
            await page.evaluate(() => {
              const dialog = document.createElement('div');
              dialog.id = 'copilot-roster-inspection';
              dialog.setAttribute('role', 'dialog');
              dialog.setAttribute('aria-modal', 'true');
              dialog.textContent = 'Inspecting the accepted edit and its undo.';
              document.body.appendChild(dialog);
              window.store.dispatch(window.reducers.setCopilotBlocked(true));
            });
            result.post_preview_automatic_requests =
              'An inspection dialog holds normal modal eligibility during undo/export; prediction and visible-preview timing precede it and use normal production eligibility.';
            const after = await state();
            expect(after.document).toEqual(expected);
            expect(after.sourceHashes).toEqual(before.sourceHashes);
            write(`${name}-accepted.json`, after);
            fs.writeFileSync(
              path.join(copilotUIOutput, `${name}-before.vtt`),
              contentToVtt(before.document.content, false, false, null).toString()
            );
            fs.writeFileSync(
              path.join(copilotUIOutput, `${name}-after.vtt`),
              contentToVtt(after.document.content, false, false, null).toString()
            );
            await page.evaluate(() => window.store.dispatch({ type: '@@redux-undo/UNDO' }));
            expect((await state()).document).toEqual(before.document);
            expect((await state()).sourceHashes).toEqual(before.sourceHashes);
          }
          result.execution = 'passed';
        } else {
          expect(predicted.document).toEqual(before.document);
          result.execution = 'no_visible_proposal';
        }
        const exported = await exportSession(documentId, name);
        expect(exported.status).toBe(200);
        const events = exported.text
          .trim()
          .split('\n')
          .map((line) => JSON.parse(line));
        expect(events[events.length - 1].event_hash).toBe(exported.head);
        expect(events.filter((e) => e.kind === 'fixture_context_seed')).toHaveLength(1);
        result.export_sha256 = hash(exported.text);
        result.model_attempted = actual && events.some((e) => e.kind === 'prediction_generated');
        result.model_response_received =
          actual && events.some((e) => e.kind === 'prediction_generated');
        result.live_feedback = events
          .filter((e) => e.kind === 'feedback')
          .map((e) => e.feedback.outcome);
        const final = await state();
        expect(final.copilot.events.every((e) => e.persistence === 'saved')).toBe(true);
        write(`${name}-final.json`, final);
        if (actual && result.execution === 'passed')
          result.real_model_action_coverage = [result.proposed_action];
      } catch (error) {
        result.status = result.request_issued
          ? 'ui_execution_failure'
          : 'hydration_binding_failure';
        result.error = String(error);
        write(`${name}-failure-state.json`, await state());
      } finally {
        const requestState = await page.evaluate(() => ({
          documentId: window.store.getState().copilot.documentId,
          measurements: window.store.getState().copilot.measurements,
        }));
        if (requestState.documentId === documentId) {
          const requestIds = requestState.measurements
            .filter((m) => m.event === 'request')
            .map((m) => m.request_id);
          result.issued_request_ids = requestIds;
          result.issued_request_count = requestIds.length;
          result.request_issued = requestIds.length > 0;
          result.unexpected_request_count = Math.max(0, requestIds.length - 1);
          if (
            result.error &&
            result.request_issued &&
            result.status === 'hydration_binding_failure'
          )
            result.status = 'ui_execution_failure';
        }
        const evidence = bindings().filter(
          (row) => row.case_id === entry.case_id || row.request?.request?.document_id === documentId
        );
        result.model_attempted = evidence.some((row) => row.kind === 'model_call_started');
        const bindingFailure = evidence.find((row) => row.kind.endsWith('binding_failure'));
        if (bindingFailure) {
          result.status = bindingFailure.kind;
          result.error = bindingFailure.error || 'Prepared input differs from registration.';
        }
        result.binding_evidence = evidence;
        result.finished_at = new Date().toISOString();
        write(`${name}-result.json`, result);
        write('results.json', results);
      }
    }
    expect(results).toHaveLength(roster.length);
    if (!actual) {
      const issued = results.filter((r) => r.request_issued);
      expect(issued.length).toBeGreaterThan(0);
      expect(issued.every((r) => r.status === 'no_action')).toBe(true);
    }
    expect(results.filter((r) => r.status === 'ui_execution_failure')).toEqual([]);
  }
);

afterAll(async () => {
  if (!enabled) return;
  try {
    await page.evaluate(() => window.store.dispatch(window.reducers.setCopilotEnabled(false)));
  } finally {
    if (bridge && bridge.exitCode === null) {
      const closed = new Promise<void>((resolve) => bridge.once('close', () => resolve()));
      bridge.stdin.end();
      await closed;
    }
    const feedback = path.join(copilotUITemp, 'feedback');
    const retained = path.join(copilotUIOutput, 'feedback-store');
    if (fs.existsSync(feedback)) {
      fs.mkdirSync(retained);
      for (const name of fs.readdirSync(feedback))
        if (name.endsWith('.jsonl'))
          fs.copyFileSync(path.join(feedback, name), path.join(retained, name));
    }
    write('network.json', network);
    write('page-errors.json', pageErrors);
    const planned = JSON.parse(
      fs.readFileSync(path.join(rosterPath, 'manifest.json'), 'utf8')
    ).cases;
    const metrics = summarizeUIObservations(planned, results);
    write('summary.json', {
      kind: actual
        ? 'actual_model_registered_ui_roster'
        : 'scripted_native_tokenizer_roster_engineering',
      actual_model_latency_measured:
        actual && metrics.model_attempts > 0 && metrics.renderer_response_complete.n > 0,
      model_quality_evaluated: false,
      roster_manifest_sha256: rosterSha,
      latency_protocol: { path: protocolPath, sha256: protocolSha },
      identity,
      ...metrics,
      hydration_failures: results.filter((r) => r.status === 'hydration_binding_failure').length,
      input_binding_failures: results.filter((r) => r.status === 'input_binding_failure').length,
      ui_execution_failures: results.filter((r) => r.status === 'ui_execution_failure').length,
      unexpected_request_count: results.reduce(
        (count, row) => count + (row.unexpected_request_count || 0),
        0
      ),
      completion_scope:
        'Renderer response measurements include successful API completions. CDP preserves HTTP/error/body completion separately; absent measurements are never imputed.',
      cache_scope:
        'Sequential requests, no deliberate exact-input prime; UI reset is not cold. Raw runtime_cached_tokens is retained when returned.',
      results,
      page_errors: pageErrors,
      source_sha256: Object.fromEntries(
        [
          'src/pages/Editor/CopilotPersistenceRoster.pup.spec.ts',
          'scripts/copilot_ui_fixture.ts',
          'scripts/copilot_ui_results.ts',
          'scripts/copilot_ui_environment.cjs',
          'scripts/copilot_ui_electron.cjs',
          'src/state/editor/selectors.ts',
          'src/state/copilot.ts',
          'src/server_api/copilot.ts',
          'src/core/copilot.ts',
          'src/core/editing_session.ts',
          'src/core/webvtt.ts',
          '../server/app/copilot.py',
          '../server/app/copilot_store.py',
          '../server/tests/copilot_p6_bridge.py',
          '../server/tests/copilot_p6_roster.py',
        ].map((file) => [file, hash(fs.readFileSync(file))])
      ),
    });
  }
});
