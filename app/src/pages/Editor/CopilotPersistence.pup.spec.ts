/** @jest-environment ./scripts/copilot_ui_environment.cjs */

import { spawn, ChildProcessWithoutNullStreams } from 'child_process';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { createInterface, Interface } from 'readline';
import { getPortPromise } from 'portfinder';
import { contentToVtt } from '../../core/webvtt';

declare const copilotUIOutput: string;
declare const copilotUITemp: string;

jest.setTimeout(90000);

interface BridgeResult {
  status: number;
  body?: any;
  text?: string;
  headers?: Record<string, string>;
  bridge_pid?: number;
}

const token = 'isolated-synthetic-ui-test';
const transport = process.env.COPILOT_UI_TRANSPORT || 'stdio';
let httpPort: number | undefined;
const checks: Record<string, unknown>[] = [];
const requests: Record<string, any>[] = [];
const pageErrors: string[] = [];
const pending = new Map<number, (value: BridgeResult) => void>();
let bridge: ChildProcessWithoutNullStreams;
let lines: Interface;
let counter = 0;
let bridgeErrors = '';
let holdNextResponse = false;
let responseHeld: (() => void) | undefined;
let releaseResponse: (() => void) | undefined;

const write = (name: string, data: unknown) =>
  fs.writeFileSync(path.join(copilotUIOutput, name), JSON.stringify(data, null, 2) + '\n');
const hash = (data: string) => crypto.createHash('sha256').update(data).digest('hex');

function exchange(message: Record<string, unknown>): Promise<BridgeResult> {
  return new Promise((resolve) => {
    const id = ++counter;
    pending.set(id, resolve);
    bridge.stdin.write(JSON.stringify({ id, ...message }) + '\n');
  });
}

function startBridge() {
  bridge = spawn(path.resolve('../server/.venv/bin/python'), [
    '-u',
    path.resolve('../server/tests/copilot_p6_bridge.py'),
    '--feedback-directory',
    path.join(copilotUITemp, 'feedback'),
    '--output',
    path.join(copilotUIOutput, 'bridge'),
    ...(httpPort ? ['--http-port', String(httpPort)] : []),
  ]);
  bridge.stderr.on('data', (data) => {
    bridgeErrors += data.toString();
    fs.appendFileSync(path.join(copilotUIOutput, 'bridge-stderr.log'), data);
  });
  lines = createInterface({ input: bridge.stdout });
  lines.on('line', (line) => {
    const result = JSON.parse(line);
    pending.get(result.id)?.(result);
    pending.delete(result.id);
  });
  bridge.on('exit', (code) => {
    for (const resolve of pending.values())
      resolve({ status: 500, body: { detail: `Bridge exited (${code}): ${bridgeErrors}` } });
    pending.clear();
  });
}

async function stopBridge() {
  if (bridge && bridge.exitCode === null) {
    const exit = new Promise<void>((resolve) => bridge.once('close', () => resolve()));
    bridge.stdin.end();
    await exit;
  }
  lines?.close();
}

beforeAll(async () => {
  if (!['stdio', 'http'].includes(transport)) throw new Error('Unknown UI transport.');
  if (transport === 'http') httpPort = await getPortPromise({ host: '127.0.0.1', port: 16000 });
  page.setDefaultTimeout(7000);
  startBridge();
  const identity = await exchange({ command: 'info' });
  expect(identity.status).toBe(200);
  write('identity.json', identity);
  const denied = await exchange({ method: 'GET', route: '/copilot/health' });
  expect(denied.status).toBe(401);
  await page.waitForFunction(() => !!window.store);
  await page.exposeFunction(
    'copilotPersistenceHoldResponse',
    () =>
      new Promise<void>((resolve) => {
        releaseResponse = resolve;
        responseHeld?.();
      })
  );
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await page.exposeFunction('copilotPersistenceBridge', async (message: any) => {
    const row: Record<string, any> = {
      request: message,
      renderer_received_at: new Date().toISOString(),
      node_start_ns: process.hrtime.bigint().toString(),
    };
    requests.push(row);
    const result = await exchange(message);
    Object.assign(row, { result, node_completed_ns: process.hrtime.bigint().toString() });
    if (holdNextResponse && message.route.replace(/\/$/, '') === '/copilot/predict') {
      holdNextResponse = false;
      await new Promise<void>((resolve) => {
        releaseResponse = resolve;
        responseHeld?.();
      });
      row.artificial_response_hold_released_ns = process.hrtime.bigint().toString();
    }
    return result;
  });
  await page.evaluate(
    ({ token, transport, port }) => {
      localStorage.setItem('intro/editor', 'true');
      const originalFetch = window.fetch.bind(window);
      if (transport === 'stdio')
        window.fetch = ((input: RequestInfo, init?: RequestInit) => {
          const url = new URL(typeof input === 'string' ? input : input.url, window.location.href);
          if (!url.pathname.startsWith('/copilot/')) return originalFetch(input, init);
          return (window as any)
            .copilotPersistenceBridge({
              route: url.pathname + url.search,
              method: init?.method || 'GET',
              body: init?.body ? JSON.parse(String(init.body)) : undefined,
              headers: init?.headers || {},
              renderer_sent_at_ms: performance.now(),
              renderer_time_origin: performance.timeOrigin,
            })
            .then(
              (result: BridgeResult) =>
                new Response(result.text ?? JSON.stringify(result.body), {
                  status: result.status,
                  headers: result.headers || { 'content-type': 'application/json' },
                })
            );
        }) as typeof window.fetch;
      window.store.dispatch({
        type: 'server/setLocalServer',
        payload: { hostname: 'http://127.0.0.1', port, token, state: 'running' },
      });
      window.store.dispatch(window.reducers.setCopilotFeedbackOrigin('synthetic'));
    },
    { token, transport, port: httpPort || 44251 }
  );
  if (httpPort)
    expect(
      await page.evaluate(
        (port) =>
          fetch(`http://127.0.0.1:${port}/copilot/health`).then((response) => response.status),
        httpPort
      )
    ).toBe(401);
});

afterAll(async () => {
  releaseResponse?.();
  try {
    await page.evaluate(() => window.store.dispatch(window.reducers.setCopilotEnabled(false)));
    await page.waitForFunction(() =>
      window.store.getState().copilot.events.every((event) => event.persistence !== 'pending')
    );
  } finally {
    await stopBridge();
    write('requests.json', requests);
    write('checks.json', checks);
    write('page-errors.json', pageErrors);
    const files = [
      'src/pages/Editor/CopilotPersistence.pup.spec.ts',
      'scripts/copilot_ui_environment.cjs',
      'scripts/copilot_ui_electron.cjs',
      '../server/tests/copilot_p6_bridge.py',
      'src/state/copilot.ts',
      'src/server_api/copilot.ts',
      'src/pages/Editor/Copilot.tsx',
      '../server/app/copilot.py',
      '../server/app/copilot_store.py',
    ];
    write('summary.json', {
      kind: 'scripted_known_action_product_checks',
      actor: 'synthetic',
      model_quality_evaluated: false,
      native_token_budget_evaluated: false,
      actual_model_latency_measured: false,
      transport:
        transport === 'http'
          ? 'renderer fetch -> authenticated loopback uvicorn HTTP'
          : 'renderer fetch -> Node stdio -> authenticated FastAPI TestClient',
      abort_transport:
        'Only the late-response test delays a completed response. HTTP mode otherwise uses the original renderer fetch and AbortSignal.',
      source_audio:
        'Three synthetic bytes; source preservation is checked, acoustic quality is not.',
      checks,
      page_errors: pageErrors,
      files_sha256: Object.fromEntries(
        files.map((file) => [file, hash(fs.readFileSync(file, 'utf8'))])
      ),
    });
  }
});

async function configure(prediction: Record<string, unknown>, delay_ms = 0) {
  expect((await exchange({ command: 'configure', prediction, delay_ms })).status).toBe(200);
}

async function openScene(
  language: 'en' | 'ko',
  content?: any[],
  { automatic = false, cursorIndex = 2 } = {}
) {
  await page.evaluate(
    ({ language, content, automatic, cursorIndex }) => {
      if (!automatic) window.store.dispatch(window.reducers.setCopilotEnabled(false));
      window.store.dispatch(window.reducers.clearCopilotMeasurements());
      const words =
        language === 'en' ? ['We', 'recieve', 'updates.'] : ['새로운', '데이타', '입니다.'];
      const items: any[] =
        content ||
        words.map((text, index) => ({
          type: 'text',
          uuid: ['a', 'b', 'c'][index],
          source: 's',
          sourceStart: index * 1.25,
          length: 1.25,
          text,
          conf: 1,
        }));
      if (!content) {
        items.unshift({ type: 'paragraph_start', uuid: 'p', speaker: 'Speaker', language });
        items.push({ type: 'paragraph_end', uuid: 'e' });
      }
      return window.store
        .dispatch(
          window.reducers.openDocumentFromMemory({
            content: items,
            metadata: { display_video: false, display_speaker_names: true },
            sources: { s: { objectUrl: '', fileContents: new Uint8Array([1, 2, 3]).buffer } },
          })
        )
        .then(() => {
          window.store.dispatch(window.reducers.setCopilotFeedbackOrigin('synthetic'));
          window.store.dispatch(window.reducers.setUserIndex(cursorIndex));
          document.getElementById('document')?.focus();
        });
    },
    { language, content: content || null, automatic, cursorIndex }
  );
}

const current = () =>
  page.evaluate(() => {
    const editor = window.store.getState().editor.present;
    if (!editor) throw new Error('No current document.');
    return {
      content: editor.document.content,
      copilot: window.store.getState().copilot,
      source: Array.from(new Uint8Array(editor.document.sources.s.fileContents)),
    };
  });

const predict = () =>
  page.evaluate(() => {
    document.getElementById('document')?.focus();
    window.store.dispatch(window.reducers.setCopilotEnabled(true));
    return window.store.dispatch(window.reducers.requestCopilot('explicit'));
  });

async function settled() {
  await page.waitForFunction(() =>
    window.store.getState().copilot.events.every((event) => event.persistence !== 'pending')
  );
}

async function exportFeedback(name: string) {
  await settled();
  const state = await current();
  const result = await page.evaluate(
    ({ documentId, token, port }) => {
      return fetch(
        `http://127.0.0.1:${port}/copilot/feedback/export?document_id=${encodeURIComponent(
          documentId
        )}`,
        { headers: { Authorization: `Bearer ${token}` } }
      ).then((response) =>
        response.text().then((text) => ({
          status: response.status,
          text,
          head: response.headers.get('X-Copilot-Evidence-Head'),
        }))
      );
    },
    { documentId: state.copilot.documentId, token, port: httpPort || 44251 }
  );
  expect(result.status).toBe(200);
  fs.writeFileSync(path.join(copilotUIOutput, `${name}.jsonl`), result.text);
  const rows = result.text
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line));
  expect(rows[rows.length - 1].event_hash).toBe(result.head);
  expect(rows.filter((row) => row.actor).every((row) => row.actor === 'synthetic')).toBe(true);
  write(`${name}-state.json`, state);
  return { ...result, rows, state };
}

test('opening a document automatically shows a stable suggestion without public activation controls', async () => {
  expect(await page.evaluate(() => window.store.getState().copilot.enabled)).toBe(true);
  const content = Array.from({ length: 32 }, (_, index) => [
    { type: 'paragraph_start', uuid: `p-${index}`, speaker: 'Speaker', language: 'en' },
    {
      type: 'text',
      uuid: `away-${index}`,
      source: 's',
      sourceStart: index * 1.25,
      length: 1.25,
      text: index === 31 ? 'recieve' : `A neutral sentence in paragraph ${index + 1}.`,
      conf: 1,
    },
    { type: 'paragraph_end', uuid: `e-${index}` },
  ]).flat();
  await configure({ action: 'correct_text', target: [31], text: 'receive' }, 600);
  await openScene('en', content, { automatic: true, cursorIndex: 1 });
  const original = await current();
  expect(await page.$('[aria-label="Enable local copilot"]')).toBeNull();
  expect(
    await page.$$eval('button', (buttons) =>
      buttons.some((button) => button.textContent === 'Predict next edit')
    )
  ).toBe(false);
  expect(
    await page.$eval('#item-away-31', (node) => {
      const viewport = document.getElementById('scroll-container');
      return Boolean(
        viewport && node.getBoundingClientRect().top >= viewport.getBoundingClientRect().bottom
      );
    })
  ).toBe(true);
  await page.waitForFunction(() => window.store.getState().copilot.proposal?.shown);
  const shown = await current();
  const proposal = shown.copilot.proposal;
  if (!proposal) throw new Error('No automatic proposal.');
  expect(proposal.response.target_uuids).toEqual(['away-31']);
  expect(shown.content).toEqual(original.content);
  expect(shown.copilot.measurements.filter((item) => item.event === 'request')).toHaveLength(1);
  expect(
    shown.copilot.measurements.some(
      (item) => item.event === 'visible_complete_preview' && item.trigger === 'automatic'
    )
  ).toBe(true);
  await page.evaluate(() => {
    window.store.dispatch(window.reducers.setUserIndex(1));
    window.store.dispatch(window.reducers.setUserIndex(1));
    return new Promise((resolve) => setTimeout(resolve, 1250));
  });
  const stable = await current();
  expect(stable.copilot.proposal?.response.proposal_id).toBe(proposal.response.proposal_id);
  expect(stable.copilot.proposal?.shown).toBe(true);
  expect(stable.copilot.measurements.filter((item) => item.event === 'request')).toHaveLength(1);
  expect(stable.copilot.events.map((event) => event.outcome)).toEqual(['shown']);
  expect(
    await page.$eval('[data-testid="copilot-preview"]', (node) => {
      const viewport = document.getElementById('scroll-container');
      if (!viewport) return false;
      const rect = node.getBoundingClientRect();
      const bounds = viewport.getBoundingClientRect();
      return rect.height > 0 && rect.bottom > bounds.top && rect.top < bounds.bottom;
    })
  ).toBe(true);
  await page.screenshot({
    path: path.join(copilotUIOutput, 'automatic-stable-preview.png'),
    captureBeyondViewport: false,
  });
  await configure({ action: 'no_action' });
  await page.keyboard.press('Tab');
  await page.waitForFunction(() =>
    window.store.getState().copilot.events.some((event) => event.outcome === 'accepted')
  );
  expect((await current()).content).toEqual(proposal.response.result_document?.content);
  expect((await current()).source).toEqual(original.source);
  await page.waitForFunction(() => window.store.getState().copilot.status === 'no_action');
  const accepted = await exportFeedback('automatic-default-accepted');
  const acceptedRequests = accepted.rows.filter((row) => row.kind === 'prediction_requested');
  expect(acceptedRequests).toHaveLength(2);
  expect(acceptedRequests.every((row) => row.input_request.trigger === 'automatic')).toBe(true);
  expect(
    accepted.rows.filter((row) => row.kind === 'feedback').map((row) => row.feedback.outcome)
  ).toEqual(['shown', 'accepted']);

  await configure({ action: 'correct_text', target: [1], text: 'receive' });
  await openScene('en', undefined, { automatic: true });
  const beforeDismiss = await current();
  await page.waitForFunction(() => window.store.getState().copilot.proposal?.shown);
  await page.keyboard.press('Escape');
  await page.evaluate(() => new Promise((resolve) => setTimeout(resolve, 1000)));
  const dismissed = await exportFeedback('automatic-default-dismissed');
  expect(dismissed.state.content).toEqual(beforeDismiss.content);
  expect(dismissed.state.copilot.proposal).toBeNull();
  expect(dismissed.rows.filter((row) => row.kind === 'prediction_requested')).toHaveLength(1);
  expect(
    dismissed.rows.filter((row) => row.kind === 'feedback').map((row) => row.feedback.outcome)
  ).toEqual(['shown', 'dismissed']);
  checks.push({
    case: 'automatic-default-open-stable-tab-escape',
    result: 'passed',
    public_activation_controls: false,
    preview_retained_ms: 1250,
    target_uuid: 'away-31',
    initial_focus_uuid: 'away-0',
    unchanged_context_requests: 1,
  });
});

test('rapid cursor moves cancel the old automatic request and debounce one request for the final cursor', async () => {
  await configure({ action: 'correct_text', target: [1], text: 'receive' }, 1200);
  const started = (await exchange({ command: 'info' })).body.generation_started;
  await openScene('en', undefined, { automatic: true });
  const original = (await current()).content;
  await page.waitForFunction(() => window.store.getState().copilot.status === 'requesting');
  for (let attempts = 0; attempts < 100; attempts++) {
    if ((await exchange({ command: 'info' })).body.generation_started > started) break;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  expect((await exchange({ command: 'info' })).body.generation_started).toBe(started + 1);
  const first = (await current()).copilot.request;
  if (!first) throw new Error('No initial automatic request.');
  await page.keyboard.press('ArrowRight');
  await new Promise((resolve) => setTimeout(resolve, 100));
  await page.keyboard.press('ArrowLeft');
  await new Promise((resolve) => setTimeout(resolve, 100));
  await page.keyboard.press('ArrowLeft');
  const lastMove = await page.evaluate(() => ({
    at_ms: performance.now(),
    cursor: window.store.getState().editor.present?.cursor.userIndex,
    epoch: window.store.getState().copilot.epoch,
  }));
  expect(lastMove.cursor).toBe(1);
  expect((await current()).content).toEqual(original);
  expect(await page.$('[data-testid="copilot-preview"]')).toBeNull();
  await page.waitForFunction(() => window.store.getState().copilot.proposal?.shown);
  const state = (await current()).copilot;
  const sent = state.measurements.filter((item) => item.event === 'request');
  expect(sent).toHaveLength(2);
  expect(sent.every((item) => item.trigger === 'automatic')).toBe(true);
  expect(sent[1].at_ms - lastMove.at_ms).toBeGreaterThanOrEqual(600);
  expect(state.proposal?.response.request.request_id).toBe(sent[1].request_id);
  expect(state.proposal?.response.request.interaction_epoch).toBe(lastMove.epoch);
  expect(
    state.measurements.some(
      (item) => item.event === 'cancel_requested' && item.request_id === first.request_id
    )
  ).toBe(true);
  expect(
    state.events.some(
      (event) => event.outcome === 'shown' && event.request.request_id === first.request_id
    )
  ).toBe(false);
  expect((await current()).content).toEqual(original);
  await page.keyboard.press('Escape');
  const exported = await exportFeedback('automatic-cursor-cancel-debounce');
  const sentRequests = exported.rows.filter((row) => row.kind === 'prediction_requested');
  expect(sentRequests).toHaveLength(2);
  expect(sentRequests[1].input_request.focus_uuid).toBe('a');
  expect(
    exported.rows.some(
      (row) => row.kind === 'prediction_failed' && row.request.request_id === first.request_id
    )
  ).toBe(true);
  expect(
    exported.rows.some(
      (row) => row.kind === 'prediction_completed' && row.request.request_id === first.request_id
    )
  ).toBe(false);
  checks.push({
    case: 'automatic-cursor-cancel-debounce',
    result: 'passed',
    cancelled_request_id: first.request_id,
    latest_request_id: sent[1].request_id,
    latest_focus_uuid: 'a',
    quiet_time_ms: sent[1].at_ms - lastMove.at_ms,
    requests: sent.length,
  });
});

test('all seven action outcomes pass the real UI, with exact apply, one undo, durable evidence and caption export', async () => {
  const fixtures = fs
    .readFileSync(path.resolve('../ml/fixtures/execution-v2.jsonl'), 'utf8')
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line));
  const scenes: { id: string; language: 'en' | 'ko'; content?: any[]; prediction: any }[] = [
    {
      id: 'correct-en',
      language: 'en',
      prediction: { action: 'correct_text', target: [1], text: 'receive' },
    },
    {
      id: 'correct-ko',
      language: 'ko',
      prediction: { action: 'correct_text', target: [1], text: '데이터' },
    },
  ];
  for (const id of [
    'delete-with-gap',
    'trim-pause',
    'remove-pause',
    'split-caption',
    'merge-caption',
  ]) {
    const fixture = fixtures.find((row) => row.id === id);
    if (id.endsWith('caption')) {
      let start = 0;
      for (const item of fixture.before.content)
        if (item.type === 'text') {
          item.sourceStart = start;
          item.length = 1.25;
          start += item.length;
        }
    }
    scenes.push({
      id,
      language: 'ko',
      content: fixture.before.content.map((item: any) =>
        'source' in item ? { ...item, source: 's' } : item
      ),
      prediction: fixture.prediction,
    });
  }
  scenes.push({ id: 'no-action', language: 'en', prediction: { action: 'no_action' } });
  for (const scene of scenes) {
    await configure(scene.prediction);
    await openScene(scene.language, scene.content);
    const before = await current();
    await predict();
    const predicted = await current();
    expect({
      case: scene.id,
      status: predicted.copilot.status,
      error: predicted.copilot.error,
    }).toEqual({
      case: scene.id,
      status: scene.prediction.action === 'no_action' ? 'no_action' : 'preview',
      error: null,
    });
    if (scene.prediction.action === 'no_action') {
      expect(predicted.copilot.status).toBe('no_action');
      expect(await page.$('[data-testid="copilot-preview"]')).toBeNull();
      expect(predicted.content).toEqual(before.content);
    } else {
      await page.waitForSelector('[data-testid="copilot-preview"]');
      await page.waitForFunction(() => window.store.getState().copilot.proposal?.shown);
      expect(predicted.content).toEqual(before.content);
      await page.screenshot({
        path: path.join(copilotUIOutput, `${scene.id}-preview.png`),
        captureBeyondViewport: false,
      });
      const expected = predicted.copilot.proposal?.response.result_document;
      if (!expected) throw new Error(`No executable proposal for ${scene.id}.`);
      await page.keyboard.press('Tab');
      await page.waitForFunction(() =>
        window.store.getState().copilot.events.some((event) => event.outcome === 'accepted')
      );
      const after = await current();
      expect(after.content).toEqual(expected.content);
      expect(after.source).toEqual(before.source);
      const vttBefore = contentToVtt(before.content, false, false, null).toString();
      const vttAfter = contentToVtt(after.content, false, false, null).toString();
      expect(vttAfter).not.toBe(vttBefore);
      fs.writeFileSync(path.join(copilotUIOutput, `${scene.id}-before.vtt`), vttBefore);
      fs.writeFileSync(path.join(copilotUIOutput, `${scene.id}-after.vtt`), vttAfter);
      await page.evaluate(() => window.store.dispatch({ type: '@@redux-undo/UNDO' }));
      expect((await current()).content).toEqual(before.content);
      expect((await current()).source).toEqual(before.source);
    }
    const exported = await exportFeedback(scene.id);
    const feedback = exported.rows.filter((row) => row.kind === 'feedback');
    expect(feedback.map((row) => row.feedback.outcome)).toEqual(
      scene.prediction.action === 'no_action' ? [] : ['shown', 'accepted', 'undone']
    );
    expect(exported.state.copilot.events.every((event) => event.persistence === 'saved')).toBe(
      true
    );
    for (const row of feedback.filter((row) => row.feedback.transition)) {
      expect(row.feedback.transition.before).toBeTruthy();
      expect(row.feedback.transition.after).toBeTruthy();
    }
    checks.push({
      case: scene.id,
      action: scene.prediction.action,
      result: 'passed',
      export_sha256: hash(exported.text),
    });
  }
  expect(pageErrors).toEqual([]);
});

test('explicit dismissal survives a gateway process restart and suppresses the same proposal', async () => {
  const action = { action: 'correct_text', target: [1], text: 'receive' };
  await configure(action);
  await openScene('en');
  const original = (await current()).content;
  await predict();
  await page.waitForFunction(() => window.store.getState().copilot.proposal?.shown);
  await page.keyboard.press('Escape');
  const before = await exportFeedback('restart-before');
  expect(
    before.rows.filter((row) => row.kind === 'feedback').map((row) => row.feedback.outcome)
  ).toEqual(['shown', 'dismissed']);
  const pidBefore = (await exchange({ command: 'info' })).body.pid;
  await stopBridge();
  startBridge();
  const pidAfter = (await exchange({ command: 'info' })).body.pid;
  expect(pidAfter).not.toBe(pidBefore);
  const reloaded = await exportFeedback('restart-reloaded');
  expect(reloaded.text).toBe(before.text);
  await configure(action);
  await predict();
  expect((await current()).copilot.status).toBe('suppressed');
  expect((await current()).content).toEqual(original);
  expect(await page.$('[data-testid="copilot-preview"]')).toBeNull();
  await exportFeedback('restart-after-suppression');
  checks.push({ case: 'durable-restart-repeat', result: 'passed', pidBefore, pidAfter });
});

test('late validated responses and in-flight cancellation cannot display or apply stale edits', async () => {
  await configure({ action: 'correct_text', target: [1], text: 'receive' });
  await openScene('en');
  const original = (await current()).content;
  holdNextResponse = true;
  const held = new Promise<void>((resolve) => {
    responseHeld = resolve;
  });
  if (httpPort)
    await page.evaluate(() => {
      const originalFetch = window.fetch.bind(window);
      window.fetch = ((input: RequestInfo, init?: RequestInit) => {
        const url = new URL(typeof input === 'string' ? input : input.url, window.location.href);
        if (url.pathname.replace(/\/$/, '') !== '/copilot/predict')
          return originalFetch(input, init);
        window.fetch = originalFetch;
        return originalFetch(input, init).then((response: Response) =>
          response.text().then((text) =>
            (window as any).copilotPersistenceHoldResponse().then(
              () =>
                new Response(text, {
                  status: response.status,
                  headers: response.headers,
                })
            )
          )
        );
      }) as typeof window.fetch;
    });
  await page.evaluate(() => {
    window.store.dispatch(window.reducers.setCopilotEnabled(true));
    void window.store.dispatch(window.reducers.requestCopilot('explicit'));
  });
  await held;
  await page.evaluate(() => window.store.dispatch(window.reducers.setUserIndex(3)));
  releaseResponse?.();
  await page.waitForFunction(() =>
    window.store.getState().copilot.measurements.some((item) => item.event === 'stale_discard')
  );
  await page.keyboard.press('Tab');
  expect((await current()).content).toEqual(original);
  const stale = await exportFeedback('late-response');
  expect(
    stale.rows.filter((row) => row.kind === 'feedback').map((row) => row.feedback.outcome)
  ).toEqual(['invalidated']);
  await configure({ action: 'correct_text', target: [1], text: 'receive' }, 800);
  await openScene('en');
  const started = (await exchange({ command: 'info' })).body.generation_started;
  await page.evaluate(() => {
    window.store.dispatch(window.reducers.setCopilotEnabled(true));
    void window.store.dispatch(window.reducers.requestCopilot('explicit'));
  });
  for (let attempts = 0; attempts < 100; attempts++) {
    if ((await exchange({ command: 'info' })).body.generation_started > started) break;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  expect((await exchange({ command: 'info' })).body.generation_started).toBeGreaterThan(started);
  await page.evaluate(() => window.store.dispatch(window.reducers.setUserIndex(3)));
  await page.waitForFunction(() =>
    window.store
      .getState()
      .copilot.measurements.some((item) => item.event === 'request_settled_after_cancel')
  );
  for (let attempts = 0; attempts < 100; attempts++) {
    if (!(await exchange({ command: 'info' })).body.active_request) break;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  expect((await exchange({ command: 'info' })).body.active_request).toBeNull();
  const cancelled = await exportFeedback('cancelled-request');
  expect(cancelled.rows.some((row) => row.kind === 'prediction_failed')).toBe(true);
  expect(cancelled.rows.some((row) => row.kind === 'prediction_completed')).toBe(false);
  expect(await page.$('[data-testid="copilot-preview"]')).toBeNull();
  expect((await current()).content).toEqual(original);
  checks.push({ case: 'late-response-and-cancel', result: 'passed', artificial_delay_ms: 800 });
});

test('automatic trigger reaches a visible preview and an expired outcome remains distinct from dismissal', async () => {
  await configure({ action: 'correct_text', target: [1], text: 'receive' });
  await openScene('en');
  await page.evaluate(() => {
    document.getElementById('document')?.focus();
    window.store.dispatch(window.reducers.setCopilotEnabled(true));
    window.store.dispatch(window.reducers.clearCopilotMeasurements());
    window.store.dispatch(window.reducers.setText({ absoluteIndex: 3, text: 'messages.' }));
  });
  await page.waitForFunction(() => window.store.getState().copilot.proposal?.shown);
  const state = await current();
  expect(
    state.copilot.measurements.some(
      (item) => item.event === 'visible_complete_preview' && item.trigger === 'automatic'
    )
  ).toBe(true);
  await page.evaluate(() =>
    window.store.dispatch({
      type: 'copilot/expire',
      payload: window.store.getState().copilot.proposal?.response.proposal_id,
    })
  );
  const expired = await exportFeedback('automatic-expired');
  expect(
    expired.rows.filter((row) => row.kind === 'feedback').map((row) => row.feedback.outcome)
  ).toEqual(['shown', 'expired']);
  await predict();
  expect((await current()).copilot.status).toBe('preview');
  await page.waitForFunction(() => window.store.getState().copilot.proposal?.shown);
  await page.keyboard.press('Escape');
  await exportFeedback('automatic-after-expiry');
  checks.push({
    case: 'automatic-trigger-and-expiry',
    result: 'passed',
    expiry_trigger: 'explicit test dispatch, not a wall-clock timeout measurement',
  });
});

test('a durable write failure stays failed in the visible UI and cannot masquerade as saved feedback', async () => {
  await configure({ action: 'correct_text', target: [1], text: 'receive' });
  await openScene('en');
  expect((await exchange({ command: 'fail_feedback' })).status).toBe(200);
  await predict();
  await page.waitForFunction(() =>
    window.store.getState().copilot.events.some((event) => event.persistence === 'failed')
  );
  const state = await current();
  expect(state.copilot.feedbackError).toContain('Injected durable feedback write failure');
  expect(await page.evaluate(() => document.body.textContent)).toContain(
    'Suggestions paused because feedback could not be saved.'
  );
  expect(state.copilot.events.some((event) => event.persistence === 'saved')).toBe(false);
  const exported = await exportFeedback('failed-feedback');
  expect(exported.rows.some((row) => row.kind === 'feedback')).toBe(false);
  checks.push({
    case: 'durable-failure-visible',
    result: 'passed',
    fault: 'injected EvidenceError before feedback append',
  });
});
