/** @jest-environment ./scripts/copilot_ui_environment.cjs */

import { spawn, ChildProcessWithoutNullStreams } from 'child_process';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { createInterface } from 'readline';
import { getPortPromise } from 'portfinder';

declare const copilotUIOutput: string;
declare const copilotUITemp: string;

jest.setTimeout(90000);

const inputDirectory = path.resolve(
  '../artifacts/ui/passive-actions-v3/b003-roster-actual-attempt-002'
);
const seedFile = path.join(inputDirectory, 'case-04-seed.json');
const predictionFile = path.join(inputDirectory, 'case-04-predicted.json');
const seed = JSON.parse(fs.readFileSync(seedFile, 'utf8')).body;
const prediction = JSON.parse(fs.readFileSync(predictionFile, 'utf8')).copilot.proposal.response;
const token = 'isolated-synthetic-ui-test';
const hash = (data: string | Buffer) => crypto.createHash('sha256').update(data).digest('hex');
const write = (name: string, value: unknown) =>
  fs.writeFileSync(path.join(copilotUIOutput, name), JSON.stringify(value, null, 2) + '\n');
const errors: string[] = [];
const pending = new Map<number, (result: any) => void>();
let bridge: ChildProcessWithoutNullStreams;
let counter = 0;
let port: number;

function exchange(message: Record<string, unknown>): Promise<any> {
  return new Promise((resolve) => {
    const id = ++counter;
    pending.set(id, resolve);
    bridge.stdin.write(JSON.stringify({ id, ...message }) + '\n');
  });
}

beforeAll(async () => {
  port = await getPortPromise({ host: '127.0.0.1', port: 16000 });
  bridge = spawn(path.resolve('../server/.venv/bin/python'), [
    '-u',
    path.resolve('../server/tests/copilot_p6_bridge.py'),
    '--feedback-directory',
    path.join(copilotUITemp, 'feedback'),
    '--output',
    path.join(copilotUIOutput, 'bridge'),
    '--http-port',
    String(port),
  ]);
  bridge.stderr.on('data', (data) =>
    fs.appendFileSync(path.join(copilotUIOutput, 'bridge-stderr.log'), data)
  );
  const lines = createInterface({ input: bridge.stdout });
  lines.on('line', (line) => {
    const result = JSON.parse(line);
    pending.get(result.id)?.(result);
    pending.delete(result.id);
  });
  bridge.on('exit', (code) => {
    for (const resolve of pending.values()) resolve({ status: 500, body: { error: code } });
    pending.clear();
    lines.close();
  });
  expect((await exchange({ command: 'info' })).status).toBe(200);
  expect(
    (
      await exchange({
        command: 'configure',
        prediction: prediction.prediction,
        delay_ms: Number(process.env.COPILOT_SCROLL_DELAY_MS || 4500),
      })
    ).status
  ).toBe(200);
  write('binding.json', {
    kind: 'engineering_replay_of_saved_B003_prediction_not_model_quality_evidence',
    model_executed: false,
    source_seed: seedFile,
    source_seed_sha256: hash(fs.readFileSync(seedFile)),
    source_prediction: predictionFile,
    source_prediction_sha256: hash(fs.readFileSync(predictionFile)),
    prediction: prediction.prediction,
    media: seed.media,
    tokenizer: 'existing_scripted_bridge_constant_stub_not_native',
    generation_delay_ms: Number(process.env.COPILOT_SCROLL_DELAY_MS || 4500),
    screenshot_mode: process.env.COPILOT_SCROLL_SCREENSHOT || 'default',
    files_sha256: Object.fromEntries(
      ['src/state/copilot.ts', 'src/pages/Editor/Copilot.tsx', 'src/pages/Editor/Cursor.tsx'].map(
        (file) => [file, hash(fs.readFileSync(file))]
      )
    ),
  });
  page.on('pageerror', (error) => errors.push(error.message));
  await page.waitForFunction(() => !!window.store);
  await page.addScriptTag({
    type: 'module',
    content: `import { installScrollDiagnostic } from ${JSON.stringify(
      '/@fs/' + path.resolve('scripts/copilot_scroll_diagnostic.ts')
    )}; window.installScrollDiagnostic = installScrollDiagnostic;`,
  });
  await page.waitForFunction(() => typeof (window as any).installScrollDiagnostic === 'function');
  await page.evaluate(
    ({ token, port, target }) => {
      localStorage.setItem('intro/editor', 'true');
      window.store.dispatch({
        type: 'server/setLocalServer',
        payload: { hostname: 'http://127.0.0.1', port, token, state: 'running' },
      });
      (window as any).scrollDiagnostic = (window as any).installScrollDiagnostic(target);
    },
    { token, port, target: prediction.target_uuids[0] }
  );
});

afterAll(async () => {
  try {
    write('telemetry.json', await page.evaluate(() => (window as any).scrollDiagnostic?.finish()));
    write('page-errors.json', errors);
    await page.evaluate(() => window.store.dispatch(window.reducers.setCopilotEnabled(false)));
  } finally {
    if (bridge && bridge.exitCode === null) {
      const exited = new Promise<void>((resolve) => bridge.once('close', () => resolve()));
      bridge.stdin.end();
      await exited;
    }
  }
});

const state = () =>
  page.evaluate(() => {
    const editor = window.store.getState().editor.present;
    if (!editor) throw new Error('No document.');
    const digest = (window as any).require('crypto');
    return {
      document: { content: editor.document.content, metadata: {} },
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
      diagnostic: (window as any).scrollDiagnostic.snapshot(),
    };
  });

function hydrateScene(scene: any) {
  return page.evaluate(({ fixture, media }) => {
    window.store.dispatch(window.reducers.setCopilotEnabled(false));
    const fs = (window as any).require('fs');
    const crypto = (window as any).require('crypto');
    const bytes = fs.readFileSync(media.path);
    const digest = crypto.createHash('sha256').update(bytes).digest('hex');
    if (digest !== media.sha256) throw new Error('Original recording differs.');
    const buffer = new Uint8Array(bytes).buffer;
    const sources = Object.fromEntries(
      media.source_ids.map((id: string) => [
        id,
        { fileContents: buffer, objectUrl: URL.createObjectURL(new Blob([buffer])) },
      ])
    );
    return window.store
      .dispatch(
        window.reducers.openDocumentFromMemory({
          content: fixture.document.content,
          metadata: Object.assign(
            { display_video: false, display_speaker_names: true },
            fixture.document.metadata
          ),
          sources,
        })
      )
      .then(() => {
        const focus = fixture.document.content.findIndex(
          (item: any) => item.uuid === fixture.runtime.focus_uuid
        );
        window.store.dispatch(window.reducers.setSelection(null));
        window.store.dispatch(window.reducers.setUserIndex(focus));
        window.store.dispatch(window.reducers.setCopilotEnabled(true));
        window.store.dispatch(window.reducers.setCopilotBlocked(false));
        window.store.dispatch(window.reducers.setCopilotFeedbackOrigin('synthetic'));
        window.store.dispatch(window.reducers.clearCopilotMeasurements());
        document.getElementById('document')?.focus();
        (window as any).scrollDiagnostic.mark('hydrated');
        return { bytes: buffer.byteLength, sha256: digest, focus };
      });
  }, scene);
}

test('replay original case04 document and saved prediction with viewport telemetry', async () => {
  if (process.env.COPILOT_SCROLL_PRIME === 'case03') {
    const primeSeedFile = path.join(inputDirectory, 'case-03-seed.json');
    const primePredictionFile = path.join(inputDirectory, 'case-03-predicted.json');
    const primeSeed = JSON.parse(fs.readFileSync(primeSeedFile, 'utf8')).body;
    const primePrediction = JSON.parse(fs.readFileSync(primePredictionFile, 'utf8')).copilot
      .proposal.response.prediction;
    await exchange({ command: 'configure', prediction: primePrediction, delay_ms: 4500 });
    write('prime-binding.json', {
      seed_sha256: hash(fs.readFileSync(primeSeedFile)),
      prediction_sha256: hash(fs.readFileSync(primePredictionFile)),
      media: primeSeed.media,
      prediction: primePrediction,
    });
    write('prime-source.json', await hydrateScene(primeSeed));
    write('prime-before.json', await state());
    await page.evaluate(() => window.store.dispatch(window.reducers.requestCopilot('explicit')));
    write('prime-predicted.json', await state());
    await page.waitForFunction(() => window.store.getState().copilot.proposal?.shown);
    write('prime-shown.json', await state());
    await page.screenshot({ path: path.join(copilotUIOutput, 'prime-shown.png') });
    await page.keyboard.press('Escape');
    write('prime-final.json', await state());
    await page.waitForFunction(() =>
      window.store.getState().copilot.events.every((event) => event.persistence !== 'pending')
    );
    await exchange({
      command: 'configure',
      prediction: prediction.prediction,
      delay_ms: Number(process.env.COPILOT_SCROLL_DELAY_MS || 4500),
    });
  }
  const source = await hydrateScene(seed);
  write('source.json', source);
  expect(source.bytes).toBe(seed.media.bytes);
  write('before.json', await state());
  await page.evaluate(() => {
    document.getElementById('document')?.focus();
    (window as any).scrollDiagnostic.mark('explicit-request');
    return window.store.dispatch(window.reducers.requestCopilot('explicit')).then(() => {
      (window as any).scrollDiagnostic.mark('request-returned');
    });
  });
  write('predicted.json', await state());
  await page.waitForFunction(() => window.store.getState().copilot.proposal?.shown);
  write('shown.json', await state());
  await page.evaluate(() => (window as any).scrollDiagnostic.mark('before-screenshot'));
  if (process.env.COPILOT_SCROLL_SCREENSHOT === 'none')
    await new Promise((resolve) => setTimeout(resolve, 1250));
  else
    await page.screenshot({
      path: path.join(copilotUIOutput, 'shown.png'),
      ...(process.env.COPILOT_SCROLL_SCREENSHOT === 'viewport' && { captureBeyondViewport: false }),
    });
  if (process.env.COPILOT_SCROLL_SCREENSHOT === 'viewport')
    await new Promise((resolve) => setTimeout(resolve, 1250));
  await page.evaluate(() => (window as any).scrollDiagnostic.mark('after-screenshot'));
  await page.keyboard.press('Tab');
  await page.evaluate(() => (window as any).scrollDiagnostic.mark('after-tab'));
  await new Promise((resolve) => setTimeout(resolve, 1500));
  const final = await state();
  write('final.json', final);
  await page.screenshot({ path: path.join(copilotUIOutput, 'final.png') });
  write('result.json', {
    status: final.copilot.status,
    feedback: final.copilot.events.map((event) => event.outcome),
    invalidations: final.copilot.measurements.filter((m) => m.event === 'preview_invalidated'),
    accepted: final.copilot.events.some((event) => event.outcome === 'accepted'),
    page_errors: errors,
  });
  expect(errors).toEqual([]);
});
