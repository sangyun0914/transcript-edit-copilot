/** @jest-environment ./src/util/jest_puppeteer_electron_environment */

import fs from 'fs';
import http from 'http';
import os from 'os';
import path from 'path';
import { createHash } from 'crypto';
import { TranscriptionContent, TranscriptionMetadata } from '../core/transcription';

jest.setTimeout(60000);

async function showOptions(configured: boolean, media = '/sample.wav') {
  await page.waitForFunction(() => !!window.store);
  await page.evaluate(
    (configured, media) => {
      window.store.dispatch({ type: 'nav/openLanding' });
      window.store.dispatch({
        type: 'models/fetchModelState/fulfilled',
        payload: { downloaded: {}, all: [], languages: {}, scribeConfigured: configured },
      });
      window.store.dispatch({ type: 'transcribe/transcribeFile/fulfilled', payload: media });
      window.store.dispatch({ type: 'nav/openTranscribe/pending' });
    },
    configured,
    media
  );
  await page.waitForSelector('#start');
}

test('configured Scribe can import without a downloaded Vosk model', async () => {
  await showOptions(true);
  expect(await page.$eval('#start', (button) => (button as HTMLButtonElement).disabled)).toBe(
    false
  );
  expect(await page.$eval('select', (select) => (select as HTMLSelectElement).value)).toBe(
    'scribe'
  );
  expect(await page.$eval('body', (body) => body.textContent)).toContain(
    'Audio is sent to ElevenLabs'
  );
  const screenshot = path.resolve('../artifacts/phase-1/transcription-options.png');
  fs.mkdirSync(path.dirname(screenshot), { recursive: true });
  await page.screenshot({ path: screenshot });
});

test('unconfigured Scribe and local mode without models cannot start a job', async () => {
  await showOptions(false);
  expect(await page.$eval('#start', (button) => (button as HTMLButtonElement).disabled)).toBe(true);
  await page.select('select', 'vosk');
  expect(await page.$eval('#start', (button) => (button as HTMLButtonElement).disabled)).toBe(true);
  expect(await page.$eval('body', (body) => body.textContent)).toContain(
    'Download a local transcription model first.'
  );
});

test('import failure leaves the progress screen with an actionable error', async () => {
  await showOptions(true);
  await page.evaluate(() => {
    window.store.dispatch({ type: 'nav/openTranscribing' });
    window.store.dispatch({
      type: 'transcribing/upload/rejected',
      error: { message: 'ElevenLabs rejected the request (HTTP 429).' },
    });
  });
  await page.waitForFunction(() =>
    document.body.textContent?.includes('ElevenLabs rejected the request')
  );
  expect(await page.$eval('body', (body) => body.textContent)).toContain(
    'Back to transcription options'
  );
  expect(await page.$eval('body', (body) => body.textContent)).not.toContain('%');
});

test('a downloaded model cannot start local ASR without its runtime', async () => {
  await showOptions(false);
  await page.evaluate(() => {
    window.store.dispatch({
      type: 'models/fetchModelState/fulfilled',
      payload: {
        scribeConfigured: false,
        localAsrAvailable: false,
        downloaded: { test: { lang: 'ko', name: 'Test model', model_id: 'test' } },
      },
    });
  });
  await page.select('select', 'vosk');
  expect(await page.$eval('#start', (button) => (button as HTMLButtonElement).disabled)).toBe(true);
  expect(await page.$eval('body', (body) => body.textContent)).toContain(
    'uv sync --extra local-asr'
  );
});

test('media conversion, upload, polling and editor import preserve the source and alignment flags', async () => {
  if (!!process.env.SCRIBE_REPLAY_MEDIA !== !!process.env.SCRIBE_REPLAY_RESULT) {
    throw new Error('Set both SCRIBE_REPLAY_MEDIA and SCRIBE_REPLAY_RESULT for a local replay.');
  }
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'audapolis-import-test-'));
  const media = process.env.SCRIBE_REPLAY_MEDIA || path.join(temporary, 'sample.WAV');
  if (!process.env.SCRIBE_REPLAY_MEDIA) {
    const wav = Buffer.alloc(44 + 3 * 16000 * 2);
    wav.write('RIFF');
    wav.writeUInt32LE(wav.length - 8, 4);
    wav.write('WAVEfmt ', 8);
    wav.writeUInt32LE(16, 16);
    wav.writeUInt16LE(1, 20);
    wav.writeUInt16LE(1, 22);
    wav.writeUInt32LE(16000, 24);
    wav.writeUInt32LE(32000, 28);
    wav.writeUInt16LE(2, 32);
    wav.writeUInt16LE(16, 34);
    wav.write('data', 36);
    wav.writeUInt32LE(wav.length - 44, 40);
    fs.writeFileSync(media, wav);
  }
  const fixture: { content: TranscriptionContent; transcription: TranscriptionMetadata } = process
    .env.SCRIBE_REPLAY_RESULT
    ? JSON.parse(fs.readFileSync(process.env.SCRIBE_REPLAY_RESULT, 'utf8'))
    : {
        content: [
          {
            speaker: 'speaker_0',
            content: [
              {
                type: 'word',
                word: '어 네',
                sourceStart: 0,
                length: 1,
                conf: null,
                asr: { word_indices: [0, 1], boundary_status: 'needs_alignment' },
              },
              { type: 'silence', sourceStart: 1, length: 2 },
            ],
          },
        ],
        transcription: {
          version: 1,
          provider: 'elevenlabs',
          model: 'scribe_v2',
          language_code: 'kor',
          duration: 3,
          request: { no_verbatim: 'false' },
          raw_response: {
            text: '어 네',
            words: [
              { type: 'word', text: '어', start: 0, end: 0 },
              { type: 'word', text: '네', start: 0.5, end: 1 },
            ],
          },
        },
      };
  let releaseUpload: () => void = () => {};
  const uploadGate = new Promise<void>((resolve) => {
    releaseUpload = resolve;
  });
  let complete = false;
  let uploadBytes = 0;
  let uploadHeader = Buffer.alloc(0);
  const startedAt = Date.now();
  let uploadCount = 0;
  let finishDeletion: () => void = () => {};
  const deleted = new Promise<void>((resolve) => {
    finishDeletion = resolve;
  });
  const server = http.createServer(async (request, response) => {
    response.setHeader('Access-Control-Allow-Origin', '*');
    response.setHeader('Access-Control-Allow-Headers', 'authorization,content-type');
    response.setHeader('Access-Control-Allow-Methods', 'POST,GET,DELETE,OPTIONS');
    response.setHeader('Content-Type', 'application/json');
    response.setHeader('Connection', 'close');
    if (request.method === 'OPTIONS') return void response.end();
    if (request.url?.startsWith('/transcription/capabilities'))
      return void response.end(JSON.stringify({ scribe: true, vosk: false }));
    if (request.method === 'POST') {
      uploadCount++;
      for await (const chunk of request) {
        uploadBytes += chunk.length;
        if (uploadHeader.length < 4096)
          uploadHeader = Buffer.concat([uploadHeader, chunk]).subarray(0, 4096);
      }
      await uploadGate;
      response.end(JSON.stringify({ uuid: 'replay', state: 'queued', progress: 0 }));
    } else if (request.method === 'DELETE') {
      response.end('{}');
      finishDeletion();
    } else {
      response.end(
        JSON.stringify(
          complete
            ? { uuid: 'replay', state: 'done', progress: 1, ...fixture }
            : { uuid: 'replay', state: 'transcribing', progress: 0 }
        )
      );
    }
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Test server did not start.');
  try {
    await showOptions(true, media);
    await page.evaluate((port) => {
      window.store.dispatch({
        type: 'server/setLocalServer',
        payload: {
          hostname: 'http://127.0.0.1',
          port,
          token: 'offline-test',
          state: 'ready',
        },
      });
    }, address.port);
    await page.click('#start');
    await page.waitForFunction(() => window.store.getState().transcribe.state === 'uploading');
    expect(await page.$eval('body', (body) => body.textContent)).not.toContain('%');
    releaseUpload();
    await page.waitForFunction(() => window.store.getState().transcribe.state === 'transcribing');
    expect(await page.$eval('body', (body) => body.textContent)).not.toContain('0 %');
    complete = true;
    await page.waitForFunction(
      () => !!window.store.getState().editor.present?.document.metadata.transcriptions?.length
    );
    await page.waitForSelector(
      '[title="Some words have no individual timing. Listen before cutting this group."]'
    );
    const imported = await page.evaluate(() => {
      const document = window.store.getState().editor.present?.document;
      return {
        content: document?.content,
        metadata: document?.metadata,
        sources: Object.keys(document?.sources || {}),
      };
    });
    expect(imported.metadata?.transcriptions?.[0].raw_response).toEqual(
      fixture.transcription.raw_response
    );
    expect(imported.sources).toEqual([
      createHash('sha256').update(fs.readFileSync(media)).digest('hex'),
    ]);
    let cursor = 0;
    for (const item of imported.content || []) {
      if ('source' in item) {
        expect(item.sourceStart).toBeCloseTo(cursor, 6);
        expect(item.length).toBeGreaterThan(0);
        cursor += item.length;
      }
    }
    expect(cursor).toBeCloseTo(fixture.transcription.duration, 6);
    expect(uploadCount).toBe(1);
    expect(uploadBytes).toBeGreaterThan(44);
    const fmt = uploadHeader.indexOf('WAVEfmt ');
    expect(fmt).toBeGreaterThan(0);
    expect(uploadHeader.readUInt16LE(fmt + 14)).toBe(1);
    expect(uploadHeader.readUInt32LE(fmt + 16)).toBe(16000);
    expect(uploadHeader.readUInt16LE(fmt + 26)).toBe(16);
    if (process.env.SCRIBE_REPLAY_METRICS) {
      fs.writeFileSync(
        process.env.SCRIBE_REPLAY_METRICS,
        JSON.stringify(
          {
            duration_seconds: fixture.transcription.duration,
            upload_bytes_with_multipart: uploadBytes,
            conversion_mock_upload_import_ms: Date.now() - startedAt,
            original_bytes: fs.statSync(media).size,
            sources: imported.sources,
            item_count: imported.content?.length,
          },
          null,
          2
        )
      );
    }
    await deleted;
    if (process.env.SCRIBE_REPLAY_PROJECT) {
      const output = path.resolve(process.env.SCRIBE_REPLAY_PROJECT);
      if (fs.existsSync(output)) throw new Error('Choose a new project output path.');
      await page.evaluate((output) => {
        const document = window.store.getState().editor.present?.document;
        window.store.dispatch({
          type: 'editor/saveDocument/fulfilled',
          payload: { document, path: output },
        });
        return window.store
          .dispatch(window.reducers.saveDocument(false))
          .unwrap()
          .then(() => {});
      }, output);
    }
    const screenshot = path.resolve('../artifacts/phase-1/transcription-import.png');
    await page.screenshot({ path: screenshot });
  } finally {
    releaseUpload();
    await new Promise<void>((resolve) => server.close(() => resolve()));
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});
