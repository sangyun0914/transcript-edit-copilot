/** @jest-environment ./src/util/jest_puppeteer_electron_environment */

import fs from 'fs';
import path from 'path';
import { replaySession, sessionDocument } from '../../core/editing_session';
import { Document } from '../../core/document';

jest.setTimeout(60000);

test('local session UI saves replayable edits, undo and gap trims without overwriting a log', async () => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.stack || error.message));
  await page.waitForFunction(() => !!window.store);
  await page.waitForSelector('h1');
  const directory = path.resolve('../artifacts/phase-3');
  fs.mkdirSync(directory, { recursive: true });
  const logPath = path.join(directory, `automated-ui-${Date.now()}.jsonl`);
  await page.evaluate((logPath) => {
    localStorage.setItem('intro/editor', 'true');
    const nodeRequire = (window as any).require;
    const ipc = nodeRequire('electron').ipcRenderer;
    const invoke = ipc.invoke.bind(ipc);
    ipc.invoke = (channel: string, options: unknown) =>
      channel === 'save-file'
        ? Promise.resolve({ canceled: false, filePath: logPath })
        : invoke(channel, options);
    const document = {
      content: [
        { type: 'paragraph_start', uuid: 'p', speaker: 'Test speaker', language: 'en' },
        { type: 'text', uuid: 'a', source: 's', sourceStart: 0, length: 1, text: 'Hello', conf: 1 },
        { type: 'non_text', uuid: 'b', source: 's', sourceStart: 1, length: 3 },
        { type: 'text', uuid: 'c', source: 's', sourceStart: 4, length: 1, text: 'world', conf: 1 },
        { type: 'paragraph_end', uuid: 'e' },
      ],
      metadata: { display_video: false, display_speaker_names: true },
      sources: { s: { objectUrl: '', fileContents: new Uint8Array([1, 2, 3]).buffer } },
    };
    window.store.dispatch(window.reducers.openDocumentFromMemory(document));
  }, logPath);
  await page
    .waitForSelector('[aria-label="Editing session recording"] summary')
    .catch(async (error) => {
      await page.screenshot({ path: path.join(directory, 'session-recorder-failure.png') });
      const details = await page.evaluate(() => ({
        text: document.body.textContent,
        nav: window.store.getState().nav,
      }));
      throw new Error(`${error.message}\n${JSON.stringify(details)}\n${pageErrors.join('\n')}`);
    });
  await page.click('[aria-label="Editing session recording"] summary');
  await page.select('[aria-label="Session type"]', 'synthetic');
  await page.click('[aria-label="Editing brief"]', { clickCount: 3 });
  await page.keyboard.press('Backspace');
  await page.keyboard.type('AUTOMATED UI TEST. Not human behavior or training data.');
  expect(
    await page.$eval('[aria-label="Editing brief"]', (input) => (input as HTMLInputElement).value)
  ).toBe('AUTOMATED UI TEST. Not human behavior or training data.');
  await page.evaluate(() => {
    const button = Array.from(document.querySelectorAll('button')).find(
      (b) => b.textContent === 'Start recording session'
    );
    button!.click();
  });
  await page.waitForFunction(() => window.store.getState().recording.status === 'recording');
  await page.evaluate(() => {
    window.store.dispatch(
      window.reducers.setSelection({ startIndex: 2, length: 1, headPosition: 'right' })
    );
  });
  await page.click('[data-gap="b"] button');
  await page.waitForSelector('[aria-label="Gap duration in seconds"]');
  await page.keyboard.type('0.725');
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => window.store.getState().recording.edits === 1);
  await page.evaluate(() => {
    window.store.dispatch({ type: '@@redux-undo/UNDO' });
    window.store.dispatch({ type: '@@redux-undo/REDO' });
    window.store.dispatch(window.reducers.setUserIndex(4));
    window.store.dispatch(window.reducers.insertParagraphEnd());
  });
  await page.screenshot({ path: path.join(directory, 'session-recorder.png') });
  await page.evaluate(() => {
    Array.from(document.querySelectorAll('button'))
      .find((b) => b.textContent === 'Stop and save session')!
      .click();
  });
  await page.waitForFunction(() => window.store.getState().recording.status === 'idle');
  const events = fs
    .readFileSync(logPath, 'utf8')
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line));
  const final = await page.evaluate(() => {
    const document = window.store.getState().editor.present!.document;
    return { content: document.content, metadata: document.metadata };
  });
  expect(replaySession(events)).toEqual(sessionDocument(final as unknown as Document));
  expect(events[0]).toMatchObject({
    collection_mode: 'synthetic',
    actor: 'synthetic',
    task_brief: 'AUTOMATED UI TEST. Not human behavior or training data.',
  });
  expect(events.filter((e) => e.kind === 'transaction').map((e) => e.operation)).toEqual([
    'trim_silence',
    'other_transaction',
    'other_transaction',
    'split_paragraph',
  ]);
  expect(events[0].sources.s.bytes).toBe(3);
  expect(events.filter((e) => e.kind === 'transaction').every((e) => e.actor === 'synthetic')).toBe(
    true
  );
  expect(events[events.length - 1].kind).toBe('end');
  const original = fs.readFileSync(logPath);
  await page.evaluate(() => {
    Array.from(document.querySelectorAll('button'))
      .find((b) => b.textContent === 'Start recording session')!
      .click();
  });
  await page.waitForFunction(() => window.store.getState().recording.status === 'failed');
  expect(fs.readFileSync(logPath)).toEqual(original);
  fs.writeFileSync(path.join(directory, 'latest-session-path.txt'), logPath + '\n');
});
