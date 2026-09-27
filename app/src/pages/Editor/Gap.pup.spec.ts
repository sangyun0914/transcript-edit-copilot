/** @jest-environment ./src/util/jest_puppeteer_electron_environment */

import fs from 'fs';
import path from 'path';

jest.setTimeout(60000);

test('all pauses show editable seconds with atomic trimming and ordinary caption shortcuts', async () => {
  await page.waitForSelector('h1');
  await page.evaluate(() => {
    localStorage.setItem('intro/editor', 'true');
    window.store.dispatch(
      window.reducers.openDocumentFromMemory({
        content: [
          { type: 'paragraph_start', uuid: 'p', speaker: 'Speaker', language: 'en' },
          {
            type: 'text',
            uuid: 'a',
            source: 's',
            sourceStart: 0,
            length: 1,
            text: 'Hello',
            conf: 1,
          },
          { type: 'non_text', uuid: 'short', source: 's', sourceStart: 1, length: 0.1 },
          {
            type: 'text',
            uuid: 'b',
            source: 's',
            sourceStart: 1.1,
            length: 1,
            text: 'world',
            conf: 1,
          },
          { type: 'non_text', uuid: 'long', source: 's', sourceStart: 2.1, length: 3 },
          { type: 'paragraph_end', uuid: 'e' },
          { type: 'paragraph_start', uuid: 'p2', speaker: 'Speaker', language: 'ko' },
          {
            type: 'text',
            uuid: 'ko',
            source: 's',
            sourceStart: 5.1,
            length: 1,
            text: '안녕하세요',
            conf: 1,
          },
          { type: 'artificial_silence', uuid: 'artificial', length: 0.25 },
          { type: 'paragraph_end', uuid: 'e2' },
        ],
        metadata: { display_video: false, display_speaker_names: false },
        sources: { s: { objectUrl: '', fileContents: new Uint8Array([1, 2, 3]).buffer } },
      })
    );
  });
  await page.waitForSelector('[data-gap="short"] button');
  expect(
    await page.$$eval('[data-gap] button', (buttons) => buttons.map((b) => b.textContent))
  ).toEqual(['0.1s', '3s', '0.25s']);
  expect(await page.$eval('[data-gap="short"]', (gap) => gap.textContent?.trim())).toBe('[_ 0.1s]');
  expect(await page.$$('[aria-label="Pause"]')).toHaveLength(3);
  expect(await page.$('[aria-label="Retain gap milliseconds"]')).toBeNull();

  const content = () =>
    page.evaluate(() => {
      const editor = window.store.getState().editor.present;
      if (!editor) throw new Error('Expected an open document.');
      return editor.document.content;
    });
  const original = await content();
  async function enter(seconds: string) {
    await page.click('[data-gap="short"] button');
    await page.waitForSelector('[aria-label="Gap duration in seconds"]');
    await page.keyboard.type(seconds);
    await page.keyboard.press('Enter');
  }
  await enter('0.04');
  await page.waitForFunction(
    () => document.querySelector('[data-gap="short"] button')?.textContent === '0.04s'
  );
  const shortened = await content();
  expect(shortened.slice(2, 4)).toMatchObject([
    { type: 'non_text', source: 's', sourceStart: 1, length: 0.02 },
    { type: 'non_text', source: 's', sourceStart: 1.08, length: 0.02 },
  ]);
  expect(shortened.filter((i) => i.type === 'paragraph_start')).toHaveLength(2);
  expect(await page.$$('[aria-label="Pause"]')).toHaveLength(3);
  await enter('0.02');
  await page.waitForFunction(
    () => document.querySelector('[data-gap="short"] button')?.textContent === '0.02s'
  );
  expect((await content()).slice(2, 4)).toMatchObject([
    { sourceStart: 1, length: 0.01 },
    { sourceStart: 1.09, length: 0.01 },
  ]);
  await page.evaluate(() => window.store.dispatch({ type: '@@redux-undo/UNDO' }));
  expect(await content()).toEqual(shortened);
  await page.evaluate(() => window.store.dispatch({ type: '@@redux-undo/UNDO' }));
  expect(await content()).toEqual(original);
  await page.evaluate(() => window.store.dispatch({ type: '@@redux-undo/REDO' }));
  expect(await content()).toEqual(shortened);

  for (const invalid of ['0', '-1', '2', '0.0001', 'not a number']) {
    await enter(invalid);
    expect(
      await page.$eval('input[aria-label="Gap duration in seconds"]', (input) =>
        input.getAttribute('aria-invalid')
      )
    ).toBe('true');
    expect(await content()).toEqual(shortened);
    await page.keyboard.press('Escape');
  }
  await page.click('[data-gap="short"] button');
  await page.keyboard.type('0.01');
  await page.keyboard.press('Escape');
  expect(await content()).toEqual(shortened);

  await page.click('[data-gap="long"] button');
  await page.keyboard.type('0.5');
  await page.click('#item-ko');
  await page.waitForFunction(
    () => document.querySelector('[data-gap="long"] button')?.textContent === '0.5s'
  );
  await page.click('[data-gap="artificial"] button');
  await page.keyboard.type('0.125');
  await page.keyboard.press('Enter');
  await page.waitForFunction(
    () => document.querySelector('[data-gap="artificial"] button')?.textContent === '0.125s'
  );

  await page.evaluate(() => {
    window.store.dispatch(window.reducers.setSelection(null));
    window.store.dispatch(window.reducers.setUserIndex(1));
    document.getElementById('document')?.focus();
  });
  await page.keyboard.press('Enter');
  expect((await content()).filter((i) => i.type === 'paragraph_start')).toHaveLength(3);
  await page.keyboard.press('Backspace');
  expect((await content()).filter((i) => i.type === 'paragraph_start')).toHaveLength(2);

  const directory = path.resolve('../artifacts/inline-gap');
  fs.mkdirSync(directory, { recursive: true });
  await page.screenshot({ path: path.join(directory, 'inline-gap-seconds.png') });
  await page.click('[data-gap="long"] button');
  await page.waitForSelector('[aria-label="Gap duration in seconds"]');
  await page.screenshot({ path: path.join(directory, 'inline-gap-editing.png') });
});
