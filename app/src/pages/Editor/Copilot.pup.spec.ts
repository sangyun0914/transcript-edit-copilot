/** @jest-environment ./src/util/jest_puppeteer_electron_environment */

import { spawn, ChildProcessWithoutNullStreams } from 'child_process';
import fs from 'fs';
import path from 'path';
import { createInterface, Interface } from 'readline';

jest.setTimeout(90000);

interface BridgeResult {
  status: number;
  body: Record<string, any>;
}

let bridge: ChildProcessWithoutNullStreams;
let bridgeLines: Interface;
let counter = 0;
let bridgeErrors = '';
const pending = new Map<number, (result: BridgeResult) => void>();
const requests: { route: string; body: any; result: BridgeResult }[] = [];
const exchange = (message: Record<string, unknown>): Promise<BridgeResult> =>
  new Promise((resolve) => {
    const id = ++counter;
    pending.set(id, resolve);
    bridge.stdin.write(JSON.stringify({ id, ...message }) + '\n');
  });

beforeAll(async () => {
  bridge = spawn(path.resolve('../server/.venv/bin/python'), [
    '-u',
    path.resolve('../server/tests/copilot_ui_bridge.py'),
  ]);
  bridge.stderr.on('data', (data) => (bridgeErrors += data.toString()));
  bridgeLines = createInterface({ input: bridge.stdout });
  bridgeLines.on('line', (line) => {
    const result = JSON.parse(line);
    pending.get(result.id)?.(result);
    pending.delete(result.id);
  });
  bridge.on('exit', (code) => {
    for (const resolve of pending.values())
      resolve({ status: 500, body: { detail: `Test bridge exited (${code}): ${bridgeErrors}` } });
    pending.clear();
  });
  await page.waitForFunction(() => !!window.store);
  await page.exposeFunction('copilotTestBridge', async (message: any) => {
    const result = await exchange(message);
    requests.push({ route: message.route, body: message.body, result });
    return result;
  });
  await page.evaluate(() => {
    localStorage.setItem('intro/editor', 'true');
    const originalFetch = window.fetch.bind(window);
    window.fetch = ((input: RequestInfo, init?: RequestInit) => {
      const url = new URL(typeof input === 'string' ? input : input.url, window.location.href);
      if (!url.pathname.startsWith('/copilot/')) return originalFetch(input, init);
      return (window as any)
        .copilotTestBridge({
          route: url.pathname,
          method: init?.method || 'GET',
          body: init?.body ? JSON.parse(String(init.body)) : undefined,
        })
        .then(
          (result: BridgeResult) =>
            new Response(JSON.stringify(result.body), {
              status: result.status,
              headers: { 'content-type': 'application/json' },
            })
        );
    }) as typeof window.fetch;
    window.store.dispatch({
      type: 'server/setLocalServer',
      payload: { hostname: 'http://127.0.0.1', port: 44251, token: null, state: 'running' },
    });
  });
});

afterAll(async () => {
  await page.evaluate(() => {
    window.store.dispatch(window.reducers.setCopilotEnabled(false));
    return window.store.dispatch(window.reducers.requestCopilot('explicit'));
  });
  bridgeLines?.close();
  if (bridge && bridge.exitCode === null) {
    const exited = new Promise<void>((resolve) => bridge.once('close', () => resolve()));
    bridge.stdin.end();
    await exited;
  }
});

async function openScene(language: 'en' | 'ko', documentContent?: Record<string, any>[]) {
  await page.evaluate(
    ({ language, documentContent }) => {
      window.store.dispatch(window.reducers.setCopilotEnabled(false));
      const words =
        language === 'en' ? ['We', 'recieve', 'updates.'] : ['새로운', '데이타', '입니다.'];
      const items: Record<string, unknown>[] =
        documentContent ||
        words.map((text, index) => ({
          type: 'text',
          uuid: ['a', 'b', 'c'][index],
          source: 's',
          sourceStart: index * 1.25,
          length: 1.25,
          text,
          conf: 1,
        }));
      if (!documentContent) {
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
          window.store.dispatch(window.reducers.setUserIndex(2));
          document.getElementById('document')?.focus();
        });
    },
    { language, documentContent: documentContent || null }
  );
}

async function predict() {
  await page.evaluate(() => {
    document.getElementById('document')?.focus();
    window.store.dispatch(window.reducers.setCopilotEnabled(true));
    return window.store.dispatch(window.reducers.requestCopilot('explicit'));
  });
}

const content = () => page.evaluate(() => window.store.getState().editor.present!.document.content);
const waitPreview = async () => {
  await page.waitForSelector('[data-testid="copilot-preview"]', { visible: true });
  await page.waitForFunction(() => window.store.getState().copilot.proposal?.shown);
};
const hoverActions = async () => {
  await page.hover('[data-testid="copilot-preview"]');
  await page.waitForSelector('#copilot-suggestion-actions', { visible: true });
};

test('real gateway validates EN/KO preview, Tab acceptance, one undo and Esc feedback', async () => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await openScene('en');
  const original = await content();
  await predict();
  await waitPreview();
  expect(await content()).toEqual(original);
  expect(await page.$eval('[data-testid="copilot-preview"]', (node) => node.textContent)).toContain(
    'receive'
  );
  expect(await page.$eval('#item-b del', (node) => node.textContent)).toBe('recieve');
  expect(await page.$eval('#item-b del', (node) => getComputedStyle(node).textDecorationLine)).toBe(
    'line-through'
  );
  expect(await page.$eval('[data-copilot-added]', (node) => node.textContent)).toBe('receive');
  expect(await page.$('#copilot-suggestion-actions')).toBeNull();
  await page.evaluate(() => document.fonts.ready.then(() => null));
  await page.waitForFunction(() =>
    document
      .getElementById('document')
      ?.getAnimations({ subtree: true })
      .every((animation) => animation.playState !== 'running')
  );
  const nextWordPosition = () =>
    page.$eval('#item-c', (node) => {
      const rect = node.getBoundingClientRect();
      return { x: rect.x, y: rect.y };
    });
  const positionBeforeHover = await nextWordPosition();
  expect(
    await page.$eval('[data-testid="copilot-preview"]', (node) => {
      const original = document.getElementById('item-b');
      return (
        original && node.getBoundingClientRect().height <= original.getBoundingClientRect().height
      );
    })
  ).toBe(true);
  const output = path.resolve('../artifacts/phase-l-ui');
  fs.mkdirSync(output, { recursive: true });
  await page.screenshot({
    path: path.join(output, 'english-preview.png'),
    captureBeyondViewport: false,
  });
  const clip = await page.$eval('#item-b', (node) => {
    const paragraph = node.parentElement?.parentElement;
    if (!paragraph) throw new Error('Missing paragraph.');
    const rect = paragraph.getBoundingClientRect();
    return { x: rect.x, y: rect.y - 8, width: rect.width, height: rect.height + 16 };
  });
  await page.screenshot({
    path: path.join(output, 'compact-preview.png'),
    clip,
    captureBeyondViewport: false,
  });
  await hoverActions();
  expect(await nextWordPosition()).toEqual(positionBeforeHover);
  await page.screenshot({
    path: path.join(output, 'english-hover-preview.png'),
    captureBeyondViewport: false,
  });
  await page.mouse.move(0, 0);
  await page.waitForSelector('#copilot-suggestion-actions', { hidden: true });
  await page.keyboard.press('Tab');
  await page.waitForFunction(
    () => (window.store.getState().editor.present!.document.content[2] as any).text === 'receive'
  );
  const edited = await content();
  expect(edited[2]).toEqual({ ...original[2], text: 'receive' });
  expect(
    await page.evaluate(() =>
      Array.from(
        new Uint8Array(window.store.getState().editor.present!.document.sources.s.fileContents)
      )
    )
  ).toEqual([1, 2, 3]);
  await page.evaluate(() => window.store.dispatch({ type: '@@redux-undo/UNDO' }));
  expect(await content()).toEqual(original);
  await predict();
  await page.waitForFunction(() => !document.querySelector('[data-testid="copilot-preview"]'));

  await openScene('ko');
  const korean = await content();
  await predict();
  await waitPreview();
  expect(await page.$eval('[data-testid="copilot-preview"]', (node) => node.textContent)).toContain(
    '데이터'
  );
  await page.screenshot({
    path: path.join(output, 'korean-preview.png'),
    captureBeyondViewport: false,
  });
  await page.keyboard.press('Escape');
  await page.waitForFunction(() => !document.querySelector('[data-testid="copilot-preview"]'));
  expect(await content()).toEqual(korean);
  await predict();
  await page.waitForFunction(() => !document.querySelector('[data-testid="copilot-preview"]'));
  const feedback = requests.filter((request) => request.route.includes('/feedback'));
  expect(feedback.map((request) => request.body.outcome)).toEqual(
    expect.arrayContaining(['shown', 'accepted', 'undone', 'dismissed'])
  );
  expect(feedback.every((request) => request.result.status === 200)).toBe(true);
  expect(
    requests.some(
      (request) => request.route.includes('/predict') && request.result.body.status === 'suppressed'
    )
  ).toBe(true);
  expect(requests.filter((request) => request.result.status >= 400)).toEqual([]);
  expect(pageErrors).toEqual([]);
});

test('IME invalidates a preview and Tab remains an ordinary key without a suggestion', async () => {
  await openScene('ko');
  const original = await content();
  await predict();
  await waitPreview();
  await page.evaluate(() => {
    document
      .getElementById('document')!
      .dispatchEvent(new CompositionEvent('compositionstart', { bubbles: true }));
  });
  await page.waitForFunction(() => !document.querySelector('[data-testid="copilot-preview"]'));
  await page.keyboard.press('Tab');
  expect(await content()).toEqual(original);
  await page.evaluate(() => {
    document
      .getElementById('document')!
      .dispatchEvent(new CompositionEvent('compositionend', { bubbles: true }));
    window.store.dispatch(window.reducers.setCopilotEnabled(false));
  });
  const prevented = await page.evaluate(() => {
    const node = document.getElementById('document')!;
    node.focus();
    const event = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
    node.dispatchEvent(event);
    return event.defaultPrevented;
  });
  expect(prevented).toBe(false);
  expect(await content()).toEqual(original);
});

test('multi-word replacement follows the original, and hover controls accept or dismiss it', async () => {
  await exchange({
    command: 'action',
    prediction: { action: 'correct_text', target: [0, 1], text: 'We receive' },
  });
  await openScene('en');
  const original = await content();
  await predict();
  await waitPreview();
  expect(
    await page.$$eval('[data-copilot-removed]', (nodes) => nodes.map((node) => node.textContent))
  ).toEqual(['We', 'recieve']);
  expect(
    await page.$eval('#item-b', (node) => {
      const addition = document.querySelector('[data-copilot-added]');
      return Boolean(
        addition && node.compareDocumentPosition(addition) & Node.DOCUMENT_POSITION_FOLLOWING
      );
    })
  ).toBe(true);
  await hoverActions();
  await page.click('[aria-label="Accept suggestion"]');
  await page.waitForFunction(() => !document.querySelector('[data-testid="copilot-preview"]'));
  expect(
    (await content()).filter((item) => item.type === 'text').map((item: any) => item.text)
  ).toEqual(['We receive', 'updates.']);
  expect(await page.evaluate(() => document.activeElement?.id)).toBe('document');
  await page.evaluate(() => window.store.dispatch({ type: '@@redux-undo/UNDO' }));
  expect(await content()).toEqual(original);
  await openScene('en');
  await predict();
  await waitPreview();
  await hoverActions();
  await page.click('[aria-label="Dismiss suggestion"]');
  expect(await content()).toEqual(original);
  expect(await page.$('[data-copilot-removed]')).toBeNull();
  await exchange({ command: 'action', prediction: null });
});

test('audio and caption previews mark only the changed content and remain usable in a narrow window', async () => {
  const fixtures = fs
    .readFileSync(path.resolve('../ml/fixtures/execution-v2.jsonl'), 'utf8')
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line));
  for (const [id, label, removed, added] of [
    ['delete-with-gap', 'Delete text + audio', 3, null],
    ['trim-pause', '0.725s pause', 1, '0.725s pause'],
    ['remove-pause', 'Remove pause', 1, null],
    ['split-caption', 'New caption', 0, '¶ New caption'],
    ['merge-caption', 'Join with previous caption', 1, null],
  ] as const) {
    const fixture = fixtures.find((row) => row.id === id);
    if (id === 'split-caption' || id === 'merge-caption') {
      let sourceStart = 0;
      for (const item of fixture.before.content) {
        if (item.type === 'text') {
          item.sourceStart = sourceStart;
          item.length = 1.25;
          sourceStart += item.length;
        }
      }
    }
    if (id === 'trim-pause') {
      const gap = fixture.before.content[2];
      fixture.before.content.splice(
        2,
        1,
        { ...gap, length: 1 },
        { ...gap, uuid: 'gap-tail', sourceStart: gap.sourceStart + 1, length: 0.2 }
      );
    }
    await exchange({ command: 'action', prediction: fixture.prediction });
    await openScene(
      'ko',
      fixture.before.content.map((item: any) =>
        'source' in item ? { ...item, source: 's' } : item
      )
    );
    const original = await content();
    await predict();
    const state = await page.evaluate(() => {
      const copilot = window.store.getState().copilot;
      return { status: copilot.status, error: copilot.error };
    });
    expect({ id, ...state }).toEqual({ id, status: 'preview', error: null });
    await waitPreview();
    expect(await content()).toEqual(original);
    expect(
      await page.$eval('[data-testid="copilot-preview"]', (node) => node.textContent)
    ).toContain(label);
    expect(await page.$$('[data-copilot-removed]')).toHaveLength(removed);
    expect(
      await page.$eval(
        '#document',
        (node) => node.querySelector('[data-copilot-added]')?.textContent || null
      )
    ).toBe(added);
    if (id === 'merge-caption') {
      expect(await page.$eval('[data-copilot-removed]', (node) => node.textContent)).toBe('¶');
    }
    if (id === 'trim-pause') {
      expect(await page.$eval('[data-copilot-removed] button', (node) => node.textContent)).toBe(
        '1s'
      );
      expect(await page.$eval('#item-gap-tail', (node) => node.closest('del'))).toBeNull();
    }
    await page.screenshot({
      path: path.resolve(`../artifacts/phase-l-ui/${id}-preview.png`),
      captureBeyondViewport: false,
    });
    await hoverActions();
    await page.click('[aria-label="Dismiss suggestion"]');
    expect(await content()).toEqual(original);
  }

  await exchange({ command: 'action', prediction: null });
  const viewport = page.viewport();
  await page.setViewport({ width: 560, height: 760 });
  await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }]);
  try {
    await openScene('en');
    await predict();
    await waitPreview();
    await page.screenshot({
      path: path.resolve('../artifacts/phase-l-ui/narrow-light-preview.png'),
      captureBeyondViewport: false,
    });
    await page.focus('[data-testid="copilot-preview"]');
    await page.waitForSelector('#copilot-suggestion-actions', { visible: true });
    const bounds = await page.$eval('[data-testid="copilot-preview"]', (node) => {
      const preview = node.getBoundingClientRect();
      const actions = document.getElementById('copilot-suggestion-actions');
      if (!actions) throw new Error('Missing suggestion controls.');
      const menu = actions.getBoundingClientRect();
      return {
        right: preview.right,
        menuRight: menu.right,
        menuLeft: menu.left,
        width: innerWidth,
      };
    });
    expect(Math.max(bounds.right, bounds.menuRight)).toBeLessThanOrEqual(bounds.width);
    expect(bounds.menuLeft).toBeGreaterThanOrEqual(0);
    await page.screenshot({
      path: path.resolve('../artifacts/phase-l-ui/narrow-light-focused-preview.png'),
      captureBeyondViewport: false,
    });
    await page.keyboard.press('ArrowDown');
    expect(await page.evaluate(() => document.activeElement?.getAttribute('aria-label'))).toBe(
      'Accept suggestion'
    );
    await page.keyboard.press('Tab');
    expect(await page.evaluate(() => document.activeElement?.getAttribute('aria-label'))).toBe(
      'Dismiss suggestion'
    );
    await page.keyboard.press('Enter');
    expect(await page.$('[data-testid="copilot-preview"]')).toBeNull();
  } finally {
    if (viewport) await page.setViewport(viewport);
    await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
  }
});
