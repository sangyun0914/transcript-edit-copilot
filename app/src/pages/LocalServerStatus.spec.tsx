import React from 'react';
import ReactDOM from 'react-dom';
import { act } from 'react-dom/test-utils';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import { LandingPage } from './Landing';
import { CopilotControls } from './Editor/Copilot';
import server, { receiveLocalServerInfo, serverSlice } from '../state/server';
import copilot, { createCopilotMiddleware } from '../state/copilot';
import { Page } from '../state/nav';
import * as api from '../server_api/copilot';

jest.mock('../../ipc/ipc_renderer');
jest.mock('../server_api/copilot');
jest.mock('../components/TitleBar', () => ({ TitleBar: () => null }));
jest.mock('../tour/LandingTour', () => ({ LandingTour: () => null }));
jest.mock('../components/Menu', () => ({
  MenuBar: () => null,
  MenuGroup: () => null,
  MenuItem: () => null,
}));

let container: HTMLDivElement;

function setup() {
  return configureStore({
    reducer: {
      server,
      copilot,
      editor: () => ({ present: null }),
      nav: () => ({ page: Page.Editor }),
    },
    middleware: (getDefault) => getDefault().concat(createCopilotMiddleware()),
  });
}

function mount(store: ReturnType<typeof setup>, child: JSX.Element) {
  act(() => {
    ReactDOM.render(<Provider store={store}>{child}</Provider>, container);
  });
}

function button(text: string) {
  const result = Array.from(container.querySelectorAll('button')).find(
    (item) => item.textContent === text
  );
  if (!result) throw new Error(`Missing button: ${text}`);
  return result;
}

beforeEach(() => {
  jest.useFakeTimers();
  jest.clearAllMocks();
  (api.copilotHealth as jest.Mock).mockResolvedValue({ status: 'ready' });
  container = document.createElement('div');
  document.body.appendChild(container);
});

afterEach(() => {
  act(() => {
    ReactDOM.unmountComponentAtNode(container);
  });
  container.remove();
  jest.clearAllTimers();
  jest.useRealTimers();
});

test.each(['not started', 'starting'])('normal %s still shows the waiting screen', (state) => {
  const store = setup();
  store.dispatch(receiveLocalServerInfo({ state, port: null, token: null }));
  mount(store, <LandingPage />);
  expect(container.textContent).toContain('Starting app...');
  expect(container.textContent).not.toContain('Open Existing');
  expect(container.querySelector('[role="alert"]')).toBeNull();
});

test('cold null-port failure leaves document editing available and renders suggestions unavailable', () => {
  const store = setup();
  mount(store, <LandingPage />);
  act(() => {
    store.dispatch(receiveLocalServerInfo({ state: 'failed to start', port: null, token: null }));
  });
  expect(container.textContent).not.toContain('Starting app...');
  expect(container.querySelector('[role="alert"]')?.textContent).toContain(
    'local server is unavailable'
  );
  expect(button('Open Existing').disabled).toBe(false);
  expect(button('New Blank Document').disabled).toBe(false);
  expect(button('Import & Transcribe').disabled).toBe(true);
  expect(store.getState().server.servers).toEqual([]);
  mount(store, <CopilotControls />);
  expect(container.querySelector('#copilot-status')?.textContent).toBe(
    'Local suggestions are unavailable. You can continue editing.'
  );
  expect(api.copilotHealth).not.toHaveBeenCalled();
});

test('running recovery restores the normal landing controls and ready status', async () => {
  const store = setup();
  store.dispatch(receiveLocalServerInfo({ state: 'stopped', port: null, token: null }));
  mount(store, <LandingPage />);
  await act(async () => {
    store.dispatch(receiveLocalServerInfo({ state: 'running', port: 45001, token: 'fixture' }));
    await Promise.resolve();
  });
  expect(container.querySelector('[role="alert"]')).toBeNull();
  expect(button('Import & Transcribe').disabled).toBe(false);
  mount(store, <CopilotControls />);
  await act(async () => {
    await Promise.resolve();
  });
  expect(container.querySelector('#copilot-status')?.textContent).toBe(
    'Suggestions appear automatically as you edit.'
  );
});

test('an explicitly selected remote server stays usable when the local process fails', () => {
  const store = setup();
  store.dispatch(
    serverSlice.actions.addServer({
      hostname: 'https://asr.example.com',
      port: 443,
      token: 'remote',
      name: 'Remote ASR',
    })
  );
  store.dispatch(receiveLocalServerInfo({ state: 'failed to start', port: null, token: null }));
  mount(store, <LandingPage />);
  expect(container.querySelector('[role="alert"]')).toBeNull();
  expect(button('Import & Transcribe').disabled).toBe(false);
  expect(store.getState().server.servers[0].hostname).toBe('https://asr.example.com');
});
