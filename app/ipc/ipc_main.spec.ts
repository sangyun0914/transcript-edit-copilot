import { BrowserWindow, dialog, ipcMain } from 'electron';
import { publishServerInfo } from './ipc_main';
import { serverInfo } from '../main_process/server';
import { sendAll } from '../main_process/windowList';
import { setMenuBar, showContextMenu, showMenuBar } from '../main_process/menu';

jest.mock('electron', () => ({
  ipcMain: { handle: jest.fn(), on: jest.fn() },
  BrowserWindow: { fromWebContents: jest.fn() },
  dialog: { showOpenDialog: jest.fn(), showSaveDialog: jest.fn() },
}));
jest.mock('../main_process/server', () => ({
  serverInfo: { port: null, token: null, state: 'not started' },
}));
jest.mock('../main_process/windowList', () => ({ sendAll: jest.fn() }));
jest.mock('../main_process/menu', () => ({
  setMenuBar: jest.fn(),
  showMenuBar: jest.fn(),
  showContextMenu: jest.fn(),
}));
jest.mock('../src/util/log', () => ({ logFilePath: null }));

beforeEach(() => {
  Object.assign(serverInfo, { port: null, token: null, state: 'not started' });
  (sendAll as jest.Mock).mockClear();
  for (const callback of [
    BrowserWindow.fromWebContents,
    dialog.showOpenDialog,
    dialog.showSaveDialog,
    setMenuBar,
    showMenuBar,
    showContextMenu,
  ])
    (callback as jest.Mock).mockReset();
});

function ipcListener(method: 'handle' | 'on', channel: string) {
  const subscription = (ipcMain[method] as jest.Mock).mock.calls.find(([name]) => name === channel);
  if (!subscription) throw new Error(`Missing IPC subscription: ${channel}`);
  return subscription[1];
}

test.each(['detached', 'destroyed'])('late IPC from a %s window is harmless', (state) => {
  const sender = {};
  (BrowserWindow.fromWebContents as jest.Mock).mockReturnValue(
    state === 'detached' ? null : { isDestroyed: () => true }
  );
  for (const channel of ['set-menubar', 'show-menubar', 'show-contextmenu'])
    expect(() => ipcListener('on', channel)({ sender }, [])).not.toThrow();
  expect(ipcListener('handle', 'open-file')({ sender }, {})).toEqual({
    canceled: true,
    filePaths: [],
  });
  expect(ipcListener('handle', 'save-file')({ sender }, {})).toEqual({
    canceled: true,
  });
  for (const callback of [
    setMenuBar,
    showMenuBar,
    showContextMenu,
    dialog.showOpenDialog,
    dialog.showSaveDialog,
  ])
    expect(callback).not.toHaveBeenCalled();
});

test('live-window menu and file IPC keeps the requesting window and native dialog result', () => {
  const window = { isDestroyed: () => false };
  const sender = {};
  const items = [{ label: 'File' }];
  const options = { title: 'Choose a document' };
  const opened = Promise.resolve({ canceled: false, filePaths: ['/fixture.aup'] });
  const saved = Promise.resolve({ canceled: false, filePath: '/fixture.aup' });
  (BrowserWindow.fromWebContents as jest.Mock).mockReturnValue(window);
  (dialog.showOpenDialog as jest.Mock).mockReturnValue(opened);
  (dialog.showSaveDialog as jest.Mock).mockReturnValue(saved);

  ipcListener('on', 'set-menubar')({ sender }, items);
  ipcListener('on', 'show-menubar')({ sender });
  ipcListener('on', 'show-contextmenu')({ sender }, items);
  expect(setMenuBar).toHaveBeenCalledWith(window, items);
  expect(showMenuBar).toHaveBeenCalledWith(window);
  expect(showContextMenu).toHaveBeenCalledWith(window, items);
  expect(ipcListener('handle', 'open-file')({ sender }, options)).toBe(opened);
  expect(ipcListener('handle', 'save-file')({ sender }, options)).toBe(saved);
  expect(dialog.showOpenDialog).toHaveBeenCalledWith(window, options);
  expect(dialog.showSaveDialog).toHaveBeenCalledWith(window, options);
  expect(BrowserWindow.fromWebContents).toHaveBeenCalledWith(sender);
});

test('a cold startup failure is broadcast and replayed without inventing a port', () => {
  publishServerInfo({ state: 'failed to start' });
  expect(sendAll).toHaveBeenCalledWith('local-server-info', {
    port: null,
    token: null,
    state: 'failed to start',
  });
  const request = (ipcMain.on as jest.Mock).mock.calls.find(
    ([event]) => event === 'local-server-request'
  );
  if (!request) throw new Error('Missing local server request subscription.');
  request[1]();
  expect(sendAll).toHaveBeenCalledTimes(2);
  expect((sendAll as jest.Mock).mock.calls[1][1].port).toBeNull();
});

test('starting, running and stopped updates keep their actual connection information', () => {
  publishServerInfo({ state: 'starting', port: 45001 });
  expect(sendAll).toHaveBeenLastCalledWith('local-server-info', {
    state: 'starting',
    port: 45001,
    token: null,
  });
  publishServerInfo({ state: 'running', token: 'fixture-token' });
  expect(sendAll).toHaveBeenLastCalledWith('local-server-info', {
    state: 'running',
    port: 45001,
    token: 'fixture-token',
  });
  publishServerInfo({ state: 'stopped' });
  expect(sendAll).toHaveBeenLastCalledWith('local-server-info', {
    state: 'stopped',
    port: 45001,
    token: 'fixture-token',
  });
});
