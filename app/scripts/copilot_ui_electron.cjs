const fs = require('fs');
const path = require('path');
const { app, BrowserWindow, ipcMain } = require('electron');
const { validateBootstrap } = require('./copilot_ui_helpers.cjs');

const root = process.env.COPILOT_UI_TEMP;
const actualAppStartup = validateBootstrap(process.env);
if (process.env.COPILOT_UI_SHUTDOWN_TRACE === '1') {
  const windows = () =>
    BrowserWindow.getAllWindows().map((window) => ({
      id: window.id,
      destroyed: window.isDestroyed(),
      visible: window.isVisible(),
      focused: window.isFocused(),
      web_contents_id: window.webContents.id,
      devtools_open: window.webContents.isDevToolsOpened(),
    }));
  const record = (event, details = {}) =>
    fs.appendFileSync(
      path.join(process.env.COPILOT_UI_OUTPUT, 'shutdown-trace.jsonl'),
      JSON.stringify({ at: new Date().toISOString(), event, ...details }) + '\n'
    );
  let quitting = false;
  for (const event of ['before-quit', 'will-quit', 'quit', 'window-all-closed', 'activate'])
    app.on(event, (value) => {
      if (event === 'before-quit') quitting = true;
      record(event, { windows: windows() });
      if (event === 'before-quit' || event === 'will-quit')
        setImmediate(() =>
          record(`${event}-after-listeners`, {
            default_prevented: value.defaultPrevented,
            windows: windows(),
          })
        );
    });
  app.on('browser-window-created', (_event, window) => {
    const windowId = window.id;
    record('window-created', { window_id: windowId });
    for (const event of ['close', 'closed', 'unresponsive', 'responsive'])
      window.on(event, (value) => {
        record(`window-${event}`, { window_id: windowId });
        if (event === 'close')
          setImmediate(() =>
            record('window-close-after-listeners', {
              window_id: windowId,
              default_prevented: value.defaultPrevented,
              destroyed: window.isDestroyed(),
              windows: windows(),
            })
          );
      });
    const id = window.webContents.id;
    for (const event of ['destroyed', 'devtools-opened', 'devtools-closed', 'will-prevent-unload'])
      window.webContents.on(event, () => record(`web-contents-${event}`, { id }));
    window.webContents.on('render-process-gone', (_event, details) =>
      record('render-process-gone', { id, reason: details.reason, exit_code: details.exitCode })
    );
  });
  for (const signal of ['SIGINT', 'SIGTERM'])
    process.on(signal, () => record('signal', { signal }));
  process.on('exit', (code) => record('process-exit', { code }));
  process.on('uncaughtExceptionMonitor', (error) =>
    record('uncaught-exception', {
      error: String(error),
      stack_frames: String(error.stack).split('\n').slice(1, 9),
    })
  );
  for (const channel of ['set-menubar', 'show-menubar', 'show-contextmenu'])
    ipcMain.prependListener(channel, (event) => {
      if (quitting)
        record('ipc-during-quit', {
          channel,
          sender_destroyed: event.sender.isDestroyed(),
          window_id: BrowserWindow.fromWebContents(event.sender)?.id ?? null,
        });
    });
  const quit = app.quit.bind(app);
  app.quit = () => {
    record('app-quit-call', { windows: windows() });
    quit();
    record('app-quit-return', { windows: windows() });
  };
  const childProcess = require('child_process');
  const spawn = childProcess.spawn;
  childProcess.spawn = (...args) => {
    const child = spawn(...args);
    record('child-spawn', { command: args[0], pid: child.pid });
    child.on('exit', (code, signal) => record('child-exit', { pid: child.pid, code, signal }));
    child.on('close', (code, signal) => record('child-close', { pid: child.pid, code, signal }));
    child.on('error', (error) => record('child-error', { pid: child.pid, error: String(error) }));
    if (args[0] === 'uv')
      child.stderr.on('data', (data) => record('server-stderr', { text: data.toString() }));
    return child;
  };
}
for (const name of ['userData', 'logs', 'cache', 'data'])
  fs.mkdirSync(path.join(root, name), { recursive: true });
app.setPath('userData', path.join(root, 'userData'));
app.setPath('logs', path.join(root, 'logs'));
process.env.AUDAPOLIS_DATA_DIR = path.join(root, 'data');
process.env.AUDAPOLIS_CACHE_DIR = path.join(root, 'cache');
delete process.env.AUDAPOLIS_NEP_CONFIG;
delete process.env.AUDAPOLIS_NEP_DEFAULT_DEPLOYMENT;
process.env.AUDAPOLIS_NEP_AUTOSTART = actualAppStartup ? '1' : '0';
delete process.env.AUDAPOLIS_NEP_AUTOSTART_TEST;

// The test uses the real renderer/main entrypoint, without an unrelated update request.
require('electron-updater').autoUpdater.checkForUpdatesAndNotify = async () => null;
require('electron-devtools-installer').default = async () => 'disabled for isolated UI checks';
fs.writeFileSync(
  path.join(process.env.COPILOT_UI_OUTPUT, 'electron-isolation.json'),
  JSON.stringify(
    {
      pid: process.pid,
      userData: app.getPath('userData'),
      data: process.env.AUDAPOLIS_DATA_DIR,
      cache: process.env.AUDAPOLIS_CACHE_DIR,
      default_model_config: null,
      default_deployment_selection: actualAppStartup
        ? path.resolve('../artifacts/local-runtime/gemma-actions-v3/default-deployment.json')
        : null,
      fixture_kind: process.env.COPILOT_UI_MODE || 'scripted_known_action_product_checks',
      node_env: process.env.NODE_ENV,
      model_autostart: actualAppStartup,
      updater_disabled_for_test: true,
      devtools_extension_download_disabled_for_test: true,
    },
    null,
    2
  )
);
require('../build/main_process/start.cjs.js');
