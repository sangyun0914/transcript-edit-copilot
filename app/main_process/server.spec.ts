import { EventEmitter } from 'events';

describe('owned local server startup and shutdown', () => {
  const originalEnv = process.env;
  const originalPlatform = process.platform;
  let child: EventEmitter & {
    pid: number;
    stdout: EventEmitter;
    stderr: EventEmitter;
    kill: jest.Mock;
  };
  let app: EventEmitter & { quit: jest.Mock };
  let spawn: jest.Mock;
  let treeKill: jest.Mock;
  let killGroup: jest.SpyInstance;

  beforeEach(() => {
    jest.resetModules();
    jest.useFakeTimers();
    process.env = { ...originalEnv, NODE_ENV: 'test' };
    Object.defineProperty(process, 'platform', { value: 'darwin' });
    delete process.env.AUDAPOLIS_NEP_AUTOSTART;
    delete process.env.AUDAPOLIS_NEP_AUTOSTART_TEST;
    child = Object.assign(new EventEmitter(), {
      pid: 12345,
      stdout: new EventEmitter(),
      stderr: new EventEmitter(),
      kill: jest.fn(),
    });
    app = Object.assign(new EventEmitter(), { quit: jest.fn() });
    spawn = jest.fn(() => child);
    treeKill = jest.fn((_pid, _signal, callback) => callback());
    jest.doMock('electron', () => ({ app, dialog: {} }));
    jest.doMock('child_process', () => ({ spawn }));
    jest.doMock('tree-kill', () => ({ __esModule: true, default: treeKill }));
    jest.doMock('../ipc/ipc_main', () => ({
      publishServerInfo: jest.fn(),
      publishServerStderr: jest.fn(),
    }));
    jest.doMock('../src/util', () => ({
      isRunningInTest: () => process.env.NODE_ENV === 'test',
    }));
    jest.doMock('../src/util/log', () => ({ logLine: undefined }));
    jest.spyOn(process, 'on').mockReturnValue(process);
    killGroup = jest.spyOn(process, 'kill').mockReturnValue(true);
    jest.spyOn(console, 'log').mockImplementation(() => {});
  });

  afterEach(() => {
    process.env = originalEnv;
    Object.defineProperty(process, 'platform', { value: originalPlatform });
    jest.clearAllTimers();
    jest.useRealTimers();
    jest.restoreAllMocks();
  });

  async function loadServer() {
    await import('./server');
    return spawn.mock.calls[0][2];
  }

  it('starts the ordinary development gateway with offline automatic model startup', async () => {
    process.env.NODE_ENV = 'development';
    process.env.AUDAPOLIS_NEP_CONFIG = '/explicit/external/gateway.json';
    const options = await loadServer();
    expect(options.env.AUDAPOLIS_NEP_AUTOSTART).toBe('1');
    expect(options.env.AUDAPOLIS_NEP_CONFIG).toBe('/explicit/external/gateway.json');
    expect(options.env.UV_NO_SYNC).toBe('1');
    expect(options.env.UV_OFFLINE).toBe('1');
    expect(spawn).toHaveBeenCalledWith('uv', ['run', 'python', 'run.py'], options);
  });

  it('suppresses model autostart for normal tests, even with inherited autostart', async () => {
    process.env.AUDAPOLIS_NEP_AUTOSTART = '1';
    const options = await loadServer();
    expect(options.env.AUDAPOLIS_NEP_AUTOSTART).toBe('0');
  });

  it('preserves an explicit runtime autostart override for ordinary development', async () => {
    process.env.NODE_ENV = 'development';
    process.env.AUDAPOLIS_NEP_AUTOSTART = '0';
    const options = await loadServer();
    expect(options.env.AUDAPOLIS_NEP_AUTOSTART).toBe('0');
  });

  it('allows an explicit actual-startup test to opt in', async () => {
    process.env.AUDAPOLIS_NEP_AUTOSTART_TEST = '1';
    const options = await loadServer();
    expect(options.env.AUDAPOLIS_NEP_AUTOSTART).toBe('1');
  });

  it('publishes a complete startup token without copying it into logs', async () => {
    const logLine = jest.fn();
    jest.doMock('../src/util/log', () => ({
      logLine,
      LogSource: { ServerProcess: 'server' },
      LogLevel: { Log: 'log' },
    }));
    await loadServer();
    child.stdout.emit('data', Buffer.from('{"msg":"server_started","token":"fixture-'));
    child.stdout.emit('data', Buffer.from('secret"}\n'));
    const { publishServerInfo } = await import('../ipc/ipc_main');
    expect(publishServerInfo).toHaveBeenCalledWith({ state: 'running', token: 'fixture-secret' });
    expect(JSON.stringify(logLine.mock.calls)).not.toContain('fixture-secret');
    expect(logLine).toHaveBeenCalledWith(
      'server',
      'log',
      '{"msg":"server_started","token":"[redacted]"}'
    );
  });

  it('waits for the owned server tree to close before quitting and does not stop twice', async () => {
    await loadServer();
    const event = { preventDefault: jest.fn() };
    app.emit('before-quit', event);
    app.emit('before-quit', event);
    expect(killGroup).toHaveBeenCalledTimes(1);
    expect(killGroup).toHaveBeenCalledWith(-child.pid, 'SIGTERM');
    expect(treeKill).not.toHaveBeenCalled();
    expect(app.quit).not.toHaveBeenCalled();
    child.emit('close', 0);
    expect(app.quit).toHaveBeenCalledTimes(1);
    jest.advanceTimersByTime(30000);
    expect(killGroup).toHaveBeenCalledTimes(1);
  });

  it('bounds shutdown by killing only the owned tree if it does not close', async () => {
    await loadServer();
    app.emit('before-quit', { preventDefault: jest.fn() });
    jest.advanceTimersByTime(30000);
    expect(killGroup).toHaveBeenLastCalledWith(-child.pid, 'SIGKILL');
    expect(app.quit).toHaveBeenCalledTimes(1);
  });

  it('retains the process-tree fallback on Windows', async () => {
    Object.defineProperty(process, 'platform', { value: 'win32' });
    await loadServer();
    app.emit('before-quit', { preventDefault: jest.fn() });
    expect(treeKill).toHaveBeenCalledWith(child.pid, 'SIGTERM', expect.any(Function));
    expect(killGroup).not.toHaveBeenCalled();
  });
});
