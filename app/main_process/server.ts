import path from 'path';
import process from 'process';
import fs from 'fs';
import { ChildProcessWithoutNullStreams, spawn } from 'child_process';
import { app, dialog } from 'electron';
import treeKill from 'tree-kill';
import { publishServerInfo, publishServerStderr } from '../ipc/ipc_main';
import { ServerInfo } from './types';
import { isRunningInTest } from '../src/util';
import { LogLevel, LogSource, logLine } from '../src/util/log';

function findServer() {
  const possibilities = [
    // In packaged app
    path.join(process.resourcesPath, 'server', 'server'),
    path.join(process.resourcesPath, 'server', 'server.exe'),
    // In development
    path.join(process.cwd(), 'server', 'server'),
    path.join(process.cwd(), 'server', 'server.exe'),
  ];
  for (const path of possibilities) {
    if (fs.existsSync(path)) {
      return path;
    }
  }
  return null;
}

function getServerProcess() {
  const runtimeEnvironment = process.env;
  const env = {
    ...runtimeEnvironment,
    AUDAPOLIS_NEP_AUTOSTART:
      isRunningInTest() && runtimeEnvironment.AUDAPOLIS_NEP_AUTOSTART_TEST !== '1'
        ? '0'
        : runtimeEnvironment.AUDAPOLIS_NEP_AUTOSTART ?? '1',
    UV_NO_SYNC: '1',
    UV_OFFLINE: '1',
  };
  if (process.env.NODE_ENV === 'development' || isRunningInTest()) {
    return spawn('uv', ['run', 'python', 'run.py'], {
      stdio: 'pipe',
      cwd: path.resolve(process.cwd(), '../server'),
      env,
      shell: process.platform == 'win32',
      detached: process.platform !== 'win32',
    });
  }
  const serverPath = findServer();
  if (serverPath === null) {
    dialog.showMessageBoxSync({
      type: 'error',
      message:
        'Failed to find local executable for server. Please report this issue to https://github.com/bugbakery/audapolis/issues',
    });
    app.quit();
    return null;
  }
  console.log('Starting server from', serverPath);
  return spawn(serverPath, {
    stdio: 'pipe',
    env,
    detached: process.platform !== 'win32',
  });
}

interface ServerStartingMessage {
  msg: 'server_starting';
  port: number;
}

interface ServerStartedMessage {
  msg: 'server_started';
  token: string;
}

export const serverInfo: ServerInfo = {
  port: null,
  token: null,
  state: 'not started',
};

let serverProcess: ChildProcessWithoutNullStreams | null = null;
let stopping = false;
let shutdownTimer: ReturnType<typeof setTimeout> | null = null;

function stopServerProcess(signal: NodeJS.Signals, done?: () => void) {
  const owned = serverProcess;
  if (!owned || !owned.pid) {
    done?.();
    return;
  }
  if (process.platform !== 'win32') {
    try {
      process.kill(-owned.pid, signal);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ESRCH')
        console.error('Failed to signal the owned local server process group', error);
    }
    done?.();
    return;
  }
  treeKill(owned.pid, signal, (error) => {
    if (error) {
      console.error('Failed to stop the owned local server process tree', error);
      owned.kill(signal);
    }
    done?.();
  });
}

app.on('before-quit', (event) => {
  if (!serverProcess) return;
  event.preventDefault();
  if (stopping) return;
  stopping = true;
  shutdownTimer = setTimeout(() => {
    stopServerProcess('SIGKILL', () => {
      serverProcess = null;
      app.quit();
    });
  }, 30000);
  stopServerProcess('SIGTERM');
});

for (const signal of ['SIGINT', 'SIGTERM'] as const) {
  process.on(signal, () => app.quit());
}
process.on('exit', () => {
  if (!serverProcess?.pid) return;
  try {
    if (process.platform === 'win32') serverProcess.kill();
    else process.kill(-serverProcess.pid, 'SIGTERM');
  } catch (error) {
    console.error('Failed to signal the owned local server on exit', error);
  }
});

function startServer() {
  serverProcess = getServerProcess();
  if (!serverProcess) {
    return;
  }
  let stdoutBuffer = '';
  serverProcess.stdout.on('data', (data: Buffer) => {
    stdoutBuffer += data.toString();

    const lines = stdoutBuffer.split('\n');
    stdoutBuffer = lines.pop() ?? '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }

      try {
        const parsed_data: ServerStartingMessage | ServerStartedMessage = JSON.parse(trimmed);
        logLine &&
          logLine(
            LogSource.ServerProcess,
            LogLevel.Log,
            JSON.stringify({
              ...parsed_data,
              ...('token' in parsed_data && { token: '[redacted]' }),
            })
          );
        if (parsed_data.msg == 'server_starting') {
          publishServerInfo({ state: 'starting', port: parsed_data.port });
        } else if (parsed_data.msg == 'server_started') {
          publishServerInfo({ state: 'running', token: parsed_data.token });
        }
      } catch {
        console.log('error decoding server stdout json');
      }
    }
  });

  serverProcess.stderr.on('data', (data: Buffer) => {
    logLine && logLine(LogSource.ServerProcess, LogLevel.Error, data);
    publishServerStderr(data.toString());
  });

  serverProcess.on('error', (error: Error) => {
    publishServerStderr(`Local server could not start: ${error.message}`);
    publishServerInfo({ state: 'failed to start' });
  });

  serverProcess.on('close', (code: number | null) => {
    if (code == 0) {
      publishServerInfo({ state: 'stopped' });
    } else {
      publishServerInfo({ state: `exited - error code ${code}` });
    }
    serverProcess = null;
    if (shutdownTimer) clearTimeout(shutdownTimer);
    console.log(`child process exited with code ${code}`);
    if (stopping) app.quit();
  });
}
startServer();
