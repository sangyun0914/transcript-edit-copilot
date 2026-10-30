const { execFileSync } = require('child_process');
const treeKill = require('tree-kill');

const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const actualStartupMode = (env) => env.COPILOT_UI_MODE === 'app-startup-actual';
const exited = (child) => child.exitCode !== null || child.signalCode != null;

function buildMode(env) {
  return actualStartupMode(env) ? 'development' : 'test';
}

function isolatedModelEnvironment(env) {
  const actual = actualStartupMode(env);
  const result = {
    ...env,
    NODE_ENV: buildMode(env),
    MODE: buildMode(env),
    AUDAPOLIS_NEP_AUTOSTART: actual ? '1' : '0',
  };
  delete result.AUDAPOLIS_NEP_CONFIG;
  delete result.AUDAPOLIS_NEP_DEFAULT_DEPLOYMENT;
  delete result.AUDAPOLIS_NEP_AUTOSTART_TEST;
  return result;
}

function validateBootstrap(env) {
  const actual = actualStartupMode(env);
  if (
    !env.COPILOT_UI_TEMP ||
    !env.COPILOT_UI_OUTPUT ||
    env.NODE_ENV !== buildMode(env) ||
    env.MODE !== buildMode(env) ||
    (actual &&
      (env.AUDAPOLIS_NEP_AUTOSTART !== '1' || env.AUDAPOLIS_NEP_AUTOSTART_TEST !== undefined))
  )
    throw new Error('Use an isolated test profile and an explicit app-startup-actual opt-in.');
  return actual;
}

function readProcessTable(timeout = 1000) {
  const text = execFileSync('ps', ['-eo', 'pid=,ppid=,lstart=,stat=,comm='], {
    encoding: 'utf8',
    env: { ...process.env, LC_ALL: 'C' },
    timeout,
  });
  const rows = text
    .split('\n')
    .map((line) =>
      line.match(
        /^\s*(\d+)\s+(\d+)\s+(\w{3}\s+\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4})\s+(\S+)\s+(.+)$/
      )
    )
    .filter(Boolean)
    .map((match) => ({
      pid: Number(match[1]),
      ppid: Number(match[2]),
      started: match[3].replace(/\s+/g, ' '),
      state: match[4],
      command: match[5],
    }));
  if (!rows.length) throw new Error('Could not read process identities for cleanup.');
  return rows;
}

function sameProcess(left, right) {
  return left.pid === right.pid && left.started === right.started;
}

function trackOwnedTree(child, known, table) {
  const identities = new Map(known.map((row) => [row.pid, row]));
  const roots = new Set(
    table.filter((row) => known.some((old) => sameProcess(old, row))).map((row) => row.pid)
  );
  if (child?.pid && !exited(child)) roots.add(child.pid);
  let changed = true;
  while (changed) {
    changed = false;
    for (const row of table) {
      if (roots.has(row.ppid) && !roots.has(row.pid)) {
        roots.add(row.pid);
        changed = true;
      }
    }
  }
  for (const row of table) if (roots.has(row.pid)) identities.set(row.pid, row);
  return [...identities.values()];
}

async function stopOwnedElectron(child, options = {}) {
  const readTable = options.readTable || readProcessTable;
  const killTree = options.killTree || treeKill;
  const now = options.now || Date.now;
  const wait = options.pause || pause;
  const start = now();
  const grace = options.graceMs ?? 30000;
  const deadline = start + grace + (options.forceMs ?? 5000);
  let known = options.known || [];
  let table = [];
  let snapshotAvailable = false;
  const result = {
    method: 'electron_sigterm_then_owned_tree_sigkill',
    electron_pid: child?.pid || null,
    graceful_wait_ms: grace,
    forced_tree_cleanup: false,
    forced_tree_roots: [],
    errors: [],
  };
  const refresh = () => {
    if (now() >= deadline) {
      snapshotAvailable = false;
      return;
    }
    try {
      table = readTable(Math.max(1, Math.min(1000, deadline - now())));
      known = trackOwnedTree(child, known, table);
      snapshotAvailable = true;
    } catch (error) {
      snapshotAvailable = false;
      const message = String(error);
      if (!result.errors.includes(message)) result.errors.push(message);
    }
  };
  const remaining = () =>
    table.filter((row) => !row.state.startsWith('Z') && known.some((old) => sameProcess(old, row)));
  if (child?.pid || known.length) {
    refresh();
    if (child?.pid && !exited(child)) {
      try {
        child.kill('SIGTERM');
      } catch (error) {
        if (error.code !== 'ESRCH') result.errors.push(String(error));
      }
      while (!exited(child) && now() < start + grace)
        await wait(Math.min(100, start + grace - now()));
    }
    refresh();
    const survivors = snapshotAvailable ? remaining().map((row) => row.pid) : [];
    if (
      child?.pid &&
      !exited(child) &&
      !survivors.includes(child.pid) &&
      (!snapshotAvailable || table.some((row) => row.pid === child.pid))
    )
      survivors.push(child.pid);
    if (survivors.length) {
      result.forced_tree_cleanup = true;
      result.forced_tree_roots = survivors;
      await Promise.all(
        survivors.map(
          (pid) =>
            new Promise((resolve) => {
              const timer = setTimeout(resolve, Math.max(0, deadline - now()));
              const complete = (error) => {
                clearTimeout(timer);
                if (error && error.code !== 'ESRCH') result.errors.push(String(error));
                resolve();
              };
              try {
                killTree(pid, 'SIGKILL', complete);
              } catch (error) {
                complete(error);
              }
            })
        )
      );
      do {
        refresh();
        if (snapshotAvailable && !remaining().length && (!child || exited(child))) break;
        await wait(Math.min(100, Math.max(0, deadline - now())));
      } while (now() < deadline);
    }
  } else snapshotAvailable = true;
  result.electron_exited = !child?.pid || exited(child);
  result.electron_exit_code = child?.exitCode ?? null;
  result.electron_signal = child?.signalCode ?? null;
  result.tracked_owned_processes = known;
  result.remaining_owned_processes = remaining();
  result.process_snapshot_available = snapshotAvailable;
  result.verified =
    snapshotAvailable && result.electron_exited && result.remaining_owned_processes.length === 0;
  result.elapsed_ms = now() - start;
  return result;
}

module.exports = {
  actualStartupMode,
  buildMode,
  isolatedModelEnvironment,
  validateBootstrap,
  readProcessTable,
  trackOwnedTree,
  stopOwnedElectron,
};
