const test = require('node:test');
const assert = require('node:assert/strict');
const {
  isolatedModelEnvironment,
  validateBootstrap,
  stopOwnedElectron,
} = require('./copilot_ui_helpers.cjs');

const processRow = (pid, ppid, started = 'Sat Oct 24 12:00:00 2026') => ({
  pid,
  ppid,
  started,
  state: 'S',
  command: `fixture-${pid}`,
});

function cleanupFixture() {
  let now = 0;
  const fixture = {
    table: [processRow(10, 1), processRow(20, 10), processRow(99, 1)],
    killed: [],
    child: {
      pid: 10,
      exitCode: null,
      signalCode: null,
      kill(signal) {
        fixture.killed.push([10, signal]);
        this.exitCode = 0;
        fixture.table = fixture.table.filter((row) => row.pid === 99);
      },
    },
  };
  fixture.options = {
    graceMs: 10,
    forceMs: 5,
    now: () => now,
    pause: async (ms) => {
      now += ms;
    },
    readTable: () => fixture.table,
    killTree: (pid, signal, callback) => {
      fixture.killed.push([pid, signal]);
      fixture.table = fixture.table.filter((row) => row.pid !== pid);
      if (pid === 10) fixture.child.signalCode = signal;
      callback();
    },
  };
  return fixture;
}

test('actual app startup uses development subscriptions and only the default workspace selection', () => {
  const env = isolatedModelEnvironment({
    COPILOT_UI_MODE: 'app-startup-actual',
    COPILOT_UI_TEMP: '/isolated/profile',
    COPILOT_UI_OUTPUT: '/isolated/output',
    AUDAPOLIS_NEP_CONFIG: '/external/gateway.json',
    AUDAPOLIS_NEP_DEFAULT_DEPLOYMENT: '/external/selection.json',
  });
  assert.equal(env.NODE_ENV, 'development');
  assert.equal(env.MODE, 'development');
  assert.equal(env.AUDAPOLIS_NEP_AUTOSTART, '1');
  assert.equal(env.AUDAPOLIS_NEP_AUTOSTART_TEST, undefined);
  assert.equal(env.AUDAPOLIS_NEP_CONFIG, undefined);
  assert.equal(env.AUDAPOLIS_NEP_DEFAULT_DEPLOYMENT, undefined);
  assert.equal(validateBootstrap(env), true);
  assert.throws(() => validateBootstrap({ ...env, AUDAPOLIS_NEP_AUTOSTART_TEST: '1' }));
});

test('scripted modes cannot inherit actual model startup from the calling shell', () => {
  const env = isolatedModelEnvironment({
    COPILOT_UI_MODE: 'roster-scripted',
    COPILOT_UI_TEMP: '/isolated/profile',
    COPILOT_UI_OUTPUT: '/isolated/output',
    AUDAPOLIS_NEP_AUTOSTART: '1',
    AUDAPOLIS_NEP_AUTOSTART_TEST: '1',
  });
  assert.equal(env.NODE_ENV, 'test');
  assert.equal(env.AUDAPOLIS_NEP_AUTOSTART, '0');
  assert.equal(env.AUDAPOLIS_NEP_AUTOSTART_TEST, undefined);
  assert.equal(validateBootstrap(env), false);
  assert.throws(() => validateBootstrap({ ...env, NODE_ENV: 'development' }));
  assert.throws(() => validateBootstrap({ ...env, COPILOT_UI_MODE: 'app-startup-actual' }));
});

test('graceful Electron shutdown waits for the owned detached server without force killing', async () => {
  const fixture = cleanupFixture();
  const result = await stopOwnedElectron(fixture.child, fixture.options);
  assert.deepEqual(fixture.killed, [[10, 'SIGTERM']]);
  assert.equal(result.verified, true);
  assert.equal(result.forced_tree_cleanup, false);
  assert.deepEqual(result.remaining_owned_processes, []);
  assert.deepEqual(
    result.tracked_owned_processes.map((row) => row.pid),
    [10, 20]
  );
});

test('a detached server that survives Electron is killed through its captured ownership', async () => {
  const fixture = cleanupFixture();
  fixture.child.kill = function (signal) {
    fixture.killed.push([10, signal]);
    this.exitCode = 0;
    fixture.table = [processRow(20, 1), processRow(99, 1)];
  };
  const result = await stopOwnedElectron(fixture.child, fixture.options);
  assert.deepEqual(fixture.killed, [
    [10, 'SIGTERM'],
    [20, 'SIGKILL'],
  ]);
  assert.equal(result.forced_tree_cleanup, true);
  assert.equal(result.verified, true);
  assert.deepEqual(
    fixture.table.map((row) => row.pid),
    [99]
  );
});

test('the shutdown deadline escalates only the owned process tree', async () => {
  const fixture = cleanupFixture();
  fixture.child.kill = (signal) => fixture.killed.push([10, signal]);
  const result = await stopOwnedElectron(fixture.child, fixture.options);
  assert.equal(result.forced_tree_cleanup, true);
  assert.deepEqual(result.forced_tree_roots, [10, 20]);
  assert.equal(result.verified, true);
  assert.ok(result.elapsed_ms <= 15);
  assert.deepEqual(
    fixture.table.map((row) => row.pid),
    [99]
  );
});

test('a reused PID is not killed after its original owned process exits', async () => {
  const fixture = cleanupFixture();
  fixture.child.exitCode = 0;
  fixture.options.known = [processRow(10, 1), processRow(20, 10)];
  fixture.table = [processRow(20, 1, 'Sat Oct 24 12:01:00 2026'), processRow(99, 1)];
  const result = await stopOwnedElectron(fixture.child, fixture.options);
  assert.deepEqual(fixture.killed, []);
  assert.equal(result.verified, true);
});

test('missing process evidence remains unverified even after Electron exits', async () => {
  const fixture = cleanupFixture();
  fixture.options.readTable = () => {
    throw new Error('process inspection unavailable');
  };
  const result = await stopOwnedElectron(fixture.child, fixture.options);
  assert.equal(result.electron_exited, true);
  assert.equal(result.verified, false);
  assert.match(result.errors[0], /inspection unavailable/);
});

test('a missing child during setup failure needs no process cleanup', async () => {
  const result = await stopOwnedElectron(undefined, {
    readTable: () => assert.fail('No process inspection is needed without an owned process.'),
  });
  assert.equal(result.verified, true);
  assert.equal(result.electron_pid, null);
});
