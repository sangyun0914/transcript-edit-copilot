import reducer, { localServerStatus, receiveLocalServerInfo, serverSlice } from './server';

test('cold failure and startup stay outside routable server configurations until running', () => {
  let state = reducer(
    undefined,
    receiveLocalServerInfo({
      state: 'failed to start',
      port: null,
      token: null,
    })
  );
  expect(state.servers).toEqual([]);
  expect(state.localServerInfo?.port).toBeNull();
  expect(localServerStatus(state)).toBe('unavailable');
  state = reducer(state, receiveLocalServerInfo({ state: 'starting', port: 45001, token: null }));
  expect(state.servers).toEqual([]);
  expect(localServerStatus(state)).toBe('starting');
  state = reducer(state, receiveLocalServerInfo({ state: 'running', port: 45001, token: 'token' }));
  expect(state.servers).toEqual([
    {
      name: 'local server',
      hostname: 'http://localhost',
      port: 45001,
      token: 'token',
      state: 'running',
    },
  ]);
  state = reducer(state, receiveLocalServerInfo({ state: 'stopped', port: 45001, token: 'token' }));
  expect(state.servers[0].state).toBe('stopped');
  expect(localServerStatus(state)).toBe('unavailable');
});

test('local lifecycle updates preserve an explicitly selected remote server', () => {
  const remote = {
    name: 'Remote ASR',
    hostname: 'https://asr.example.com',
    port: 443,
    token: 'remote',
  };
  let state = reducer(undefined, serverSlice.actions.addServer(remote));
  state = reducer(
    state,
    receiveLocalServerInfo({ state: 'failed to start', port: null, token: null })
  );
  expect(state.servers).toEqual([remote]);
  state = reducer(state, receiveLocalServerInfo({ state: 'running', port: 45001, token: 'local' }));
  expect(state.servers[state.selectedServer]).toEqual(remote);
  expect(state.servers).toHaveLength(2);
});
