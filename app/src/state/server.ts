import { createAsyncThunk, createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from './index';
import { fetchModelState } from './models';
import {
  requestLocalServerInfo,
  subscribeLocalServerInfo,
  subscribeLocalServerStderr,
} from '../../ipc/ipc_renderer';
import { isRunningInTest } from '../util';
import type { ServerInfo } from '../../main_process/types';

export interface ServerState {
  servers: ServerConfig[];
  selectedServer: number;
  localServerInfo?: ServerInfo;
}

export interface ServerConfig {
  hostname: string;
  port: number;
  token: string | null;
  name: string;
  state?: string;
}

function updateLocalServer(state: ServerState, config: Omit<ServerConfig, 'name'>) {
  const name = 'local server';
  const index = state.servers.findIndex((server) => server.name === name);
  const server = { name, ...config };
  if (index < 0) state.servers.push(server);
  else state.servers[index] = server;
}

export function localServerStatus(state: ServerState): 'starting' | 'running' | 'unavailable' {
  const info = state.localServerInfo;
  if (!info || ['not started', 'starting'].includes(info.state)) return 'starting';
  return info.state === 'running' && info.port !== null ? 'running' : 'unavailable';
}

export const changeServer = createAsyncThunk<void, number, { state: RootState }>(
  'models/changeServer',
  async (server, { dispatch }) => {
    await dispatch(selectServer(server));
    dispatch(fetchModelState());
  }
);

export const serverSlice = createSlice({
  name: 'server',
  initialState: {
    servers: [],
    selectedServer: 0,
  } as ServerState,
  reducers: {
    addServer: (state, args: PayloadAction<ServerConfig>) => {
      state.servers.push(args.payload);
    },
    setLocalServer: (state, args: PayloadAction<Omit<ServerConfig, 'name'>>) => {
      updateLocalServer(state, args.payload);
    },
    receiveLocalServerInfo: (state, { payload: info }: PayloadAction<ServerInfo>) => {
      state.localServerInfo = info;
      if (info.state === 'running' && info.port !== null) {
        updateLocalServer(state, {
          hostname: 'http://localhost',
          port: info.port,
          token: info.token,
          state: info.state,
        });
      } else {
        const existing = state.servers.find((server) => server.name === 'local server');
        if (existing) existing.state = info.state;
      }
    },
    removeServer: (state, args: PayloadAction<string>) => {
      state.servers = state.servers.filter((config) => config.hostname != args.payload);
    },
    selectServer: (state, args: PayloadAction<number>) => {
      state.selectedServer = args.payload;
    },
  },
});
const { selectServer } = serverSlice.actions;
export const { receiveLocalServerInfo } = serverSlice.actions;
export default serverSlice.reducer;

export const getServer = (state: RootState): ServerConfig => {
  return state.server.servers[state.server.selectedServer];
};

if (!isRunningInTest()) {
  subscribeLocalServerStderr((stderr) => {
    console.groupCollapsed('server stderr');
    stderr.split('\n').forEach((line) => {
      console.log(line);
    });
    console.groupEnd();
  });
  subscribeLocalServerInfo((info) => {
    window.store.dispatch(receiveLocalServerInfo(info));
  });
  requestLocalServerInfo();
}
