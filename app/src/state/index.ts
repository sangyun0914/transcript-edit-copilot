import { configureStore } from '@reduxjs/toolkit';
import { player } from '../core/player';

import nav from './nav';
import transcribe from './transcribe';
import editor from './editor';
import models from './models';
import server from './server';
import recording, { recordingMiddleware } from './recording';
import copilot, { copilotMiddleware } from './copilot';

export const store = configureStore({
  reducer: {
    nav,
    transcribe,
    editor,
    models,
    server,
    recording,
    copilot,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: false,
    }).concat(copilotMiddleware, recordingMiddleware),
});

window.store = store;
player.setStore(store);

export type RootState = ReturnType<typeof store.getState>;
