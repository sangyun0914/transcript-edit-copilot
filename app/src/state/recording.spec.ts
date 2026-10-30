import { configureStore } from '@reduxjs/toolkit';
import { promises as fs } from 'fs';
import os from 'os';
import path from 'path';
import copilot from './copilot';
import recording, { startRecording, stopRecording } from './recording';
import { defaultEditorState } from './editor/types';

test('automated recording scopes the copilot actor and a failed start keeps the previous actor', async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'copilot-recording-'));
  const logPath = path.join(directory, 'session.jsonl');
  const store = configureStore({
    reducer: {
      editor: () => ({ present: defaultEditorState }),
      copilot,
      recording,
    },
    middleware: (getDefault) => getDefault({ serializableCheck: false }),
  });
  try {
    await store.dispatch(
      startRecording({ path: logPath, mode: 'synthetic', brief: 'Test only.' }) as any
    );
    expect(store.getState().recording.status).toBe('recording');
    expect(store.getState().copilot.feedbackOrigin).toBe('synthetic');
    await store.dispatch(stopRecording('test_complete') as any);
    expect(store.getState().copilot.feedbackOrigin).toBe('human');
    const saved = await fs.readFile(logPath, 'utf8');
    expect(JSON.parse(saved.split('\n')[0]).actor).toBe('synthetic');
    await store.dispatch(
      startRecording({ path: logPath, mode: 'ai_computer_use', brief: 'Test only.' }) as any
    );
    expect(store.getState().recording.status).toBe('failed');
    expect(store.getState().copilot.feedbackOrigin).toBe('human');
    expect(await fs.readFile(logPath, 'utf8')).toBe(saved);
  } finally {
    await store.dispatch(stopRecording('test_cleanup') as any);
    await fs.rm(directory, { recursive: true, force: true });
  }
});
