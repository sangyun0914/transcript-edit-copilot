import { createAsyncThunk, createSlice, Middleware } from '@reduxjs/toolkit';
import { createHash } from 'crypto';
import { promises as fs } from 'fs';
import type { FileHandle } from 'fs/promises';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import type { RootState } from './index';
import { SessionEvent, SessionJournal, SessionMode } from '../core/editing_session';
import type { Document } from '../core/document';
import type { EditorState } from './editor/types';

interface RecordingState {
  status: 'idle' | 'starting' | 'recording' | 'stopping' | 'failed';
  path: string | null;
  edits: number;
  error: string | null;
}
const initialState: RecordingState = { status: 'idle', path: null, edits: 0, error: null };
const slice = createSlice({
  name: 'recording',
  initialState,
  reducers: {
    starting: (state) => {
      state.status = 'starting';
      state.error = null;
    },
    started: (state, action) => {
      state.status = 'recording';
      state.path = action.payload;
      state.edits = 0;
    },
    progress: (state, action) => {
      state.edits = action.payload;
    },
    stopping: (state) => {
      state.status = 'stopping';
    },
    stopped: (state) => {
      state.status = 'idle';
    },
    failed: (state, action) => {
      state.status = 'failed';
      state.error = action.payload;
    },
  },
});
export default slice.reducer;

let writer: {
  file: FileHandle;
  queue: Promise<void>;
  journal: SessionJournal | null;
  sources: Set<string>;
  error: string | null;
  stopped: boolean;
} | null = null;

function sourceManifest(document: Document) {
  return Object.fromEntries(
    Object.entries(document.sources).map(([key, source]) => [
      key,
      {
        sha256: createHash('sha256').update(Buffer.from(source.fileContents)).digest('hex'),
        bytes: source.fileContents.byteLength,
      },
    ])
  );
}

export const startRecording = createAsyncThunk<
  void,
  { path: string; mode: SessionMode; brief: string },
  { state: RootState }
>('recording/start', async (args, { getState, dispatch }) => {
  if (writer || getState().recording.status === 'starting') return;
  dispatch(slice.actions.starting());
  let file: FileHandle | undefined;
  try {
    await fs.mkdir(path.dirname(args.path), { recursive: true });
    file = await fs.open(args.path, 'wx');
    const editor = getState().editor.present;
    if (!editor) throw new Error('Open a project before recording.');
    const sources = sourceManifest(editor.document);
    const active = {
      file,
      queue: Promise.resolve(),
      journal: null as SessionJournal | null,
      sources: new Set(Object.keys(sources)),
      error: null as string | null,
      stopped: false,
    };
    writer = active;
    active.journal = new SessionJournal(
      editor,
      {
        session_id: uuidv4(),
        editor_id: ['ai_computer_use', 'synthetic'].includes(args.mode)
          ? 'local-automation'
          : 'local-user',
        collection_mode: args.mode,
        task_brief: args.brief,
        sources,
      },
      (event: SessionEvent) => {
        const line = JSON.stringify(event) + '\n';
        active.queue = active.queue
          .then(async () => {
            if (active.error) return;
            await active.file.writeFile(line);
          })
          .catch((error) => {
            active.error = String(error.message || error);
            dispatch(slice.actions.failed(active.error));
          });
      }
    );
    await active.queue;
    if (active.error) throw new Error(active.error);
    if (writer !== active || active.stopped) return;
    dispatch(slice.actions.started(args.path));
  } catch (error) {
    if (file) await file.close();
    writer = null;
    dispatch(slice.actions.failed(error.message || String(error)));
  }
});

export const stopRecording = createAsyncThunk<void, string | undefined, { state: RootState }>(
  'recording/stop',
  async (reason, { dispatch }) => {
    const active = writer;
    if (!active) {
      dispatch(slice.actions.stopped());
      return;
    }
    if (active.stopped) return;
    active.stopped = true;
    dispatch(slice.actions.stopping());
    active.journal?.finish(reason || 'user_stopped');
    try {
      await active.queue;
      if (active.error) throw new Error(active.error);
      await active.file.sync();
      dispatch(slice.actions.stopped());
    } catch (error) {
      dispatch(slice.actions.failed(error.message || String(error)));
    } finally {
      await active.file.close();
      if (writer === active) writer = null;
    }
  }
);

type RecordingRoot = { editor: { present: EditorState | null }; recording: RecordingState };

export const recordingMiddleware: Middleware<Record<string, never>, RecordingRoot> =
  (store) => (next) => (action) => {
    const before = store.getState().editor.present;
    const active = writer;
    const result = next(action);
    const after = store.getState().editor.present;
    if (
      !active ||
      active.stopped ||
      active.error ||
      !before ||
      !after ||
      typeof action.type !== 'string' ||
      action.type.startsWith('recording/')
    )
      return result;
    if (
      [
        'editor/openDocumentFromDisk/fulfilled',
        'editor/openDocumentFromMemory/fulfilled',
        'editor/closeDocument/pending',
      ].includes(action.type)
    ) {
      store.dispatch(stopRecording('document_changed') as any);
      return result;
    }
    try {
      for (const [key, source] of Object.entries(after.document.sources)) {
        if (!active.sources.has(key)) {
          active.journal?.sourceAdded(
            key,
            createHash('sha256').update(Buffer.from(source.fileContents)).digest('hex'),
            source.fileContents.byteLength
          );
          active.sources.add(key);
        }
      }
      active.journal?.observe(before, after, action.type);
      if (active.journal && store.getState().recording.edits !== active.journal.edits)
        store.dispatch(slice.actions.progress(active.journal.edits));
    } catch (error) {
      active.error = error.message || String(error);
      store.dispatch(slice.actions.failed(active.error));
      store.dispatch(stopRecording('recording_error') as any);
    }
    return result;
  };
