import { createAsyncThunk, createSlice, PayloadAction } from '@reduxjs/toolkit';
import { openLanding, openTranscribe, openTranscribing } from './nav';
import { RootState } from './index';
import { readFileSync } from 'fs';
import { basename, extname } from 'path';
import { sleep } from '../util';
import { importTranscript } from '../core/transcription';
import { fetchModelState, Model } from './models';
import { getServer } from './server';
import { createHash } from 'crypto';
import { copyToMp4, convertToWav } from '../core/ffmpeg';
import {
  deleteTask,
  getTask,
  startTranscription as startTranscriptionApiCall,
  TranscriptionState,
  getTranscriptionCapabilities,
} from '../server_api/api';
import { openFile } from '../../ipc/ipc_renderer';
import { openDocumentFromMemory } from './editor/io';

export interface TranscribeState {
  file?: string;
  progress: number;
  state?: TranscriptionState;
  error?: string;
}

export const transcribeFile = createAsyncThunk<string | undefined, void, { state: RootState }>(
  'transcribe/transcribeFile',
  async (_, { dispatch }) => {
    await dispatch(fetchModelState()).unwrap();
    const file = await openFile({
      title: 'Import media file...',
      properties: ['openFile', 'promptToCreate', 'createDirectory'],
      filters: [
        {
          name: 'Audio & Video Files',
          extensions: ['mp3', 'wav', 'ogg', 'wma', 'aac', 'mp4', 'mkv', 'mov', 'webm'],
        },
        { name: 'All Files', extensions: ['*'] },
      ],
    });
    if (file.canceled) {
      return;
    }
    dispatch(openTranscribe());
    return file.filePaths[0];
  }
);

function workingExtensions(extension: string): boolean {
  return ['.wav', '.mp4'].includes(extension.toLowerCase());
}
export const startTranscription = createAsyncThunk<
  void,
  {
    provider: 'scribe' | 'vosk';
    transcription_model?: Model;
    language_code?: string;
    diarize: boolean;
    diarize_max_speakers: number | null;
  },
  { state: RootState }
>(
  'transcribing/upload',
  async (
    { provider, transcription_model, language_code, diarize, diarize_max_speakers },
    { dispatch, getState }
  ) => {
    const state = getState();
    const server = getServer(state);
    const path = state?.transcribe?.file;
    if (path === undefined) {
      throw Error('Failed to start transcription: No file for transcription given.');
    }
    dispatch(openTranscribing());

    const capabilities = await getTranscriptionCapabilities(server);
    if (!capabilities[provider]) {
      throw new Error(
        provider === 'vosk'
          ? 'Local ASR is unavailable. In server/, run: uv sync --extra local-asr'
          : 'ElevenLabs is not configured on this server.'
      );
    }

    dispatch(setProgress(0));
    dispatch(setState(TranscriptionState.CONVERTING));

    const fileName = basename(path);
    let wavFileContent;
    try {
      wavFileContent = await convertToWav(path, (p) => dispatch(setProgress(p)));
    } catch (err) {
      if (err.code == 'ENOSPC') {
        alert(
          'The import failed because your disk is (almost) full.\n' +
            'Note: During import, audapolis may need some additional free disk space.'
        );
      } else {
        alert(err);
      }
      dispatch(openLanding());
      return;
    }

    const file = new File([wavFileContent], 'input.wav');
    dispatch(setProgress(0));
    dispatch(setState(TranscriptionState.UPLOADING));
    const task = await startTranscriptionApiCall(
      server,
      provider,
      transcription_model?.model_id,
      language_code,
      diarize,
      diarize_max_speakers,
      file,
      fileName
    );

    dispatch(setState(task.state));

    try {
      while (true) {
        const { content, state, progress, transcription, error } = await getTask(server, task);

        dispatch(setProgress(progress));
        dispatch(setState(state));

        if (state === TranscriptionState.FAILED) {
          throw new Error(error || 'Transcription failed.');
        }
        if (state == 'done') {
          if (!content) throw new Error('Transcription completed without content.');
          const fileContent = readFileSync(path);

          let storedFileContent = fileContent;
          if (!workingExtensions(extname(path))) {
            try {
              storedFileContent = await copyToMp4(path);
            } catch (err) {
              alert(
                'Error: Your file is not in a format that is known to work (.wav or .mp4) and could ' +
                  'not be copied to an mp4.\n\nContinue at your own risk.'
              );
            }
          }

          const objectUrl = URL.createObjectURL(new Blob([storedFileContent]));
          const hash = createHash('sha256');
          hash.update(storedFileContent);
          const hashValue = hash.digest('hex');
          const sources = {
            [hashValue]: {
              fileName,
              fileContents: storedFileContent,
              objectUrl,
            },
          };

          const imported = importTranscript(
            content,
            hashValue,
            transcription?.language_code || language_code || transcription_model?.lang || '',
            transcription
          );

          await dispatch(
            openDocumentFromMemory({
              sources: sources,
              content: imported.content,
              metadata: { ...imported.metadata, display_speaker_names: diarize },
            })
          ).unwrap();
          break;
        }
        await sleep(0.5);
      }
    } finally {
      await deleteTask(server, task.uuid).catch(() => undefined);
    }
  }
);

export const importSlice = createSlice({
  name: 'nav',
  initialState: { progress: 0 } as TranscribeState,
  reducers: {
    setState: (state, args: PayloadAction<TranscriptionState>) => {
      state.state = args.payload;
    },
    setProgress: (state, args: PayloadAction<number>) => {
      state.progress = args.payload;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(startTranscription.pending, (state) => {
      state.error = undefined;
    });
    builder.addCase(startTranscription.rejected, (state, action) => {
      state.state = TranscriptionState.FAILED;
      state.error = action.error.message || 'Transcription failed.';
    });
    builder.addCase(transcribeFile.fulfilled, (state, action) => {
      state.file = action.payload;
    });
  },
});
const { setState, setProgress } = importSlice.actions;
export default importSlice.reducer;
