import { fetchFromServer } from './index';
import { ServerConfig } from '../state/server';
import { Language, Model } from '../state/models';
import { TranscriptionContent, TranscriptionMetadata } from '../core/transcription';

export interface Task {
  uuid: string;
}

export enum TranscriptionState {
  QUEUED = 'queued',
  LOADING = 'loading',
  LOADING_TRANSCRIPTION_MODEL = 'loading transcription model',
  DIARIZING = 'diarizing',
  TRANSCRIBING = 'transcribing',
  POST_PROCESSING = 'post_processing',
  DONE = 'done',
  FAILED = 'failed',
  CONVERTING = 'converting',
  UPLOADING = 'uploading',
}

export interface TranscriptionTask extends Task {
  progress: number;
  state: TranscriptionState;
  content?: TranscriptionContent;
  transcription?: TranscriptionMetadata | null;
  error?: string | null;
}

export enum DownloadingModelState {
  QUEUED = 'queued',
  DOWNLOADING = 'downloading',
  EXTRACTING = 'extracting',
  DONE = 'done',
}

export interface DownloadModelTask extends Task {
  lang: string;
  name: string;
  state: DownloadingModelState;
  progress: number;
}

export function startTranscription(
  server: ServerConfig,
  provider: 'scribe' | 'vosk',
  transcription_model: string | undefined,
  language_code: string | undefined,
  diarize: boolean,
  diarize_max_speakers: number | null,
  file: File,
  fileName: string
): Promise<TranscriptionTask> {
  const opts: Record<string, string | boolean | number> = { provider, diarize };
  if (transcription_model) opts.transcription_model = transcription_model;
  if (language_code) opts.language_code = language_code;
  if (diarize_max_speakers !== null) {
    opts['diarize_max_speakers'] = diarize_max_speakers;
  }
  return fetchFromServer(server, 'POST', 'tasks/start_transcription', opts, {
    form: { file, fileName },
  }).then((x) => x.json());
}

export function getTranscriptionCapabilities(
  server: ServerConfig
): Promise<{ scribe: boolean; vosk: boolean }> {
  return fetchFromServer(server, 'GET', 'transcription/capabilities').then((x) => x.json());
}

export function downloadModel(server: ServerConfig, model_id: string): Promise<DownloadModelTask> {
  return fetchFromServer(server, 'POST', 'tasks/download_model', { model_id }).then((x) =>
    x.json()
  );
}

export function getTask<T extends Task>(server: ServerConfig, task: T): Promise<T> {
  return fetchFromServer(server, 'GET', `tasks/${task.uuid}`).then((x) => x.json());
}

export function deleteTask(server: ServerConfig, uuid: string): Promise<void> {
  return fetchFromServer(server, 'DELETE', `tasks/${uuid}`).then(() => {});
}

export function deleteModel(server: ServerConfig, model_id: string): Promise<void> {
  return fetchFromServer(server, 'POST', 'models/delete', { model_id }).then(() => {});
}

export function getAvailableModels(server: ServerConfig): Promise<Record<string, Language>> {
  return fetchFromServer(server, 'GET', 'models/available').then((x) => x.json());
}

export function getDownloadedModels(server: ServerConfig): Promise<Record<string, Model>> {
  return fetchFromServer(server, 'GET', 'models/downloaded').then((x) => x.json());
}

export interface OtioSegment {
  speaker: string;
  source_file: string;
  source_length: number;
  has_video: boolean;
  source_start: number;
  length: number;
}

export function convertOtio(
  server: ServerConfig,
  name: string,
  adapter: string,
  timeline: OtioSegment[]
): Promise<ArrayBuffer> {
  return fetchFromServer(
    server,
    'POST',
    'util/otio/convert',
    { name, adapter },
    { json: timeline }
  ).then((x) => x.arrayBuffer());
}
