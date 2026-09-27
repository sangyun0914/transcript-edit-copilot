import { convertV1toV3, Document, V1Paragraph, V1V2Silence, V1V2Word } from './document';
import { v4 as uuidv4 } from 'uuid';
import { createHash } from 'crypto';
import { createSelector } from '@reduxjs/toolkit';

export interface AsrItemReference {
  word_indices: number[];
  boundary_status: 'unverified' | 'overlap' | 'needs_alignment';
  speaker_ids?: string[];
}

export interface TranscriptAnnotation {
  type: 'audio_event' | 'unaligned_word';
  text: string;
  start: number;
  end: number;
  word_indices: number[];
  speaker_id: string | null;
}

export interface AsrRegion extends AsrItemReference {
  sourceStart: number;
  length: number;
}

export type TranscriptionContent = V1Paragraph<
  (Omit<V1V2Word, 'source'> | Omit<V1V2Silence, 'source'>) & { asr?: AsrItemReference }
>[];

export interface TranscriptionMetadata {
  version: 1;
  provider: 'elevenlabs';
  model: string;
  language_code: string;
  duration: number;
  request: Record<string, string>;
  raw_response: Record<string, unknown>;
  annotations?: TranscriptAnnotation[];
}

export interface LegacyTranscriptionProvenance extends TranscriptionMetadata {
  source: string;
  items: Record<string, AsrItemReference>;
}

export interface TranscriptionProvenance extends TranscriptionMetadata {
  id: string;
  source: string;
  regions: AsrRegion[];
}

export function migrateTranscription(document: Omit<Document, 'sources'>): void {
  const legacy = document.metadata.transcription;
  if (!legacy) return;
  const { items, ...metadata } = legacy;
  const regions = document.content.flatMap((item) =>
    'source' in item && item.source === legacy.source && items[item.uuid]
      ? [{ ...items[item.uuid], sourceStart: item.sourceStart, length: item.length }]
      : []
  );
  document.metadata.transcriptions = [
    ...(document.metadata.transcriptions || []),
    {
      ...metadata,
      id: createHash('sha256').update(JSON.stringify(legacy)).digest('hex'),
      regions,
    },
  ];
  delete document.metadata.transcription;
}

export function mergeTranscriptions(
  document: Document,
  incoming: Pick<Document, 'content'> & Partial<Pick<Document, 'metadata'>>
): void {
  migrateTranscription(document);
  if (!incoming.metadata) return;
  const imported = { content: incoming.content, metadata: { ...incoming.metadata } };
  migrateTranscription(imported);
  const records = [
    ...(document.metadata.transcriptions || []),
    ...(imported.metadata.transcriptions || []),
  ];
  if (!records.length) return;
  const merged = new Map<string, TranscriptionProvenance>();
  for (const record of records) {
    const regions = [...(merged.get(record.id)?.regions || []), ...record.regions];
    merged.set(record.id, {
      ...record,
      regions: [...new Map(regions.map((region) => [JSON.stringify(region), region])).values()],
    });
  }
  document.metadata.transcriptions = [...merged.values()];
}

export const transcriptionWarnings = createSelector(
  [(records?: TranscriptionProvenance[]) => records],
  (records) =>
    (records || []).flatMap((record) =>
      record.regions
        .filter((region) => region.boundary_status !== 'unverified')
        .map((region) => ({ ...region, source: record.source }))
    )
);

export function alignmentWarning(
  item: { source?: string; sourceStart?: number; length?: number },
  regions: ReturnType<typeof transcriptionWarnings>
): string | undefined {
  const { source, sourceStart, length } = item;
  if (!source || sourceStart === undefined || length === undefined) return;
  const matches = regions.filter(
    (region) =>
      region.source === source &&
      region.sourceStart < sourceStart + length &&
      sourceStart < region.sourceStart + region.length
  );
  if (!matches.length) return;
  const speakers = [...new Set(matches.flatMap((region) => region.speaker_ids || []))];
  const detail = matches.some((region) => region.boundary_status === 'needs_alignment')
    ? 'Some words have no individual timing.'
    : 'Speech intervals overlap.';
  return `${detail} Listen before cutting this group.${
    speakers.length > 1 ? ` Original speakers: ${speakers.join(', ')}.` : ''
  }`;
}

export function importTranscript(
  content: TranscriptionContent,
  source: string,
  language: string,
  transcription?: TranscriptionMetadata | null
): Omit<Document, 'sources'> {
  const paragraphs: V1Paragraph[] = content.map((paragraph) => ({
    speaker: paragraph.speaker,
    content: paragraph.content.map((item) => ({ ...item, source })),
  }));
  const document = convertV1toV3(paragraphs);
  document.content = document.content.map((item) =>
    item.type === 'paragraph_start' ? { ...item, language } : item
  );
  if (transcription) {
    const originalItems = content.flatMap((paragraph) => paragraph.content);
    const regions: AsrRegion[] = [];
    let index = 0;
    for (const item of document.content) {
      if ('source' in item) {
        const reference = originalItems[index++].asr;
        if (reference)
          regions.push({ ...reference, sourceStart: item.sourceStart, length: item.length });
      }
    }
    document.metadata.transcriptions = [{ ...transcription, id: uuidv4(), source, regions }];
  }
  return document;
}
