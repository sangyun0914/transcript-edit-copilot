import { createHash } from 'crypto';
import type { Document, V3DocumentItem } from './document';
import type { EditorState } from '../state/editor/types';

export interface SessionDocument {
  content: V3DocumentItem[];
  metadata: Record<string, unknown>;
}

export interface ContentPatch {
  index: number;
  remove: V3DocumentItem[];
  insert: V3DocumentItem[];
  metadata?: Record<string, unknown>;
  previous_metadata?: Record<string, unknown>;
}

export type SessionEvent = Record<string, any> & { kind: string; seq: number };
export type SessionMode = 'unguided' | 'reference_guided' | 'ai_computer_use' | 'synthetic';
export type SessionActor = 'human' | 'ai_computer_use' | 'synthetic';

// Encode numbers as IEEE-754 bytes so Python and JavaScript hash source times identically.
function canonical(value: any): string {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error('Session numbers must be finite.');
    const bytes = Buffer.alloc(8);
    bytes.writeDoubleBE(value === 0 ? 0 : value);
    return `#${bytes.toString('hex')}`;
  }
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value !== null && typeof value === 'object') {
    return `{${Object.keys(value)
      .filter((key) => value[key] !== undefined)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`)
      .join(',')}}`;
  }
  return JSON.stringify(value);
}

export function sessionHash(value: unknown): string {
  return createHash('sha256').update(canonical(value)).digest('hex');
}

export function sessionDocument(document: Document): SessionDocument {
  const metadata: Record<string, unknown> = {};
  if (document.metadata.transcription) metadata.transcription = document.metadata.transcription;
  if (document.metadata.transcriptions) metadata.transcriptions = document.metadata.transcriptions;
  return { content: document.content, metadata };
}

export function sessionContext(editor: EditorState): Record<string, unknown> {
  return {
    cursor: editor.cursor,
    selection: editor.selection,
    playing: editor.playing,
    correcting_text: editor.transcriptCorrectionState !== null,
  };
}

export function contentPatch(before: SessionDocument, after: SessionDocument): ContentPatch {
  const equal = (a: unknown, b: unknown) => a === b || JSON.stringify(a) === JSON.stringify(b);
  let first = 0;
  while (
    first < before.content.length &&
    first < after.content.length &&
    equal(before.content[first], after.content[first])
  )
    first++;
  let endBefore = before.content.length,
    endAfter = after.content.length;
  while (
    endBefore > first &&
    endAfter > first &&
    equal(before.content[endBefore - 1], after.content[endAfter - 1])
  ) {
    endBefore--;
    endAfter--;
  }
  const patch: ContentPatch = {
    index: first,
    remove: before.content.slice(first, endBefore),
    insert: after.content.slice(first, endAfter),
  };
  if (!equal(before.metadata, after.metadata)) {
    patch.previous_metadata = before.metadata;
    patch.metadata = after.metadata;
  }
  return patch;
}

function operation(patch: ContentPatch, action: string): string {
  const { remove, insert } = patch;
  if (action === 'transcript_correction/finishTranscriptCorrection' || action === 'editor/setWord')
    return 'correct_text';
  if (action === 'editor/insertParagraphEnd') return 'split_paragraph';
  if (action === 'editor/acceptCopilot') {
    if (insert.length === 1 && insert[0].type === 'text' && remove.length) return 'correct_text';
    if (
      remove.length === 1 &&
      remove[0].type === 'non_text' &&
      insert.length === 2 &&
      insert.every((item) => item.type === 'non_text')
    )
      return 'trim_silence';
    if (
      !remove.length &&
      insert.length === 2 &&
      insert[0].type === 'paragraph_end' &&
      insert[1].type === 'paragraph_start'
    )
      return 'split_paragraph';
  }
  if (
    remove.length === 2 &&
    remove[0].type === 'paragraph_end' &&
    remove[1].type === 'paragraph_start' &&
    !insert.length
  )
    return 'merge_paragraph';
  if (remove.length && remove.every((item) => item.type === 'non_text') && !insert.length)
    return 'remove_silence';
  if (action === 'editor/trimSelectedGap') return 'trim_silence';
  if (remove.some((item) => item.type === 'text') && !insert.length) return 'delete_span';
  if (!remove.length && !insert.length) return 'metadata_change';
  return 'other_transaction';
}

export class SessionJournal {
  seq = 0;
  revision = 0;
  epoch = 0;
  edits = 0;
  hash: string;
  private emit: (event: SessionEvent) => void;
  private actor: SessionActor;

  constructor(
    editor: EditorState,
    info: Record<string, unknown> & { collection_mode?: SessionMode },
    emit: (event: SessionEvent) => void
  ) {
    this.emit = emit;
    this.actor =
      info.collection_mode === 'ai_computer_use' || info.collection_mode === 'synthetic'
        ? info.collection_mode
        : 'human';
    const document = sessionDocument(editor.document);
    this.hash = sessionHash(document);
    this.write('start', {
      ...info,
      schema_version: 2,
      hash_format: 'typed-json-f64-v1',
      actor: this.actor,
      document,
      document_hash: this.hash,
      context: sessionContext(editor),
    });
    this.opportunity(editor, 'session_start');
  }

  private write(kind: string, fields: Record<string, unknown>) {
    this.emit({ kind, seq: this.seq++, at: new Date().toISOString(), ...fields });
  }

  private opportunity(editor: EditorState, trigger: string) {
    this.write('opportunity', {
      trigger,
      policy: 'recorded-stable-context-v1',
      revision: this.revision,
      interaction_epoch: this.epoch,
      document_hash: this.hash,
      context: sessionContext(editor),
      eligible: !editor.playing && editor.transcriptCorrectionState === null,
      prediction_shown: false,
    });
  }

  observe(before: EditorState, after: EditorState, action: string): void {
    const oldDocument = sessionDocument(before.document),
      newDocument = sessionDocument(after.document);
    const oldContext = sessionContext(before),
      newContext = sessionContext(after);
    const documentChanged = before.document !== after.document;
    const beforeHash = documentChanged ? sessionHash(oldDocument) : this.hash;
    const afterHash = documentChanged ? sessionHash(newDocument) : this.hash;
    if (beforeHash !== this.hash) throw new Error('Recording state diverged from the editor.');
    const changed = beforeHash !== afterHash;
    const contextChanged = JSON.stringify(oldContext) !== JSON.stringify(newContext);
    if (contextChanged) this.epoch++;
    if (changed) {
      const patch = contentPatch(oldDocument, newDocument);
      const origin = action.startsWith('@@redux-undo/')
        ? action.endsWith('REDO')
          ? 'redo'
          : 'undo'
        : action.includes('filterContent')
        ? 'bulk'
        : this.actor;
      this.write('transaction', {
        revision: ++this.revision,
        interaction_epoch: this.epoch,
        action_type: action,
        operation: operation(patch, action),
        origin,
        actor: this.actor,
        before_hash: beforeHash,
        after_hash: afterHash,
        patch,
        before_context: oldContext,
        after_context: newContext,
      });
      this.hash = afterHash;
      this.edits++;
      this.opportunity(after, 'committed_transaction');
    } else if (contextChanged && action !== 'editor/setTimePlayer') {
      this.write('context', {
        action_type: action,
        revision: this.revision,
        interaction_epoch: this.epoch,
        before: oldContext,
        after: newContext,
      });
      this.opportunity(after, 'context_change');
    }
  }

  sourceAdded(key: string, sha256: string, bytes: number): void {
    this.write('source', { source: key, sha256, bytes });
  }

  finish(reason: string): void {
    this.write('end', {
      reason,
      revision: this.revision,
      document_hash: this.hash,
      transactions: this.edits,
    });
  }
}

export function replaySession(events: SessionEvent[]): SessionDocument {
  const first = events[0];
  if (
    !first ||
    first.kind !== 'start' ||
    ![1, 2].includes(first.schema_version) ||
    first.hash_format !== 'typed-json-f64-v1'
  )
    throw new Error('Unsupported editing session.');
  if (
    first.schema_version === 2 &&
    !['human', 'ai_computer_use', 'synthetic'].includes(first.actor)
  )
    throw new Error('Missing or unknown session actor.');
  const document: SessionDocument = JSON.parse(JSON.stringify(first.document));
  if (sessionHash(document) !== first.document_hash)
    throw new Error('Initial snapshot hash mismatch.');
  let revision = 0;
  let ended = false;
  events.forEach((event, seq) => {
    if (ended) throw new Error('Event after session end.');
    if (event.seq !== seq) throw new Error('Missing or reordered session event.');
    if (!['start', 'transaction', 'context', 'opportunity', 'source', 'end'].includes(event.kind))
      throw new Error('Unknown session event.');
    if (event.kind === 'start' && seq !== 0) throw new Error('Repeated session start.');
    if (event.kind === 'transaction') {
      if (first.schema_version === 2 && event.actor !== first.actor)
        throw new Error('Transaction actor differs from the session actor.');
      if (event.revision !== ++revision || sessionHash(document) !== event.before_hash)
        throw new Error('Transaction history mismatch.');
      const patch: ContentPatch = event.patch;
      if (
        !Number.isInteger(patch.index) ||
        patch.index < 0 ||
        patch.index > document.content.length
      )
        throw new Error('Invalid patch index.');
      if (
        sessionHash(document.content.slice(patch.index, patch.index + patch.remove.length)) !==
        sessionHash(patch.remove)
      )
        throw new Error('Patch does not match original items.');
      if (
        patch.previous_metadata &&
        sessionHash(document.metadata) !== sessionHash(patch.previous_metadata)
      )
        throw new Error('Metadata before-image mismatch.');
      document.content.splice(patch.index, patch.remove.length, ...patch.insert);
      if (patch.metadata) document.metadata = patch.metadata;
      if (sessionHash(document) !== event.after_hash)
        throw new Error('Replayed snapshot hash mismatch.');
    }
    if (
      ['end', 'opportunity'].includes(event.kind) &&
      (event.revision !== revision || sessionHash(document) !== event.document_hash)
    )
      throw new Error('Final snapshot hash mismatch.');
    if (event.kind === 'end') ended = true;
  });
  return document;
}
