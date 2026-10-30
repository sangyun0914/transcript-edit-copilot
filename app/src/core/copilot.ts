import type { V3DocumentItem } from './document';
import { contentPatch, SessionDocument, sessionHash } from './editing_session';
import { lintDocumentContent } from '../util/document_linter';

export interface CopilotRequestIdentity {
  request_id: string;
  document_id: string;
  revision: number;
  interaction_epoch: number;
}

export interface CopilotPrediction {
  action:
    | 'no_action'
    | 'correct_text'
    | 'delete_span'
    | 'trim_silence'
    | 'remove_silence'
    | 'split_paragraph'
    | 'merge_paragraph';
  target?: number[];
  text?: string;
  keep_ms?: number;
}

export interface CommittedTransition {
  sequence: number;
  origin: 'human' | 'ai_computer_use' | 'synthetic';
  transaction_id: string;
  before: SessionDocument;
  after: SessionDocument;
  prediction: { action: string; target?: number[]; text?: string; keep_ms?: number };
  undo_of: string | null;
}

export type CopilotOutcome =
  | 'shown'
  | 'accepted'
  | 'dismissed'
  | 'modified'
  | 'undone'
  | 'expired'
  | 'invalidated';

export interface CopilotFeedback {
  event_id: string;
  document_id: string;
  request: CopilotRequestIdentity;
  proposal_id: string;
  outcome: CopilotOutcome;
  sequence: number;
  origin: 'human' | 'ai_computer_use' | 'synthetic';
  reason: 'unspecified';
  observed_at: string;
  revision: number | null;
  interaction_epoch: number | null;
  transition?: CommittedTransition;
}

export interface CopilotHistoryTip {
  transaction_id: string | null;
  sequence: number;
  count: number;
}

export interface CopilotResponse {
  request: CopilotRequestIdentity;
  status: 'proposal' | 'no_action' | 'suppressed';
  proposal_id: string | null;
  prediction: CopilotPrediction;
  target_uuids: string[];
  document_hash: string;
  result_document: SessionDocument | null;
  result_hash: string | null;
  context: Record<string, unknown>;
  model: Record<string, unknown>;
  timings: Record<string, number | string>;
  history_tip?: CopilotHistoryTip;
}

export function sameRequest(a: CopilotRequestIdentity, b: CopilotRequestIdentity): boolean {
  return (
    a.request_id === b.request_id &&
    a.document_id === b.document_id &&
    a.revision === b.revision &&
    a.interaction_epoch === b.interaction_epoch
  );
}

export const isTimed = (item: V3DocumentItem): boolean =>
  ['text', 'non_text', 'artificial_silence'].includes(item.type);
const equal = (a: unknown, b: unknown): boolean => sessionHash(a) === sessionHash(b);

function structuralSessionEqual(a: unknown, b: unknown): boolean {
  const supported = (value: any): boolean => {
    if (typeof value === 'number') {
      if (!Number.isFinite(value)) throw new Error('Session numbers must be finite.');
      return true;
    }
    if (value === null || ['undefined', 'boolean', 'string'].includes(typeof value)) return true;
    if (typeof value !== 'object') return false;
    const prototype = Object.getPrototypeOf(value);
    if (Array.isArray(value)) {
      if (prototype !== Array.prototype || value.map !== Array.prototype.map) return false;
      for (let index = 0; index < value.length; index++) {
        const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
        if (!descriptor) {
          if (index in value) return false;
        } else if (!('value' in descriptor) || !supported(descriptor.value)) return false;
      }
      return true;
    }
    if (prototype !== Object.prototype && prototype !== null) return false;
    for (const key of Object.keys(value)) {
      const descriptor = Object.getOwnPropertyDescriptor(value, key)!;
      if (!('value' in descriptor) || !supported(descriptor.value)) return false;
    }
    return true;
  };
  const compare = (left: any, right: any): boolean => {
    if (left === right) return true;
    if (left === null || right === null || typeof left !== 'object' || typeof right !== 'object')
      return false;
    if (Array.isArray(left) || Array.isArray(right)) {
      if (!Array.isArray(left) || !Array.isArray(right)) return false;
      // The current canonical array join makes [] and [undefined] identical.
      const length = left.length === 1 && left[0] === undefined ? 0 : left.length;
      const otherLength = right.length === 1 && right[0] === undefined ? 0 : right.length;
      if (length !== otherLength) return false;
      for (let index = 0; index < length; index++)
        if (!compare(left[index], right[index])) return false;
      return true;
    }
    const keys = Object.keys(left).filter((key) => left[key] !== undefined);
    if (keys.length !== Object.keys(right).filter((key) => right[key] !== undefined).length)
      return false;
    return keys.every(
      (key) => Object.prototype.hasOwnProperty.call(right, key) && compare(left[key], right[key])
    );
  };
  // Keep the original canonicalizer for executable or non-plain JS values.
  if (a === undefined || b === undefined || !supported(a) || !supported(b)) return equal(a, b);
  return compare(a, b);
}

type SourceItem = Extract<V3DocumentItem, { type: 'text' | 'non_text' }>;

function sourceSpan(document: SessionDocument, indices: number[]): SourceItem[] | null {
  const selected = indices.map((index) => document.content[index]);
  if (
    !selected.length ||
    indices.some((index, i) => index !== indices[0] + i) ||
    selected.some(
      (item) =>
        (item.type !== 'text' && item.type !== 'non_text') ||
        typeof item.source !== 'string' ||
        !item.source ||
        !Number.isFinite(item.sourceStart) ||
        item.sourceStart < 0 ||
        !Number.isFinite(item.length) ||
        item.length <= 0
    )
  )
    return null;
  const items = selected as SourceItem[];
  if (
    items.some(
      (item, i) =>
        item.source !== items[0].source ||
        (i > 0 &&
          Math.abs(items[i - 1].sourceStart + items[i - 1].length - item.sourceStart) > 1e-6)
    )
  )
    return null;
  return items;
}

export function copilotPreviewText(prediction: CopilotPrediction): string {
  switch (prediction.action) {
    case 'no_action':
      return 'No edit suggested.';
    case 'correct_text':
      return `→ ${prediction.text}`;
    case 'delete_span':
      return 'Delete this span and its source audio';
    case 'trim_silence':
      return `Shorten this pause to ${prediction.keep_ms} ms (keep both edges)`;
    case 'remove_silence':
      return 'Remove this entire pause';
    case 'split_paragraph':
      return '¶ Start a new caption here';
    case 'merge_paragraph':
      return '↳ Join this caption with the previous one';
  }
}

// The server decides proposal eligibility. This boundary verifies the exact UUID-bound
// document change, including source intervals, before it enters undoable editor history.
export function validCopilotResult(document: SessionDocument, response: CopilotResponse): boolean {
  return validateCopilotResult(document, response);
}

function validateCopilotResult(
  document: SessionDocument,
  response: CopilotResponse,
  hashes?: [string, string]
): boolean {
  try {
    const result = response.result_document,
      prediction = response.prediction;
    if (response.status !== 'proposal' || !response.proposal_id || !result) return false;
    const [documentHash, resultHash] = hashes || [sessionHash(document), sessionHash(result)];
    if (
      documentHash !== response.document_hash ||
      resultHash !== response.result_hash ||
      !structuralSessionEqual(document.metadata, result.metadata) ||
      !lintDocumentContent(document.content).pass ||
      !lintDocumentContent(result.content).pass ||
      new Set(document.content.map((item) => item.uuid)).size !== document.content.length ||
      new Set(result.content.map((item) => item.uuid)).size !== result.content.length ||
      !response.target_uuids.length ||
      new Set(response.target_uuids).size !== response.target_uuids.length ||
      prediction.target?.length !== response.target_uuids.length
    )
      return false;
    const timed = document.content.filter(isTimed);
    if (
      prediction.target?.some(
        (alias, i) =>
          !Number.isSafeInteger(alias) ||
          alias < 0 ||
          timed[alias]?.uuid !== response.target_uuids[i]
      )
    )
      return false;
    const indices = response.target_uuids.map((uuid) =>
      document.content.findIndex((item) => item.uuid === uuid && isTimed(item))
    );
    if (indices.some((index) => index < 0)) return false;
    const first = indices[0],
      expected = [...document.content];
    if (prediction.action === 'correct_text') {
      const selected = sourceSpan(document, indices);
      if (!prediction.text?.trim() || !selected?.some((item) => item.type === 'text')) return false;
      expected.splice(first, indices.length, {
        ...selected[0],
        type: 'text',
        text: prediction.text,
        conf: 1,
        length: selected.reduce((sum, item) => sum + item.length, 0),
      });
    } else if (
      prediction.action === 'delete_span' ||
      prediction.action === 'trim_silence' ||
      prediction.action === 'remove_silence'
    ) {
      const selected = sourceSpan(document, indices);
      if (!selected || prediction.text !== undefined) return false;
      if (prediction.action === 'delete_span') {
        if (!selected.some((item) => item.type === 'text') || prediction.keep_ms !== undefined)
          return false;
        expected.splice(first, selected.length);
      } else {
        const gap = selected[0];
        if (selected.length !== 1 || gap.type !== 'non_text') return false;
        if (prediction.action === 'remove_silence') {
          if (prediction.keep_ms !== undefined) return false;
          expected.splice(first, 1);
        } else {
          const keep = prediction.keep_ms,
            right = result.content[first + 1];
          if (
            keep === undefined ||
            !Number.isSafeInteger(keep) ||
            keep <= 0 ||
            keep / 1000 >= gap.length ||
            right?.type !== 'non_text' ||
            typeof right.uuid !== 'string' ||
            !right.uuid ||
            document.content.some((item) => item.uuid === right.uuid)
          )
            return false;
          const half = keep / 2000;
          expected.splice(
            first,
            1,
            { ...gap, length: half },
            {
              ...gap,
              uuid: right.uuid,
              sourceStart: gap.sourceStart + gap.length - half,
              length: half,
            }
          );
        }
      }
    } else if (prediction.action === 'split_paragraph') {
      if (indices.length !== 1) return false;
      const marker = [...document.content.slice(0, first)]
        .reverse()
        .find((item) => item.type === 'paragraph_start');
      const end = result.content[first],
        start = result.content[first + 1];
      if (!marker || end?.type !== 'paragraph_end' || start?.type !== 'paragraph_start')
        return false;
      expected.splice(
        first,
        0,
        { type: 'paragraph_end', uuid: end.uuid },
        { ...marker, uuid: start.uuid }
      );
    } else if (prediction.action === 'merge_paragraph') {
      if (
        indices.length !== 1 ||
        document.content[first - 2]?.type !== 'paragraph_end' ||
        document.content[first - 1]?.type !== 'paragraph_start'
      )
        return false;
      expected.splice(first - 2, 2);
    } else return false;
    return structuralSessionEqual(expected, result.content) && documentHash !== resultHash;
  } catch {
    return false;
  }
}

export function inferManualPrediction(
  before: SessionDocument,
  after: SessionDocument
): CommittedTransition['prediction'] | null {
  return inferManualPredictionWithHashes(before, after);
}

export function inspectManualEdit(
  before: SessionDocument,
  after: SessionDocument
): { changed: boolean; prediction: CommittedTransition['prediction'] | null } {
  const hashes: [string, string] = [sessionHash(before), sessionHash(after)];
  const changed = hashes[0] !== hashes[1];
  return {
    changed,
    prediction: changed ? inferManualPredictionWithHashes(before, after, hashes) : null,
  };
}

function inferManualPredictionWithHashes(
  before: SessionDocument,
  after: SessionDocument,
  hashes?: [string, string]
): CommittedTransition['prediction'] | null {
  if (!equal(before.metadata, after.metadata)) return null;
  const patch = contentPatch(before, after),
    timed = before.content.filter(isTimed);
  const aliasAt = (index: number) =>
    timed.findIndex((item) => item.uuid === before.content[index]?.uuid);
  const targets = patch.remove.map((item) =>
    timed.findIndex((timedItem) => timedItem.uuid === item.uuid)
  );
  const validate = (prediction: CopilotPrediction): CopilotPrediction | null => {
    const [documentHash, resultHash] = hashes || [sessionHash(before), sessionHash(after)];
    const response = {
      status: 'proposal',
      proposal_id: 'manual',
      prediction,
      target_uuids: prediction.target?.map((alias) => timed[alias]?.uuid),
      document_hash: documentHash,
      result_document: after,
      result_hash: resultHash,
    } as CopilotResponse;
    return validateCopilotResult(before, response, [documentHash, resultHash]) ? prediction : null;
  };
  if (
    patch.remove.length > 0 &&
    targets.every((n) => n >= 0) &&
    patch.insert.length === 1 &&
    patch.insert[0].type === 'text' &&
    patch.remove.every((item) => item.type === 'text' || item.type === 'non_text')
  ) {
    return validate({
      action: 'correct_text',
      target: targets,
      text: patch.insert[0].text,
    });
  }
  if (
    !patch.remove.length &&
    patch.insert.length === 2 &&
    patch.insert[0].type === 'paragraph_end' &&
    patch.insert[1].type === 'paragraph_start' &&
    aliasAt(patch.index) >= 0
  )
    return validate({ action: 'split_paragraph', target: [aliasAt(patch.index)] });
  if (
    !patch.insert.length &&
    patch.remove.length === 2 &&
    patch.remove[0].type === 'paragraph_end' &&
    patch.remove[1].type === 'paragraph_start' &&
    aliasAt(patch.index + 2) >= 0
  )
    return validate({ action: 'merge_paragraph', target: [aliasAt(patch.index + 2)] });
  if (
    patch.remove.length === 1 &&
    patch.remove[0].type === 'non_text' &&
    patch.insert.length === 2 &&
    patch.insert.every((item) => item.type === 'non_text')
  )
    return validate({
      action: 'trim_silence',
      target: targets,
      keep_ms: Math.round(
        patch.insert.reduce((sum, item) => sum + ('length' in item ? item.length : 0), 0) * 1000
      ),
    });
  if (!patch.insert.length && patch.remove.length && targets.every((n) => n >= 0)) {
    if (patch.remove.length === 1 && patch.remove[0].type === 'non_text')
      return validate({ action: 'remove_silence', target: targets });
    if (patch.remove.some((item) => item.type === 'text'))
      return validate({ action: 'delete_span', target: targets });
  }
  return null;
}
