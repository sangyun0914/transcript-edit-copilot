import { UuidExtension, V3ArtificialSilence, V3DocumentItem, V3NonTextItem } from './document';

export type GapItem = (V3NonTextItem | V3ArtificialSilence) & UuidExtension;

export function isGap(item: V3DocumentItem | undefined): item is GapItem {
  return item?.type === 'non_text' || item?.type === 'artificial_silence';
}

export function sameGap(left: V3DocumentItem | undefined, right: V3DocumentItem): boolean {
  return (
    isGap(left) &&
    isGap(right) &&
    left.type === right.type &&
    (left.type === 'artificial_silence' ||
      (right.type === 'non_text' && left.source === right.source))
  );
}

export function gapSeconds(length: number): string {
  return length < 0.001 ? '<0.001' : String(Number(length.toFixed(3)));
}
