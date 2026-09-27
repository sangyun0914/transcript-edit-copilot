import _ from 'lodash';
import { defaultEditorState } from '../state/editor/types';
import { trimSelectedGap } from '../state/editor/edit';
import { gapSeconds, GapItem } from './gaps';

const sourceGap: GapItem = {
  type: 'non_text',
  uuid: 'gap',
  source: 's',
  sourceStart: 1,
  length: 3,
};

function stateFor(gaps: GapItem[]) {
  const state = _.cloneDeep(defaultEditorState);
  state.document.content = [
    { type: 'paragraph_start', uuid: 'p', speaker: 'Speaker', language: 'en' },
    ..._.cloneDeep(gaps),
    { type: 'paragraph_end', uuid: 'e' },
  ];
  state.selection = { startIndex: 1, length: gaps.length, headPosition: 'right' };
  return state;
}

test('repeated inline trims preserve the outside source edges across existing cuts', () => {
  const state = stateFor([sourceGap]);
  trimSelectedGap.reducer(state, { uuid: 'gap', keepMs: 725, expectedLength: 3 });
  expect(state.document.content.slice(1, 3)).toMatchObject([
    { sourceStart: 1, length: 0.3625 },
    { sourceStart: 3.6375, length: 0.3625 },
  ]);
  state.selection = { startIndex: 1, length: 2, headPosition: 'right' };
  trimSelectedGap.reducer(state, { uuid: 'gap', keepMs: 400, expectedLength: 0.725 });
  expect(state.document.content.slice(1, 3)).toMatchObject([
    { sourceStart: 1, length: 0.2 },
    { sourceStart: 3.8, length: 0.2 },
  ]);
  expect(new Set(state.document.content.map((item) => item.uuid)).size).toBe(4);
});

test('retained edges can span several source fragments without filling a removed interval', () => {
  const state = stateFor([
    { ...sourceGap, length: 0.1 },
    { ...sourceGap, uuid: 'middle', sourceStart: 2, length: 1 },
    { ...sourceGap, uuid: 'last', sourceStart: 4, length: 0.2 },
  ]);
  trimSelectedGap.reducer(state, { uuid: 'gap', keepMs: 800 });
  const pieces = state.document.content.slice(1, -1) as GapItem[];
  expect(pieces).toHaveLength(4);
  expect(pieces.reduce((sum, item) => sum + item.length, 0)).toBeCloseTo(0.8);
  expect(pieces[0]).toMatchObject({ sourceStart: 1, length: 0.1 });
  expect(pieces[3]).toMatchObject({ sourceStart: 4, length: 0.2 });
  expect(new Set(pieces.map((item) => item.uuid)).size).toBe(4);
});

test('artificial pauses can also be shortened without inventing source mappings', () => {
  const state = stateFor([{ type: 'artificial_silence', uuid: 'gap', length: 0.1 }]);
  trimSelectedGap.reducer(state, { uuid: 'gap', keepMs: 40 });
  expect(state.document.content.slice(1, 3)).toEqual([
    { type: 'artificial_silence', uuid: 'gap', length: 0.02 },
    { type: 'artificial_silence', uuid: expect.any(String), length: 0.02 },
  ]);
});

test('invalid, stale and cross-source inline requests preserve the document', () => {
  for (const keepMs of [0, -1, 3000, NaN, Infinity, 1.5]) {
    const state = stateFor([sourceGap]);
    const before = _.cloneDeep(state);
    trimSelectedGap.reducer(state, { uuid: 'gap', keepMs });
    expect(state).toEqual(before);
  }
  for (const request of [
    { uuid: 'missing', keepMs: 300 },
    { uuid: 'gap', keepMs: 300, expectedLength: 4 },
  ]) {
    const state = stateFor([sourceGap]);
    const before = _.cloneDeep(state);
    trimSelectedGap.reducer(state, request);
    expect(state).toEqual(before);
  }
  const state = stateFor([sourceGap, { ...sourceGap, uuid: 'other', source: 'other' }]);
  const before = _.cloneDeep(state);
  trimSelectedGap.reducer(state, { uuid: 'gap', keepMs: 300 });
  expect(state).toEqual(before);
});

test('seconds labels retain short pauses without showing a positive gap as zero', () => {
  expect(gapSeconds(0.1)).toBe('0.1');
  expect(gapSeconds(0.725)).toBe('0.725');
  expect(gapSeconds(0.0005)).toBe('<0.001');
});
