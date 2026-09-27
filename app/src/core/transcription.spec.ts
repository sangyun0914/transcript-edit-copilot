import { importTranscript, TranscriptionContent, TranscriptionMetadata } from './transcription';
import { deserializeDocument, serializeDocument } from './document';
import { contentToVtt } from './webvtt';
import { memoizedDocumentRenderItems } from '../state/editor/selectors';

const content: TranscriptionContent = [
  {
    speaker: 'speaker_0',
    content: [
      { type: 'silence', sourceStart: 0, length: 0.25 },
      {
        type: 'word',
        word: '어',
        sourceStart: 0.25,
        length: 0.5,
        conf: null,
        asr: { word_indices: [0], boundary_status: 'unverified' },
      },
      { type: 'silence', sourceStart: 0.75, length: 0.25 },
      {
        type: 'word',
        word: '데이터를',
        sourceStart: 1,
        length: 1,
        conf: null,
        asr: { word_indices: [2, 3], boundary_status: 'needs_alignment' },
      },
    ],
  },
  {
    speaker: 'speaker_1',
    content: [
      {
        type: 'word',
        word: '분석합니다.',
        sourceStart: 2,
        length: 1,
        conf: null,
        asr: { word_indices: [4], boundary_status: 'overlap' },
      },
    ],
  },
];
const metadata: TranscriptionMetadata = {
  version: 1,
  provider: 'elevenlabs',
  model: 'scribe_v2',
  language_code: 'kor',
  duration: 3,
  request: { no_verbatim: 'false' },
  raw_response: { text: '어 데이터를 분석합니다.', words: [{ text: '어', logprob: -0.4 }] },
};

test('import, save and reopen preserve source intervals, unknown confidence and raw provenance', async () => {
  const imported = importTranscript(content, 'source-1', 'ko', metadata);
  const document = {
    ...imported,
    sources: { 'source-1': { objectUrl: 'test', fileContents: new ArrayBuffer(0) } },
  };
  global.URL.createObjectURL = jest.fn(() => 'restored');
  const zip = await serializeDocument(document).generateAsync({ type: 'nodebuffer' });
  const restored = await deserializeDocument(zip);
  expect(restored.content).toEqual(document.content);
  expect(restored.metadata).toEqual(document.metadata);
  const words = restored.content.filter((item) => item.type === 'text');
  expect(words).toHaveLength(3);
  expect(restored.metadata.transcriptions?.[0].regions[0].word_indices).toEqual([0]);
  expect(restored.metadata.transcriptions?.[0].regions[1]).toEqual({
    word_indices: [2, 3],
    boundary_status: 'needs_alignment',
    sourceStart: 1,
    length: 1,
  });
  expect(restored.metadata.transcriptions?.[0].raw_response).toEqual(metadata.raw_response);
  expect(restored.content.filter((item) => item.type === 'paragraph_start')).toEqual([
    expect.objectContaining({ language: 'ko', speaker: 'speaker_0' }),
    expect.objectContaining({ language: 'ko', speaker: 'speaker_1' }),
  ]);
  const segments = memoizedDocumentRenderItems(restored.content, false);
  expect(segments.reduce((sum, item) => sum + item.length, 0)).toBeCloseTo(3);
  expect(contentToVtt(restored.content, false, true, null).toString()).toContain(
    '00:00:02.000 --> 00:00:03.000\n<v speaker_1>분석합니다.'
  );
});

test('legacy Vosk imports retain numeric confidence without fabricating provenance', () => {
  const document = importTranscript(
    [
      {
        speaker: 'sample',
        content: [{ type: 'word', word: 'hello', conf: 0.75, sourceStart: 0, length: 1 }],
      },
    ],
    'source-1',
    'en'
  );
  expect(document.content[1]).toEqual(expect.objectContaining({ type: 'text', conf: 0.75 }));
  expect(document.metadata.transcriptions).toBeUndefined();
});
