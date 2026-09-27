import { getTask, startTranscription } from './api';

const server = { hostname: 'http://localhost', port: 8000, name: 'test', token: 'test-token' };

afterEach(() => jest.restoreAllMocks());

test('Scribe requests preserve the selected speaker count and require no local model', async () => {
  global.fetch = jest
    .fn()
    .mockResolvedValue({ ok: true, json: async () => ({ uuid: 'task', state: 'queued' }) });
  await startTranscription(
    server,
    'scribe',
    undefined,
    'ko',
    true,
    4,
    new File(['audio'], 'input.wav'),
    'sample.wav'
  );
  const [url, options] = (fetch as jest.Mock).mock.calls[0];
  expect(url).toContain('provider=scribe');
  expect(url).toContain('language_code=ko');
  expect(url).toContain('diarize_max_speakers=4');
  expect(url).not.toContain('transcription_model');
  expect(options.body.get('fileName')).toBe('sample.wav');
});

test('HTTP failures reject instead of being treated as transcription tasks', async () => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: false,
    status: 503,
    text: async () => JSON.stringify({ detail: 'ElevenLabs is not configured on this server.' }),
  });
  await expect(getTask(server, { uuid: 'task' })).rejects.toThrow('ElevenLabs is not configured');
});
