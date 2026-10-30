import { fetchFromServer } from './index';

test('structured server errors retain their recovery code and readable message', async () => {
  const previous = global.fetch;
  global.fetch = jest.fn().mockResolvedValue({
    ok: false,
    status: 409,
    text: async () =>
      JSON.stringify({
        detail: { code: 'history_resync_required', message: 'History is missing.' },
      }),
  });
  try {
    await expect(
      fetchFromServer(
        { hostname: 'http://localhost', port: 8000, token: 'test', name: 'local' },
        'POST',
        'copilot/predict'
      )
    ).rejects.toMatchObject({
      message: 'History is missing.',
      status: 409,
      code: 'history_resync_required',
    });
  } finally {
    global.fetch = previous;
  }
});
