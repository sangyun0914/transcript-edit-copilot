import { ServerConfig } from '../state/server';

export async function fetchFromServer(
  server: ServerConfig,
  method: 'POST' | 'GET' | 'DELETE',
  route: string,
  getParameters?: Record<string, string | boolean | number>,
  body?: {
    form?: Record<string, any>;
    json?: any;
  },
  signal?: AbortSignal
): Promise<Response> {
  const options: RequestInit = {
    method,
    headers: { Authorization: `Bearer ${server.token}` },
    signal,
  };

  const getString = getParameters ? '?' + encodeGetParams(getParameters) : '';

  if (body?.form) {
    const formData = new FormData();
    Object.entries(body.form).forEach(([k, v]) => {
      formData.append(k, v);
    });
    options.body = formData;
  } else if (body?.json) {
    options.body = JSON.stringify(body.json);
    options.headers = { ...options.headers, 'content-type': 'application/json' };
  }

  const response = await fetch(`${server.hostname}:${server.port}/${route}/${getString}`, options);
  if (!response.ok) {
    const body = await response.text();
    let message = `Server request failed (HTTP ${response.status}).`;
    let code: string | undefined;
    try {
      const parsed = JSON.parse(body);
      if (typeof parsed.detail === 'string') message = parsed.detail;
      else if (typeof parsed.detail?.message === 'string') message = parsed.detail.message;
      if (typeof parsed.detail?.code === 'string') code = parsed.detail.code;
    } catch {
      // Older endpoints return plain text errors.
      if (body) message = body;
    }
    throw Object.assign(new Error(message), { status: response.status, code });
  }
  return response;
}

function encodeGetParams(params: Record<string, string | boolean | number>): string {
  return Object.entries(params)
    .map((kv) => kv.map(encodeURIComponent).join('='))
    .join('&');
}
