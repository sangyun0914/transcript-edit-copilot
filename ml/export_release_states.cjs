/* Audit actual exported bytes against each executed document, using a persistent JSONL process. */
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const readline = require('node:readline');
const {createRequire} = require('node:module');
const root = path.resolve(__dirname, '..');
const appRequire = createRequire(path.join(root, 'app/package.json'));
const swc = appRequire('@swc/core');
require.extensions['.ts'] = (module, filename) => {
  const result = swc.transformSync(fs.readFileSync(filename, 'utf8'), {
    filename, jsc: {parser: {syntax: 'typescript'}, target: 'es2020'},
    module: {type: 'commonjs'},
  });
  module._compile(result.code, filename);
};
const {contentToVtt} = require(path.join(root, 'app/src/core/webvtt.ts'));
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const milliseconds = seconds => Math.floor(seconds) * 1000 + Math.floor((seconds % 1) * 1000);
const stamp = value => {
  const match = /^(\d{2,}):(\d{2}):(\d{2})\.(\d{3})$/.exec(value);
  assert(match, `Malformed caption timestamp: ${value}`);
  assert(Number(match[2]) < 60 && Number(match[3]) < 60);
  return ((Number(match[1]) * 60 + Number(match[2])) * 60 + Number(match[3])) * 1000 + Number(match[4]);
};
const decode = value => value.replace(/&(amp|lt|gt);/g, (_, name) => ({amp: '&', lt: '<', gt: '>'})[name]);

function parseVtt(value) {
  const lines = value.split('\n');
  assert(lines[0].startsWith('WEBVTT'));
  const cues = [];
  let index = 1;
  while (index < lines.length) {
    if (lines[index] === '') { index++; continue; }
    const match = /^(\d{2,}:\d{2}:\d{2}\.\d{3}) --> (\d{2,}:\d{2}:\d{2}\.\d{3})$/.exec(lines[index++]);
    assert(match, `Malformed cue header or broken multiline payload: ${lines[index - 1]}`);
    const payload = [];
    while (index < lines.length && lines[index] !== '') payload.push(lines[index++]);
    cues.push({start_ms: stamp(match[1]), end_ms: stamp(match[2]), payload: payload.join('\n')});
  }
  return cues;
}

function audit(document, lineLength) {
  const groups = [];
  let current = null, elapsed = 0, characterLength = 0;
  const ids = new Set();
  for (const item of document.content) {
    assert(!ids.has(item.uuid), 'Duplicate document UUID');
    ids.add(item.uuid);
    if (item.type === 'paragraph_start') {
      assert(current === null, 'Nested or unterminated paragraph');
      current = {paragraph_id: item.uuid, segment: 0, items: []};
      characterLength = 0;
    } else if (item.type === 'paragraph_end') {
      assert(current !== null, 'Unmatched paragraph end');
      if (current.items.length) groups.push(current);
      current = null;
    } else {
      assert(['text', 'non_text', 'artificial_silence'].includes(item.type), 'Unknown exported document item');
      assert(current !== null && Number.isFinite(item.length) && item.length > 0, 'Invalid timed item');
      if (item.type === 'text') {
        assert.equal(typeof item.text, 'string');
        if (lineLength !== null && characterLength + item.text.length > lineLength) {
          if (current.items.length) groups.push(current);
          current = {paragraph_id: current.paragraph_id, segment: current.segment + 1, items: []};
          characterLength = 0;
        }
        characterLength += item.text.length;
      }
      const start = elapsed;
      elapsed += item.length;
      current.items.push({...item, start_seconds: start, end_seconds: elapsed});
    }
  }
  assert(current === null, 'Missing final paragraph end');
  const exported = contentToVtt(document.content, false, false, lineLength).toString();
  const actual = parseVtt(exported);
  assert.equal(actual.length, groups.length, 'Exporter dropped or invented a cue');
  const cues = actual.map((cue, index) => {
    const group = groups[index];
    const first = group.items[0], last = group.items.at(-1);
    const words = group.items.filter(item => item.type === 'text');
    const text = words.map(item => item.text).join(' ');
    assert.equal(cue.start_ms, milliseconds(first.start_seconds), 'Cue start differs from executed timeline');
    assert.equal(cue.end_ms, milliseconds(last.end_seconds), 'Cue end differs from executed timeline');
    assert.equal(decode(cue.payload), text, 'Exported cue payload dropped or changed executed text');
    assert(!/[<>]/.test(cue.payload), 'Unescaped markup in a plain-text caption');
    assert(cue.end_ms >= cue.start_ms, 'Reversed exported cue');
    if (index) assert(cue.start_ms >= actual[index - 1].end_ms, 'Overlapping or reordered cue');
    return {
      paragraph_id: group.paragraph_id, segment: group.segment,
      start_ms: cue.start_ms, end_ms: cue.end_ms,
      start_seconds: first.start_seconds, end_seconds: last.end_seconds,
      payload_sha256: hash(cue.payload), text_sha256: hash(text), text_utf16: text.length,
      timed_ids: group.items.map(item => item.uuid), text_ids: words.map(item => item.uuid),
      empty: text.length === 0, zero_duration: cue.start_ms === cue.end_ms,
    };
  });
  if (cues.length) {
    assert.equal(cues[0].start_ms, 0, 'Export omitted the start of the executed timeline');
    assert.equal(cues.at(-1).end_ms, milliseconds(elapsed), 'Export omitted the end of the executed timeline');
    for (let index = 1; index < cues.length; index++) {
      assert.equal(cues[index].start_ms, cues[index - 1].end_ms, 'A timed document interval is absent from export');
    }
  }
  return {
    status: 'passed', export_sha256: hash(exported), cues,
    duration_seconds: elapsed, exported_end_ms: cues.at(-1)?.end_ms || 0,
    timed_items: cues.reduce((n, cue) => n + cue.timed_ids.length, 0),
    text_items: cues.reduce((n, cue) => n + cue.text_ids.length, 0),
  };
}

const input = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
input.on('line', line => {
  try {
    const request = JSON.parse(line);
    process.stdout.write(JSON.stringify(audit(request.document, request.line_length)) + '\n');
  } catch (error) {
    process.stdout.write(JSON.stringify({status: 'failed', error: String(error.stack || error)}) + '\n');
  }
});
