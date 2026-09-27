const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
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
const directory = path.resolve(process.argv[2] || path.join(root, 'artifacts/iterations/source-qualification-v1'));
const rows = name => fs.readFileSync(path.join(directory, name), 'utf8').trim().split('\n').map(JSON.parse);
const cases = [...rows('train-cases.jsonl'), ...rows('dev-cases.jsonl')];
const digest = filename => crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
function cues(document) {
  return contentToVtt(document.content, false, false, null).toString().split('\n\n').slice(1).map(block => {
    const [timing, ...lines] = block.trim().split('\n');
    const seconds = value => value.split(':').reduce((n, v) => n * 60 + Number(v), 0);
    const [start, end] = timing.split(' --> ').map(seconds);
    return {start, end, text: lines.join(' ')};
  });
}
const results = [];
for (const c of cases.filter(c => ['split_paragraph', 'merge_paragraph'].includes(c.expectedOutput.action))) {
  const events = rows(c.metadata.session);
  const before = events[0].document;
  const after = structuredClone(before);
  for (const event of events.filter(e => e.kind === 'transaction')) {
    const patch = event.patch;
    after.content.splice(patch.index, patch.remove.length, ...structuredClone(patch.insert));
    after.metadata = patch.metadata;
  }
  const left = cues(before), right = cues(after);
  const split = c.expectedOutput.action === 'split_paragraph';
  assert.equal(right.length - left.length, split ? 1 : -1);
  assert.equal(left.map(c => c.text).join(' '), right.map(c => c.text).join(' '));
  assert.equal(left[0].start, right[0].start);
  assert.equal(left.at(-1).end, right.at(-1).end);
  const paragraph = c.input.items.find(i => i.id === c.expectedOutput.target[0]).paragraph;
  const offset = split ? paragraph : paragraph - 1;
  const affectedBefore = left.slice(offset, offset + (split ? 1 : 2));
  const affectedAfter = right.slice(offset, offset + (split ? 2 : 1));
  assert.equal(affectedBefore[0].start, affectedAfter[0].start);
  assert.equal(affectedBefore.at(-1).end, affectedAfter.at(-1).end);
  assert(affectedAfter.every(c => c.end - c.start >= 1));
  if (!split) {
    assert(affectedAfter[0].text.length <= 84);
    assert(affectedAfter[0].end - affectedAfter[0].start <= 7);
  }
  results.push({id: c.id, action: c.expectedOutput.action, before: affectedBefore, after: affectedAfter});
}
assert(results.length > 0);
const report = {
  status: 'passed', cases: results, exporter: 'actual Audapolis contentToVtt',
  export_settings: {wordTimings: false, includeSpeakerNames: false, limitLineLength: null},
  scope: 'Caption payload, cue count and timings verified; unchanged source intervals do not certify perceptual audio quality.',
  inputs_sha256: Object.fromEntries([
    path.join(root, 'app/src/core/webvtt.ts'), path.join(root, 'app/src/state/editor/selectors.ts'),
    path.join(directory, 'train-cases.jsonl'), path.join(directory, 'dev-cases.jsonl'), __filename,
  ].map(p => [path.relative(root, p), digest(p)])),
};
fs.writeFileSync(path.join(directory, 'caption-export-check.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report));
