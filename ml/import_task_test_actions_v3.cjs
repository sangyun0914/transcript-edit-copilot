// Execute the existing app importer; the caller freezes the generated UUIDs separately.
const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const root = path.resolve(__dirname, '..');
const appRequire = createRequire(path.join(root, 'app/package.json'));
const swc = appRequire('@swc/core');
require.extensions['.ts'] = (module, filename) => {
  const result = swc.transformSync(fs.readFileSync(filename, 'utf8'), {
    filename, jsc: {parser: {syntax: 'typescript'}, target: 'es2020'}, module: {type: 'commonjs'},
  });
  module._compile(result.code, filename);
};
const {importTranscript} = require(path.join(root, 'app/src/core/transcription.ts'));
const request = JSON.parse(fs.readFileSync(0, 'utf8'));
const document = importTranscript(request.content, request.source, request.language, request.transcription);
process.stdout.write(JSON.stringify(document));
