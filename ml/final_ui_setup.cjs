/* Explicit devconsole setup only. No model calls, request actions, toggles, or product changes. */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const assert = require('assert').strict;

async function create({ window, registration, importModule }) {
  assert.equal(window.location.protocol, 'http:');
  assert(['localhost', '127.0.0.1', '[::1]'].includes(window.location.hostname));
  assert.equal(window.process.env.MODE, 'development', 'Use ordinary npm start, not test mode.');
  const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
  const bytes = binding => {
    const data = fs.readFileSync(binding.path);
    assert.equal(hash(data), binding.sha256, `Changed registered file: ${binding.path}`);
    return data;
  };
  const read = binding => JSON.parse(bytes(binding));
  const roster = read(registration);
  assert.equal(roster.version, 'final-ui-latency-measurement-1');
  assert.equal(roster.normal_app_setup.path, __filename);
  bytes(roster.normal_app_setup);
  bytes(roster.measurement_plan);
  for (const [relative, expected] of Object.entries(roster.trace_sources)) {
    assert.equal(hash(fs.readFileSync(path.resolve(__dirname, '..', relative))), expected);
  }
  const names = ['nav', 'transcribe', 'editor', 'models', 'server', 'recording', 'copilot'];
  const modules = await Promise.all(names.map(name => importModule(`/state/${name === 'editor' ? 'editor/index' : name}.ts`)));
  const slices = Object.fromEntries(names.map((name, index) => [name, modules[index]]));
  const production = window.require('@reduxjs/toolkit').combineReducers(
    Object.fromEntries(names.map(name => [name, slices[name].default]))
  );
  const { store } = await importModule('/state/index.ts');
  assert.equal(store, window.store);
  const documentApi = await importModule('/core/document.ts');
  const { sessionDocument, sessionHash } = await importModule('/core/editing_session.ts');
  const { defaultEditorState } = await importModule('/state/editor/types.ts');
  const original = store.getState();
  assert(!original.editor.present?.playing && !original.copilot.request && !original.copilot.proposal,
    'Capture the original while playback, requests and previews are idle.');
  assert(!original.copilot.events.some(event => event.persistence === 'pending'));
  const audit = [], loaded = new Map();
  const clone = value => JSON.parse(JSON.stringify(value));
  const facts = state => ({
    document_hash: state.editor.present ? sessionHash(sessionDocument(state.editor.present.document)) : null,
    history_hash: sessionHash(state.copilot.history), history_count: state.copilot.history.length,
    cursor: state.editor.present?.cursor, selection: state.editor.present?.selection,
    undo_past: state.editor.past.length, undo_future: state.editor.future.length,
    document_id: state.copilot.documentId, path: state.editor.present?.path,
    unsaved: state.editor.present?.document !== state.editor.present?.lastSavedDocument,
  });
  const originalFacts = facts(original);
  async function replace(parts, label) {
    audit.push({ kind: 'before_setup', label, facts: facts(store.getState()),
      measurements: clone(store.getState().copilot.measurements), feedback: clone(store.getState().copilot.events) });
    store.dispatch(slices.nav.openLanding());
    const deadline = Date.now() + 35000;
    while (store.getState().copilot.events.some(event => event.persistence === 'pending')) {
      assert(Date.now() < deadline, 'Setup feedback did not settle; keep the failed setup evidence.');
      await new Promise(resolve => setTimeout(resolve, 25));
    }
    assert(!store.getState().copilot.feedbackError, 'Do not hide failed feedback by installing a new seed.');
    store.replaceReducer((state, action) => {
      const next = production(state, action);
      return action.type === 'copilot/finalUILatencySetup' ? { ...next, ...parts } : next;
    });
    try { store.dispatch({ type: 'copilot/finalUILatencySetup' }); }
    finally { store.replaceReducer(production); }
    const value = { kind: 'setup_complete', label, facts: facts(store.getState()),
      health: store.getState().copilot.healthStatus, model_request_synthesized: false };
    audit.push(value);
    return value;
  }
  async function install(name, { focus = 'alternate', restoreOriginal = false } = {}) {
    assert(name === 'shuka' || Object.prototype.hasOwnProperty.call(roster.selected_cases, name));
    assert(['alternate', 'registered', 'saved'].includes(focus));
    assert(!restoreOriginal || name === 'shuka');
    assert(store.getState().copilot.enabled, 'The ordinary app should already enable copilot.');
    const source = name === 'shuka' ? roster.shuka : roster.selected_cases[name];
    if (!loaded.has(name)) {
      const archive = source.archive;
      const data = bytes(archive);
      loaded.set(name, await documentApi.deserializeDocument(data));
    }
    const mediaDocument = loaded.get(name);
    const saved = read(name === 'shuka' ? source.snapshot : source.case);
    assert.deepEqual(mediaDocument.content, saved.document.content);
    const expected = { ...saved.document, sources: mediaDocument.sources };
    const runtime = name === 'shuka' ? { history: saved.copilot.history, sequence: saved.copilot.sequence } : saved.runtime;
    let editor;
    if (name === 'shuka') {
      editor = read(source.editor_history);
      for (const state of [...editor.past, editor.present, ...editor.future, editor._latestUnfiltered]) {
        if (!state) continue;
        state.document.sources = mediaDocument.sources;
        if (state.lastSavedDocument) state.lastSavedDocument.sources = mediaDocument.sources;
        if (!restoreOriginal) state.path = null;
      }
      assert.deepEqual(sessionDocument(editor.present.document), sessionDocument(expected));
      editor._latestUnfiltered = editor.present;
    } else {
      const present = { ...defaultEditorState, document: { ...mediaDocument, content: clone(saved.document.content) }, path: null,
        lastSavedDocument: null, cursor: { playerTime: 0, userIndex: 0, current: 'user' }, selection: null };
      editor = { ...slices.editor.default(undefined, { type: '@@INIT' }), past: [], present, future: [],
        group: null, _latestUnfiltered: present, index: 0, limit: 1 };
    }
    if (name !== 'shuka') {
      const uuid = focus === 'alternate' ? source.alternate_focus_uuid : source.focus_uuid;
      const index = editor.present.document.content.findIndex(item => item.uuid === uuid);
      assert(index >= 0);
      editor.present.cursor = { playerTime: 0, userIndex: index, current: 'user' };
    } else if (!restoreOriginal) {
      const index = editor.present.document.content.findIndex(item => item.uuid === source.operations.split.target_uuids[0]);
      assert(index >= 0);
      editor.present.cursor = { playerTime: 0, userIndex: index, current: 'user' };
      editor.present.selection = null;
    }
    editor.present.playing = false;
    editor.present.transcriptCorrectionState = null;
    editor.present.exportPopup = 'hidden';
    editor.present.filterPopup = false;
    const current = store.getState().copilot;
    const copilot = { ...slices.copilot.default(undefined, { type: '@@INIT' }),
      health: current.health, healthStatus: current.healthStatus, history: clone(runtime.history), sequence: runtime.sequence,
      documentId: window.require('uuid').v4(), feedbackOrigin: 'ai_computer_use' };
    const value = await replace({ editor, copilot, nav: slices.nav.default(store.getState().nav, slices.nav.openEditor()) }, name);
    assert.deepEqual(sessionDocument(store.getState().editor.present.document), sessionDocument(expected));
    assert.deepEqual(store.getState().copilot.history, runtime.history);
    return { ...value, seed: source,
      focus_for_measured_action: restoreOriginal ? null : name === 'shuka' ? source.operations.split.target_uuids[0] : source.focus_uuid };
  }
  return {
    install,
    capture: () => ({ registration, original: originalFacts, setup: clone(audit), current: facts(store.getState()),
      measurements: clone(store.getState().copilot.measurements), feedback: clone(store.getState().copilot.events),
      status: store.getState().copilot.status, error: store.getState().copilot.error }),
    restore: async () => {
      await replace({ editor: original.editor, copilot: original.copilot, nav: original.nav }, 'restore-original-in-memory');
      assert.deepEqual(facts(store.getState()), originalFacts);
      return { restored: true, facts: originalFacts };
    },
    restoreSavedShuka: () => install('shuka', { focus: 'saved', restoreOriginal: true }),
  };
}

module.exports = { create };
