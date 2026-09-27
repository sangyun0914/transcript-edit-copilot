import React, { useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Button, Pane, Text } from 'evergreen-ui';
import path from 'path';
import { RootState } from '../../state';
import { startRecording, stopRecording } from '../../state/recording';
import { getHomePath, saveFile } from '../../../ipc/ipc_renderer';
import { SessionMode } from '../../core/editing_session';

export function SessionRecorder(): JSX.Element {
  const dispatch = useDispatch();
  const recording = useSelector((state: RootState) => state.recording);
  const projectPath = useSelector((state: RootState) => state.editor.present?.path);
  const [mode, setMode] = useState<SessionMode>('unguided');
  const [brief, setBrief] = useState('Preserve meaning and conversational style.');
  const active = recording.status === 'recording';
  const pending = recording.status === 'starting' || recording.status === 'stopping';
  async function start() {
    const directory = projectPath ? path.dirname(projectPath) : await getHomePath();
    const result = await saveFile({
      title: 'Save editing session log',
      defaultPath: path.join(
        directory,
        `editing-session-${new Date().toISOString().replace(/[:.]/g, '-')}.jsonl`
      ),
      filters: [{ name: 'Editing session', extensions: ['jsonl'] }],
      properties: ['createDirectory'],
    });
    if (result.filePath) dispatch(startRecording({ path: result.filePath, mode, brief }));
  }
  return (
    <Pane
      paddingX={16}
      paddingY={8}
      borderBottom="muted"
      onKeyDown={(event: React.KeyboardEvent) => event.stopPropagation()}
      aria-label="Editing session recording"
    >
      <details>
        <summary>
          <Text fontWeight={600}>
            Editing session ·{' '}
            {active
              ? `${recording.edits} edits recorded`
              : pending
              ? recording.status
              : 'local research capture'}
          </Text>
        </summary>
        <Pane display="flex" gap={12} alignItems="center" marginY={10}>
          <label>
            Session type{' '}
            <select
              aria-label="Session type"
              value={mode}
              disabled={active || pending}
              onChange={(event) => setMode(event.target.value as typeof mode)}
            >
              <option value="unguided">My own editing</option>
              <option value="reference_guided">Recreating a reference edit</option>
              <option value="ai_computer_use">AI operating the editor</option>
              <option value="synthetic">Scripted synthetic session</option>
            </select>
          </label>
          <label>
            Editing brief{' '}
            <input
              aria-label="Editing brief"
              value={brief}
              maxLength={2000}
              disabled={active || pending}
              onChange={(event) => setBrief(event.target.value)}
            />
          </label>
          {active || (recording.status === 'failed' && recording.path) ? (
            <Button disabled={pending} onClick={() => dispatch(stopRecording())}>
              Stop and save session
            </Button>
          ) : (
            <Button disabled={pending} onClick={start}>
              Start recording session
            </Button>
          )}
        </Pane>
        <Text display="block" color="muted" size={300}>
          Records transcript changes, caption boundaries, selections and undo locally. Source media
          stays in the project. A reference-guided session is labeled separately.
        </Text>
        {recording.path && (
          <Text display="block" size={300}>
            Log: {recording.path}
            {recording.status === 'idle' ? ' · saved' : ''}
          </Text>
        )}
        {recording.error && (
          <Text display="block" color="danger" role="alert">
            Recording failed: {recording.error}
          </Text>
        )}
      </details>
    </Pane>
  );
}
