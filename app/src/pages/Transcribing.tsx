import * as React from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { TitleBar } from '../components/TitleBar';
import { AppContainer, MainMaxWidthContainer } from '../components/Util';
import { RootState } from '../state';
import { Line } from 'rc-progress';
import * as path from 'path';
import { Button, FormField, majorScale, Pane, Spinner, Text } from 'evergreen-ui';
import { openTranscribe } from '../state/nav';
import { TranscriptionState } from '../server_api/api';

export function TranscribingPage(): JSX.Element {
  const dispatch = useDispatch();
  const error = useSelector((state: RootState) => state.transcribe.error);
  const file = useSelector((state: RootState) => state.transcribe.file) || '';
  const progress = useSelector((state: RootState) => state.transcribe.progress) || 0;
  const server_state = useSelector((state: RootState) => state.transcribe.state) || '';
  const indeterminate =
    server_state === TranscriptionState.UPLOADING ||
    (progress === 0 && server_state !== TranscriptionState.CONVERTING);

  return (
    <AppContainer>
      <TitleBar />
      <MainMaxWidthContainer width={500} centerVertically>
        <FormField label="Transcribing File" marginBottom={majorScale(3)}>
          <Text color="muted">{path.basename(file)}</Text>
        </FormField>

        {error ? (
          <Pane>
            <Text display="block" marginBottom={majorScale(2)}>
              {error}
            </Text>
            <Button onClick={() => dispatch(openTranscribe())}>
              Back to transcription options
            </Button>
          </Pane>
        ) : (
          <Pane textAlign="center">
            {indeterminate ? (
              <Spinner marginX="auto" marginBottom={majorScale(2)} />
            ) : (
              <Line percent={progress * 100} style={{ width: '100%' }} />
            )}
            <Text color="muted">
              {!indeterminate && `${(progress * 100).toFixed(0)} % - `}
              {server_state}
            </Text>
          </Pane>
        )}
      </MainMaxWidthContainer>
    </AppContainer>
  );
}
