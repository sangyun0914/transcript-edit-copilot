import * as React from 'react';
import { AppContainer, MainCenterColumn } from '../../components/Util';
import styled from 'styled-components';
import { EditorTitleBar } from './TitleBar';
import { Document } from './Document';
import { Player } from './Player';
import { KeyboardEventHandler } from 'react';
import { useDispatch, useStore } from 'react-redux';
import { ExportDocumentDialog } from './ExportDocumentDialog';
import { EditorTour } from '../../tour/EditorTour';
import { togglePlaying } from '../../state/editor/play';
import { insertParagraphEnd } from '../../state/editor/edit';
import { EditorMenuBar } from './MenuBar';
import { FilterDialog } from './Filter';
import { TranscriptionNotes } from './TranscriptionNotes';
import { SessionRecorder } from './SessionRecorder';
import { CopilotControls } from './Copilot';
import { acceptCopilot, copilotKey, dismissCopilot } from '../../state/copilot';
import type { RootState } from '../../state';

const MainContainer = styled(MainCenterColumn)`
  justify-content: start;
  overflow-y: auto !important;

  &:focus {
    outline: none;
  }
`;
export function EditorPage(): JSX.Element {
  const dispatch = useDispatch();
  const store = useStore<RootState>();
  const handleCopilotKey: KeyboardEventHandler = (event) => {
    const decision = copilotKey(
      { ...event, isComposing: event.nativeEvent.isComposing, keyCode: event.keyCode },
      store.getState()
    );
    if (!decision) return;
    event.preventDefault();
    event.stopPropagation();
    dispatch(decision === 'accept' ? acceptCopilot() : dismissCopilot());
  };
  const handleKeyPress: KeyboardEventHandler = (e) => {
    if (e.key === ' ') {
      dispatch(togglePlaying());
      e.preventDefault();
    } else if (e.key === 'Enter') {
      dispatch(insertParagraphEnd());
    }
  };

  return (
    <AppContainer
      tabIndex={-1}
      onKeyDown={handleKeyPress}
      onKeyDownCapture={handleCopilotKey}
      ref={(ref) => ref?.focus()}
    >
      <EditorMenuBar />
      <EditorTour />
      <Player />

      <EditorTitleBar />
      <SessionRecorder />
      <CopilotControls />
      <ExportDocumentDialog />
      <FilterDialog />

      <MainContainer id={'scroll-container'}>
        <TranscriptionNotes />
        <Document />
      </MainContainer>
    </AppContainer>
  );
}
