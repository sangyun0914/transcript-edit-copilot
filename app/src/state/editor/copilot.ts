import { createActionWithReducer } from '../util';
import { EditorState } from './types';
import { CopilotResponse, validCopilotResult } from '../../core/copilot';
import { sessionContext, sessionDocument, sessionHash } from '../../core/editing_session';

export const acceptCopilotTransaction = createActionWithReducer<
  EditorState,
  { proposal: CopilotResponse; context_hash: string }
>('editor/acceptCopilot', (state, { proposal, context_hash }) => {
  if (
    state.playing ||
    state.transcriptCorrectionState !== null ||
    state.exportPopup !== 'hidden' ||
    state.filterPopup ||
    sessionHash(sessionContext(state)) !== context_hash ||
    !validCopilotResult(sessionDocument(state.document), proposal) ||
    !proposal.result_document
  )
    return;
  state.document.content = proposal.result_document.content;
  const target = state.document.content.findIndex((item) => item.uuid === proposal.target_uuids[0]);
  state.cursor = { ...state.cursor, current: 'user', userIndex: Math.max(0, target) };
  state.selection = null;
});
