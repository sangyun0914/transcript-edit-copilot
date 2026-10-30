import { combineReducers } from '@reduxjs/toolkit';
import nav from '../src/state/nav';
import transcribe from '../src/state/transcribe';
import editor from '../src/state/editor';
import models from '../src/state/models';
import server from '../src/state/server';
import recording from '../src/state/recording';
import copilot from '../src/state/copilot';
import type { CommittedTransition } from '../src/core/copilot';
import type { V3DocumentItem } from '../src/core/document';
import { memoizedDocumentRenderItems } from '../src/state/editor/selectors';

const production = combineReducers({ nav, transcribe, editor, models, server, recording, copilot });

export function hydrateCopilotFixture(fixture: {
  document_id: string;
  renderer_sequence_before_request: number;
  runtime: { history: CommittedTransition[] };
}): void {
  if (process.env.NODE_ENV !== 'test') throw new Error('Fixture hydration requires test mode.');
  const store = window.store;
  if (!store.getState().copilot.enabled || store.getState().copilot.request)
    throw new Error('Fixture hydration requires an enabled idle session.');
  // Install one fixture state, then immediately restore the same production reducers.
  // No completion or feedback action is synthesized by this initialization.
  store.replaceReducer((state, action) => {
    const next = production(state, action);
    if (action.type !== 'copilot/p6HydrateFixture') return next;
    return {
      ...next,
      copilot: {
        ...next.copilot,
        documentId: fixture.document_id,
        history: fixture.runtime.history,
        sequence: fixture.renderer_sequence_before_request,
        feedbackOrigin: 'synthetic' as const,
      },
    };
  });
  try {
    store.dispatch({ type: 'copilot/p6HydrateFixture' });
  } finally {
    store.replaceReducer(production);
  }
}

export function validateCopilotFixtureDocument(content: V3DocumentItem[]): void {
  memoizedDocumentRenderItems(content, false);
}
