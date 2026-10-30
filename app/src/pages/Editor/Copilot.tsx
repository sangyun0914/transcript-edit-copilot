import * as React from 'react';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useDispatch, useSelector, useStore } from 'react-redux';
import { Button, getPosition, Pane, StackingOrder, Text } from 'evergreen-ui';
import styled from 'styled-components';
import type { RootState } from '../../state';
import { useTheme } from '../../components/theme';
import { copilotPreviewText } from '../../core/copilot';
import { gapSeconds } from '../../core/gaps';
import {
  acceptCopilot,
  currentCopilotProposal,
  dismissCopilot,
  expireCopilot,
  markCopilotShown,
  localCopilotServer,
  refreshCopilotHealth,
  retryCopilotFeedback,
  setCopilotBlocked,
  copilotPreviewVisible,
  invalidateCopilot,
} from '../../state/copilot';

export function CopilotControls(): JSX.Element {
  const dispatch = useDispatch(),
    store = useStore<RootState>();
  const state = useSelector((state: RootState) => state.copilot);
  const server = useSelector(localCopilotServer);
  useEffect(() => {
    if (!state.enabled) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    const refresh = async () => {
      await dispatch(refreshCopilotHealth());
      if (!disposed)
        timer = setTimeout(
          refresh,
          store.getState().copilot.healthStatus === 'ready' ? 10000 : 2000
        );
    };
    void refresh();
    return () => {
      disposed = true;
      clearTimeout(timer);
    };
  }, [
    state.enabled,
    server?.hostname,
    server?.port,
    server?.token,
    server?.state,
    dispatch,
    store,
  ]);
  useEffect(() => {
    let composing = false;
    const refresh = () => {
      const focus = document.activeElement;
      const blocked =
        composing ||
        document.hidden ||
        Boolean(
          document.querySelector('[role="dialog"], [aria-modal="true"]') ||
            focus?.closest(
              'input:not([type="checkbox"]), textarea, select, [contenteditable="true"]'
            )
        );
      if (store.getState().copilot.blocked !== blocked) dispatch(setCopilotBlocked(blocked));
    };
    const compositionStart = () => {
      composing = true;
      refresh();
    };
    const compositionEnd = () => {
      composing = false;
      refresh();
    };
    const focusOut = () => {
      Promise.resolve().then(refresh);
    };
    document.addEventListener('compositionstart', compositionStart, true);
    document.addEventListener('compositionend', compositionEnd, true);
    document.addEventListener('focusin', refresh, true);
    document.addEventListener('focusout', focusOut, true);
    document.addEventListener('visibilitychange', refresh);
    const observer = new MutationObserver(refresh);
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['role', 'aria-modal', 'contenteditable'],
    });
    refresh();
    return () => {
      observer.disconnect();
      document.removeEventListener('compositionstart', compositionStart, true);
      document.removeEventListener('compositionend', compositionEnd, true);
      document.removeEventListener('focusin', refresh, true);
      document.removeEventListener('focusout', focusOut, true);
      document.removeEventListener('visibilitychange', refresh);
      dispatch(setCopilotBlocked(true));
    };
  }, [dispatch, store]);

  const status = state.feedbackError
    ? 'Suggestions paused because feedback could not be saved. Your edits are retained.'
    : state.error
    ? 'Could not suggest an edit here. Suggestions will resume when you move or edit.'
    : state.status === 'requesting'
    ? 'Finding the next edit…'
    : state.status === 'preview'
    ? 'Suggestion ready. Tab to accept, Esc to dismiss.'
    : state.healthStatus === null || state.healthStatus === 'starting'
    ? 'Preparing local suggestions…'
    : state.healthStatus === 'busy'
    ? 'Waiting for the local model…'
    : !['ready', 'busy'].includes(state.healthStatus)
    ? 'Local suggestions are unavailable. You can continue editing.'
    : state.status === 'no_action' || state.status === 'suppressed'
    ? 'No edit suggested here.'
    : 'Suggestions appear automatically as you edit.';
  return (
    <Pane display="flex" alignItems="center" flexWrap="wrap" gap={12} paddingX={24} paddingY={12}>
      <Text
        id="copilot-status"
        role="status"
        size={300}
        title={state.error || state.feedbackError || state.health || undefined}
      >
        {status}
      </Text>
      {state.feedbackError && (
        <Button size="small" onClick={() => dispatch(retryCopilotFeedback())}>
          Retry saving feedback
        </Button>
      )}
    </Pane>
  );
}

const RemovedText = styled.del`
  border-radius: 3px;
  text-decoration: line-through;
  text-decoration-thickness: 1.5px;
  box-decoration-break: clone;

  button {
    text-decoration: line-through;
  }
`;

export function CopilotOriginal({ children }: { children: React.ReactNode }): JSX.Element {
  const theme = useTheme();
  return (
    <RemovedText
      data-copilot-removed
      style={{ color: theme.colors.diffRemoved, background: theme.colors.diffRemovedBackground }}
      title="This will be removed or replaced if you accept the suggestion."
    >
      {children}
    </RemovedText>
  );
}

function Shortcut({ children }: { children: string }): JSX.Element {
  return (
    <kbd
      aria-hidden="true"
      style={{
        marginLeft: 6,
        padding: '0 3px',
        border: '1px solid currentColor',
        borderRadius: 3,
        fontSize: 10,
        opacity: 0.7,
        lineHeight: '16px',
      }}
    >
      {children}
    </kbd>
  );
}

export function CopilotPreview(): JSX.Element | null {
  const dispatch = useDispatch();
  const theme = useTheme();
  const ref = useRef<HTMLSpanElement>(null);
  const actionsRef = useRef<HTMLDivElement>(null);
  const closeTimer = useRef<number>();
  const [showActions, setShowActions] = useState(false);
  const [position, setPosition] = useState({ left: 0, top: 0 });
  const proposal = useSelector(currentCopilotProposal);
  const response = proposal?.response;
  const revealActions = () => {
    window.clearTimeout(closeTimer.current);
    setShowActions(true);
  };
  const hideActions = () => {
    window.clearTimeout(closeTimer.current);
    closeTimer.current = window.setTimeout(() => {
      const focused = document.activeElement;
      if (
        !ref.current?.contains(focused) &&
        !actionsRef.current?.contains(focused) &&
        !ref.current?.matches(':hover') &&
        !actionsRef.current?.matches(':hover')
      )
        setShowActions(false);
    }, 150);
  };
  useEffect(() => () => window.clearTimeout(closeTimer.current), []);
  useLayoutEffect(() => {
    if (!showActions) return;
    const updatePosition = () => {
      if (!ref.current || !actionsRef.current) return;
      const { rect } = getPosition({
        position: 'top-left',
        dimensions: actionsRef.current.getBoundingClientRect(),
        targetRect: ref.current.getBoundingClientRect(),
        viewport: { width: window.innerWidth, height: window.innerHeight },
        targetOffset: 4,
        viewportOffset: 8,
      });
      setPosition({ left: rect.left, top: rect.top });
    };
    updatePosition();
    const observer = new ResizeObserver(updatePosition);
    if (ref.current) observer.observe(ref.current);
    if (actionsRef.current) observer.observe(actionsRef.current);
    window.addEventListener('resize', updatePosition);
    document.addEventListener('scroll', updatePosition, true);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', updatePosition);
      document.removeEventListener('scroll', updatePosition, true);
    };
  }, [showActions]);
  useEffect(() => {
    if (!response?.proposal_id || !proposal) return;
    let exposed = false;
    let firstFrame = 0,
      secondFrame = 0;
    const updateExposure = () => {
      if (copilotPreviewVisible()) {
        if (!exposed && !firstFrame && !secondFrame) {
          // A visible DOM node is not yet a painted preview. Allow a frame before measuring it.
          firstFrame = requestAnimationFrame(() => {
            firstFrame = 0;
            secondFrame = requestAnimationFrame(() => {
              secondFrame = 0;
              if (copilotPreviewVisible()) {
                exposed = true;
                dispatch(markCopilotShown(response.proposal_id as string));
              }
            });
          });
        }
      } else if (exposed) dispatch(invalidateCopilot(response.proposal_id as string));
    };
    ref.current?.scrollIntoView({ block: 'nearest' });
    const observer = new IntersectionObserver(updateExposure, {
      root: document.getElementById('scroll-container'),
    });
    if (ref.current) observer.observe(ref.current);
    updateExposure();
    const timer = setTimeout(
      () => dispatch(expireCopilot(response.proposal_id as string)),
      Math.max(0, proposal.expiresAt - Date.now())
    );
    return () => {
      cancelAnimationFrame(firstFrame);
      cancelAnimationFrame(secondFrame);
      clearTimeout(timer);
      observer.disconnect();
    };
  }, [response?.proposal_id, proposal?.expiresAt, dispatch]);
  if (!response) return null;
  const prediction = response.prediction;
  const text =
    prediction.action === 'correct_text'
      ? `Replace the marked transcript text with “${prediction.text}”. Audio stays unchanged.`
      : copilotPreviewText(prediction);
  const addition =
    prediction.action === 'correct_text'
      ? prediction.text
      : prediction.action === 'trim_silence'
      ? `${gapSeconds((prediction.keep_ms as number) / 1000)}s pause`
      : prediction.action === 'split_paragraph'
      ? '¶ New caption'
      : null;
  const dismiss = () => {
    dispatch(dismissCopilot());
    document.getElementById('document')?.focus();
  };
  return (
    <span
      ref={ref}
      data-testid="copilot-preview"
      role="group"
      tabIndex={0}
      aria-label={`Suggested edit: ${text}`}
      aria-controls={showActions ? 'copilot-suggestion-actions' : undefined}
      onMouseEnter={revealActions}
      onMouseLeave={hideActions}
      onFocus={revealActions}
      onBlur={hideActions}
      style={{
        display: 'inline',
        color: theme.colors.default,
        marginLeft: 4,
        overflowWrap: 'anywhere',
      }}
      onMouseDown={(event) => {
        event.preventDefault();
        event.stopPropagation();
      }}
      onMouseMove={(event) => event.stopPropagation()}
      onClick={(event) => {
        event.stopPropagation();
        if (event.target instanceof Node && ref.current?.contains(event.target)) revealActions();
      }}
      onDoubleClick={(event) => event.stopPropagation()}
      onContextMenu={(event) => event.stopPropagation()}
      onKeyDown={(event) => {
        event.stopPropagation();
        if (
          event.nativeEvent.isComposing ||
          event.repeat ||
          event.altKey ||
          event.ctrlKey ||
          event.metaKey
        )
          return;
        if (event.key === 'Escape') {
          event.preventDefault();
          dismiss();
        } else if (event.key === 'ArrowDown' && event.target === ref.current) {
          event.preventDefault();
          actionsRef.current?.querySelector('button')?.focus();
        }
      }}
    >
      <span role="status" aria-label={text} style={{ minWidth: 0, overflowWrap: 'anywhere' }}>
        {addition ? (
          <ins
            data-copilot-added
            title={
              prediction.action === 'correct_text'
                ? 'Replace the highlighted transcript text. Audio stays unchanged.'
                : text
            }
            style={{
              color: theme.colors.diffAdded,
              background: theme.colors.diffAddedBackground,
              borderRadius: 3,
              padding: '0 2px',
              whiteSpace: 'pre-wrap',
              textDecoration: 'underline',
              textUnderlineOffset: '3px',
            }}
          >
            {addition}
          </ins>
        ) : prediction.action === 'merge_paragraph' ? (
          <>
            <CopilotOriginal>¶</CopilotOriginal>{' '}
            <span style={{ fontSize: 13 }}>Join with previous caption</span>
          </>
        ) : (
          <span style={{ color: theme.colors.diffRemoved, fontSize: 13 }}>
            {prediction.action === 'delete_span' ? 'Delete text + audio' : 'Remove pause'}
          </span>
        )}
      </span>
      {showActions &&
        createPortal(
          <div
            ref={actionsRef}
            id="copilot-suggestion-actions"
            role="group"
            aria-label="Suggestion actions"
            onMouseEnter={revealActions}
            onMouseLeave={hideActions}
            style={{
              position: 'fixed',
              ...position,
              zIndex: StackingOrder.POSITIONER,
              display: 'flex',
              gap: 2,
              padding: 2,
              border: `1px solid ${theme.colors.border.default}`,
              borderRadius: 5,
              background: theme.colors.overlayBackgroundColor,
              boxShadow: theme.shadows[1],
            }}
          >
            <Button
              type="button"
              size="small"
              appearance="minimal"
              aria-label="Accept suggestion"
              title="Accept this entire suggestion (Tab). Undo restores the original."
              onClick={() => {
                if (copilotPreviewVisible()) dispatch(acceptCopilot());
                document.getElementById('document')?.focus();
              }}
            >
              Accept <Shortcut>Tab</Shortcut>
            </Button>
            <Button
              type="button"
              size="small"
              appearance="minimal"
              aria-label="Dismiss suggestion"
              title="Dismiss this suggestion (Esc)."
              onClick={dismiss}
            >
              Dismiss <Shortcut>Esc</Shortcut>
            </Button>
          </div>,
          document.body
        )}
    </span>
  );
}
