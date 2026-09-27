import React, { useEffect, useRef, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import styled from 'styled-components';
import { TimedItemExtension } from '../../core/document';
import { GapItem, gapSeconds } from '../../core/gaps';
import { RootState } from '../../state';
import { trimSelectedGap } from '../../state/editor/edit';
import { setSelection } from '../../state/editor/selection';
import { setPlay } from '../../state/editor/play';

const DurationButton = styled.button`
  padding: 0;
  border: 0;
  background: none;
  color: inherit;
  font: inherit;
  cursor: text;
  font-variant-numeric: tabular-nums;
`;

const DurationInput = styled.input`
  width: 5em;
  padding: 0 2px;
  border: 1px solid currentColor;
  border-radius: 3px;
  background: transparent;
  color: inherit;
  font: inherit;
  font-variant-numeric: tabular-nums;
`;

export function Gap({
  items,
  warning,
}: {
  items: (GapItem & TimedItemExtension)[];
  warning?: string;
}): JSX.Element {
  return (
    <span
      data-gap={items[0].uuid}
      title={warning}
      style={{
        whiteSpace: 'nowrap',
        ...(warning && { textDecoration: 'underline dotted', textUnderlineOffset: '3px' }),
      }}
    >
      {items.map((item, index) => (
        <span key={item.uuid} id={`item-${item.uuid}`}>
          {index === 0 && (
            <>
              {' ['}
              <span aria-label="Pause" style={{ fontFamily: 'quarter_rest' }}>
                _
              </span>
            </>
          )}
          {index > 0 && index < items.length - 1 && '\u200b'}
          {index === items.length - 1 && (
            <>
              {' '}
              <GapDuration items={items} />
              {']'}
            </>
          )}
        </span>
      ))}
    </span>
  );
}

function GapDuration({ items }: { items: (GapItem & TimedItemExtension)[] }): JSX.Element {
  const dispatch = useDispatch();
  const content = useSelector((state: RootState) => state.editor.present?.document.content);
  const playing = useSelector((state: RootState) => state.editor.present?.playing);
  const selection = useSelector((state: RootState) => state.editor.present?.selection);
  const length = items.reduce((total, item) => total + item.length, 0);
  const seconds = gapSeconds(length);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(seconds);
  const initialContent = useRef(content);
  const input = useRef<HTMLInputElement>(null);
  const finished = useRef(false);
  const keepMs = Math.round(Number(draft) * 1000);
  const valid =
    /^(?:\d+(?:\.\d*)?|\.\d+)$/.test(draft) &&
    Number.isSafeInteger(keepMs) &&
    Math.abs(Number(draft) * 1000 - keepMs) < 1e-7 &&
    keepMs > 0 &&
    keepMs / 1000 < length;

  useEffect(() => {
    if (editing) input.current?.select();
  }, [editing]);

  useEffect(() => {
    if (
      editing &&
      (playing ||
        content !== initialContent.current ||
        selection?.startIndex !== items[0].absoluteIndex ||
        selection?.length !== items.length)
    ) {
      finished.current = true;
      setEditing(false);
    }
  }, [content, playing, selection, editing, items]);

  function finish(accept: boolean, focusDocument: boolean) {
    if (finished.current) return;
    if (accept && draft !== seconds && !valid && focusDocument) return;
    finished.current = true;
    if (accept && draft !== seconds && valid && content === initialContent.current) {
      dispatch(trimSelectedGap({ uuid: items[0].uuid, keepMs, expectedLength: length }));
    }
    setEditing(false);
    if (focusDocument) document.getElementById('document')?.focus();
  }

  return (
    <span
      onMouseDown={(event) => event.stopPropagation()}
      onMouseMove={(event) => event.stopPropagation()}
      onContextMenu={(event) => event.stopPropagation()}
      onKeyDown={(event) => event.stopPropagation()}
    >
      {editing ? (
        <>
          <DurationInput
            ref={input}
            autoFocus
            type="text"
            inputMode="decimal"
            aria-label="Gap duration in seconds"
            aria-invalid={draft !== seconds && !valid}
            title={`Enter a positive duration shorter than ${seconds} seconds (up to 3 decimal places).`}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onBlur={() => finish(true, false)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === 'Escape') {
                event.preventDefault();
                finish(event.key === 'Enter', true);
              }
            }}
          />
          {'s'}
        </>
      ) : (
        <DurationButton
          type="button"
          aria-label={`Edit gap duration, ${seconds} seconds`}
          title="Click to shorten this gap in seconds. Enter applies; Escape cancels."
          onClick={() => {
            dispatch(setPlay(false));
            dispatch(
              setSelection({
                startIndex: items[0].absoluteIndex,
                length: items.length,
                headPosition: 'right',
              })
            );
            initialContent.current = content;
            finished.current = false;
            setDraft(seconds);
            setEditing(true);
          }}
        >
          {seconds}s
        </DurationButton>
      )}
    </span>
  );
}
