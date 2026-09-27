import * as React from 'react';
import { useSelector } from 'react-redux';
import { Pane, Text } from 'evergreen-ui';
import { RootState } from '../../state';

export function TranscriptionNotes(): JSX.Element | null {
  const document = useSelector((state: RootState) => state.editor.present?.document);
  if (!document) return null;
  const annotations = (document.metadata.transcriptions || []).flatMap((record) =>
    (record.annotations || [])
      .filter((annotation) =>
        document.content.some(
          (item) =>
            'source' in item &&
            item.source === record.source &&
            item.sourceStart <= annotation.end &&
            annotation.start <= item.sourceStart + item.length
        )
      )
      .map((annotation) => ({
        ...annotation,
        key: `${record.id}-${annotation.word_indices.join('-')}`,
      }))
  );
  if (!annotations.length) return null;
  const unresolved = annotations.filter(
    (annotation) => annotation.type === 'unaligned_word'
  ).length;
  return (
    <Pane padding={16} maxWidth={950} aria-label="Transcription notes">
      <details>
        <summary>
          <Text fontWeight={600}>
            Transcription notes · {unresolved} words need timing review ·{' '}
            {annotations.length - unresolved} audio events
          </Text>
        </summary>
        <Text display="block" marginY={8}>
          Times refer to the original recording. Untimed words stay in these notes; text and
          subtitle exports include timed items only.
        </Text>
        {annotations.map((annotation) => (
          <Text display="block" key={annotation.key}>
            {annotation.start.toFixed(3)}s · {annotation.speaker_id && `${annotation.speaker_id}: `}
            {annotation.text} ·{' '}
            {annotation.type === 'unaligned_word'
              ? 'Timing unresolved; align before cutting.'
              : 'Audio event'}
          </Text>
        ))}
      </details>
    </Pane>
  );
}
