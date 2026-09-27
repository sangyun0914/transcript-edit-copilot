import * as React from 'react';
import { useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { startTranscription } from '../state/transcribe';
import { TitleBar } from '../components/TitleBar';
import { AppContainer } from '../components/Util';
import { RootState } from '../state';
import { openLanding, openModelManager } from '../state/nav';
import {
  Button,
  Combobox,
  Dialog,
  FormField,
  Link,
  majorScale,
  SelectField,
  Text,
  TextInputField,
} from 'evergreen-ui';
import * as path from 'path';
import { TranscribeTour } from '../tour/TranscribeTour';
import { useTheme } from '../components/theme';
import { getDefaultModel } from '../state/models';

export function TranscribePage(): JSX.Element {
  const dispatch = useDispatch();
  const file = useSelector((state: RootState) => state.transcribe.file) || '';
  const scribeConfigured = useSelector((state: RootState) => state.models.scribeConfigured);
  const localAsrAvailable = useSelector((state: RootState) => state.models.localAsrAvailable);
  const models = useSelector((state: RootState) => Object.values(state.models.downloaded));
  const languages = Array.from(new Set(models.map((model) => model.lang)));
  const [provider, setProvider] = useState<'scribe' | 'vosk'>(
    scribeConfigured || !localAsrAvailable || !models.length ? 'scribe' : 'vosk'
  );
  const [cloudLanguage, setCloudLanguage] = useState('ko');
  const [languageChoice, setLocalLanguage] = useState<string>();
  const localLanguage = languageChoice || languages[0];
  const [modelId, setModelId] = useState<string>();
  const languageModels = models.filter((model) => model.lang === localLanguage);
  const selectedModel =
    languageModels.find(
      (model) => model.model_id === (modelId || getDefaultModel(localLanguage, 'transcription'))
    ) || languageModels[0];
  const [diarizationMode, setDiarizationMode] = useState('on');
  const [speakers, setSpeakers] = useState('4');
  const [animationDone, setAnimationDone] = useState(false);
  const count = Number(speakers);
  const validSpeakers =
    diarizationMode !== 'advanced' || (Number.isInteger(count) && count >= 1 && count <= 32);
  const canStart =
    validSpeakers &&
    (provider === 'scribe' ? scribeConfigured : localAsrAvailable && !!selectedModel);
  const theme = useTheme();

  return (
    <AppContainer>
      <TranscribeTour animationDone={animationDone} />
      <TitleBar />
      <Dialog
        isShown={true}
        title="Transcription Options"
        topOffset={32}
        onCloseComplete={() => dispatch(openLanding())}
        onOpenComplete={() => setAnimationDone(true)}
        containerProps={{ backgroundColor: theme.colors.overlayBackgroundColor }}
        footer={({ close }) => (
          <>
            <Button onClick={() => close()}>Cancel</Button>
            <Button
              id="start"
              marginLeft={8}
              appearance="primary"
              disabled={!canStart}
              onClick={() =>
                dispatch(
                  startTranscription({
                    provider,
                    transcription_model: provider === 'vosk' ? selectedModel : undefined,
                    language_code: provider === 'scribe' ? cloudLanguage || undefined : undefined,
                    diarize: diarizationMode !== 'off',
                    diarize_max_speakers: diarizationMode === 'advanced' ? count : null,
                  })
                )
              }
            >
              Transcribe
            </Button>
          </>
        )}
      >
        <FormField label="Opened File" marginBottom={majorScale(3)}>
          <Text color="muted">{path.basename(file)}</Text>
        </FormField>
        <SelectField
          label="Transcription service"
          value={provider}
          onChange={(event: React.ChangeEvent<HTMLSelectElement>) =>
            setProvider(event.target.value as 'scribe' | 'vosk')
          }
        >
          <option value="scribe">ElevenLabs Scribe v2 (cloud)</option>
          <option value="vosk">Vosk (local)</option>
        </SelectField>
        {provider === 'vosk' && !localAsrAvailable && (
          <Text display="block" color="danger" marginBottom={majorScale(2)}>
            Local ASR is unavailable. In server/, run: uv sync --extra local-asr
          </Text>
        )}
        {provider === 'scribe' ? (
          <>
            <Text display="block" marginBottom={majorScale(2)}>
              Audio is sent to ElevenLabs for transcription.
            </Text>
            {!scribeConfigured && (
              <Text display="block" color="danger">
                ElevenLabs is not configured on this server.
              </Text>
            )}
            <SelectField
              id="model"
              label="Language"
              value={cloudLanguage}
              onChange={(event: React.ChangeEvent<HTMLSelectElement>) =>
                setCloudLanguage(event.target.value)
              }
            >
              <option value="ko">Korean</option>
              <option value="en">English</option>
              <option value="">Detect automatically</option>
            </SelectField>
          </>
        ) : (
          <FormField
            id="model"
            label="Language and model"
            description={
              <Link onClick={() => dispatch(openModelManager())}>Manage Languages & Models</Link>
            }
            marginBottom={majorScale(3)}
          >
            {models.length === 0 ? (
              <Text>Download a local transcription model first.</Text>
            ) : (
              <>
                <Combobox
                  width="100%"
                  selectedItem={localLanguage}
                  items={languages}
                  onChange={(language) => {
                    if (language) {
                      setLocalLanguage(language);
                      setModelId(undefined);
                    }
                  }}
                  marginBottom={majorScale(1)}
                />
                <Combobox
                  width="100%"
                  selectedItem={selectedModel}
                  items={languageModels}
                  itemToString={(model) => model?.name || ''}
                  onChange={(model) => {
                    if (model) setModelId(model.model_id);
                  }}
                />
              </>
            )}
          </FormField>
        )}
        <SelectField
          label="Speaker separation"
          value={diarizationMode}
          onChange={(event: React.ChangeEvent<HTMLSelectElement>) =>
            setDiarizationMode(event.target.value)
          }
        >
          <option value="off">Off</option>
          <option value="on">Detect speakers</option>
          <option value="advanced">Set a maximum</option>
        </SelectField>
        {diarizationMode === 'advanced' && (
          <TextInputField
            label="Maximum number of speakers"
            type="number"
            min={1}
            max={32}
            value={speakers}
            isInvalid={!validSpeakers}
            validationMessage={!validSpeakers ? 'Enter a whole number from 1 to 32.' : undefined}
            onChange={(event: React.ChangeEvent<HTMLInputElement>) =>
              setSpeakers(event.target.value)
            }
          />
        )}
      </Dialog>
    </AppContainer>
  );
}
