/** @jest-environment ./scripts/copilot_ui_environment.cjs */

import crypto from 'crypto';
import fs from 'fs';
import path from 'path';

declare const copilotUIOutput: string;

jest.setTimeout(240000);
const enabled = process.env.COPILOT_UI_MODE === 'app-startup-actual';
const protocolPath = path.resolve(
  '../artifacts/ui/passive-actions-v3/automatic-startup-protocol-001.json'
);
const hash = (value: Buffer | string) => crypto.createHash('sha256').update(value).digest('hex');
const write = (name: string, value: unknown) =>
  fs.writeFileSync(path.join(copilotUIOutput, name), JSON.stringify(value, null, 2) + '\n');

(enabled ? test : test.skip)(
  'ordinary app starts the selected model and suggests edits automatically',
  async () => {
    const protocol = JSON.parse(fs.readFileSync(protocolPath, 'utf8'));
    const selectionPath = path.resolve(
      '../artifacts/local-runtime/gemma-actions-v3/default-deployment.json'
    );
    expect(hash(fs.readFileSync(selectionPath))).toBe(protocol.selection_sha256);
    await page.waitForFunction(() => !!window.store);
    expect(await page.evaluate(() => window.store.getState().copilot.enabled)).toBe(true);
    await page.waitForFunction(
      () =>
        window.store
          .getState()
          .server.servers.some(
            (server) =>
              server.name === 'local server' && server.state === 'running' && !!server.token
          ),
      { timeout: 60000 }
    );
    await page.evaluate(() => {
      localStorage.setItem('intro/editor', 'true');
      window.store.dispatch(window.reducers.setCopilotFeedbackOrigin('ai_computer_use'));
    });
    const results: unknown[] = [];
    try {
      for (const entry of protocol.cases) {
        const raw = fs.readFileSync(path.resolve('..', entry.seed_path));
        expect(hash(raw)).toBe(entry.seed_sha256);
        const { fixture, media } = JSON.parse(raw.toString()).body;
        expect(fixture.runtime.history).toHaveLength(0);
        await page.bringToFront();
        await page.evaluate(
          ({ fixture, media }) => {
            const fs = (window as any).require('fs');
            const crypto = (window as any).require('crypto');
            const bytes = fs.readFileSync(media.path);
            if (crypto.createHash('sha256').update(bytes).digest('hex') !== media.sha256)
              throw new Error('Source media differs from the registered fixture.');
            const buffer = new Uint8Array(bytes).buffer;
            const sources = Object.fromEntries(
              media.source_ids.map((id: string) => [
                id,
                {
                  fileContents: buffer,
                  objectUrl: URL.createObjectURL(new Blob([buffer])),
                },
              ])
            );
            window.store.dispatch(window.reducers.clearCopilotMeasurements());
            return window.store
              .dispatch(
                window.reducers.openDocumentFromMemory({
                  content: fixture.document.content,
                  metadata: Object.assign(
                    { display_video: false, display_speaker_names: true },
                    fixture.document.metadata
                  ),
                  sources,
                })
              )
              .then(() => {
                const focus = fixture.document.content.findIndex(
                  (item: any) => item.uuid === fixture.runtime.focus_uuid
                );
                window.store.dispatch(window.reducers.setUserIndex(focus));
                document.getElementById('document')?.focus();
              });
          },
          { fixture, media }
        );
        expect(await page.$('[aria-label="Enable local copilot"]')).toBeNull();
        expect(
          await page.$$eval('button', (buttons) =>
            buttons.some((button) => button.textContent === 'Predict next edit')
          )
        ).toBe(false);
        await page.waitForFunction(
          () => ['preview', 'no_action', 'error'].includes(window.store.getState().copilot.status),
          { timeout: 120000 }
        );
        const predicted = await page.evaluate(() => window.store.getState().copilot);
        write(`${entry.language}-predicted.json`, predicted);
        expect(predicted.status).toBe('preview');
        await page.waitForFunction(() => window.store.getState().copilot.proposal?.shown);
        await page.evaluate(() => new Promise((resolve) => setTimeout(resolve, 1250)));
        const visible = await page.evaluate(() => window.store.getState().copilot);
        expect(visible.proposal?.shown).toBe(true);
        expect(visible.measurements.filter((item) => item.event === 'request')).toHaveLength(1);
        expect(visible.measurements.filter((item) => item.event === 'request')[0].trigger).toBe(
          'automatic'
        );
        expect(visible.proposal?.response.model.adapter_sha256).toBe(protocol.adapter_sha256);
        await page.screenshot({
          path: path.join(copilotUIOutput, `${entry.language}-automatic-preview.png`),
          captureBeyondViewport: false,
        });
        await page.keyboard.press(entry.intervention === 'accept' ? 'Tab' : 'Escape');
        await page.waitForFunction(
          (outcome: string) =>
            window.store
              .getState()
              .copilot.events.some(
                (event) => event.outcome === outcome && event.persistence === 'saved'
              ),
          {},
          entry.intervention === 'accept' ? 'accepted' : 'dismissed'
        );
        const after = await page.evaluate(() => ({
          content: window.store.getState().editor.present?.document.content,
          copilot: window.store.getState().copilot,
        }));
        expect(after.content).toEqual(
          entry.intervention === 'accept'
            ? visible.proposal?.response.result_document?.content
            : fixture.document.content
        );
        const evidence = await page.evaluate((documentId) => {
          const server = window.store
            .getState()
            .server.servers.find((item) => item.name === 'local server');
          if (!server) throw new Error('No ordinary app server.');
          const headers = { Authorization: `Bearer ${server.token}` };
          const url = `${server.hostname}:${server.port}/copilot/`;
          return Promise.all([
            fetch(`${url}feedback/export?document_id=${encodeURIComponent(documentId)}`, {
              headers,
            }).then((response) =>
              response.text().then((journal) => ({ status: response.status, journal }))
            ),
            fetch(`${url}health`, { headers }).then((response) => response.json()),
          ]).then((responses) => Object.assign({}, responses[0], { health: responses[1] }));
        }, after.copilot.documentId);
        expect(evidence.status).toBe(200);
        expect(evidence.health.process_lifecycle).toBe('app_owned');
        expect(evidence.health.model.adapter_sha256).toBe(protocol.adapter_sha256);
        const rows = evidence.journal
          .trim()
          .split('\n')
          .map((line: string) => JSON.parse(line));
        expect(
          rows
            .filter((row: any) => row.kind === 'prediction_requested')
            .every((row: any) => row.input_request.trigger === 'automatic')
        ).toBe(true);
        fs.writeFileSync(
          path.join(copilotUIOutput, `${entry.language}-feedback.jsonl`),
          evidence.journal
        );
        write(`${entry.language}-health.json`, evidence.health);
        results.push({
          case_id: fixture.case_id,
          language: entry.language,
          automatic: true,
          intervention: entry.intervention,
          measurements: visible.measurements,
          result: 'passed',
        });
        await page.evaluate(() => window.store.dispatch(window.reducers.closeDocument()));
      }
    } finally {
      write('startup-results.json', {
        protocol_sha256: hash(fs.readFileSync(protocolPath)),
        model_quality_evaluated: false,
        results,
      });
      await page.evaluate(() => window.store.dispatch(window.reducers.closeDocument()));
    }
  }
);
