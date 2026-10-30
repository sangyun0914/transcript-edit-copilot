const NodeEnvironment = require('jest-environment-node');
const { spawn, execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const puppeteer = require('puppeteer-core');
const { getPortPromise } = require('portfinder');
const { createServer, build } = require('vite');
const {
  actualStartupMode,
  buildMode,
  isolatedModelEnvironment,
  readProcessTable,
  trackOwnedTree,
  stopOwnedElectron,
} = require('./copilot_ui_helpers.cjs');

const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

module.exports = class CopilotUIEnvironment extends NodeEnvironment {
  async setup() {
    await super.setup();
    this.started = new Date().toISOString();
    this.fixtureKind = process.env.COPILOT_UI_MODE || 'scripted_known_action_product_checks';
    this.actualAppStartup = actualStartupMode(process.env);
    this.mode = buildMode(process.env);
    this.previousEnvironment = Object.fromEntries(
      ['NODE_ENV', 'MODE', 'VITE_DEV_SERVER_URL'].map((key) => [key, process.env[key]])
    );
    this.output = path.resolve(
      process.env.COPILOT_UI_OUTPUT || `../artifacts/ui/passive-actions-v3/scripted-${Date.now()}`
    );
    fs.mkdirSync(path.dirname(this.output), { recursive: true });
    fs.mkdirSync(this.output, { recursive: false });
    this.temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'audapolis-copilot-ui-'));
    this.global.copilotUIOutput = this.output;
    this.global.copilotUITemp = this.temporary;
    this.log = fs.openSync(path.join(this.output, 'electron.log'), 'wx');
    this.initialLoad = os.loadavg();
    try {
      try {
        fs.writeFileSync(
          path.join(this.output, 'processes-start.txt'),
          execFileSync('ps', ['-eo', 'pid,pcpu,pmem,comm'])
        );
      } catch (error) {
        fs.writeFileSync(path.join(this.output, 'processes-start.txt'), `Unavailable: ${error}\n`);
      }
      process.env.NODE_ENV = this.mode;
      process.env.MODE = this.mode;
      this.vite = await createServer({
        mode: this.mode,
        configFile: 'src/vite_renderer.config.js',
        logLevel: 'warn',
      });
      await this.vite.listen(0, '127.0.0.1');
      process.env.VITE_DEV_SERVER_URL = `http://127.0.0.1:${this.vite.httpServer.address().port}/`;
      await build({
        mode: this.mode,
        configFile: 'main_process/vite_main.config.js',
        logLevel: 'warn',
      });
      const port = await getPortPromise({ host: '127.0.0.1', port: 12000 });
      const env = {
        ...isolatedModelEnvironment(process.env),
        COPILOT_UI_OUTPUT: this.output,
        COPILOT_UI_TEMP: this.temporary,
        AUDAPOLIS_DATA_DIR: path.join(this.temporary, 'data'),
        AUDAPOLIS_CACHE_DIR: path.join(this.temporary, 'cache'),
        UV_NO_SYNC: '1',
        UV_OFFLINE: '1',
        UV_CACHE_DIR: path.join(this.temporary, 'uv-cache'),
      };
      this.child = spawn(
        require('electron'),
        ['scripts/copilot_ui_electron.cjs', `--remote-debugging-port=${port}`],
        {
          env,
          detached: true,
          stdio: ['ignore', this.log, this.log],
        }
      );
      const deadline = Date.now() + 45000;
      while (Date.now() < deadline) {
        if (this.child.exitCode !== null)
          throw new Error(`Electron exited: ${this.child.exitCode}`);
        try {
          if (!this.browser)
            this.browser = await puppeteer.connect({ browserURL: `http://127.0.0.1:${port}/` });
          for (const page of await this.browser.pages()) {
            if ((await page.title()) === 'audapolis') {
              this.global.page = page;
              try {
                this.ownedProcesses = trackOwnedTree(this.child, [], readProcessTable());
              } catch (error) {
                this.processSnapshotFailure = String(error);
              }
              return;
            }
          }
        } catch (error) {
          this.lastConnectionError = String(error);
        }
        await pause(100);
      }
      throw new Error(`Electron did not become ready: ${this.lastConnectionError}`);
    } catch (error) {
      this.failure = String(error);
      await this.cleanup();
      throw error;
    }
  }

  async cleanup() {
    if (!this.cleanupPromise) this.cleanupPromise = this.cleanupResources();
    return this.cleanupPromise;
  }

  async cleanupResources() {
    const failures = [];
    try {
      this.browser?.disconnect();
    } catch (error) {
      failures.push(String(error));
    }
    try {
      this.processCleanup = await stopOwnedElectron(this.child, {
        known: this.ownedProcesses || [],
      });
      if (!this.processCleanup.verified)
        failures.push('Owned Electron/server process cleanup could not be verified.');
    } catch (error) {
      failures.push(String(error));
    }
    try {
      await this.vite?.close();
    } catch (error) {
      failures.push(String(error));
    }
    if (this.log !== undefined) {
      fs.closeSync(this.log);
      this.log = undefined;
    }
    let temporaryRemoved = false;
    if (this.temporary && this.processCleanup?.verified) {
      try {
        fs.rmSync(this.temporary, { recursive: true, force: true });
        temporaryRemoved = true;
      } catch (error) {
        failures.push(String(error));
      }
    }
    for (const [key, value] of Object.entries(this.previousEnvironment || {})) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    this.cleanupFailure = failures.length ? failures.join('\n') : null;
    if (this.output)
      fs.writeFileSync(
        path.join(this.output, 'environment.json'),
        JSON.stringify(
          {
            started_at: this.started,
            finished_at: new Date().toISOString(),
            platform: os.platform(),
            release: os.release(),
            arch: os.arch(),
            node: process.version,
            cpus: os.cpus().length,
            load_start: this.initialLoad,
            load_end: os.loadavg(),
            concurrent_load:
              'Other project work may run concurrently; this is engineering evidence, not an isolated benchmark.',
            setup_failure: this.failure || null,
            cleanup_failure: this.cleanupFailure || null,
            temporary_profile: this.temporary,
            temporary_profile_removed: temporaryRemoved,
            fixture_kind: this.fixtureKind,
            build_mode: this.mode,
            model_autostart: this.actualAppStartup,
            process_snapshot_failure: this.processSnapshotFailure || null,
            owned_process_cleanup: this.processCleanup || null,
            transport: this.actualAppStartup
              ? 'ordinary_app_gateway_http'
              : process.env.COPILOT_UI_TRANSPORT || 'stdio',
          },
          null,
          2
        )
      );
  }

  async teardown() {
    await this.cleanup();
    await super.teardown();
  }
};
