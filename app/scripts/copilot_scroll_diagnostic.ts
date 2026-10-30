import { combineReducers } from '@reduxjs/toolkit';
import nav from '../src/state/nav';
import transcribe from '../src/state/transcribe';
import editor from '../src/state/editor';
import models from '../src/state/models';
import server from '../src/state/server';
import recording from '../src/state/recording';
import copilot from '../src/state/copilot';

const production = combineReducers({ nav, transcribe, editor, models, server, recording, copilot });

export function installScrollDiagnostic(targetUuid: string) {
  if (process.env.NODE_ENV !== 'test') throw new Error('Engineering replay requires test mode.');
  const rows: Record<string, any>[] = [];
  let sampling = true;
  const rect = (node: Element | null) => {
    if (!node) return null;
    const bounds = node.getBoundingClientRect();
    return { x: bounds.x, y: bounds.y, width: bounds.width, height: bounds.height };
  };
  const identify = (node: Element | null) =>
    node && {
      tag: node.tagName,
      id: node.id,
      testid: node.getAttribute('data-testid'),
      className: String(node.className).slice(0, 180),
    };
  const snapshot = (state = window.store.getState()) => {
    const viewport = document.getElementById('scroll-container');
    const preview = document.querySelector('[data-testid="copilot-preview"]');
    const target = document.getElementById(`item-${targetUuid}`);
    const cursor = document.getElementById('cursor');
    return {
      at_ms: performance.now(),
      status: state.copilot.status,
      blocked: state.copilot.blocked,
      epoch: state.copilot.epoch,
      request_id: state.copilot.request?.request_id,
      proposal_id: state.copilot.proposal?.response.proposal_id,
      shown: state.copilot.proposal?.shown,
      cursor: state.editor.present?.cursor,
      hidden: document.hidden,
      focus: identify(document.activeElement),
      hasFocus: document.hasFocus(),
      inner: [innerWidth, innerHeight],
      viewport: rect(viewport),
      scrollTop: viewport?.scrollTop,
      scrollHeight: viewport?.scrollHeight,
      viewportClientWidth: viewport?.clientWidth,
      viewportOverflow: viewport && getComputedStyle(viewport).overflow,
      document: rect(document.getElementById('document')),
      preview: rect(preview),
      target: rect(target),
      cursorRect: rect(cursor),
      cursorTransform: cursor && getComputedStyle(cursor).transform,
    };
  };
  const log = (kind: string, detail: unknown = {}, state?: any) => {
    if (rows.length < 15000) rows.push({ kind, detail, ...snapshot(state) });
  };
  window.store.replaceReducer((state, action) => {
    const next = production(state, action);
    log(
      'redux',
      {
        action: action.type,
        payload:
          action.type === 'copilot/measured' ||
          action.type === 'copilot/setCopilotBlocked' ||
          action.type === 'editor/setUserIndex' ||
          action.type === 'copilot/invalidated'
            ? action.payload
            : undefined,
      },
      next
    );
    return next;
  });
  const originals: [string, any][] = [];
  for (const method of ['scrollTo', 'scroll', 'scrollIntoView'] as const) {
    const original = Element.prototype[method];
    originals.push([method, original]);
    (Element.prototype as any)[method] = function (...args: any[]) {
      log('scroll-call', { method, node: identify(this), args, stack: new Error().stack });
      return original.apply(this, args as any);
    };
  }
  const originalObserver = window.IntersectionObserver;
  window.IntersectionObserver = class extends originalObserver {
    constructor(callback: IntersectionObserverCallback, options?: IntersectionObserverInit) {
      super((entries, observer) => {
        log('intersection', {
          root: identify(options?.root instanceof Element ? options.root : null),
          entries: entries.map((entry) => ({
            target: identify(entry.target),
            intersecting: entry.isIntersecting,
            ratio: entry.intersectionRatio,
            rect: entry.boundingClientRect.toJSON(),
            intersection: entry.intersectionRect.toJSON(),
            rootBounds: entry.rootBounds?.toJSON(),
          })),
        });
        callback(entries, observer);
      }, options);
    }
  };
  const onEvent = (event: Event) =>
    log(event.type, { target: identify(event.target instanceof Element ? event.target : null) });
  for (const event of ['scroll', 'focusin', 'focusout', 'visibilitychange', 'resize'])
    document.addEventListener(event, onEvent, true);
  const frame = () => {
    if (!sampling) return;
    log('frame');
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
  return {
    mark: (label: string) => log('mark', label),
    snapshot,
    finish: () => {
      sampling = false;
      log('finish');
      for (const [method, original] of originals) (Element.prototype as any)[method] = original;
      window.IntersectionObserver = originalObserver;
      for (const event of ['scroll', 'focusin', 'focusout', 'visibilitychange', 'resize'])
        document.removeEventListener(event, onEvent, true);
      window.store.replaceReducer(production);
      return rows;
    },
  };
}
