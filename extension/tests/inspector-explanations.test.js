/**
 * Right-inspector explanation states — read-first panel, honest failures,
 * no duplicate retries. No browser required.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import './dom-shim.js';

const { InspectorPanel } = await import('../viewer/js/features/panel.js');

const EXPLAIN = 'ONT (Optical Network Terminal) terminates the ISP fibre at the home and converts the optical signal for the local network.';

class FakeModel extends EventTarget {
  constructor(node) {
    super();
    this.node = node;
    this.meta = { videoId: 'vid', title: 'Networking' };
    this.applied = [];
  }

  get(id) { return this.node.id === id ? this.node : null; }
  parent() { return null; }
  nodes() { return [this.node]; }

  apply(op) {
    this.applied.push(op);
    this.node = { ...this.node, ...(op.patch || {}) };
    this.dispatchEvent(new CustomEvent('change', { detail: { origin: 'remote', ops: [op] } }));
  }

  toJSON() { return { meta: this.meta, root: { ...this.node, children: [] } }; }
}

class FakeRenderer extends EventTarget {
  constructor() { super(); this.selection = new Set(); }
  select(ids) { this.selection = new Set(ids); }
  reveal(id) { return id; }
  centerOn() {}
  render() {}
  startEdit() {}
}

function mount(node, api) {
  const host = document.createElement('aside');
  const model = new FakeModel(node);
  const renderer = new FakeRenderer();
  const panel = new InspectorPanel({ host, model, renderer, api, seek() {}, gamify: null, ask: null });
  return { host, model, renderer, panel };
}

const node = (extra = {}) => ({
  id: 'c1', type: 'concept', text: 'ONT and Router', start: 1913.31, summary: '', children: [],
  explain: 'Explanation unavailable: no verified transcript evidence or AI study aid.',
  ...extra,
});
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

test('saved explanation is displayed read-only, not in an empty textarea', () => {
  const { host, panel } = mount(node({ explain: EXPLAIN, steps: ['Fibre reaches the ONT.', 'The ONT converts the signal.'], why_it_matters: 'You know which box to troubleshoot.' }), { online: false });
  panel.show(['c1']);
  assert.match(host.textContent, /What it means/);
  assert.match(host.textContent, /ONT \(Optical Network Terminal\)/);
  assert.match(host.textContent, /How it works/);
  assert.match(host.textContent, /Why it matters/);
  assert.equal(host.querySelector('textarea'), null, 'learn mode has no explanation textarea');
});

test('missing explanation with offline backend shows an honest unavailable state', () => {
  const { host, panel } = mount(node(), { online: false });
  panel.show(['c1']);
  assert.match(host.textContent, /Explanation unavailable/);
  assert.ok(!host.textContent.includes(EXPLAIN));
  assert.equal(host.querySelector('textarea'), null);
});

test('loading skeleton appears once and a successful recovery updates the concept', async () => {
  let resolve;
  const api = {
    online: true,
    recoverExplanations: () => new Promise((r) => { resolve = r; }),
  };
  const { host, model, panel } = mount(node(), api);
  panel.show(['c1']);
  assert.match(host.textContent, /Preparing explanation/);
  assert.ok(host.querySelector('.insp-skeleton'));
  panel.show(['c1']); // duplicate selection while pending
  await tick();
  assert.equal(host.textContent.includes('Preparing explanation'), true);
  resolve({ map: { root: { id: 'root', children: [{ ...node(), explain: EXPLAIN, lesson_available: true }] } }, report: { repaired: ['c1'], failed: [] } });
  await tick();
  assert.match(host.textContent, /ONT \(Optical Network Terminal\)/);
  assert.equal(model.applied.length, 1, 'the recovered lesson is applied exactly once');
});

test('retry does not fire a second concurrent request', async () => {
  let calls = 0;
  let resolve;
  const api = { online: true, recoverExplanations: () => { calls += 1; return new Promise((r) => { resolve = r; }); } };
  const { panel } = mount(node(), api);
  panel.show(['c1']);
  panel.show(['c1']);
  panel.show(['c1']);
  await tick();
  assert.equal(calls, 1);
  resolve({ map: { root: { id: 'root', children: [{ ...node(), explain: EXPLAIN }] } }, report: { repaired: ['c1'], failed: [] } });
  await tick();
});

test('failed recovery preserves the reason and offers a retry', async () => {
  const api = {
    online: true,
    recoverExplanations: async () => ({
      map: { root: { id: 'root', children: [node()] } },
      report: { repaired: [], failed: [{ id: 'c1', reason: 'Lesson failed validation.' }] },
    }),
  };
  const { host, model, panel } = mount(node(), api);
  panel.show(['c1']);
  await tick();
  assert.match(host.textContent, /Lesson failed validation/);
  assert.match(host.textContent, /Retry explanation/);
  assert.equal(model.applied.length, 0, 'a rejected lesson is never written as successful');
});

test('offline unavailable state never renders a literal null and keeps honest copy', () => {
  const { host, panel } = mount(node({ evidence: [{ start: 1913.31, text: 'an ONT, Optical Network Terminal' }] }), { online: false });
  panel.show(['c1']);
  const text = host.textContent;
  assert.ok(!/\bnull\b/i.test(text), 'native append must not stringify an absent retry button');
  assert.match(text, /Explanation unavailable/);
  assert.match(text, /No validated explanation is saved for this concept yet/);
  assert.match(text, /Transcript evidence/);
  assert.equal(host.querySelector('textarea'), null, 'no empty explanation textarea in read mode');
});

const configuredApi = (overrides = {}) => ({
  online: true, health: { llm: 'openai-compatible' },
  recoverExplanations: async () => ({ map: { root: { id: 'root', children: [node()] } }, report: { failed: [] } }),
  generateExplanations: async () => ({ map: { root: { id: 'root', children: [node({ explain: EXPLAIN })] } }, report: { generated: ['c1'], failed: [] } }),
  ...overrides,
});

const clickByText = (host, text) => [...host.querySelectorAll('button')].find((b) => b.textContent.includes(text));

test('configured provider offers Generate; no provider gives an honest message', async () => {
  const { host, panel } = mount(node(), configuredApi());
  panel.show(['c1']);
  await tick(); // initial cache recovery settles first
  assert.ok(clickByText(host, 'Generate explanation'));
  const offline = mount(node(), configuredApi({ health: { llm: 'heuristic' } }));
  offline.panel.show(['c1']);
  await tick();
  assert.equal(clickByText(offline.host, 'Generate explanation'), undefined);
  assert.match(offline.host.textContent, /No LLM provider is configured/);
});

test('Generate is single-flight, applies validated output and reports failure', async () => {
  let calls = 0;
  const api = configuredApi({
    generateExplanations: async () => {
      calls += 1;
      return { map: { root: { id: 'root', children: [node({ explain: EXPLAIN })] } }, report: { generated: ['c1'], failed: [] } };
    },
  });
  const { host, model, panel } = mount(node(), api);
  panel.show(['c1']);
  await tick(); // let the initial cache check finish
  clickByText(host, 'Generate explanation').click();
  await tick();
  assert.equal(calls, 1);
  assert.equal(model.applied.length, 1);
  assert.equal(model.applied[0].type, 'update');
  assert.equal(model.applied[0].patch.explain, EXPLAIN);
  assert.match(host.textContent, /ONT \(Optical Network Terminal\)/);
});

test('failed generation keeps the node honest and shows the reason', async () => {
  const api = configuredApi({
    generateExplanations: async () => ({ map: { root: { id: 'root', children: [node()] } }, report: { failed: [{ id: 'c1', reason: 'Lesson failed validation.' }] } }),
  });
  const { host, model, panel } = mount(node(), api);
  panel.show(['c1']);
  await tick();
  clickByText(host, 'Generate explanation').click();
  await tick();
  assert.match(host.textContent, /Lesson failed validation/);
  assert.equal(model.applied.length, 0);
});

test('recall nodes show their prompt and source, never a lesson generation', async () => {
  let recoverCalls = 0;
  let generateCalls = 0;
  const api = configuredApi({
    recoverExplanations: async () => { recoverCalls += 1; return { map: { root: { id: 'root', children: [] } }, report: {} }; },
    generateExplanations: async () => { generateCalls += 1; return { map: { root: { id: 'root', children: [] } }, report: {} }; },
  });
  const recall = {
    id: 'r1', type: 'concept', text: 'Q: A network is not the internet', recall: true, qa: true,
    question: 'What is the difference between a network and the internet?',
    answer: 'A network connects local devices. The internet links many independent networks worldwide.',
    source: 'A network connects devices.', start: 42, summary: '', children: [],
  };
  const { host, panel } = mount(recall, api);
  panel.show(['r1']);
  assert.match(host.textContent, /Recall question/);
  assert.match(host.textContent, /What is the difference/);
  assert.equal(clickByText(host, 'Generate explanation'), undefined);
  await tick();
  assert.equal(recoverCalls, 0);
  assert.equal(generateCalls, 0);
});

test('edit mode is explicit and keeps the editing controls', () => {
  const { host, panel } = mount(node({ explain: EXPLAIN }), { online: false });
  panel.show(['c1']);
  const edit = [...host.querySelectorAll('.icon-btn')].find((b) => b.getAttribute('aria-label') === 'Edit mode');
  assert.ok(edit);
  edit.click();
  assert.match(host.textContent, /Editing/);
  assert.ok(host.querySelector('textarea'), 'editing mode exposes editable fields');
  const done = [...host.querySelectorAll('button')].find((b) => b.textContent === 'Done');
  done.click();
  assert.equal(host.querySelector('textarea'), null, 'Done returns to read-only learning mode');
});

test('local unauthenticated provider (can_generate) enables fresh generation', async () => {
  const api = configuredApi({
    health: { llm: 'heuristic', openai_compatible: false, llm_local: true, can_generate: true },
  });
  const { host, panel } = mount(node(), api);
  panel.show(['c1']);
  await tick();
  assert.ok(clickByText(host, 'Generate explanation'), 'can_generate: true wins over a stale provider label');
});

test('configured-but-unavailable provider does not offer generation', async () => {
  const api = configuredApi({
    health: { llm: 'openai-compatible', openai_compatible: true, can_generate: false },
  });
  const { host, panel } = mount(node(), api);
  panel.show(['c1']);
  await tick();
  assert.equal(clickByText(host, 'Generate explanation'), undefined);
  assert.match(host.textContent, /unavailable for this backend configuration/);
});

test('generation failure reason from the backend is shown verbatim', async () => {
  const reason = 'The configured inference server is unreachable. ConnectError: connection refused';
  const api = configuredApi({
    generateExplanations: async () => ({ map: { root: { id: 'root', children: [node()] } }, report: { failed: [{ id: 'c1', reason }] } }),
  });
  const { host, panel } = mount(node(), api);
  panel.show(['c1']);
  await tick();
  clickByText(host, 'Generate explanation').click();
  await tick();
  assert.match(host.textContent, /unreachable/);
  assert.notEqual(clickByText(host, 'Generate explanation').disabled, true, 'retry stays available');
});
