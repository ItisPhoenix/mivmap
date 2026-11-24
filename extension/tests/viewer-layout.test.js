/**
 * Redesigned viewer shell: Outline / Recall tabs, compact header, concise
 * canvas labels, and navigation controls. DOM-shim only, no browser.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import './dom-shim.js';

const read = (path) => readFileSync(new URL(`../${path}`, import.meta.url), 'utf8');
const { MindMapModel } = await import('../viewer/js/mindmap/model.js');
const { MindMapRenderer } = await import('../viewer/js/mindmap/renderer.js');
const { ContentMap } = await import('../viewer/js/features/contentmap.js');
const fixture = JSON.parse(read('tests/fixtures/sketchbook-map.json'));

function mount() {
  const model = new MindMapModel();
  const host = document.createElement('div');
  const renderer = new MindMapRenderer(host, model);
  model.load(fixture);
  const stage = document.createElement('main');
  const toggle = document.createElement('button');
  const seeks = [];
  localStorage.removeItem('tm-content-map-tab');
  const cm = new ContentMap({
    host: document.createElement('aside'), stage, toggle, model, renderer,
    seek: (node) => seeks.push(node.id),
  });
  cm.render();
  return { model, renderer, cm, host: cm.host, seeks };
}

test('outline tab is default and has a compact numbered section list', () => {
  const { host } = mount();
  assert.equal(cmTab(host), 'Outline');
  assert.match(host.textContent, /Video outline/);
  assert.match(host.textContent, /\d+ sections/);
  assert.match(host.textContent, /Jump to/);
  assert.match(host.textContent, /Section detail/);
  assert.ok(host.querySelectorAll('.cm-jump-item').length >= 3);
  assert.ok(host.querySelectorAll('.cm-section-detail').length >= 3);
});

test('recall tab moves recall out of the outline and reveals prompts', () => {
  const { host } = mount();
  const tabs = host.querySelectorAll('.cm-tab');
  assert.equal(tabs.length, 2);
  tabs.find((b) => b.textContent === 'Recall').click();
  assert.equal(cmTab(host), 'Recall');
  assert.match(host.textContent, /Try to say each point/);
  assert.ok(host.querySelectorAll('.cm-recall-item').length >= 3);
  const answer = host.querySelector('.cm-recall-answer');
  assert.equal(answer.hidden, true);
  // Legacy fragment items have no distinct answer: show an honest state, not a fake reveal.
  assert.match(answer.textContent, /No distinct answer is saved/);
  assert.match(host.textContent, /Start quiz from recall/);
});

test('section navigation keeps the map state and offers a video jump', () => {
  const { model, renderer, host, seeks } = mount();
  const section = model.root.children[1];
  host.querySelector(`li.cm-jump-item[data-node-id="${section.id}"]`).querySelector('button').click();
  assert.deepEqual([...renderer.selection], [section.id]);
  assert.deepEqual(seeks, [section.id]);
  assert.ok(host.querySelector(`li.cm-section[data-node-id="${section.id}"]`).classList.contains('cm-active'));
});

test('canvas labels are concise but the underlying concept text is unchanged', () => {
  const { model, renderer } = mount();
  const concept = model.nodes().find((n) => n.type === 'concept' && /:/.test(n.text));
  assert.ok(concept, 'fixture has a "Term: meaning" concept');
  const display = renderer.displayText(concept);
  assert.ok(display.length < concept.text.length);
  assert.ok(!display.includes(':'));
  assert.equal(model.get(concept.id).text, concept.text, 'display never mutates map data');
});

test('header keeps one primary study action and moves secondary tools into More', () => {
  const html = read('viewer/viewer.html');
  assert.match(html, /id="btn-study" class="btn btn-accent"/);
  for (const id of ['btn-share', 'btn-export', 'btn-voice', 'layer-seg']) {
    assert.ok(html.indexOf(`id="${id}"`) > html.indexOf('id="btn-more"'), `${id} lives in the More menu`);
  }
  assert.ok(!/location\.reload/.test(read('viewer/js/app.js')));
});

test('Focus Section shows one branch, Full Map restores every section', () => {
  const { model, renderer } = mount();
  const sections = model.root.children.filter((s) => s.type === 'section' && !s.recall);
  assert.ok(sections.length >= 3);
  renderer.setFocus(sections[0].id);
  assert.equal(renderer.focusId, sections[0].id);
  assert.ok(renderer.svgEl.classList.contains('focus-mode'));
  for (const s of sections.slice(1)) assert.equal(renderer.positions.has(s.id), false, 'other sections are hidden');
  assert.ok(renderer.positions.has(sections[0].id));
  const inFocus = [...renderer.layers.nodes.children].filter((g) => g.classList.contains('in-focus'));
  assert.ok(inFocus.length >= 1, 'focused branch is marked');
  renderer.clearFocus();
  assert.equal(renderer.focusId, null);
  assert.ok(sections.every((s) => renderer.positions.has(s.id)), 'all sections return');
});

test('collapsed children use an accessible expansion affordance, not "+1"', () => {
  const { model, renderer } = mount();
  const collapsed = model.nodes().find((n) => n.collapsed && n.children?.length);
  assert.ok(collapsed, 'fixture has a collapsed node');
  const toggle = renderer.layers.nodes.querySelector(`[data-node-id="${collapsed.id}"]`).querySelector('.toggle');
  assert.ok(toggle);
  const glyph = toggle.textContent.replace(toggle.querySelector('title')?.textContent || '', '').trim();
  assert.equal(glyph, '▸');
  assert.match(toggle.getAttribute('aria-label'), /hidden detail/);
  assert.equal(toggle.getAttribute('role'), 'button');
  assert.equal(toggle.getAttribute('tabindex'), '0');
  assert.equal(toggle.getAttribute('aria-expanded'), 'false');
  let fired = false;
  renderer.addEventListener('toggle', () => { fired = true; });
  const ev = new Event('keydown', { bubbles: true }); ev.key = 'Enter';
  toggle.dispatchEvent(ev);
  assert.equal(fired, true, 'keyboard activation expands without changing map data');
});

test('map navigation has compact zoom and focus controls', () => {
  const html = read('viewer/viewer.html');
  for (const id of ['zoom-in', 'zoom-out', 'zoom-focus', 'zoom-fit']) assert.match(html, new RegExp(`id="${id}"`));
  assert.match(read('viewer/viewer.css'), /\.zoom-controls \{[^}]*flex-direction: row/);
});

function cmTab(host) {
  return host.querySelector('.cm-tab.on')?.textContent || '';
}
