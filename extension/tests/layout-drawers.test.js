/**
 * Layout regression tests for the right-inspector occlusion and responsive
 * drawer fixes. The pure state machine is tested directly; the CSS/DOM contract
 * is asserted here, and real geometry is verified with browser screenshots.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

const read = (path) => readFileSync(new URL(`../${path}`, import.meta.url), 'utf8');
const { layoutState, LAYOUT_BREAKPOINTS } = await import('../viewer/js/features/layout.js');
const css = read('viewer/viewer.css');
const html = read('viewer/viewer.html');
const app = read('viewer/js/app.js');
const renderer = read('viewer/js/mindmap/renderer.js');

test('desktop reserves inspector width; smaller viewports never do', () => {
  assert.deepEqual(LAYOUT_BREAKPOINTS, { medium: 1100, narrow: 760 });
  const desktop = layoutState(1440, { inspectorOpen: true, outlineOpen: true });
  assert.equal(desktop.mode, 'desktop');
  assert.equal(desktop.reserveInspector, true, 'root and sections must not sit under the panel');
  assert.equal(desktop.drawersActive, false);

  const desktopClosed = layoutState(1440, { inspectorOpen: false, outlineOpen: true });
  assert.equal(desktopClosed.reserveInspector, false, 'closing the inspector restores full canvas width');

  for (const width of [820, 600]) {
    const state = layoutState(width, { inspectorOpen: true, outlineOpen: true });
    assert.notEqual(state.mode, 'desktop');
    assert.equal(state.reserveInspector, false, 'drawers must not permanently consume canvas width');
    assert.equal(state.drawersActive, true);
    assert.equal(layoutState(width).drawersActive, false, 'both drawers closed');
  }
  assert.equal(layoutState(820, { outlineOpen: true }).drawersActive, false,
    'medium keeps the outline in flow, so no backdrop is shown');
  assert.equal(layoutState(600, { outlineOpen: true }).drawersActive, true,
    'narrow turns the outline into a drawer');
});

test('CSS reserves real canvas space on desktop and overrides it in drawer modes', () => {
  assert.match(css, /\.stage\s*\{[^}]*--rail-width:\s*370px[^}]*--rail-reserve:\s*calc\(var\(--rail-width\)\s*\+\s*20px\)/);
  assert.match(css, /\.stage\[data-layout='desktop'\] #canvas,\s*\.stage\.inspector-open #canvas\s*\{\s*right:\s*var\(--rail-reserve\);\s*\}/);
  const mediumStart = css.indexOf('@media (max-width: 1100px)');
  const narrowStart = css.indexOf('@media (max-width: 760px)', mediumStart);
  const medium = css.slice(mediumStart, narrowStart);
  assert.match(medium, /\.stage\.inspector-open #canvas\s*\{\s*right:\s*0;\s*\}/);
  assert.match(medium, /#btn-inspector\s*\{\s*display:\s*inline-flex;/);
  const narrow = css.slice(narrowStart);
  assert.match(narrow, /\.stage\.cm-open #canvas\s*\{\s*left:\s*0;/);
});

test('header has separate outline and inspector controls plus a drawer backdrop', () => {
  assert.match(html, /id="btn-content-map"/);
  assert.match(html, /id="btn-inspector"[^>]*aria-pressed="false"[^>]*aria-controls="inspector"/);
  assert.match(html, /id="drawer-backdrop" class="drawer-backdrop"[^>]*hidden/);
  assert.match(css, /\.drawer-backdrop\s*\{[^}]*position:\s*absolute[^}]*inset:\s*0/);
});

test('app shell syncs drawer state, refits on visibility change, and never reloads', () => {
  assert.match(app, /layoutState\(window\.innerWidth/);
  assert.match(app, /stage\.classList\.toggle\('inspector-open',\s*state\.reserveInspector\)/);
  assert.match(app, /drawerBackdrop\.hidden\s*=\s*!state\.drawersActive/);
  assert.match(app, /inspectorEmpty\.hidden\s*=\s*!\(state\.mode === 'desktop' && !state\.inspectorOpen\)/);
  assert.match(app, /window\.addEventListener\('resize'/);
  assert.match(app, /syncLayout\(\{ refit: wasHidden !== inspectorPanel\.host\.hidden \}\)/);
  assert.match(app, /inspectorDrawerClosed = true; \/\/ hide the drawer but keep the selected concept/);
  assert.match(css, /\.stage\.inspector-drawer-closed \.right-rail \{ display: none; \}/);
  assert.ok(!/location\.reload|location\.href|window\.location/.test(app));
  assert.match(app, /ctxMenu\.replaceChildren\(\.\.\.entries\.filter\(Boolean\)\)/);
  // fit()/centerOn() must use the (now correctly sized) canvas box.
  assert.match(renderer, /fit\(animate = true\)\s*\{[\s\S]{0,200}this\.container\.getBoundingClientRect\(\)/);
  assert.match(renderer, /centerOn\(id[\s\S]{0,200}this\.container\.getBoundingClientRect\(\)/);
});
