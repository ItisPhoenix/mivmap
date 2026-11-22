/**
 * The floating YouTube player is infrastructure we must not break. These tests
 * exercise the real module (no mocks of its behaviour) with the repo DOM shim.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import './dom-shim.js'; // installs the minimal document/window used by the viewer modules

globalThis.window = globalThis.window || { innerWidth: 1024, innerHeight: 768, addEventListener() {}, removeEventListener() {} };
document.addEventListener = document.addEventListener || (() => {});
document.querySelector = document.querySelector || ((sel) => document.documentElement.querySelector(sel));
document.querySelectorAll = document.querySelectorAll || ((sel) => document.documentElement.querySelectorAll(sel));

const { showVideoPreview, closePreview, hasPreview } = await import('../viewer/js/features/video-preview.js');
const { MindMapModel } = await import('../viewer/js/mindmap/model.js');
const { ContentMap, outline } = await import('../viewer/js/features/contentmap.js');

test('floating player opens once at an exact timestamp and reuses the iframe', () => {
  assert.equal(hasPreview(), false);
  assert.equal(showVideoPreview({ videoId: '4-Ea5EI6LrM', seconds: 739, x: 400, y: 300 }), true);
  assert.equal(hasPreview(), true);
  const player = document.querySelector('.tm-video-preview');
  assert.ok(player, 'floating preview mounted');
  const iframe = player.querySelector('iframe');
  assert.ok(iframe, 'YouTube iframe present');
  assert.match(iframe.getAttribute('src') || iframe.src || '', /\/player\/4-Ea5EI6LrM\?t=739/);
  showVideoPreview({ videoId: '4-Ea5EI6LrM', seconds: 950, x: 420, y: 320 });
  assert.equal(document.querySelectorAll('.tm-video-preview').length, 1, 'single floating player');
  closePreview();
  assert.equal(hasPreview(), false);
});

test('self-contained map imports into viewer with concepts and Quick recall Q&A', () => {
  // Original fixture: no generated exports, private transcripts or file reads.
  const map = {
    schema: 'tubemind/1',
    meta: { videoId: 'synthetic-map-001', mode: 'deep', teachingLanguage: 'en' },
    root: {
      id: 'root', type: 'root', text: 'Synthetic networking map', summary: 'LAN · WAN',
      start: 0, end: 40, children: [
        {
          id: 's1', type: 'section', text: 'Network types', start: 0, end: 40, children: [
            { id: 'c1', type: 'concept', text: 'LAN', start: 1, end: 10, summary: '', children: [] },
            { id: 'c2', type: 'concept', text: 'WAN', start: 11, end: 20, summary: '', children: [] },
          ],
        },
        {
          id: 'recall', type: 'section', recall: true, text: 'Quick recall',
          summary: 'LAN · WAN', start: 0, end: 40, children: [
            { id: 'q0', type: 'concept', oneline: true, text: 'In one line: LAN · WAN', children: [] },
            { id: 'q1', type: 'concept', qa: true, start: 1, end: 10,
              text: 'Q: What does LAN stand for?  A: Local Area Network.', children: [] },
          ],
        },
      ],
    },
    edges: [], transcript: [],
  };

  const model = new MindMapModel();
  model.load(map);
  assert.ok(model.root, 'viewer model has a root');
  assert.equal(model.root.children.length, 2, 'section and Quick recall imported');

  const concepts = model.root.children
    .filter((s) => !s.recall)
    .flatMap((s) => s.children || [])
    .filter((c) => c.type === 'concept');
  assert.equal(concepts.length, 2, 'both concepts imported');
  assert.deepEqual(concepts.map((c) => c.text).sort(), ['LAN', 'WAN']);

  const o = outline(map);
  assert.ok(o.recall.length > 0, 'Quick recall is present in the content map');
  const qa = o.recall.filter((r) => /^Q:/.test(r.text || ''));
  assert.equal(qa.length, 1, 'one genuine Q&A entry is available');
  assert.match(qa[0].text, /Local Area Network/, 'Q&A answer is grounded in the concept');
});
