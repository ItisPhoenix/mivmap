/**
 * Node inspector (right-hand panel).
 *
 * LEARN is the default, read-only experience: concept title, timestamp, the
 * saved/validated explanation and teaching aids, source evidence, notes and
 * resources. EDIT is an explicit mode containing every existing editing,
 * visual, structure and AI-refinement control. Nothing here fabricates an
 * explanation: a missing one is reported honestly, with a safe retry that only
 * re-attaches cached lessons which pass the backend quality gate.
 */
import { captureCurrentFrame, currentVideoTime, pickImageFile, storyboardFrame } from './frames.js';
import { REFINE_ACTIONS, runRefine } from './refine.js';
import { el, fmtTime, isHttpUrl, plain, toast } from './shared.js';

const TONES = ['enthusiastic', 'critical', 'controversial', 'cautionary', 'humorous', 'instructional', 'inspirational', 'analytical', 'skeptical', 'optimistic'];
const TYPE_LABEL = { root: 'Topic', section: 'Section', concept: 'Key concept', detail: 'Detail', transcript: 'Quote' };
const PLACEHOLDER = /^(?:explanation unavailable|unavailable|no verified transcript evidence|n\/?a)\b/i;
const LESSON_KEYS = [
  'explain', 'intuition', 'why_it_matters', 'steps', 'worked_example', 'diagram',
  'misconception', 'exercise', 'knowledge_check', 'lesson_evidence', 'provenance',
  'groundingNote', 'accuracy_notes', 'acronym_expansion', 'teaching_language',
  'lesson_available', 'available', 'conceptUnavailable', 'unavailableReason', 'generatedBy',
];
const BUSY = new Set(['loading', 'retrying', 'generating', 'retrying-generate']);
const isRecall = (n) => !!(n?.recall || n?.oneline || n?.qa);
const isConceptNode = (n) => n?.type === 'concept' && !isRecall(n);
// Prefer the backend's explicit generation capability (covers unauthenticated
// loopback servers and in-process local models); fall back to the provider label
// for older backends that do not expose it.
const providerConfigured = (api) => {
  if (!api?.online || !api?.health) return false;
  if (typeof api.health.can_generate === 'boolean') return api.health.can_generate;
  return !!api.health.llm && api.health.llm !== 'heuristic';
};
const generationUnavailableNote = (api) => {
  const health = api?.health || {};
  if (health.llm && health.llm !== 'heuristic') {
    return 'Fresh generation is unavailable for this backend configuration; retry only checks the validated lesson cache.';
  }
  return 'No LLM provider is configured. Fresh generation is unavailable; retry only checks the validated lesson cache.';
};
const WEAK_LABEL = /^(?:extra|extras|\+1|untitled|n\/?a|—|-+|\d+)$/i;

function explanationText(node) {
  for (const key of ['explain', 'summary']) {
    const text = String(node?.[key] || '').trim();
    if (text && !PLACEHOLDER.test(text)) return text;
  }
  return '';
}

const normalizeText = (text) => String(text || '').toLowerCase().replace(/[^a-z0-9]+/gi, ' ').trim();

function findNode(root, id) {
  if (!root) return null;
  if (root.id === id) return root;
  for (const child of root.children || []) {
    const hit = findNode(child, id);
    if (hit) return hit;
  }
  return null;
}

export class InspectorPanel {
  constructor({ host, model, renderer, api, seek, gamify, ask, emptyHost = null }) {
    Object.assign(this, { host, model, renderer, api, seek, gamify, ask, emptyHost });
    this.ids = [];
    this.editing = false;
    this.editFocus = null;
    this.currentId = null;
    this.states = new Map(); // node id -> { status, error }
    renderer.addEventListener('select', (e) => this.show(e.detail.ids));
    model.addEventListener('change', (e) => {
      if (e.detail.origin !== 'local-panel' && this.ids.length) this.#refreshSoon();
    });
    model.addEventListener('load', () => {
      this.states.clear(); // a new map must not inherit another map's failure state
      this.editing = false;
      this.show([]);
    });
  }

  #refreshSoon() {
    if (this.host.contains(document.activeElement) && document.activeElement.matches('textarea, input')) return;
    clearTimeout(this.refreshTimer);
    this.refreshTimer = setTimeout(() => this.show(this.ids), 60);
  }

  show(ids) {
    const next = ids.filter((id) => this.model.get(id));
    if (next.length !== this.ids.length || next.some((id, i) => id !== this.ids[i])) this.editing = false;
    this.ids = next;
    if (!this.ids.length) {
      this.currentId = null;
      this.host.hidden = true;
      if (this.emptyHost) this.emptyHost.hidden = false;
      return;
    }
    this.host.hidden = false;
    if (this.emptyHost) this.emptyHost.hidden = true;
    this.openedAt = performance.now();
    const node = this.model.get(this.ids.at(-1));
    this.currentId = node.id;
    this.host.replaceChildren(this.#render(node));
    if (!this.editing && isConceptNode(node) && !explanationText(node) && this.api?.online) {
      const status = this.states.get(node.id)?.status;
      if (!status || status === 'idle') this.#recover(node, false);
    }
    if (this.editFocus) {
      const target = this.host.querySelector(`[data-section="${this.editFocus}"]`);
      target?.scrollIntoView?.({ block: 'center' });
      this.editFocus = null;
    }
  }

  close() {
    this.renderer.select([]);
  }

  #commit(id, patch) {
    this.model.apply({ type: 'update', id, patch }, { origin: 'local' });
  }

  #render(node) {
    return this.editing ? this.#renderEdit(node) : this.#renderLearn(node);
  }

  // ---------------------------------------------------------------- LEARN ---
  #renderLearn(node) {
    const meta = this.model.meta || {};
    const videoId = node.videoId || meta.videoId;
    const frag = document.createDocumentFragment();

    frag.append(el('div', { class: 'panel-head insp-head' }, [
      el('span', { class: 'panel-type' }, node.recall ? 'Quick recall' : TYPE_LABEL[node.type] || node.type),
      el('div', { class: 'panel-head-actions' }, [
        el('button', { class: 'icon-btn', title: 'Edit this concept', 'aria-label': 'Edit mode', onclick: () => { this.editing = true; this.show(this.ids); } }, '✎'),
        this.#overflowMenu(node),
        el('button', { class: 'icon-btn', title: 'Close (Esc)', 'aria-label': 'Close inspector', onclick: () => this.close() }, '✕'),
      ]),
    ]));

    if (this.ids.length > 1) frag.append(el('p', { class: 'panel-multi' }, `${this.ids.length} nodes selected — Merge or add a cross-link in Edit mode.`));

    const titleText = isRecall(node)
      ? plain(node.question || node.text).replace(/^Q:\s*/i, '')
      : plain(node.text);
    frag.append(el('h2', { class: 'insp-title' }, titleText));
    if (WEAK_LABEL.test(plain(node.text).trim())) {
      frag.append(el('p', { class: 'panel-note' },
        'This node has a weak label from the source video. The node and its evidence are preserved below; no concept name was invented.'));
    }
    const tsRow = el('div', { class: 'panel-row wrap insp-actions' });
    if (node.start !== null && node.start !== undefined) {
      tsRow.append(el('button', { class: 'btn btn-accent btn-small', onclick: () => this.seek(node) }, `▶ Watch at ${fmtTime(node.start)}`));
    }
    if (this.ask && this.ids.length === 1) {
      tsRow.append(el('button', { class: 'btn btn-small', title: 'Ask a question about this node (a)', onclick: () => this.ask(node.id) }, '💬 Ask AI'));
    }
    if (tsRow.children.length) frag.append(tsRow);

    frag.append(isRecall(node) ? this.#recall(node) : this.#explanation(node));

    const related = this.#related(node);
    if (related.length) {
      frag.append(el('section', { class: 'insp-related' }, [
        el('h3', { class: 'insp-section-title' }, 'Related concepts'),
        el('div', { class: 'insp-related-list' }, related.map((r) =>
          el('button', {
            class: 'related-chip',
            onclick: () => { this.renderer.reveal(r.id); this.renderer.select(r.id); this.renderer.centerOn(r.id); },
          }, plain(r.text)),
        )),
      ]));
    }

    frag.append(this.#disclosure('My notes', 'notes', [
      node.notes
        ? el('p', { class: 'insp-note-text' }, node.notes)
        : el('p', { class: 'panel-note' }, 'No notes yet.'),
      el('button', { class: 'btn btn-small btn-ghost', onclick: () => { this.editing = true; this.editFocus = 'notes'; this.show(this.ids); } }, 'Edit notes'),
    ], !!node.notes));

    const links = node.links || [];
    frag.append(this.#disclosure(`Resources${links.length ? ` (${links.length})` : ''}`, 'resources', [
      links.length
        ? el('ul', { class: 'panel-links' }, links.map((link) => el('li', {}, el('a', { href: link.url, target: '_blank', rel: 'noopener noreferrer' }, link.title || link.url))))
        : el('p', { class: 'panel-note' }, 'No external links yet.'),
      el('button', { class: 'btn btn-small btn-ghost', onclick: () => { this.editing = true; this.editFocus = 'links'; this.show(this.ids); } }, 'Manage resources'),
    ], false));

    const evidence = node.evidence || [];
    if (evidence.length) {
      const list = el('ul', { class: 'insp-evidence' }, evidence.slice(0, 6).map((e) =>
        el('li', {}, [
          e.start !== undefined && e.start !== null ? el('button', { class: 'ts-chip', onclick: () => this.seek({ start: e.start, videoId }) }, fmtTime(e.start)) : null,
          ` ${String(e.text || '').slice(0, 220)}`,
        ]),
      ));
      frag.append(this.#disclosure('Transcript evidence', 'evidence', [
        el('p', { class: 'panel-note' }, 'Timestamped lines from the video. Teaching aids are labelled separately and are not source claims.'),
        list,
      ], false));
    }

    if (node.videoId && node.videoId !== meta.videoId) frag.append(el('p', { class: 'panel-note' }, '🔗 From another video in the Knowledge Hub.'));
    return frag;
  }

  #disclosure(label, section, children, open = false) {
    const details = el('details', { class: 'insp-disclosure', 'data-section': section });
    if (open) details.open = true;
    details.append(el('summary', {}, label), el('div', { class: 'insp-disclosure-body' }, children));
    return details;
  }

  #overflowMenu(node) {
    const menu = el('div', { class: 'menu insp-menu' }, [
      el('button', { class: 'icon-btn', title: 'More actions', 'aria-label': 'More actions', 'aria-haspopup': 'true' }, '⋯'),
      el('div', { class: 'dropdown dropdown-right', hidden: true }),
    ]);
    const list = [...menu.children].at(-1);
    const action = (icon, label, section) => el('button', {
      class: 'menu-item',
      onclick: () => { list.hidden = true; this.editing = true; this.editFocus = section; this.show(this.ids); },
    }, [el('span', {}, icon), label]);
    list.append(
      action('✎', 'Edit explanation', 'explanation'),
      action('📝', 'Edit notes', 'notes'),
      action('🔗', 'Edit links & resources', 'links'),
      action('🖼', 'Visual / capture frame', 'visual'),
      action('🌳', 'Structure: child, sibling, delete', 'structure'),
      action('✨', 'AI refinement', 'ai'),
    );
    const toggle = [...menu.children][0];
    toggle.addEventListener('click', (e) => {
      e.stopPropagation();
      this.host.querySelectorAll('.insp-menu .dropdown').forEach((d) => { if (d !== list) d.hidden = true; });
      list.hidden = !list.hidden;
    });
    return menu;
  }

  #related(node) {
    const out = [];
    const parent = this.model.parent?.(node.id) || null;
    if (parent) out.push(parent);
    for (const id of node.children || []) {
      const child = this.model.get(id);
      if (child) out.push(child);
    }
    const siblings = parent?.children || [];
    for (const id of siblings) {
      if (id === node.id) continue;
      const sib = this.model.get(id);
      if (sib) out.push(sib);
    }
    return out.slice(0, 6);
  }

  /** Recall items are revision prompts, not concept lessons. */
  #recall(node) {
    const section = el('section', { class: 'insp-explanation', 'data-section': 'explanation' });
    const question = plain(node.question || node.knowledge_check?.question || '').replace(/^Q:\s*/i, '');
    const answer = plain(node.answer || node.knowledge_check?.answer || '').replace(/^A:\s*/i, '');
    section.append(el('h3', { class: 'insp-section-title' },
      node.oneline ? 'One-line summary' : node.qa ? 'Recall question' : 'Point to remember'));

    if (question && answer && normalizeText(question) !== normalizeText(answer)) {
      section.append(el('p', { class: 'insp-explain' }, question));
      const reveal = el('p', { class: 'insp-explain', hidden: true }, answer);
      const button = el('button', {
        class: 'btn btn-small btn-ghost',
        onclick: () => {
          reveal.hidden = !reveal.hidden;
          button.textContent = reveal.hidden ? 'Reveal answer' : 'Hide answer';
        },
      }, 'Reveal answer');
      section.append(button, reveal);
      if (node.knowledge_check?.explanation) section.append(el('p', {}, node.knowledge_check.explanation));
    } else if (question) {
      section.append(el('p', { class: 'insp-explain' }, question),
        el('p', { class: 'panel-note' }, 'No distinct answer is saved for this prompt.'));
    } else {
      section.append(el('p', { class: 'insp-explain' }, plain(node.text).replace(/^(In one line:|Q:)\s*/i, '')),
        el('p', { class: 'panel-note' }, 'No question/answer pair is saved for this recall item.'));
    }
    if (node.source) section.append(el('blockquote', { class: 'insp-recall-source' }, `“${plain(node.source)}”`));
    if (node.start !== null && node.start !== undefined) {
      section.append(el('button', { class: 'btn btn-small', onclick: () => this.seek(node) }, `▶ ${fmtTime(node.start)}`));
    }
    section.append(el('p', { class: 'panel-note insp-provenance' },
      'Quick recall is a revision prompt grounded in the video above. Concept lessons are not generated for recall items.'));
    return section;
  }

  #explanation(node) {
    const section = el('section', { class: 'insp-explanation', 'data-section': 'explanation' });
    const text = explanationText(node);
    if (text) {
      section.append(el('h3', { class: 'insp-section-title' }, 'What it means'));
      section.append(el('p', { class: 'insp-explain' }, text));
      if (node.intuition) section.append(el('p', { class: 'insp-intuition' }, node.intuition));
      if (Array.isArray(node.steps) && node.steps.length) {
        section.append(el('h3', { class: 'insp-section-title' }, 'How it works'));
        section.append(el('ol', { class: 'insp-steps' }, node.steps.map((s) => el('li', {}, s))));
      }
      if (node.why_it_matters) {
        section.append(el('h3', { class: 'insp-section-title' }, 'Why it matters'));
        section.append(el('p', {}, node.why_it_matters));
      }
      const example = node.worked_example || node.example;
      if (example) {
        section.append(el('h3', { class: 'insp-section-title' }, 'Practical example'));
        section.append(el('p', {}, example));
        if (node.provenance?.example_source === 'supplemental') {
          section.append(el('p', { class: 'panel-note' }, 'This example is a teaching aid, not a statement from the video.'));
        }
      }
      if (node.diagram) section.append(el('pre', { class: 'learning-diagram' }, node.diagram));
      if (node.misconception) section.append(el('p', { class: 'insp-misconception' }, `⚠ ${node.misconception}`));
      if (node.knowledge_check?.question) {
        section.append(el('div', { class: 'insp-check' }, [
          el('strong', {}, 'Check yourself'),
          el('p', {}, node.knowledge_check.question),
        ]));
      }
      section.append(el('p', { class: 'panel-note insp-provenance' },
        'Teaching aids are generated and validated; source claims are grounded in the timestamped transcript evidence below.'));
      return section;
    }

    const state = this.states.get(node.id) || { status: 'idle' };
    if (BUSY.has(state.status)) {
      const preparing = state.status === 'generating' || state.status === 'retrying-generate';
      section.append(el('div', { class: 'insp-skeleton', 'aria-live': 'polite' }, [
        el('span', { class: 'skel-line long' }), el('span', { class: 'skel-line' }),
        el('span', { class: 'skel-line short' }),
      ]), el('p', { class: 'panel-note' },
        preparing ? (state.status === 'retrying-generate' ? 'Retrying generation…' : 'Generating explanation…')
          : (state.status === 'retrying' ? 'Retrying explanation…' : 'Preparing explanation…')));
      return section;
    }
    if (state.status === 'failed' || state.status === 'generate-failed') {
      section.append(
        el('h3', { class: 'insp-section-title' }, 'Explanation unavailable'),
        el('p', { class: 'insp-error' }, state.error || 'An explanation could not be generated for this concept.'),
        el('button', {
          class: 'btn btn-small',
          disabled: !this.api?.online ? true : null,
          onclick: () => this.#recover(node, true),
        }, '↻ Retry explanation'),
      );
      if (providerConfigured(this.api)) {
        section.append(el('button', { class: 'btn btn-accent btn-small', onclick: () => this.#generate(node, true) }, '✦ Generate explanation'));
      } else {
        section.append(el('p', { class: 'panel-note' }, this.api?.online
          ? generationUnavailableNote(this.api)
          : 'Connect the backend to retry.'));
      }
      return section;
    }
    section.append(
      el('h3', { class: 'insp-section-title' }, 'Explanation unavailable'),
      el('p', { class: 'panel-note' }, 'No validated explanation is saved for this concept yet. The timestamped transcript evidence is preserved below.'),
    );
    // Native append() stringifies null/undefined; only append real nodes.
    if (providerConfigured(this.api)) {
      section.append(el('button', { class: 'btn btn-accent btn-small', onclick: () => this.#generate(node) }, '✦ Generate explanation'));
    }
    if (this.api?.online) {
      section.append(el('button', { class: 'btn btn-small btn-ghost', onclick: () => this.#recover(node, true) }, '↻ Try to recover'));
    } else {
      section.append(el('p', { class: 'panel-note' }, 'Connect the backend to look for a validated explanation.'));
    }
    if (this.api?.online && !providerConfigured(this.api)) {
      section.append(el('p', { class: 'panel-note' }, generationUnavailableNote(this.api)));
    }
    return section;
  }

  async #recover(node, retry = false) {
    if (!this.api?.online) {
      this.states.set(node.id, { status: 'failed', error: 'The backend is offline, so no validated explanation can be loaded.' });
      if (this.currentId === node.id) this.show([node.id]);
      return;
    }
    const current = this.states.get(node.id)?.status;
    if (BUSY.has(current)) return; // prevents duplicate generate/recover requests
    this.states.set(node.id, { status: retry ? 'retrying' : 'loading' });
    if (this.currentId === node.id) this.show([node.id]);
    try {
      const response = await this.api.recoverExplanations(this.model.toJSON(), [node.id]);
      const fresh = findNode(response?.map?.root, node.id);
      if (!explanationText(fresh)) {
        const reason = response?.report?.failed?.find((f) => f.id === node.id)?.reason;
        throw new Error(reason || 'The saved lesson did not pass validation, so nothing was changed.');
      }
      const patch = {};
      for (const key of LESSON_KEYS) if (key in fresh) patch[key] = fresh[key];
      this.model.apply({ type: 'update', id: node.id, patch }, { origin: 'remote' });
      this.states.set(node.id, { status: 'success' });
    } catch (err) {
      this.states.set(node.id, { status: 'failed', error: err?.message || String(err) });
    }
    if (this.currentId === node.id) this.show([node.id]);
  }

  async #generate(node, retry = false) {
    if (!isConceptNode(node) || !providerConfigured(this.api)) return;
    const current = this.states.get(node.id)?.status;
    if (BUSY.has(current)) return; // single-flight for both generate and recover
    this.states.set(node.id, { status: retry ? 'retrying-generate' : 'generating' });
    if (this.currentId === node.id) this.show([node.id]);
    try {
      const response = await this.api.generateExplanations(this.model.toJSON(), [node.id]);
      const fresh = findNode(response?.map?.root, node.id);
      if (!explanationText(fresh)) {
        const reason = response?.report?.failed?.find((f) => f.id === node.id)?.reason;
        throw new Error(reason || 'No validated explanation was produced; nothing was changed.');
      }
      const patch = {};
      for (const key of LESSON_KEYS) if (key in fresh) patch[key] = fresh[key];
      this.model.apply({ type: 'update', id: node.id, patch }, { origin: 'ai' });
      this.states.set(node.id, { status: 'success' });
    } catch (err) {
      this.states.set(node.id, { status: 'generate-failed', error: err?.message || String(err) });
    }
    if (this.currentId === node.id) this.show([node.id]);
  }

  // ----------------------------------------------------------------- EDIT ---
  #renderEdit(node) {
    const meta = this.model.meta || {};
    const videoId = node.videoId || meta.videoId;
    const frag = document.createDocumentFragment();

    frag.append(el('div', { class: 'panel-head' }, [
      el('span', { class: 'panel-type edit-badge' }, '✎ Editing'),
      el('button', {
        class: 'btn btn-small',
        onclick: () => { this.editing = false; this.show(this.ids); },
      }, 'Done'),
    ]));
    frag.append(el('p', { class: 'panel-note' }, 'Map editing mode. Changes are saved to this mindmap.'));

    const title = el('textarea', { class: 'panel-title', rows: 2, 'aria-label': 'Node text' });
    title.value = node.text;
    title.addEventListener('change', () => title.value.trim() && this.#commit(node.id, { text: title.value.trim() }));
    frag.append(el('div', { 'data-section': 'explanation' }, [
      el('h3', { class: 'insp-section-title' }, 'Label'),
      title,
    ]));

    const summary = el('textarea', { class: 'panel-summary', rows: 3, placeholder: 'Short summary (optional)…' });
    summary.value = node.summary || '';
    summary.addEventListener('change', () => this.#commit(node.id, { summary: summary.value.trim() }));
    frag.append(summary);

    // Structured lesson fields remain visible to an editor for manual repair.
    const aids = el('section', { class: 'panel-learning' });
    aids.append(el('h3', { class: 'insp-section-title' }, 'Lesson fields (read/write)'));
    for (const key of ['explain', 'intuition', 'why_it_matters', 'worked_example', 'misconception']) {
      const box = el('textarea', { class: 'panel-field', rows: 2, 'aria-label': key });
      box.value = Array.isArray(node[key]) ? node[key].join('\n') : (node[key] || '');
      box.addEventListener('change', () => this.#commit(node.id, { [key]: box.value.trim() }));
      aids.append(el('label', { class: 'field-label' }, key), box);
    }
    frag.append(aids);

    const tsRow = el('div', { class: 'panel-row wrap' });
    if (node.start !== null && node.start !== undefined) {
      tsRow.append(el('button', { class: 'btn btn-accent btn-small', onclick: () => this.seek(node) }, `Watch at ${fmtTime(node.start)}`));
    }
    if (videoId) {
      tsRow.append(el('button', {
        class: 'btn btn-small btn-ghost',
        title: 'Use the current playback time of the YouTube tab',
        onclick: async () => {
          const t = await currentVideoTime(videoId);
          if (t === null) toast('Open the video in a YouTube tab first.');
          else this.#commit(node.id, { start: Math.round(t * 10) / 10 });
        },
      }, '⏱ Set to now'));
    }
    if (tsRow.children.length) frag.append(tsRow);

    frag.append(el('h3', { class: 'insp-section-title' }, 'Tone'));
    frag.append(el('div', { class: 'chips' }, TONES.map((tone) => {
      const on = (node.tone || []).includes(tone);
      return el('button', {
        class: `chip ${on ? 'on' : ''} tone-${tone}`,
        onclick: () => this.#commit(node.id, { tone: on ? node.tone.filter((t) => t !== tone) : [...(node.tone || []), tone].slice(-3) }),
      }, tone);
    })));

    const imgBox = el('div', { class: 'panel-image', 'data-section': 'visual' });
    imgBox.append(el('h3', { class: 'insp-section-title' }, 'Visual'));
    if (node.image?.src) {
      imgBox.append(
        el('img', { src: node.image.src, alt: 'Frame from the video' }),
        el('small', {}, `${node.image.source || 'image'}${node.image.t ? ` @ ${fmtTime(node.image.t)}` : ''}`),
      );
    }
    const imgActions = el('div', { class: 'panel-row wrap' });
    if (meta.storyboardSpec && node.start !== null && node.start !== undefined) {
      imgActions.append(el('button', {
        class: 'btn btn-small',
        onclick: async () => {
          try {
            const src = await storyboardFrame(meta.storyboardSpec, meta.duration, node.start + 2);
            this.#commit(node.id, { image: { src, t: node.start + 2, source: 'storyboard' } });
          } catch (err) { toast(err.message, { type: 'error' }); }
        },
      }, '🎞 Storyboard'));
    }
    if (videoId) {
      imgActions.append(el('button', {
        class: 'btn btn-small',
        title: 'Grab the frame currently shown in the YouTube tab',
        onclick: async () => {
          try { this.#commit(node.id, { image: await captureCurrentFrame(videoId) }); }
          catch (err) { toast(err.message, { type: 'error' }); }
        },
      }, '📸 Capture frame'));
    }
    imgActions.append(el('button', {
      class: 'btn btn-small',
      onclick: async () => { const image = await pickImageFile(); if (image) this.#commit(node.id, { image }); },
    }, '⬆ Upload'));
    if (node.image) imgActions.append(el('button', { class: 'btn btn-small btn-ghost', onclick: () => this.#commit(node.id, { image: null }) }, 'Remove'));
    imgBox.append(imgActions);
    frag.append(imgBox);

    const notes = el('textarea', { class: 'panel-notes', rows: 4, placeholder: 'Write your own notes, questions, examples…', 'aria-label': 'My notes' });
    notes.value = node.notes || '';
    let notesTimer;
    notes.addEventListener('input', () => {
      clearTimeout(notesTimer);
      notesTimer = setTimeout(() => {
        if (notes.value !== (node.notes || '')) {
          const firstNote = !node.notes;
          this.#commit(node.id, { notes: notes.value });
          if (firstNote) this.gamify?.track('note_added');
        }
      }, 700);
    });
    frag.append(el('div', { 'data-section': 'notes' }, [el('h3', { class: 'insp-section-title' }, 'My notes'), notes]));

    const list = el('ul', { class: 'panel-links' }, (node.links || []).map((link, i) =>
      el('li', {}, [
        el('a', { href: link.url, target: '_blank', rel: 'noopener noreferrer' }, link.title || link.url),
        el('button', { class: 'icon-btn', title: 'Remove link', onclick: () => this.#commit(node.id, { links: node.links.filter((_, j) => j !== i) }) }, '✕'),
      ]),
    ));
    const urlInput = el('input', { type: 'url', placeholder: 'https://…' });
    const titleInput = el('input', { type: 'text', placeholder: 'Title (optional)' });
    const addLink = () => {
      const url = urlInput.value.trim();
      if (!isHttpUrl(url)) { toast('Enter a valid http(s) link.', { type: 'error' }); return; }
      this.#commit(node.id, { links: [...(node.links || []), { url, title: titleInput.value.trim() }] });
      this.gamify?.track('link_added');
    };
    urlInput.addEventListener('keydown', (e) => e.key === 'Enter' && addLink());
    frag.append(el('div', { 'data-section': 'links' }, [
      el('h3', { class: 'insp-section-title' }, 'Resources'),
      list,
      el('div', { class: 'panel-row link-form' }, [urlInput, titleInput, el('button', { class: 'btn btn-small', onclick: addLink }, 'Add')]),
    ]));

    const structure = el('div', { class: 'panel-row wrap', 'data-section': 'structure' }, [
      el('button', { class: 'btn btn-small', onclick: () => this.#addChild(node) }, '＋ Child (Tab)'),
      node.type !== 'root' ? el('button', { class: 'btn btn-small', onclick: () => this.#addSibling(node) }, '＋ Sibling (Enter)') : null,
      node.type !== 'root' ? el('button', {
        class: 'btn btn-small btn-danger',
        onclick: () => {
          if (!confirm(`Delete “${plain(node.text).slice(0, 60)}” and its children? This cannot be undone.`)) return;
          this.model.remove(this.ids);
        },
      }, '🗑 Delete') : null,
    ]);
    if (this.ids.length === 2) {
      structure.append(el('button', {
        class: 'btn btn-small',
        onclick: () => {
          const label = prompt('Relationship label (e.g. "leads to", "contrasts with")', 'related to');
          if (label !== null) this.model.addEdge(this.ids[0], this.ids[1], label || 'related to');
        },
      }, '⤳ Cross-link'));
    }
    frag.append(structure);

    const ai = el('section', { 'data-section': 'ai' }, [
      el('h3', { class: 'insp-section-title' }, this.api.online ? 'AI refinement' : 'AI refinement · offline'),
      el('input', { type: 'text', class: 'panel-instruction', placeholder: 'Optional instruction, e.g. "add real-world examples"' }),
      el('div', { class: 'panel-row wrap' }, REFINE_ACTIONS.map((action) =>
        el('button', {
          class: 'btn btn-small btn-ai', title: action.hint,
          disabled: this.ids.length < action.min ? true : null,
          onclick: () => runRefine({ api: this.api, model: this.model, action: action.id, ids: this.ids, instruction: ai.querySelector('.panel-instruction').value, gamify: this.gamify, renderer: this.renderer }),
        }, `${action.icon} ${action.label}`),
      )),
    ]);
    frag.append(ai);

    if (node.sources?.length > 1) {
      frag.append(el('h3', { class: 'insp-section-title' }, 'Merged from'),
        el('ul', { class: 'panel-sources' }, node.sources.map((s) => el('li', {}, `${s.videoId || 'map'} ${s.start !== null && s.start !== undefined ? `@ ${fmtTime(s.start)}` : ''}`))));
    }
    frag.append(el('p', { class: 'panel-foot' }, `${(node.children || []).length} children · layer ${node.layer ?? 0} · ${plain(node.text).length} chars`));
    return frag;
  }

  #addChild(node) {
    const child = this.model.addChild(node.id);
    this.renderer.render();
    this.renderer.select(child.id);
    this.renderer.startEdit(child.id);
    this.gamify?.track('node_added');
  }

  #addSibling(node) {
    const sibling = this.model.addSibling(node.id);
    this.renderer.render();
    this.renderer.select(sibling.id);
    this.renderer.startEdit(sibling.id);
    this.gamify?.track('node_added');
  }
}
