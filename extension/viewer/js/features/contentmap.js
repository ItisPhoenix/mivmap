/**
 * Content map — the video outline and quick recall, in two focused tabs.
 *
 * OUTLINE is a compact numbered revision list: title, one line, section count
 * and duration, then short section rows with a time range. Full section text,
 * key terms and tagged teaching lines live in an expandable detail view so the
 * sidebar never becomes a second reading page.
 *
 * RECALL holds the must-remember prompts with reveal + timestamps and a button
 * to start a quiz from those concepts.
 *
 * Selecting a node on the map highlights its entry here, and the reverse.
 * `outline(map)` stays pure: the Markdown export and the tests use it too.
 */
import { el, fmtTime, plain } from './shared.js';

const PREF_KEY = 'tm-content-map-open';
const TAB_KEY = 'tm-content-map-tab';

/** Plain data for the outline of a map. */
export function outline(map) {
  const root = map?.root || { children: [] };
  const sections = (root.children || []).filter((s) => s.type === 'section');
  const recall = sections.find((s) => s.recall);
  const untag = (node) => {
    const text = plain(node.text);
    return node.tag && text.startsWith(`${node.tag}:`) ? text.slice(node.tag.length + 1).trim() : text;
  };
  const item = (node) => ({ id: node.id, text: plain(node.text), start: node.start ?? null });
  return {
    topic: plain(root.text),
    oneLine: plain(recall?.summary || root.summary || ''),
    recall: recall ? (recall.children || []).filter((c) => !c.oneline).map((c) => ({
      ...item(c),
      question: plain(c.question || ''),
      answer: plain(c.answer || ''),
      source: plain(c.source || ''),
      qa: !!c.qa,
    })) : [],
    sections: sections
      .filter((s) => !s.recall)
      .map((s, i) => {
        const concepts = (s.children || []).filter((c) => c.type === 'concept');
        const terms = [...new Set(concepts.map((c) => plain(c.text).split(':')[0].trim()).filter(Boolean))].slice(0, 5);
        return {
          ...item(s),
          title: plain(s.text),
          n: i + 1,
          end: s.end ?? null,
          summary: plain(s.summary || ''),
          terms: terms.length ? terms : (s.keywords || []).slice(0, 5),
          concepts: concepts.map((c) => ({
            ...item(c),
            items: (c.children || []).filter((d) => d.type === 'detail').map((d) => ({ ...item(d), tag: d.tag || null, text: untag(d) })),
          })),
        };
      }),
  };
}

export function timeRange(start, end) {
  if (start === null || start === undefined) return '';
  return end !== null && end !== undefined && end > start ? `${fmtTime(start)}–${fmtTime(end)}` : fmtTime(start);
}

export class ContentMap {
  constructor({ host, stage, toggle, model, renderer, seek, onStudy = null }) {
    Object.assign(this, { host, stage, toggle, model, renderer, seek, onStudy });
    renderer.addEventListener('select', (e) => this.highlight(e.detail.ids));
    model.addEventListener('load', () => this.render());
    model.addEventListener('change', () => {
      clearTimeout(this.timer);
      this.timer = setTimeout(() => this.render(), 120);
    });
    toggle.addEventListener('click', () => this.setOpen(!this.open));
    let saved = null;
    let tab = null;
    try {
      saved = localStorage.getItem(PREF_KEY);
      tab = localStorage.getItem(TAB_KEY);
    } catch {
      /* storage blocked: open by default */
    }
    this.tab = tab === 'recall' ? 'recall' : 'outline';
    this.setOpen(saved !== '0', { fit: false });
  }

  setOpen(open, { fit = true } = {}) {
    this.open = open;
    this.host.hidden = !open;
    this.stage.classList.toggle('cm-open', open);
    this.toggle.setAttribute('aria-pressed', String(open));
    try {
      localStorage.setItem(PREF_KEY, open ? '1' : '0');
    } catch {
      /* per-viewer convenience only */
    }
    if (fit && this.model.map) requestAnimationFrame(() => this.renderer.fit());
  }

  setTab(tab) {
    this.tab = tab === 'recall' ? 'recall' : 'outline';
    try { localStorage.setItem(TAB_KEY, this.tab); } catch { /* optional */ }
    this.render();
  }

  render() {
    const map = this.model.map;
    if (!map) {
      this.host.replaceChildren();
      return;
    }
    const scroll = this.host.querySelector('.cm-body')?.scrollTop || 0;
    const o = outline(map);
    const duration = Number(map.meta?.duration) || 0;
    const conceptCount = o.sections.reduce((n, s) => n + s.concepts.length, 0);
    const meta = el('div', { class: 'cm-meta' }, [
      el('h2', { class: 'cm-topic-title' }, o.topic),
      o.oneLine ? el('p', { class: 'cm-oneline' }, o.oneLine.replace(/^In one line:\s*/i, '')) : null,
      el('div', { class: 'cm-facts' }, [
        el('span', {}, `${o.sections.length} sections`),
        el('span', {}, `${conceptCount} concepts`),
        duration ? el('span', {}, `◷ ${fmtTime(duration)}`) : null,
      ]),
    ]);
    const tabs = el('div', { class: 'cm-tabs', role: 'tablist' }, [
      el('button', {
        class: `cm-tab ${this.tab === 'outline' ? 'on' : ''}`,
        role: 'tab', 'aria-selected': String(this.tab === 'outline'),
        onclick: () => this.setTab('outline'),
      }, 'Outline'),
      el('button', {
        class: `cm-tab ${this.tab === 'recall' ? 'on' : ''}`,
        role: 'tab', 'aria-selected': String(this.tab === 'recall'),
        onclick: () => this.setTab('recall'),
      }, 'Recall'),
    ]);
    const body = el('div', { class: 'cm-body' },
      this.tab === 'recall'
        ? [this.#recallView(o)]
        : [meta, this.#outlineView(o)]);

    this.host.replaceChildren(
      el('header', { class: 'cm-head' }, [
        el('strong', {}, 'Video outline'),
        el('button', { class: 'icon-btn', title: 'Hide the outline', 'aria-label': 'Hide the outline', onclick: () => this.setOpen(false) }, '×'),
      ]),
      tabs,
      body,
    );
    body.scrollTop = scroll;
    this.highlight([...this.renderer.selection]);
  }

  #outlineView(o) {
    const focus = (id, play) => {
      const shown = this.renderer.reveal(id);
      this.renderer.select(shown);
      this.renderer.centerOn(shown, { zoom: Math.max(this.renderer.view.k, 0.8) });
      if (play) this.#play(id);
    };
    const nodeButton = (id, children, play = false, cls = 'cm-entry') =>
      el('button', { class: cls, onclick: () => focus(id, play) }, children);

    const jumpList = el('ol', { class: 'cm-jump-list cm-sections' }, o.sections.map((s) => {
      const range = timeRange(s.start, s.end);
      return el('li', {
        class: 'cm-jump-item cm-section',
        dataset: { nodeId: s.id, range },
      }, [
        el('button', {
          class: 'cm-entry',
          'aria-label': `${s.n}. ${s.title}${range ? `, ${range}` : ''} — show and play`,
          onclick: () => focus(s.id, true),
        }, `${s.n}. ${s.title}`),
      ]);
    }));

    const details = el('div', { class: 'cm-details' }, o.sections.map((s) => {
      const block = el('details', { class: 'cm-section-detail' }, [
        el('summary', {}, [
          el('span', { class: 'cm-num' }, String(s.n).padStart(2, '0')),
          s.title,
        ]),
        el('div', { class: 'cm-detail-body' }, [
          s.summary ? el('p', { class: 'cm-summary' }, s.summary) : null,
          s.terms.length ? el('p', { class: 'cm-terms' }, [el('span', {}, 'Key terms: '), s.terms.join(' · ')]) : null,
          el('ul', { class: 'cm-concepts' }, s.concepts.map((c) => {
            const [term, ...rest] = c.text.split(':');
            const li = el('li', { class: 'cm-concept', dataset: { nodeId: c.id } }, [
              nodeButton(c.id, rest.length ? [el('b', {}, `${term.trim()}:`), ` ${rest.join(':').trim()}`] : c.text),
            ]);
            if (c.items.length) {
              li.append(el('ul', { class: 'cm-items' }, c.items.map((d) =>
                el('li', { class: 'cm-item', dataset: { nodeId: d.id } }, [
                  nodeButton(d.id, [d.tag ? el('span', { class: `cm-tag tag-${d.tag.toLowerCase()}` }, d.tag) : null, d.text]),
                ]),
              )));
            }
            return li;
          })),
        ]),
      ]);
      return block;
    }));

    return el('div', { class: 'cm-outline' }, [
      el('div', { class: 'cm-block cm-recall-link' }, [
        el('button', { class: 'cm-entry', onclick: () => this.setTab('recall') }, 'Quick recall →'),
      ]),
      el('section', { class: 'cm-block cm-jump' }, [
        el('h3', {}, 'Jump to'),
        jumpList,
      ]),
      el('section', { class: 'cm-block cm-section-details' }, [
        el('h3', {}, 'Section detail'),
        details,
      ]),
    ]);
  }

  #recallView(o) {
    const startQuiz = () => {
      if (this.onStudy) this.onStudy(o.recall.map((r) => r.id));
      else this.host.dispatchEvent(new CustomEvent('study-quiz', { detail: { focusIds: o.recall.map((r) => r.id) } }));
    };
    if (!o.recall.length) {
      return el('div', { class: 'cm-recall-empty' }, [
        el('p', {}, 'This map has no recall prompts yet. You can still quiz yourself on its concepts.'),
        el('button', { class: 'btn btn-accent btn-small', onclick: startQuiz }, '🎓 Start a quiz'),
        el('button', { class: 'btn btn-ghost btn-small', onclick: () => this.setTab('outline') }, '← Back to Outline'),
      ]);
    }
    return el('div', { class: 'cm-recall-view' }, [
      el('p', { class: 'cm-recall-hint' }, 'Try to say each point before revealing it.'),
      el('ul', { class: 'cm-recall-list' }, o.recall.map((r) => {
        const norm = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9]+/gi, ' ').trim();
        const question = r.question || String(r.text || '').replace(/^Q:\s*/i, '');
        const answer = r.answer || '';
        const distinct = answer && norm(question) !== norm(answer);
        const answerEl = el('p', { class: 'cm-recall-answer', hidden: true }, distinct ? answer : 'No distinct answer is saved for this prompt.');
        const reveal = distinct ? el('button', {
          class: 'btn btn-small btn-ghost',
          onclick: () => {
            answerEl.hidden = !answerEl.hidden;
            reveal.textContent = answerEl.hidden ? 'Reveal' : 'Hide';
          },
        }, 'Reveal') : null;
        return el('li', { class: 'cm-recall-item', dataset: { nodeId: r.id } }, [
          el('div', { class: 'cm-recall-row' }, [
            el('button', { class: 'cm-entry cm-recall-cue', onclick: () => this.#focus(r.id) }, question),
            r.start !== null && r.start !== undefined
              ? el('button', { class: 'cm-time', title: 'Play this moment', onclick: () => this.#play(r.id) }, fmtTime(r.start)) : null,
          ]),
          answerEl,
          reveal,
        ]);
      })),
      el('button', { class: 'btn btn-accent btn-small', onclick: startQuiz }, '🎓 Start quiz from recall'),
      el('button', { class: 'btn btn-ghost btn-small', onclick: () => this.setTab('outline') }, '← Back to Outline'),
    ]);
  }

  #focus(id) {
    const shown = this.renderer.reveal(id);
    this.renderer.select(shown);
    this.renderer.centerOn(shown, { zoom: Math.max(this.renderer.view.k, 0.8) });
  }

  #play(id) {
    const node = this.model.get(id);
    if (node && node.start !== null && node.start !== undefined) this.seek(node);
  }

  /** Map → content map: mark the entries of the selected nodes and scroll the last one into view. */
  highlight(ids) {
    const wanted = new Set(ids);
    let last = null;
    for (const li of this.host.querySelectorAll('[data-node-id]')) {
      const on = wanted.has(li.dataset.nodeId);
      li.classList.toggle('cm-active', on);
      if (on && !li.classList.contains('cm-jump-item')) last = li;
    }
    last?.scrollIntoView?.({ block: 'nearest' });
  }
}
