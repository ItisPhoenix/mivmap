/**
 * Personalized Study Mode.
 *  • Flashcards with Leitner-box spaced repetition (box 1 = see often, box 5 = mastered).
 *  • Multiple-choice quizzes with explanations and "watch this moment" links.
 *  • Mastered nodes get a hand-drawn ✓ on the map; progress feeds gamification.
 *
 * A "New set" advances an eligible-concept cursor instead of regenerating the
 * same first cards. Set identity, used-question history and previous sets are
 * persisted alongside (never instead of) the Leitner boxes and quiz results.
 */
import { getProgress, saveProgress } from '../lib/storage.js';
import { el, fmtTime, modal, plain, toast, truncateText } from './shared.js';
import {
  applySet, filterFreshQuestions, isEligibleConcept, newSetId, planNewSet,
  recordQuizResult, reviewSet, seededShuffle, withStableIds,
} from './study-sets.js';

const CARD_COUNT = 14;

export class StudyMode {
  constructor({ api, model, renderer, seek, gamify }) {
    Object.assign(this, { api, model, renderer, seek, gamify });
    this.generating = false;
    this.currentTab = 'cards';
  }

  async loadProgress() {
    this.progress = await getProgress(this.model.map.id);
    this.progress.usedConceptIds = this.progress.usedConceptIds || [];
    this.progress.usedQuestions = this.progress.usedQuestions || [];
    this.progress.sets = this.progress.sets || [];
    this.progress.setNumber = this.progress.setNumber || this.progress.sets.length;
    this.renderer.setMastered(this.progress.mastered || []);
    return this.progress;
  }

  async open({ focusIds = null, tab = 'cards' } = {}) {
    await this.loadProgress();
    const { body, close } = modal('🎓 Study mode', { wide: true });
    this.body = body;
    this.close = close;
    if (!this.progress.set || focusIds) await this.#generate({ focusIds, initial: true });
    this.#tabs(tab);
  }

  /** Loading / empty / failed states that never discard the current set. */
  #state(icon, title, detail, { error = false, action = null } = {}) {
    const children = [
      el('div', { class: `study-state ${error ? 'study-state-error' : ''}` }, [
        el('div', { class: error ? 'study-state-icon' : 'scribble-spinner' }, error ? icon : null),
        el('h3', {}, title),
        el('p', {}, detail),
        action,
      ]),
    ];
    this.body.replaceChildren(...children);
  }

  #backToStudy() {
    this.#tabs(this.currentTab);
  }

  async #generate({ focusIds = null, initial = false } = {}) {
    if (this.generating) return false; // guards double-click and concurrent requests
    this.generating = true;
    const tab = this.currentTab;
    let completed = false;
    let set;
    try {
      this.#state('↻', initial ? 'Writing flashcards and quiz questions…' : 'Writing a new set…',
        'Choosing unseen concepts from this mind map.');

      const plan = planNewSet({
        model: this.model,
        usedConceptIds: this.progress.usedConceptIds,
        focusIds: focusIds ?? this.progress.set?.focusIds ?? null,
        cardCount: CARD_COUNT,
      });
      if (!plan.ok) {
        this.#showExhausted();
        return false;
      }

      const number = (this.progress.setNumber || 0) + 1;
      const offline = () => localStudySet(this.model, CARD_COUNT, plan.selectedIds, number);
      let raw;
      try {
        raw = this.api.online
          ? await this.api.study(this.model.toJSON(), CARD_COUNT, plan.selectedIds)
          : offline();
      } catch (err) {
        toast(`Backend study generation failed (${err.message}); using offline cards.`, { type: 'error' });
        raw = offline();
      }

      set = withStableIds(raw || {}, `${number}-${Date.now()}`);
      const fresh = filterFreshQuestions(set, { usedQuestions: this.progress.usedQuestions });
      if (!fresh.flashcards.length && !fresh.quiz.length) {
        this.#showExhausted();
        return false;
      }

      this.progress = applySet(
        this.progress,
        { flashcards: fresh.flashcards, quiz: fresh.quiz, provider: set.provider },
        {
          id: newSetId(), number, createdAt: Date.now(),
          focusIds: focusIds ?? this.progress.set?.focusIds ?? null,
          usedConceptIds: plan.selectedIds,
          newQuestions: fresh.newQuestions,
        },
      );
      await saveProgress(this.progress);
      completed = true;
      set = { number, count: fresh.flashcards.length, quiz: fresh.quiz.length };
      return true;
    } catch (err) {
      this.#state('!', 'Could not create a new set', err?.message || 'The existing set was kept unchanged.', {
        error: true,
        action: el('button', { class: 'btn', onclick: () => this.#backToStudy() }, 'Back to study'),
      });
      return false;
    } finally {
      this.generating = false;
      // Render only after the lock is released, otherwise New Set stays disabled.
      if (completed) {
        this.currentTab = tab;
        this.#tabs(tab);
        if (!initial) toast(`Set ${set.number} ready — ${set.count} cards, ${set.quiz} quiz questions.`, { type: 'success' });
      }
    }
  }

  #showExhausted() {
    const history = this.progress.sets || [];
    this.#state('✓', 'No new unique questions left',
      `All ${this.progress.usedConceptIds.length} eligible concepts have been used. Previous sets and their progress are unchanged.`, {
        action: el('div', { class: 'panel-row center' }, [
          history.length ? el('button', { class: 'btn', onclick: () => this.#reviewSets() }, 'Review previous sets') : null,
          el('button', { class: 'btn btn-ghost', onclick: () => this.#backToStudy() }, 'Back to current set'),
        ]),
      });
  }

  #tabs(active) {
    this.currentTab = active;
    const { flashcards = [], quiz = [] } = this.progress.set || {};
    const number = this.progress.setNumber || 1;
    const newSet = el('button', {
      class: 'btn btn-small btn-ghost push-right',
      disabled: this.generating ? true : null,
      onclick: () => this.#generate({ initial: false }),
    }, this.generating ? 'Creating…' : '↻ New set');
    const tabs = el('div', { class: 'tabs' }, [
      el('button', { class: `tab ${active === 'cards' ? 'on' : ''}`, onclick: () => this.#tabs('cards') }, `Flashcards (${flashcards.length})`),
      el('button', { class: `tab ${active === 'quiz' ? 'on' : ''}`, onclick: () => this.#tabs('quiz') }, `Quiz (${quiz.length})`),
      el('button', { class: `tab ${active === 'progress' ? 'on' : ''}`, onclick: () => this.#tabs('progress') }, 'Progress'),
      newSet,
    ]);
    const content = el('div', { class: 'study-content' }, [
      el('p', { class: 'study-set-meta' }, `Set ${number} · ${flashcards.length} cards · ${quiz.length} quiz questions`),
    ]);
    this.body.replaceChildren(tabs, content);
    if (active === 'cards') this.#cards(content);
    else if (active === 'quiz') this.#quiz(content);
    else this.#progressView(content);
  }

  // --- flashcards ----------------------------------------------------------
  #dueCards() {
    const cards = this.progress.set?.flashcards || [];
    const boxes = this.progress.cards || {};
    return cards
      .map((c) => ({ c, box: boxes[c.id]?.box || 1, seen: boxes[c.id]?.seen || 0 }))
      .filter((x) => x.box < 5)
      .sort((a, b) => a.box - b.box || a.seen - b.seen || cards.indexOf(a.c) - cards.indexOf(b.c));
  }

  #nextCard() {
    return this.#dueCards()[0]?.c || null;
  }

  #cards(host) {
    const card = this.#nextCard();
    if (!card) {
      host.append(el('div', { class: 'study-done' }, [
        '🏆 All cards mastered! ',
        el('button', { class: 'btn', onclick: () => this.#tabs('quiz') }, 'Take the quiz'),
      ]));
      return;
    }
    const cards = this.progress.set?.flashcards || [];
    const position = cards.indexOf(card) + 1;
    const box = this.progress.cards[card.id]?.box || 1;
    const face = el('div', { class: 'flashcard', tabindex: 0, role: 'button', 'aria-label': 'Flip card' }, [
      el('div', { class: 'flash-front' }, card.front),
      el('div', { class: 'flash-back' }, card.back),
    ]);
    const flip = () => face.classList.toggle('flipped');
    face.addEventListener('click', flip);
    face.addEventListener('keydown', (e) => (e.key === ' ' || e.key === 'Enter') && (e.preventDefault(), flip()));

    const grade = async (delta) => {
      const entry = this.progress.cards[card.id] || { box: 1, seen: 0 };
      entry.box = delta < 0 ? 1 : Math.min(5, entry.box + delta);
      entry.seen += 1;
      this.progress.cards[card.id] = entry;
      if (delta > 0) this.gamify?.track('flashcard_correct');
      if (entry.box >= 4 && card.nodeId && !this.progress.mastered.includes(card.nodeId)) {
        this.progress.mastered.push(card.nodeId);
        this.renderer.setMastered(this.progress.mastered);
      }
      await saveProgress(this.progress);
      this.#tabs('cards');
    };

    host.append(
      el('p', { class: 'card-count' }, `Card ${position} of ${cards.length}`),
      el('div', { class: 'leitner' }, [1, 2, 3, 4, 5].map((b) => el('span', { class: `box ${b === box ? 'on' : ''}`, title: b === 1 ? 'Seen often' : b === 5 ? 'Mastered' : `Review interval ${b}` }, `Box ${b}`))),
      face,
      el('p', { class: 'hint' }, 'Click the card (or press Space) to reveal the answer.'),
      el('div', { class: 'panel-row center' }, [
        el('button', { class: 'btn btn-danger', onclick: () => grade(-1) }, '↺ Again'),
        el('button', { class: 'btn', onclick: () => grade(1) }, '✓ Good'),
        el('button', { class: 'btn btn-accent', onclick: () => grade(2) }, '★ Easy'),
      ]),
      this.#nodeLinks(card),
    );
    face.focus();
  }

  #nodeLinks(item) {
    if (!item.nodeId || !this.model.get(item.nodeId)) return el('span');
    const node = this.model.get(item.nodeId);
    return el('div', { class: 'panel-row center subtle' }, [
      el('button', {
        class: 'btn btn-ghost btn-small',
        onclick: () => {
          this.close();
          this.renderer.reveal(node.id);
          this.renderer.select(node.id);
          this.renderer.centerOn(node.id);
        },
      }, '🧠 Show in map'),
      node.start !== null && node.start !== undefined ? el('button', { class: 'btn btn-ghost btn-small', onclick: () => this.seek(node) }, `▶ Watch ${fmtTime(node.start)}`) : null,
    ]);
  }

  // --- quiz ------------------------------------------------------------------
  #quiz(host, index = 0, score = 0) {
    const quiz = this.progress.set?.quiz || [];
    if (!quiz.length) {
      host.append(el('p', {}, 'Not enough concepts for a quiz yet.'));
      return;
    }
    if (index >= quiz.length) {
      const pct = Math.round((score / quiz.length) * 100);
      this.progress = recordQuizResult(this.progress, {
        setId: this.progress.set?.id, correct: score, total: quiz.length,
      });
      saveProgress(this.progress);
      host.replaceChildren(
        el('div', { class: 'study-done' }, [
          el('div', { class: 'score-burst' }, `${pct}%`),
          el('p', {}, `${score} / ${quiz.length} correct${pct >= 80 ? ' — excellent!' : pct >= 50 ? ' — keep going!' : ' — review the highlighted nodes.'}`),
          el('button', { class: 'btn', onclick: () => this.#tabs('quiz') }, 'Retry'),
        ]),
      );
      return;
    }
    const q = quiz[index];
    const feedback = el('div', { class: 'quiz-feedback' });
    const options = q.options.map((option, i) =>
      el('button', {
        class: 'quiz-option',
        onclick: async () => {
          const correct = i === q.answer;
          options.forEach((b, j) => {
            b.disabled = true;
            b.classList.toggle('right', j === q.answer);
            b.classList.toggle('wrong', j === i && !correct);
          });
          this.progress.quizTotal = (this.progress.quizTotal || 0) + 1;
          if (correct) {
            this.progress.quizCorrect = (this.progress.quizCorrect || 0) + 1;
            this.gamify?.track('quiz_correct');
          }
          await saveProgress(this.progress);
          feedback.replaceChildren(
            el('strong', {}, correct ? '✓ Correct! ' : '✗ Not quite. '),
            q.explanation || '',
            this.#nodeLinks(q),
            el('button', { class: 'btn', onclick: () => this.#quiz(host.replaceChildren() || host, index + 1, score + (correct ? 1 : 0)) }, index + 1 < quiz.length ? 'Next →' : 'See score'),
          );
        },
      }, option),
    );
    host.replaceChildren(
      el('p', { class: 'quiz-count' }, `Question ${index + 1} of ${quiz.length}`),
      el('h3', { class: 'quiz-question' }, q.question),
      el('div', { class: 'quiz-options' }, options),
      feedback,
    );
  }

  // --- progress & review ----------------------------------------------------
  #progressView(host) {
    const cards = this.progress.set?.flashcards || [];
    const boxes = [1, 2, 3, 4, 5].map((b) => cards.filter((c) => (this.progress.cards[c.id]?.box || 1) === b).length);
    const explored = this.progress.explored?.length || 0;
    const total = this.model.nodes().length;
    const due = this.#dueCards().length;
    const history = this.progress.quizHistory || [];
    const parts = [
      el('div', { class: 'progress-grid' }, [
        stat('Explored nodes', `${explored}/${total}`, explored / Math.max(total, 1)),
        stat('Mastered concepts', String(this.progress.mastered.length), this.progress.mastered.length / Math.max(cards.length, 1)),
        stat('Cards due', String(due), due / Math.max(cards.length, 1)),
        stat('Quiz accuracy', this.progress.quizTotal ? `${Math.round((this.progress.quizCorrect / this.progress.quizTotal) * 100)}%` : '—', this.progress.quizTotal ? this.progress.quizCorrect / this.progress.quizTotal : 0),
        stat('Completed sets', String((this.progress.sets || []).length), 1),
      ]),
      el('h4', {}, 'Leitner boxes'),
      el('div', { class: 'leitner big' }, boxes.map((n, i) => el('span', { class: 'box', style: `--fill:${n / Math.max(cards.length, 1)}` }, `Box ${i + 1}: ${n}`))),
      el('h4', {}, 'Quiz history'),
      history.length
        ? el('ul', { class: 'study-history' }, history.slice(-8).reverse().map((h) => el('li', {}, `Set ${this.#setNumber(h.setId)} — ${h.correct}/${h.total} correct`)))
        : el('p', { class: 'panel-note' }, 'No quiz results yet.'),
    ];
    // Native append() stringifies null/undefined: only append real nodes.
    if ((this.progress.sets || []).length > 1) {
      parts.push(el('button', { class: 'btn btn-ghost btn-small push-right', onclick: () => this.#reviewSets() }, 'Review previous sets'));
    }
    host.append(...parts);
  }

  #setNumber(id) {
    return (this.progress.sets || []).find((s) => s.id === id)?.number || '?';
  }

  async #reviewSets() {
    const sets = [...(this.progress.sets || [])].reverse();
    const list = el('div', { class: 'study-history' }, sets.map((s) =>
      el('button', {
        class: 'study-history-item',
        onclick: async () => {
          this.progress = reviewSet(this.progress, s.id);
          await saveProgress(this.progress);
          this.#tabs(this.currentTab);
        },
      }, `Set ${s.number} · ${s.flashcards} cards · ${s.quiz} questions${s.id === this.progress.set?.id ? ' · current' : ''}`),
    ));
    this.body.replaceChildren(
      el('div', { class: 'tabs' }, [
        el('button', { class: 'tab on', onclick: () => this.#backToStudy() }, '← Back to study'),
      ]),
      el('div', { class: 'study-content' }, [el('h3', {}, 'Previous sets'), list]),
    );
  }
}

function stat(label, value, ratio) {
  return el('div', { class: 'stat' }, [
    el('div', { class: 'stat-value' }, value),
    el('div', { class: 'stat-label' }, label),
    el('div', { class: 'sketch-bar' }, el('span', { style: `width:${Math.round(Math.min(1, ratio) * 100)}%` })),
  ]);
}

/** Offline generation mirroring backend/app/features/study.py fallbacks. */
export function localStudySet(model, count = 12, focusIds = null, seed = 0) {
  const focus = focusIds ? new Set(focusIds) : null;
  const concepts = [];
  model.walk((node, parent) => {
    if (focus && !focus.has(node.id)) return;
    if (['concept', 'detail'].includes(node.type) && plain(node.text) && isEligibleConcept(node)) {
      concepts.push({ node, section: parent ? plain(parent.text) : '' });
    }
  });
  const split = ({ node, section }) => {
    const text = plain(node.text);
    if (text.includes(':')) return [text.split(':')[0].trim(), text.slice(text.indexOf(':') + 1).trim()];
    return [text, node.summary || section];
  };
  const pairs = concepts.map((c) => [c, ...split(c)]).filter(([, , meaning]) => meaning);
  const id = () => crypto.randomUUID().slice(0, 8);
  const flashcards = pairs.slice(0, count).map(([c, term, meaning]) => ({ id: id(), front: `What is meant by “${term}”?`, back: meaning, nodeId: c.node.id, start: c.node.start }));
  const quiz = [];
  for (const [c, term, meaning] of pairs) {
    const distractors = pairs.filter(([, t, m]) => t !== term && m !== meaning).map(([, , m]) => m);
    if (distractors.length < 3) break;
    const options = seededShuffle(distractors, seed).slice(0, 3).concat(meaning);
    const ordered = seededShuffle(options, seed + 1);
    quiz.push({ id: id(), question: `Which statement best describes “${term}”?`, options: ordered.map((o) => truncateText(o, 140)), answer: ordered.indexOf(meaning), explanation: meaning, nodeId: c.node.id, start: c.node.start });
    if (quiz.length >= Math.max(4, count / 2)) break;
  }
  return { flashcards, quiz, provider: 'offline' };
}
