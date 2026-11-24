/**
 * Pure helpers for Study Mode set generation.
 *
 * No DOM, no storage, no framework: StudyMode uses these to plan a genuinely
 * new set, exclude already-used learning content, keep stable set identity and
 * preserve the ratings/history that live alongside the current set in
 * progress. The unit tests import the same functions.
 */

const plain = (text) => String(text || '').replace(/^[“"]|[”"]$/g, '').trim();

function hasUsableTeachingMaterial(node) {
  for (const key of ['explain', 'intuition', 'why_it_matters', 'worked_example', 'misconception', 'summary', 'source']) {
    if (String(node[key] ?? '').trim()) return true;
  }
  if ((Array.isArray(node.steps) && node.steps.length)
    || (node.knowledge_check && Object.keys(node.knowledge_check).length)) return true;
  const text = String(node.text ?? '');
  const i = text.indexOf(':');
  return i >= 0 && text.slice(i + 1).trim().length > 0;
}

/**
 * Shared eligibility rule (mirrors backend/app/features/study.py:is_eligible).
 * Explicitly unavailable concepts never enter flashcards, quizzes or New Set.
 * Missing flags stay eligible for legacy / Basic Mode maps. User-authored nodes
 * are preserved in the map but only yield questions with teachable material.
 */
export function isEligibleConcept(node) {
  if (!node || typeof node !== 'object') return false;
  if (node.conceptUnavailable === true || node.available === false || node.lesson_available === false) return false;
  const userAuthored = node.userAuthored === true || node.origin === 'user';
  if (userAuthored && !hasUsableTeachingMaterial(node)) return false;
  return true;
}

/** Normalise a question so punctuation/quote changes cannot smuggle in a duplicate. */
export function normalizeQuestion(text) {
  return String(text || '')
    .toLowerCase()
    .replace(/[\u2018\u2019\u201c\u201d]/g, '"')
    .replace(/[^a-z0-9]+/gi, ' ')
    .trim();
}

/** A unique, stable identity for a generated set (not a random card id). */
export function newSetId() {
  const rand = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
  return `set-${String(rand).replace(/[^a-z0-9]/gi, '').slice(0, 18)}`;
}

/** Concepts that can still be used for a fresh set, in map order. */
export function conceptPool(model, { usedConceptIds = [], focusIds = null } = {}) {
  const used = new Set(usedConceptIds);
  const focus = focusIds ? new Set(focusIds) : null;
  const out = [];
  model.walk((node, parent) => {
    if (node.type !== 'concept' && node.type !== 'detail') return;
    if (!plain(node.text)) return;
    if (!isEligibleConcept(node)) return;
    if (used.has(node.id)) return;
    if (focus && !focus.has(node.id)) return;
    out.push({
      id: node.id,
      text: plain(node.text),
      summary: node.summary || '',
      section: parent ? plain(parent.text) : '',
      start: node.start,
    });
  });
  return out;
}

/** Plan which eligible concepts a new set may use. Never repeats used concepts. */
export function planNewSet({ model, usedConceptIds = [], focusIds = null, cardCount = 14 }) {
  const pool = conceptPool(model, { usedConceptIds, focusIds });
  if (!pool.length) return { ok: false, reason: 'exhausted', selectedIds: [], poolSize: 0 };
  const selectedIds = pool.slice(0, Math.max(1, cardCount)).map((c) => c.id);
  return { ok: true, reason: null, selectedIds, poolSize: pool.length };
}

/** Deterministic Fisher–Yates shuffle so a set number reproduces its question order. */
export function seededShuffle(items, seed = 0) {
  const arr = [...items];
  let state = (Number(seed) || 0) >>> 0 || 0x9e3779b9;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(next() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/** Give every generated card a stable id if the generator did not supply one. */
export function withStableIds(set, salt = '') {
  const idFor = (prefix, index, item) => item?.id
    ? String(item.id)
    : `${prefix}-${salt}-${index}`;
  return {
    ...set,
    flashcards: (set.flashcards || []).map((c, i) => ({ ...c, id: idFor('fc', i, c) })),
    quiz: (set.quiz || []).map((q, i) => ({ ...q, id: idFor('qz', i, q) })),
  };
}

/**
 * Drop questions already asked in an earlier set. Comparison is on normalised
 * question text, so "What is X?" and "what is x" count as the same question.
 * Returns the fresh set plus the normalised keys that should now be remembered.
 */
export function filterFreshQuestions(set, { usedQuestions = [] } = {}) {
  const seen = new Set((usedQuestions || []).map(normalizeQuestion).filter(Boolean));
  const fresh = { flashcards: [], quiz: [], newQuestions: [], provider: set?.provider };
  for (const card of set?.flashcards || []) {
    const key = normalizeQuestion(card.front);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    fresh.flashcards.push(card);
    fresh.newQuestions.push(key);
  }
  for (const question of set?.quiz || []) {
    const key = normalizeQuestion(question.question);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    fresh.quiz.push(question);
    fresh.newQuestions.push(key);
  }
  return fresh;
}

/**
 * Commit a newly generated set without touching Leitner ratings, mastered
 * nodes, quiz results or the learning history. Previous sets stay reviewable.
 */
export function applySet(progress, set, meta) {
  const entry = {
    id: meta.id,
    number: meta.number,
    createdAt: meta.createdAt || Date.now(),
    focusIds: meta.focusIds || null,
    flashcards: set.flashcards.length,
    quiz: set.quiz.length,
    set,
  };
  return {
    ...progress,
    set: { ...set, id: meta.id, number: meta.number, createdAt: entry.createdAt, focusIds: meta.focusIds || null },
    setNumber: meta.number,
    sets: [...(progress.sets || []), entry],
    usedConceptIds: [...new Set([...(progress.usedConceptIds || []), ...(meta.usedConceptIds || [])])],
    usedQuestions: [...new Set([...(progress.usedQuestions || []), ...(meta.newQuestions || [])])],
    // progress.cards / mastered / quizTotal / quizCorrect / explored are deliberately inherited.
  };
}

/** Record a finished quiz against the set it belongs to, keeping a separate history. */
export function recordQuizResult(progress, { setId, correct, total }) {
  const history = [
    ...(progress.quizHistory || []),
    { setId, correct, total, at: Date.now() },
  ];
  return {
    ...progress,
    quizHistory: history,
    quizCorrect: (progress.quizCorrect || 0) + (correct || 0),
    quizTotal: (progress.quizTotal || 0) + (total || 0),
  };
}

/** Switch the visible set to an earlier one without changing any progress. */
export function reviewSet(progress, setId) {
  const entry = (progress.sets || []).find((s) => s.id === setId);
  return entry ? { ...progress, set: { ...entry.set, id: entry.id, number: entry.number } } : progress;
}
