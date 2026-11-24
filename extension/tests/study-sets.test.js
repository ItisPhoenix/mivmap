/**
 * Study Mode / New Set regressions — pure logic, no browser required.
 * Run with: node --test extension/tests/*.test.js
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

const {
  applySet, filterFreshQuestions, isEligibleConcept, newSetId, normalizeQuestion,
  planNewSet, recordQuizResult, reviewSet, withStableIds,
} = await import('../viewer/js/features/study-sets.js');
const { localStudySet } = await import('../viewer/js/features/study.js');

const source = readFileSync(new URL('../viewer/js/features/study.js', import.meta.url), 'utf8');

function fakeModel(count = 30) {
  const root = { id: 'root', type: 'root', text: 'Networking', children: [] };
  const concepts = Array.from({ length: count }, (_, i) => ({
    id: `c${i}`, type: 'concept', text: `Term ${i}: Meaning number ${i}`,
    summary: '', start: i * 10, children: [],
  }));
  return {
    root,
    walk(fn) {
      fn(root, null, 0);
      concepts.forEach((c) => fn(c, root, 1));
    },
  };
}

const progress = () => ({
  mapId: 'm1', explored: [], mastered: ['c0'],
  cards: { fc0: { box: 4, seen: 3 } },
  quizTotal: 4, quizCorrect: 3, sets: [], setNumber: 0,
  usedConceptIds: [], usedQuestions: [], quizHistory: [],
});

test('New Set gets a unique identity, not just a new card id', () => {
  const ids = new Set(Array.from({ length: 25 }, () => newSetId()));
  assert.equal(ids.size, 25);
});

test('New Set selects genuinely different eligible concepts', () => {
  const model = fakeModel(30);
  const first = planNewSet({ model, usedConceptIds: [], cardCount: 14 });
  assert.equal(first.ok, true);
  assert.equal(first.selectedIds.length, 14);
  const second = planNewSet({ model, usedConceptIds: first.selectedIds, cardCount: 14 });
  assert.equal(second.ok, true);
  const overlap = second.selectedIds.filter((id) => first.selectedIds.includes(id));
  assert.deepEqual(overlap, [], 'no concept repeats while unique content remains');
});

test('question dedup ignores punctuation, case and quote style', () => {
  assert.equal(normalizeQuestion('What is “ONT”?'), normalizeQuestion('what is ont'));
  const set = {
    flashcards: [
      { id: 'a', front: 'What is “ONT”?', back: 'A' },
      { id: 'b', front: 'what is ont!', back: 'B' },
      { id: 'c', front: 'What is a router?', back: 'C' },
    ],
    quiz: [{ id: 'q', question: 'Which statement best describes “ONT”?', options: [], answer: 0 }],
  };
  const fresh = filterFreshQuestions(set, { usedQuestions: [] });
  assert.deepEqual(fresh.flashcards.map((c) => c.id), ['a', 'c']);
  assert.equal(fresh.quiz.length, 1);

  const later = filterFreshQuestions(set, { usedQuestions: fresh.newQuestions });
  assert.deepEqual(later.flashcards, []);
  assert.deepEqual(later.quiz, []);
});

test('applySet keeps ratings, boxes, mastered nodes and quiz results', () => {
  const before = progress();
  const set = { flashcards: [{ id: 'fc1', front: 'Q', back: 'A' }], quiz: [], provider: 'offline' };
  const after = applySet(before, set, {
    id: 'set-1', number: 1, usedConceptIds: ['c1'], newQuestions: ['q'], focusIds: null,
  });
  assert.deepEqual(after.cards, before.cards, 'Leitner boxes survive');
  assert.deepEqual(after.mastered, before.mastered);
  assert.equal(after.quizCorrect, before.quizCorrect);
  assert.equal(after.quizTotal, before.quizTotal);
  assert.equal(after.setNumber, 1);
  assert.equal(after.sets.length, 1);
  assert.deepEqual(after.usedConceptIds, ['c1']);
  assert.deepEqual(after.usedQuestions, ['q']);
  assert.equal(after.set.id, 'set-1');
});

test('quiz history is separate and accumulates across sets', () => {
  const after = recordQuizResult(progress(), { setId: 'set-1', correct: 4, total: 7 });
  assert.equal(after.quizTotal, 11);
  assert.equal(after.quizCorrect, 7);
  assert.equal(after.quizHistory.length, 1);
  assert.equal(after.quizHistory[0].setId, 'set-1');
});

test('reviewing an earlier set changes only the visible set', () => {
  const setA = { flashcards: [{ id: 'a' }], quiz: [] };
  const setB = { flashcards: [{ id: 'b' }], quiz: [] };
  let p = applySet(progress(), setA, { id: 'set-1', number: 1 });
  p = applySet(p, setB, { id: 'set-2', number: 2 });
  const reviewed = reviewSet(p, 'set-1');
  assert.equal(reviewed.set.flashcards[0].id, 'a');
  assert.equal(reviewed.setNumber, 2);
  assert.deepEqual(reviewed.cards, p.cards);
  assert.equal(reviewed.sets.length, 2);
});

test('exhaustion is reported instead of inventing a new set', () => {
  const model = fakeModel(3);
  const first = planNewSet({ model, usedConceptIds: [], cardCount: 14 });
  const exhausted = planNewSet({ model, usedConceptIds: first.selectedIds, cardCount: 14 });
  assert.equal(exhausted.ok, false);
  assert.equal(exhausted.reason, 'exhausted');
  assert.deepEqual(exhausted.selectedIds, []);
});

test('offline generation produces different questions for different sets', () => {
  const model = fakeModel(30);
  const firstPlan = planNewSet({ model, usedConceptIds: [], cardCount: 14 });
  const secondPlan = planNewSet({ model, usedConceptIds: firstPlan.selectedIds, cardCount: 14 });
  const first = localStudySet(model, 14, firstPlan.selectedIds, 1);
  const second = localStudySet(model, 14, secondPlan.selectedIds, 2);
  assert.equal(first.flashcards.length, 14);
  assert.notDeepEqual(first.flashcards.map((c) => c.front), second.flashcards.map((c) => c.front));
  assert.ok(!first.flashcards.some((c) => second.flashcards.some((d) => d.front === c.front)), 'no repeated question');
  assert.notDeepEqual(first.quiz.map((q) => q.options), second.quiz.map((q) => q.options));
});

test('stable ids are assigned when a generator omits them', () => {
  const set = withStableIds({ flashcards: [{ front: 'Q' }], quiz: [{ question: 'Q?' }] }, 'salt');
  assert.match(set.flashcards[0].id, /^fc-salt-0$/);
  assert.match(set.quiz[0].id, /^qz-salt-0$/);
});

test('New Set never reloads the page or resets Leitner progress in source', () => {
  assert.ok(!/location\.reload|location\.href|window\.location|\.submit\(/.test(source));
  assert.ok(!/<form/.test(source));
  assert.ok(!/progress\.cards\s*=\s*\{\}/.test(source));
  assert.match(source, /if \(this\.generating\) return false/);
  assert.match(source, /planNewSet\(/);
  assert.match(source, /filterFreshQuestions\(/);
});

test('progress view never stringifies a missing control into "null"', () => {
  assert.match(source, /host\.append\(\.\.\.parts\);/);
  assert.ok(!/Review previous sets'\)\s*\n\s*: null,/.test(source));
});

// --- D5: unavailable / generated filler never enters Study Mode ------------
const eligibilityNodes = () => ({
  valid: { id: 'v', type: 'concept', text: 'Node: a network device' },
  legacy: { id: 'l', type: 'concept', text: 'Bit: a binary digit' },
  filler: { id: 'f', type: 'concept', text: 'Interesting Scenario', conceptUnavailable: true, lesson_available: false },
  rejected: { id: 'r', type: 'concept', text: 'Checksum: verifies data', available: false },
  userNoMaterial: { id: 'u', type: 'concept', text: 'My note', userAuthored: true },
  userWithMaterial: { id: 'um', type: 'concept', text: 'My note', origin: 'user', explain: 'A user supplied explanation.' },
});

test('eligibility matches the backend rule for all node states', () => {
  const nodes = eligibilityNodes();
  assert.equal(isEligibleConcept(nodes.valid), true);
  assert.equal(isEligibleConcept(nodes.legacy), true, 'missing flags keep legacy maps working');
  assert.equal(isEligibleConcept(nodes.filler), false);
  assert.equal(isEligibleConcept(nodes.rejected), false);
  assert.equal(isEligibleConcept(nodes.userNoMaterial), false);
  assert.equal(isEligibleConcept(nodes.userWithMaterial), true);
});

test('New Set never selects generated filler or rejected lessons', () => {
  const nodes = Object.values(eligibilityNodes());
  const root = { id: 'root', type: 'root', text: 'R', children: nodes };
  const model = {
    root,
    walk(fn) { fn(root, null, 0); nodes.forEach((n) => fn(n, root, 1)); },
  };
  const plan = planNewSet({ model, cardCount: 14 });
  assert.deepEqual(plan.selectedIds, ['v', 'l', 'um'], 'excluded nodes never enter New Set');
  const focused = planNewSet({ model, focusIds: ['f', 'r'], cardCount: 14 });
  assert.equal(focused.ok, false, 'focus ids cannot resurrect excluded nodes');
});

test('offline cards and quizzes skip unavailable concepts', () => {
  const nodes = Object.values(eligibilityNodes());
  const root = { id: 'root', type: 'root', text: 'R', children: nodes };
  const model = {
    root,
    walk(fn) { fn(root, null, 0); nodes.forEach((n) => fn(n, root, 1)); },
  };
  const set = localStudySet(model, 14, null, 7);
  const ids = new Set([...set.flashcards, ...set.quiz].map((c) => c.nodeId));
  assert.equal(ids.has('f'), false);
  assert.equal(ids.has('r'), false);
  assert.equal(ids.has('u'), false);
});
