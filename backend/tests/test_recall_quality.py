"""Quick Recall quality regressions: real Q/A, grounding, user-item safety."""
from __future__ import annotations

from app.features.recall import build_recall_pairs, improve_recall, valid_pair
from app.pipeline.builder import compress, make_node


def _concept(cid, text, start, answer, question="What is the difference between a network and the internet?",
             evidence_text="A network connects local devices. The internet links many independent networks worldwide.",
             validated=True):
    node = make_node(text, "concept", 2, start, start + 30,
                     source=evidence_text, topic=text,
                     evidence=[{"start": start, "end": start + 30, "text": evidence_text}])
    node["id"] = cid
    if validated:
        node["lesson_available"] = True
        node["available"] = True
    if question and answer:
        node["knowledge_check"] = {"question": question, "answer": answer, "explanation": evidence_text}
    return node


def _map(recall_children=None, concept=None):
    concept = concept or _concept("c1", "Network vs internet", 60.0,
        "A network connects local devices. The internet links many independent networks worldwide.")
    rec = make_node("Quick recall", "section", 1, 0, 300, source="", recall=True)
    rec["children"] = recall_children if recall_children is not None else []
    return {"meta": {"videoId": "v", "mode": "deep"},
            "root": {"id": "root", "text": "Topic", "type": "root", "children": [
                {"id": "s1", "text": "Section 1", "type": "section", "children": [concept]},
                rec,
            ]}}


def test_repeated_prompt_answer_and_weak_pairs_are_rejected():
    assert not valid_pair("A network is not the internet", "A network is not the internet")
    assert not valid_pair("A network is not the internet", "A network is not the internet!")
    assert not valid_pair("Describe it", "A short answer")  # not a question prompt
    assert not valid_pair("What is a network?", "network", "A network connects devices.")
    assert valid_pair(
        "What is the difference between a network and the internet?",
        "A network connects local devices. The internet links many independent networks worldwide.",
        "A network connects local devices. The internet links many independent networks worldwide.",
    )


def test_pairs_come_from_validated_lessons_and_are_deduplicated():
    duplicate = _concept("c2", "Another name", 90.0,
        "A network connects local devices. The internet links many independent networks worldwide.",
        question="What is the difference between a network and the internet?")
    mm = _map(concept=duplicate)
    pairs = build_recall_pairs(mm, use_llm=False)
    assert len(pairs) == 1
    assert pairs[0]["node"]["id"] == "c2"


def test_unsupported_or_unvalidated_concepts_are_not_used():
    weak = _concept("c1", "Weak", 10.0, "network", validated=False)
    mm = _map(concept=weak)
    assert build_recall_pairs(mm, use_llm=False) == []


def test_improve_recall_replaces_extractive_fragments_but_keeps_user_items():
    source = "A network connects local devices. The internet links many independent networks worldwide."
    fragment = make_node(compress(source, 14), "concept", 2, 61.0, 70.0,
                         source=source, recall=True)
    user = make_node("My own question", "concept", 2, 99.0, 100.0,
                     source="my note", recall=True, notes="authored by me")
    mm = _map(recall_children=[fragment, user])
    report = improve_recall(mm, use_llm=False)
    recall = mm["root"]["children"][1]
    assert report["removed"] == 1 and report["pairs"] == 1
    assert any(c.get("notes") == "authored by me" for c in recall["children"])
    qa = next(c for c in recall["children"] if c.get("qa"))
    assert qa["question"] and qa["answer"]
    assert qa["conceptId"] == "c1" and qa["start"] == 60.0 and qa["source"]
    assert recall["qaCount"] == 1


def test_no_validated_material_shows_honest_unavailable_state():
    mm = _map(concept=_concept("c1", "Weak", 10.0, "", validated=False))
    fragment = make_node(compress("A transcript sentence.", 14), "concept", 2, 1, 2,
                         source="A transcript sentence.", recall=True)
    mm["root"]["children"][1]["children"] = [fragment]
    report = improve_recall(mm, use_llm=False)
    recall = mm["root"]["children"][1]
    assert report["pairs"] == 0
    assert "No validated lesson material" in recall["recallUnavailable"]
    assert all(not c.get("qa") for c in recall["children"])
    assert not any(c["text"] == compress("A transcript sentence.", 14) for c in recall["children"])


def test_maps_without_recall_section_stay_compatible():
    mm = {"meta": {}, "root": {"id": "root", "text": "T", "type": "root", "children": []}}
    assert improve_recall(mm)["unavailable"] == "No Quick recall section."
