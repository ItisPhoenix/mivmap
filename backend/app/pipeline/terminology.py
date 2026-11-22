"""Concept terminology, fragment rejection and externally-sourced accuracy notes.

This module is deliberately small and data-driven. It exists because the generic
TF-IDF/noun-chunk extractor produces labels that are *phrases from the caption*
rather than names of things a learner can study ("Local Area", "Tower", "Cables",
"Sign of Internet"). It also records the small number of technical corrections
that real exported lessons got wrong, so the correction is reproducible and
explicitly marked as externally supplemented instead of being silently blended
into video evidence.

Nothing in here writes transcript text, timestamps or quotes. Corrections are
applied only to AI-generated teaching fields, never to `evidence` or `source`.
"""
from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Canonical labels. Keys are normalised (lower-case, single spaces).
# An empty value means "reject this label outright".
# ---------------------------------------------------------------------------
TERM_NORMALIZATIONS: dict[str, str] = {
    # fragments seen in real Deep exports
    "local area": "LAN",
    "local area network": "LAN",
    "sign of internet": "Internet (network of networks)",
    "internet": "Internet (network of networks)",
    "geographical categories": "",                 # section heading, not a concept
    "systems interconnection model": "OSI model",
    "open systems interconnection": "OSI model",
    "cdn connected": "CDNs",
    "wi fi speed": "Wi-Fi throughput",
    "wifi speed": "Wi-Fi throughput",
    "waves": "Radio waves (Wi-Fi)",
    "ont plus router": "ONT and router",
    "tower": "Cell tower",
    "cell tower": "Cell tower",
    "cables": "Submarine fibre cables",
    "submarine cables": "Submarine fibre cables",
    "provider network": "",                        # vague duplicate of ISP tiers / transit
    "fibre infrastructure": "Fibre infrastructure",
    "fiber infrastructure": "Fibre infrastructure",
    # consistent casing for acronyms
    "lan": "LAN", "pan": "PAN", "man": "MAN", "wan": "WAN",
    "isps": "ISPs", "isp": "ISP", "cdn": "CDN", "cdns": "CDNs",
    "osi": "OSI model", "osi model": "OSI model",
    "tcp ip": "TCP/IP", "tcp/ip": "TCP/IP", "arpanet": "ARPANET",
    "ont": "ONT", "http": "HTTP", "https": "HTTPS",
    "url": "URL", "uri": "URI",
}

# Generic English words / discourse scraps that are never study topics on their own.
REJECTED_LABELS = {
    "hard work", "working hard", "motivated", "motivation", "researching", "research",
    "length", "power", "travels", "gained", "relates", "maintained", "maintain",
    "person receiving", "x15 room", "short range", "network outside", "hardware level",
    "geographical categories", "provider network", "sign of internet",
    "extra infrastructure", "entire content", "previous videos compared",
    "upcoming videos", "connection at that time", "that time",
}

# Single/two-word labels that need a qualifying technical noun to mean anything.
BARE_GENERIC_LABELS = {
    "tower", "cables", "cable", "provider", "equipment", "infrastructure", "signal",
    "signals", "speed", "model", "layer", "layers", "network", "networks", "connection",
    "connections", "hardware", "software", "support", "storage", "security", "power",
    "length", "travels", "gained", "relates", "maintained", "motivated", "researching",
    "experience", "foundation", "foundations", "overview", "example", "examples",
    "question", "answer", "difference", "similarity", "idea", "point", "way", "lot",
}


# Devanagari speech with English technical words: map the spoken term to a
# stable concept name. Keeps pure-Hindi concepts meaningful without translating
# the transcript (timestamps and original text stay untouched).
HINDI_TERM_HINTS: dict[str, str] = {
    "राउटर": "Router", "राऊटर": "Router",
    "फाइबर": "Fibre cable", "फाइबर केबल": "Fibre cable",
    "इंटरनेट": "Internet (network of networks)",
    "नेटवर्क": "Computer network", "नेटवर्कों": "Computer network",
    "टावर": "Cell tower", "सेलुलर": "Cell tower",
    "लैन": "LAN", "वान": "WAN",
}


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def normalize_label(label: str, context: str = "") -> str:
    """Return the canonical label, or '' when the phrase should be rejected.

    `context` is the concept's own transcript evidence. It is used only to keep
    genuinely different concepts apart: "Local Area" means LAN when the passage
    defines a local network, but a last-mile access region when the passage talks
    about a provider extending fibre into localities.
    """
    # Devanagari has no latin-normalised key; check spoken technical terms first.
    for hint, canonical in HINDI_TERM_HINTS.items():
        if hint in str(label):
            return canonical
    key = norm(label)
    if not key:
        raw = str(label or "").strip()
        if not any(not ch.isascii() and ch.isalpha() for ch in raw):
            return ""
        # A whole caption sentence is not a concept name.
        if len(raw.split()) > 6 or "।" in raw:
            return ""
        return raw
    # A normalized mapping wins over the raw reject list: "Sign of Internet"
    # is a caption fragment, but it maps to a real topic.
    if key in TERM_NORMALIZATIONS:
        mapped = TERM_NORMALIZATIONS[key]
    else:
        if key in REJECTED_LABELS:
            return ""
        mapped = label.strip()
    if not mapped:
        return ""
    if key == "local area" and re.search(r"localit|provider|last mile|backhaul|fibre|fiber", context, re.I):
        return "Local access network (last mile)"
    if key in ("fiber", "fibre"):
        if re.search(r"outage|cut|still print|locally|local printer|local network", context, re.I):
            return "Local network during a fiber outage"
        return "Fibre cable"
    if key in BARE_GENERIC_LABELS and key not in TERM_NORMALIZATIONS:
        return ""
    # A bare generic word that survived as a label is still meaningless.
    if norm(mapped) in BARE_GENERIC_LABELS and norm(mapped) not in TERM_NORMALIZATIONS:
        return ""
    return mapped


def is_meaningful_label(label: str) -> bool:
    """True when a label is allowed to become a concept node.

    Legacy "Term: short meaning" labels are judged by the term before the colon
    so older exports remain compatible with the quality gates.
    """
    head = str(label or "").split(":", 1)[0]
    key = norm(head)
    if not key:
        # Non-Latin teaching languages: reject whole caption sentences, keep
        # short topical names.
        if not any(not ch.isascii() and ch.isalpha() for ch in head):
            return False
        return len(head.split()) <= 6 and "।" not in head
    if key in REJECTED_LABELS:
        return False
    if key in BARE_GENERIC_LABELS and key not in TERM_NORMALIZATIONS:
        return False
    # Transcript scaffolding: a sentence fragment used as a name.
    if re.search(r"\b(?:we|you|they|it|this|that|there|here|what|why|how|would|could|should|"
                 r"going to|let's|we're|i think|maybe|right now|okay)\b", key):
        return False
    if len(key.split()) > 5:
        return False
    return True


# ---------------------------------------------------------------------------
# Acronym expansions (externally supplemented, standard terminology).
# ---------------------------------------------------------------------------
ACRONYM_EXPANSIONS: dict[str, str] = {
    "lan": "Local Area Network",
    "pan": "Personal Area Network",
    "man": "Metropolitan Area Network",
    "wan": "Wide Area Network",
    "arpanet": "Advanced Research Projects Agency Network",
    "osi model": "Open Systems Interconnection",
    "osi": "Open Systems Interconnection",
    "tcp/ip": "Transmission Control Protocol / Internet Protocol",
    "cdn": "Content Delivery Network",
    "cdns": "Content Delivery Networks",
    "isp": "Internet Service Provider",
    "isps": "Internet Service Providers",
    "ont": "Optical Network Terminal",
    "ftth": "Fibre To The Home",
    "http": "Hypertext Transfer Protocol",
    "url": "Uniform Resource Locator",
    "uri": "Uniform Resource Identifier",
}


def acronym_expansion(topic: str) -> str:
    return ACRONYM_EXPANSIONS.get(norm(topic), "")


# ---------------------------------------------------------------------------
# Known technical corrections. These are used ONLY when the video's own words
# are insufficient or inaccurate. Every note is marked external.
# References: ISO/IEC 7498-1 (OSI); RFC 1122 / RFC 675 (internetworking, TCP/IP);
# ARPANET history: Advanced Research Projects Agency (ARPA/DARPA), first
# host-to-host message 29 Oct 1969, TCP/IP flag day 1 Jan 1983, decommissioned 1990.
# ---------------------------------------------------------------------------
# Real reference titles and URLs. Kept here so every external note carries a
# citable source instead of a vague "standard terminology" phrase.
REFERENCE_URLS: dict[str, tuple[str, str]] = {
    "Generic networking terminology; RFC 1122 (Requirements for Internet Hosts)": (
        "RFC 1122: Requirements for Internet Hosts - Communication Layers",
        "https://www.rfc-editor.org/rfc/rfc1122"),
    "RFC 1122; standard Internet terminology": (
        "RFC 1122: Requirements for Internet Hosts - Communication Layers",
        "https://www.rfc-editor.org/rfc/rfc1122"),
    "ARPA/DARPA ARPANET history; IEEE milestone records": (
        "ARPANET - Wikipedia (history, ARPA/DARPA, 1969, 1983, 1990)",
        "https://en.wikipedia.org/wiki/ARPANET"),
    "Standard cellular network and submarine-cable architecture": (
        "Submarine Cable Map (Telegeography) - global cable routes",
        "https://www.submarinecablemap.com/"),
    "Standard submarine cable infrastructure references": (
        "Submarine Cable Map (Telegeography) - global cable routes",
        "https://www.submarinecablemap.com/"),
    "ISO/IEC 7498-1; RFC 1122": (
        "ISO/IEC 7498-1:1994 - Open Systems Interconnection, Basic Reference Model",
        "https://www.iso.org/standard/37097.html"),
    "RFC 1122": (
        "RFC 1122: Requirements for Internet Hosts - Communication Layers",
        "https://www.rfc-editor.org/rfc/rfc1122"),
    "Standard Wi-Fi / radio-link terminology": (
        "IEEE 802.11 - Wireless LAN Medium Access Control and Physical Layer Specifications",
        "https://standards.ieee.org/ieee/802.11/6981/"),
    "Generic access-network terminology": (
        "RFC 1122: Requirements for Internet Hosts - Communication Layers",
        "https://www.rfc-editor.org/rfc/rfc1122"),
    "Generic networking terminology": (
        "IEEE 802.3 / IEEE 802.11 networking standards overview",
        "https://standards.ieee.org/"),
}


def _note(claim: str, correction: str, reference: str, required: bool = False) -> dict[str, Any]:
    title, url = REFERENCE_URLS.get(reference, (reference, ""))
    return {
        "video_claim": claim,
        "correction": correction,
        "basis": reference,
        "title": title,
        "url": url,
        "source": "external-supplement",
        "supplemented": True,
        "required": required,  # True when the video claim is inaccurate and must be corrected
    }


TECHNICAL_CORRECTIONS: dict[str, dict[str, Any]] = {
    "wan": {
        "notes": [
            _note("A WAN is described as 'a type of internet' and the Internet as the collection of all WANs.",
                  "A WAN (Wide Area Network) is a network classified by geographic scale. The Internet is one specific global internetwork that uses TCP/IP to interconnect many networks. The word 'internet' is a shortening of 'internetwork', not 'International Network'; a private WAN is not automatically part of the Internet.",
                  "Generic networking terminology; RFC 1122 (Requirements for Internet Hosts)",
                  required=True),
        ],
        # Targeted text fixes applied only to AI teaching fields, never to evidence.
        "replacements": [
            (r"\b(?:a\s+)?WAN(?:\s*\([^)]*\))?\s+is a type of internet\b",
             "WAN is a wide-area network (the Internet is one global internetwork built from many networks)"),
            (r"\btype of internet, meaning an interconnected network of networks\b",
             "internetwork, meaning a network of networks"),
            (r"\bfull form is (?:obviously )?International Network\b",
             "name comes from 'internetwork', not 'International Network'"),
        ],
    },
    "internet (network of networks)": {
        "notes": [
            _note("The video derives 'Internet' from 'International Network'.",
                  "'Internet' is a contraction of 'internetwork' (a network of networks), not of 'International Network'. The Internet is the global system of interconnected computer networks that uses the TCP/IP protocol suite.",
                  "RFC 1122; standard Internet terminology", required=True),
        ],
        "replacements": [
            (r"\bfull form is (?:obviously )?International Network\b",
             "name comes from 'internetwork', not 'International Network'"),
            (r"\binternational network, which connects the whole world\b",
             "global internetwork (a network of networks) that connects the world"),
        ],
    },
    "arpanet": {
        "notes": [
            _note("The speaker is unsure of the expansion ('Association... maybe Agency Network').",
                  "ARPANET expands to Advanced Research Projects Agency Network. It was established by ARPA (later DARPA) of the US Department of Defense; the first hosts were connected in 1969, TCP/IP replaced NCP on 1 January 1983, and ARPANET was decommissioned in 1990. It was a research network, not a public service.",
                  "ARPA/DARPA ARPANET history; IEEE milestone records", required=True),
        ],
        "replacements": [
            (r"\bthe acronym, is ARPANET\b", "acronym is ARPANET, Advanced Research Projects Agency Network"),
        ],
        "prefix_explain": (
            "ARPANET (Advanced Research Projects Agency Network) was the first wide-area "
            "packet-switched network, established by ARPA (later DARPA) of the US Department "
            "of Defense with its first hosts connected in 1969. "
        ),
    },
    "cell tower": {
        "notes": [
            _note("International traffic is described as going by fibre 'to the tower located at the US-based provider'.",
                  "A cell tower (cell site/base station) is the radio access point at the edge of a mobile network. The tower is normally backhauled over fibre to the operator's core network; an international call crosses one or more submarine fibre cables to the destination operator's core network and only then reaches a nearby cell tower serving the recipient.",
                  "Standard cellular network and submarine-cable architecture", required=True),
        ],
        "replacements": [
            (r"\btower located at the US-based providers?(?:, or whatever telecom providers are there)?\b",
             "destination operator's core network, which then reaches a nearby cell tower"),
        ],
    },
    "submarine fibre cables": {
        "notes": [
            _note("The video discusses fibre cables laid on the seabed near the Strait of Hormuz.",
                  "Submarine fibre-optic cables carry the overwhelming majority of intercontinental data traffic; satellites carry only a small share. Because cables are physically routed through narrow seas, a single chokepoint can affect several systems at once, which is why route diversity matters.",
                  "Standard submarine cable infrastructure references"),
        ],
        "replacements": [],
    },
    "osi model": {
        "notes": [
            _note("The video says 'nothing was defined in the OSI protocols' and that OSI was only used in 4.3BSD.",
                  "The OSI Reference Model (ISO/IEC 7498-1) defines seven abstraction layers and their responsibilities; the OSI programme also specified concrete protocols (for example X.25). The model is a reference/teaching framework, whereas TCP/IP is the protocol suite implemented in practice. The 4.3BSD detail is a video-specific historical claim and is not relied on here.",
                  "ISO/IEC 7498-1; RFC 1122", required=True),
        ],
        "replacements": [
            (r"\bnothing was defined in the OSI protocols\b",
             "the OSI reference model itself does not specify wire-level protocols"),
        ],
    },
    "tcp/ip": {
        "notes": [
            _note("None - the video correctly notes TCP and UDP both sit in the transport layer.",
                  "TCP/IP is a protocol suite named after its two best-known protocols; UDP, ICMP and many others belong to it. Confirmed here so the AI explanation is not misled by an out-of-context quote such as 'we cannot include UDP in it'.",
                  "RFC 1122"),
        ],
        "replacements": [],
    },
    "wi fi throughput": {
        "notes": [
            _note("The video says data speed is inversely proportional to the wavelength/range of the radio waves.",
                  "Wi-Fi throughput depends on the modulation and coding scheme, channel width, number of spatial streams and the signal-to-noise ratio at the receiver. Higher frequency bands can offer wider channels but suffer greater path loss, so range tends to fall as frequency rises. Reducing this to 'speed is inversely proportional to wavelength' is an oversimplification of the video's claim.",
                  "Standard Wi-Fi / radio-link terminology", required=True),
        ],
        "replacements": [
            (r"\bdata speed is inversely proportional to the wavelength, which corresponds to the range\b",
             "throughput depends on modulation, channel width and signal quality; higher frequencies offer wider channels but shorter range"),
            (r"\bdata speed is inversely proportional to the wavelength\b",
             "throughput depends on modulation, channel width and signal quality rather than wavelength alone"),
        ],
    },
    "local access network (last mile)": {
        "notes": [
            _note("The video describes fibre being extended from the provider's network into different localities.",
                  "This is the access or 'last mile' part of the network: the provider's backbone reaches a local area and fibre equipment is then extended locality by locality to subscribers. It is distinct from a LAN, which is a local network inside one building or campus.",
                  "Generic access-network terminology"),
        ],
        "replacements": [],
    },
    "lan": {
        "notes": [
            _note("None - the video defines LAN directly.",
                  "The geographical scale sequence is PAN (a person), LAN (a building or campus), MAN (a city) and WAN (multiple cities, countries or continents). These are different categories, not synonyms.",
                  "Generic networking terminology"),
        ],
        "replacements": [],
    },
    "pan": {
        "notes": [
            _note("None - the video names PAN as the first geographical category.",
                  "PAN (Personal Area Network) covers roughly the space around one person (a few metres), for example Bluetooth earbuds paired to a phone. It is smaller than a LAN, which covers a building or campus.",
                  "Generic networking terminology"),
        ],
        "replacements": [],
    },
    "man": {
        "notes": [
            _note("None - the video names MAN as a geographical category.",
                  "MAN (Metropolitan Area Network) covers a city or metropolitan region, sitting between a LAN (one building or campus) and a WAN (multiple cities or countries). It is a distinct category, not another name for a WAN.",
                  "Generic networking terminology"),
        ],
        "replacements": [],
    },
}


def accuracy_notes_for(topic: str) -> list[dict[str, Any]]:
    key = norm(topic)
    # A concept may be labelled "LAN" but have a last-mile topic; check both.
    notes = list(TECHNICAL_CORRECTIONS.get(key, {}).get("notes", []))
    return notes


def correction_required(topic: str) -> bool:
    """True when the video's own claim is inaccurate and needs an explicit correction."""
    return any(n.get("required") for n in accuracy_notes_for(topic))


# Raw caption lines that must not be displayed as fact under a concept whose
# lesson already contains an external correction. They are still kept as
# timestamped transcript evidence, just not as teaching details.
MISLEADING_DETAIL_PATTERNS: dict[str, tuple[str, ...]] = {
    "cell tower": ("tower located at the us-based provider", "goes through the oceans via that same fiber"),
    "osi model": ("first implemented in bsd unix", "nothing was defined in the osi protocols"),
    "internet (network of networks)": ("international network, which connects", "full form is obviously international network"),
    "wi fi throughput": ("inversely proportional to the wavelength",),
    "wan": ("a type of internet", "international network, which connects"),
    "arpanet": ("association; maybe",),
}


def misleading_detail(text: str, topic: str) -> bool:
    low = str(text or "").lower()
    return any(p in low for p in MISLEADING_DETAIL_PATTERNS.get(norm(topic), ()))


def apply_accuracy_corrections(lesson: dict[str, Any], topic: str) -> dict[str, Any]:
    """Return a copy of a *validated AI lesson* with corrections and provenance.

    Transcript evidence and quotes are never modified.
    """
    out = dict(lesson)
    key = norm(topic)
    config = TECHNICAL_CORRECTIONS.get(key, {})
    notes = accuracy_notes_for(topic)
    if key == "lan" and "localit" in norm(str(lesson.get("source") or "")):
        notes = accuracy_notes_for("Local access network (last mile)")

    text_fields = ("explain", "intuition", "why_it_matters", "worked_example",
                   "misconception", "exercise")
    for pattern, replacement in config.get("replacements", []):
        rx = re.compile(pattern, re.I)
        for field in text_fields:
            value = out.get(field)
            if isinstance(value, str):
                out[field] = rx.sub(replacement, value)
        steps = out.get("steps")
        if isinstance(steps, list):
            out["steps"] = [rx.sub(replacement, s) if isinstance(s, str) else s for s in steps]

    prefix = config.get("prefix_explain")
    if prefix and isinstance(out.get("explain"), str):
        exp = out["explain"]
        if acronym_expansion(topic).lower() not in exp.lower():
            out["explain"] = prefix + exp[0].lower() + exp[1:] if exp else prefix

    if notes:
        out["accuracy_notes"] = notes
        out["supplemented_accuracy"] = True
    return out
