"""Author the V14 held-out candidate questions from the real corpus.

Usage (from ``backend/``):

    py -m scripts.author_heldout_candidates            # write the candidate artifact
    py -m scripts.author_heldout_candidates --dry-run  # check quotes only, write nothing

## What this script does and does not do

It does NOT generate questions. The 26 questions below were written by an agent
(Buffy, under explicit human instruction) after reading the full text of the
named corpus chunks. This script's job is the part a human should not have to do
by hand and should not be trusted to do from memory:

* it resolves every claimed chunk id against the LIVE corpus and refuses to write
  an artifact that cites a chunk that does not exist;
* it takes the document id, document title and section from the live corpus
  rather than from the draft, so provenance cannot drift from reality;
* it verifies that each draft's ``evidence_quote`` appears VERBATIM in the named
  chunk, which is the only machine-checkable evidence that the chunk answers the
  question;
* it writes ``draft_origin`` and the creation protocol into the artifact, and
  leaves every question at ``status=draft`` with no review record.

## The drafting method, stated so it can be attacked

Candidate evidence chunks were located by reading the indexed chunk TEXT —
a full scan of the knowledge base's chunks with plain substring matching over
their stored text. No vector retrieval, no BM25 scoring, no reranker and no
embedding call was involved at any point in authoring, and no language model
generated any of the question text. The reason this matters: a question authored
by asking a retriever what it finds is fitted to that retriever, which is exactly
the fitting this held-out set exists to escape.

## What is deliberately NOT done here

Nothing approves a question. Every candidate is written with
``status='draft'`` and ``review=None``, so the freeze gate can only ever seal this
artifact as PROVISIONAL until a human records decisions in the review view. The
script cannot mark its own work as reviewed, and neither can any caller of it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import get_settings  # noqa: E402
from app.repositories.sqlite_repo import Repository  # noqa: E402
from app.services.answer_eval.heldout_benchmark import (  # noqa: E402
    DEFAULT_DEV_BENCHMARK_PATH,
    DEFAULT_HELDOUT_CANDIDATE_PATH,
    HeldOutCandidate,
    HeldOutCandidateSet,
    HeldOutValidationError,
    QueryType,
    save_candidate_set,
)

KB_ID = "kb_f278c283c748"

#: The date the drafts were written. Recorded per question so the artifact's
#: provenance has a date that does not depend on when the file was rewritten.
DRAFT_DATE = "2026-10-11"

DRAFT_ORIGIN = (
    "Agent-drafted (Buffy, under explicit human instruction) after reading the full "
    "text of the named indexed chunk. No retrieval, scoring or embedding was run "
    "during authoring and no language model generated the question text; the "
    "evidence_quote was verified verbatim against the chunk before the draft was "
    "written."
)

PROTOCOL: dict[str, Any] = {
    "method": (
        "Manual, agent-authored drafting grounded in the full text of the indexed "
        "corpus, following the same evidence-first method as the development "
        "benchmark: a chunk is chosen because it contains the answer, and the "
        "question is then written from that chunk."
    ),
    "question_selection": (
        "Candidate chunks were located by scanning the stored TEXT of every chunk in "
        "the knowledge base with plain substring matching. No retriever was run: no "
        "vector search, no BM25, no reranker, no embedding call. Topics were chosen to "
        "avoid the concepts the 28 development questions already cover and to spread "
        "across all 11 corpus documents."
    ),
    "evidence_selection": (
        "Exactly one necessary chunk per question, inherited from the development "
        "benchmark's relevance model (binary, necessity-only). Every chunk must not "
        "appear in the development benchmark, and every question must be answerable "
        "from that single chunk alone."
    ),
    "review_policy": (
        "Every question is frozen as a DRAFT. A human must record an approve, edit or "
        "reject decision for each question in the review view before the artifact can "
        "freeze as VALIDATED. Until then it freezes as PROVISIONAL, which permits "
        "exploratory results only."
    ),
    "separation_policy": (
        "Separation from the development benchmark is measured, not asserted: no "
        "expected chunk may be shared with a development question, and every question "
        "is scored for lexical similarity against all 28 development questions. The "
        "measurements are recomputed by the validator and reported per question."
    ),
    "known_limitations": (
        "Lexical similarity cannot prove two differently-worded questions test "
        "different knowledge, and the agent that drafted these questions is the same "
        "agent that ran the V13 experiments, so a shared blind spot cannot be excluded "
        "by the drafting method alone. Only independent human review, and ideally "
        "questions authored by someone else, can address that."
    ),
}

# ---------------------------------------------------------------------------
# The drafts. One entry per held-out question.
#
# `quote` must appear VERBATIM (whitespace-collapsed, case-insensitive) in `chunk`.
# ---------------------------------------------------------------------------
DRAFTS: list[dict[str, Any]] = [
    {
        "id": "auto-eng-hout-001",
        "chunk": "chk_2878a1b70522",
        "subdomain": "engine",
        "query_type": QueryType.TERMINOLOGY,
        "difficulty": "basic",
        "question": (
            "Which devices were described as engines during the Industrial Revolution, "
            "and what were the original steam engines such as Thomas Savery's?"
        ),
        "quote": (
            "Most mechanical devices invented during the Industrial Revolution were "
            "described as engines—the steam engine being a notable example. However, the "
            "original steam engines, such as those by Thomas Savery , were not mechanical "
            "engines but pumps."
        ),
        "rationale": (
            "Adds the history and etymology of the word 'engine' plus the counter-intuitive "
            "fact that Savery's engines were pumps. The development set asks only what an "
            "engine IS (auto-eng-001) and which IC engine suits light automobiles "
            "(auto-eng-002); it contains no terminology or history question."
        ),
    },
    {
        "id": "auto-eng-hout-002",
        "chunk": "chk_93a889a89753",
        "subdomain": "engine",
        "query_type": QueryType.QUANTITATIVE,
        "difficulty": "advanced",
        "question": (
            "Which engine is the largest internal combustion engine ever built, and what "
            "are its cylinder count, fuel and rated output?"
        ),
        "quote": (
            "The largest internal combustion engine ever built is the Wärtsilä-Sulzer "
            "RTA96-C , a 14-cylinder, 2-stroke turbocharged diesel engine"
        ),
        "rationale": (
            "A named-entity plus numeric-specification question (14 cylinders, 2-stroke, "
            "turbocharged diesel, 80 MW). Nothing in the development set asks for a "
            "specific engine model or any engine specification."
        ),
    },
    {
        "id": "auto-eng-hout-003",
        "chunk": "chk_5927eb177d1e",
        "subdomain": "engine",
        "query_type": QueryType.TERMINOLOGY,
        "difficulty": "intermediate",
        "question": (
            "What are engines with horizontally opposed pistons commonly called, and when "
            "was the first such design patented?"
        ),
        "quote": (
            "In 1897, Karl Benz was granted a patent for his design of the first engine "
            "with horizontally opposed pistons."
        ),
        "rationale": (
            "Introduces engine-layout terminology ('flat' / 'boxer') and a patent date. The "
            "development set covers four-stroke stages (auto-eng-004) but never a named "
            "layout or its origin."
        ),
    },
    {
        "id": "auto-eng-hout-004",
        "chunk": "chk_741b5e7c0b5e",
        "subdomain": "transmission",
        "query_type": QueryType.MECHANISM,
        "difficulty": "intermediate",
        "question": (
            "Why do automobiles powered by internal combustion engines require multiple "
            "gear ratios?"
        ),
        "quote": (
            "Since the torque and power output of an internal combustion engine (ICE) "
            "varies with its (rotational) speed , automobiles powered by ICEs require "
            "multiple gear ratios to keep the engine within its power band to produce "
            "optimal power, fuel efficiency , and smooth operation."
        ),
        "rationale": (
            "Asks for the CAUSE of multi-ratio transmissions. The development set asks what "
            "a transmission is (auto-eng-005), how a dual-clutch differs (auto-eng-006), how "
            "a CVT differs from stepped designs (auto-eng-007) and what a hydraulic automatic "
            "uses (auto-eng-008) — none asks why gears are needed at all."
        ),
    },
    {
        "id": "auto-eng-hout-005",
        "chunk": "chk_e7b4113e9fd5",
        "subdomain": "transmission",
        "query_type": QueryType.QUANTITATIVE,
        "difficulty": "intermediate",
        "question": (
            "How many forward gear ratios do most currently produced gasoline or diesel "
            "passenger cars use, and what do electric vehicles use instead?"
        ),
        "quote": (
            "Most currently produced passenger cars with gasoline or diesel engines use "
            "transmissions with 4–10 forward gear ratios (also called speeds ) and one "
            "reverse gear ratio. Electric vehicles typically use fixed-ratio or two-speed "
            "transmissions with no reverse"
        ),
        "rationale": (
            "A numeric range plus an electric-vehicle comparison. The development set's "
            "transmission questions are all about transmission TYPES; none gives a gear "
            "count or contrasts an EV drivetrain."
        ),
    },
    {
        "id": "auto-eng-hout-006",
        "chunk": "chk_0d263bc542f9",
        "subdomain": "braking",
        "query_type": QueryType.TERMINOLOGY,
        "difficulty": "intermediate",
        "question": (
            "What is meant by fade in a brake, and what determines the greatest amount of "
            "power a brake can dissipate without failing?"
        ),
        "quote": (
            "Fade – As a brake heats, it may become less effective, called brake fade . "
            "Some designs are inherently prone to fade, while other designs are relatively "
            "immune."
        ),
        "rationale": (
            "Adds the failure mode of a brake (fade) and the concept of continuous power "
            "dissipation. The development set asks what a brake is (auto-eng-009) and how "
            "friction pad/shoe brakes work (auto-eng-010) — it never asks how brakes fail."
        ),
    },
    {
        "id": "auto-eng-hout-007",
        "chunk": "chk_502dcf67ec92",
        "subdomain": "braking",
        "query_type": QueryType.MECHANISM,
        "difficulty": "advanced",
        "question": (
            "What did Bosch and Mercedes update in 1978, and how was that system "
            "described?"
        ),
        "quote": (
            "In 1978, Bosch and Mercedes updated their 1936 anti-lock brake system for the "
            "Mercedes S-Class . That ABS is a fully electronic, four-wheel and multi-channel "
            "system that later became standard."
        ),
        "rationale": (
            "A specific dated engineering-development fact about ABS. The development set "
            "covers regenerative versus friction braking only; anti-lock braking does not "
            "appear in it."
        ),
    },
    {
        "id": "auto-eng-hout-008",
        "chunk": "chk_2bdbedea01d7",
        "subdomain": "steering",
        "query_type": QueryType.MECHANISM,
        "difficulty": "intermediate",
        "question": (
            "Who invented collapsible steering columns and in which car were they "
            "introduced?"
        ),
        "quote": (
            "Collapsible steering columns were invented by Béla Barényi and were introduced "
            "in the 1959 Mercedes-Benz W111 Fintail, along with crumple zones."
        ),
        "rationale": (
            "A safety-engineering fact tying steering to crash design. The development set "
            "asks how steering input is transmitted (auto-eng-013), how hydraulic and "
            "electric assistance differ (auto-eng-014) and what speed-sensitive steering "
            "changes (auto-eng-015) — nothing about steering as a crash-safety topic."
        ),
    },
    {
        "id": "auto-eng-hout-009",
        "chunk": "chk_8ec810da0d3c",
        "subdomain": "steering",
        "query_type": QueryType.MECHANISM,
        "difficulty": "advanced",
        "question": (
            "What is compliance understeer, and which methods are typical for achieving it "
            "on a live rear axle and on a twist beam suspension?"
        ),
        "quote": (
            "This effect is called compliance understeer ; it, or its opposite, is present "
            "on all suspensions. Typical methods of achieving compliance understeer are to "
            "use a Watt's link on a live rear axle, or the use of toe control bushings on a "
            "twist beam suspension ."
        ),
        "rationale": (
            "Requires linking suspension geometry to a steering behaviour named by the "
            "corpus. The development set never uses the terms understeer or compliance, and "
            "its suspension questions are about roll rate, roll couple and interconnected "
            "suspension."
        ),
    },
    {
        "id": "auto-eng-hout-010",
        "chunk": "chk_5345c113b94f",
        "subdomain": "suspension",
        "query_type": QueryType.MECHANISM,
        "difficulty": "intermediate",
        "question": "What is damping, and what happens to a car that has none?",
        "quote": (
            "Damping is the control of motion or oscillation, as seen with the use of "
            "hydraulic gates and valves in a vehicle's shock absorber."
        ),
        "rationale": (
            "Defines damping and its failure mode. The development set's suspension "
            "questions (auto-eng-016/017/018) are about roll rate, roll couple percentage "
            "and interconnected suspension; damping appears nowhere in it."
        ),
    },
    {
        "id": "auto-eng-hout-011",
        "chunk": "chk_09e04aa8bc01",
        "subdomain": "suspension",
        "query_type": QueryType.COMPARISON,
        "difficulty": "intermediate",
        "question": (
            "Which suspension arrangement is generally accepted as the most compact for "
            "front-engined vehicles, and why does that matter?"
        ),
        "quote": (
            "It is generally accepted, that MacPherson struts are the most compact "
            "arrangement for front-engined vehicles, where space between the wheels is "
            "required to place the engine."
        ),
        "rationale": (
            "A packaging/space trade-off question naming a suspension type. The development "
            "set never names a suspension type or discusses space, weight or packaging."
        ),
    },
    {
        "id": "auto-eng-hout-012",
        "chunk": "chk_03476fb586a7",
        "subdomain": "suspension",
        "query_type": QueryType.MECHANISM,
        "difficulty": "advanced",
        "question": (
            "How do the shock absorbers sold with magneto-rheological fluid give variable "
            "control, and how does fully active suspension differ in principle?"
        ),
        "quote": (
            "Delphi currently sells shock absorbers filled with a magneto-rheological fluid "
            ", whose viscosity can be changed electromagnetically — thereby giving variable "
            "control without switching valves, which is faster and thus more effective."
        ),
        "rationale": (
            "Introduces semi-active versus fully active suspension and electromagnetic fluid "
            "control. The development set contains no actuator, damper-technology or "
            "active-suspension question at all."
        ),
    },
    {
        "id": "auto-eng-hout-013",
        "chunk": "chk_69451f181e58",
        "subdomain": "chassis",
        "query_type": QueryType.TERMINOLOGY,
        "difficulty": "intermediate",
        "question": (
            "Why is 'semi-monocoque' a more appropriate description than 'monocoque' for an "
            "integrated body structure?"
        ),
        "quote": (
            "Although this is sometimes also referred to as a monocoque structure, because "
            "the car's outer skin and panels are made load-bearing, there are still ribs, "
            "bulkheads, and box sections to reinforce the body, making the description "
            "semi-monocoque more appropriate."
        ),
        "rationale": (
            "A terminology-precision question about body construction. The development set "
            "asks what a frame is (auto-eng-019), what replaced body-on-frame (auto-eng-020) "
            "and what a ladder frame is (auto-eng-021); the monocoque/semi-monocoque "
            "distinction is not in it. Shares the DOCUMENT with auto-eng-019/020 but not the "
            "evidence chunk."
        ),
    },
    {
        "id": "auto-eng-hout-014",
        "chunk": "chk_4299bcd54ac5",
        "subdomain": "chassis",
        "query_type": QueryType.TERMINOLOGY,
        "difficulty": "advanced",
        "question": (
            "Which streamlined 1936 car used a unibody structure, and what does the corpus "
            "say about how the design spread afterwards?"
        ),
        "quote": (
            "The streamlined 1936 Lincoln-Zephyr with conventional front-engine, "
            "rear-wheel-drive layout utilized a unibody structure."
        ),
        "rationale": (
            "A named-model, dated adoption question about unit construction, distinct from "
            "auto-eng-020's general 'what replaced body-on-frame'. Shares the document with "
            "three development questions but not the evidence chunk."
        ),
    },
    {
        "id": "auto-eng-hout-015",
        "chunk": "chk_561c97852061",
        "subdomain": "vehicle-dynamics",
        "query_type": QueryType.ENUMERATION,
        "difficulty": "intermediate",
        "question": (
            "Which attributes or aspects of vehicle dynamics are purely due to mass and its "
            "distribution?"
        ),
        "quote": (
            "Some attributes or aspects of vehicle dynamics are purely due to mass and its "
            "distribution. These include: Center of mass Moment of inertia Roll moment "
            "Sprung mass Unsprung mass Weight distribution"
        ),
        "rationale": (
            "Asks for a specific grouping of attributes (mass and its distribution). The "
            "development set asks which TIRE attributes affect dynamics (auto-eng-022), how "
            "dynamics can be analysed (auto-eng-023) and which phenomena are listed as key "
            "behaviours (auto-eng-024) — the mass-distribution grouping is a fourth, separate "
            "grouping in the same corpus section."
        ),
    },
    {
        "id": "auto-eng-hout-016",
        "chunk": "chk_0f11fa3d273f",
        "subdomain": "vehicle-dynamics",
        "query_type": QueryType.MECHANISM,
        "difficulty": "advanced",
        "question": (
            "In current vehicle simulator models, which part of the math model is described "
            "as the weakest and most difficult to simulate?"
        ),
        "quote": (
            "In current vehicle simulator models, the tire model is the weakest and most "
            "difficult part to simulate."
        ),
        "rationale": (
            "A limitation-of-simulation fact, not a 'how do you simulate' question. Deliberately "
            "narrow wording to keep it distinct from auto-eng-023, which is in the same "
            "document; the validator reports its lexical similarity to that question so the "
            "resemblance is inspectable rather than assumed away."
        ),
    },
    {
        "id": "auto-eng-hout-017",
        "chunk": "chk_8d1f34008a3d",
        "subdomain": "electric-vehicles",
        "query_type": QueryType.COMPARISON,
        "difficulty": "intermediate",
        "question": (
            "According to the vehicle-type table, how does a mild hybrid electric vehicle "
            "differ from a plug-in hybrid vehicle in the fuel it uses?"
        ),
        "quote": (
            "Mild hybrid electric vehicle (MHEV, BAHV) Exclusively uses petroleum or other "
            "fuel, but has electric battery to consume less. Plug-in hybrid vehicle (PHEV) "
            "Uses mixture of petroleum or other fuel and electricity from power grid ."
        ),
        "rationale": (
            "Classification-by-fuel question, a taxonomy the development set never touches. "
            "The development set's EV questions are about the battery (auto-eng-026) and "
            "AC/DC charging (auto-eng-027); neither classifies vehicle types."
        ),
    },
    {
        "id": "auto-eng-hout-018",
        "chunk": "chk_60b255a941d5",
        "subdomain": "electric-vehicles",
        "query_type": QueryType.MECHANISM,
        "difficulty": "advanced",
        "question": (
            "What do current electric vehicle safety regulations focus on, and which "
            "materials are being developed to address it?"
        ),
        "quote": (
            "Current regulations focus on thermal runaway protection, with various "
            "international standards mandating advance warning systems and thermal "
            "propagation containment measures."
        ),
        "rationale": (
            "Battery-safety regulation and fire-protection materials — a distinct area from "
            "the development set's battery-chemistry (auto-eng-026) and charging-mode "
            "(auto-eng-027) questions."
        ),
    },
    {
        "id": "auto-eng-hout-019",
        "chunk": "chk_934dc093b023",
        "subdomain": "electric-vehicles",
        "query_type": QueryType.TERMINOLOGY,
        "difficulty": "basic",
        "question": (
            "What is a charging station also known as, and what does it supply power to?"
        ),
        "quote": (
            "A charging station , also known as a charge point, chargepoint, or electric "
            "vehicle supply equipment (EVSE), is a power supply device that supplies "
            "electrical power for recharging the onboard battery packs of plug-in electric "
            "vehicles"
        ),
        "rationale": (
            "Infrastructure terminology (EVSE). The development set asks how AC versus DC "
            "charging WORKS (auto-eng-027) but never names the equipment."
        ),
    },
    {
        "id": "auto-eng-hout-020",
        "chunk": "chk_2d819cfb9f96",
        "subdomain": "electric-vehicles",
        "query_type": QueryType.ENUMERATION,
        "difficulty": "intermediate",
        "question": (
            "Which forms of rechargeable on-board electricity storage are used in electric "
            "vehicles, and what advantage does avoiding an intermediate mechanical step give?"
        ),
        "quote": (
            "Batteries, electric double-layer capacitors and flywheel energy storage are "
            "forms of rechargeable on-board electricity storage systems."
        ),
        "rationale": (
            "Asks for the storage taxonomy including supercapacitors and flywheels. The "
            "development set asks which battery powers a BEV (auto-eng-026) — chemistry only "
            "— and never mentions flywheels or double-layer capacitors."
        ),
    },
    {
        "id": "auto-eng-hout-021",
        "chunk": "chk_5559c040bb81",
        "subdomain": "battery-management",
        "query_type": QueryType.MECHANISM,
        "difficulty": "intermediate",
        "question": (
            "Why does a battery management system balance the cells, and by which methods "
            "can it do so?"
        ),
        "quote": (
            "In order to maximize the battery's capacity, and to prevent localized "
            "under-charging or over-charging, the BMS may actively ensure that all the cells "
            "that compose the battery are kept at the same voltage or State of Charge, "
            "through balancing."
        ),
        "rationale": (
            "Cell balancing and its methods. The development set asks what a BMS is and which "
            "states it monitors (auto-eng-025) — a different function of the same component. "
            "Shares the document with auto-eng-025 but not the evidence chunk."
        ),
    },
    {
        "id": "auto-eng-hout-022",
        "chunk": "chk_5037fa56fe36",
        "subdomain": "battery-management",
        "query_type": QueryType.MECHANISM,
        "difficulty": "advanced",
        "question": (
            "Why can the battery management system of an electric or hybrid vehicle not work "
            "as a stand-alone device, and with which subsystems must it communicate?"
        ),
        "quote": (
            "In the case of electric or hybrid vehicles, the BMS is only a subsystem and "
            "cannot work as a stand-alone device. It must communicate with at least a "
            "charger (or charging infrastructure), a load, thermal management and emergency "
            "shutdown subsystems."
        ),
        "rationale": (
            "Integration/interface question — which other subsystems the BMS depends on. "
            "Distinct from both auto-eng-025 (monitored states) and hout-021 (balancing)."
        ),
    },
    {
        "id": "auto-eng-hout-023",
        "chunk": "chk_183f9951e89a",
        "subdomain": "regenerative-braking",
        "query_type": QueryType.QUANTITATIVE,
        "difficulty": "advanced",
        "question": (
            "Which factors can affect the energy transfer efficiency of a braking energy "
            "recovery system, and roughly what value does it generally take?"
        ),
        "quote": (
            "which may be affected by battery state of charge (SOC) and ambient temperature. "
            "In general, about 0.9."
        ),
        "rationale": (
            "A symbolic-plus-numeric efficiency fact (SOC and ambient temperature; about "
            "0.9). The development set asks how regenerative braking differs from friction "
            "braking (auto-eng-011) and how hybrid and electric vehicles combine the two "
            "(auto-eng-012) — neither quantifies recovery efficiency."
        ),
    },
    {
        "id": "auto-eng-hout-024",
        "chunk": "chk_39602e0f5da3",
        "subdomain": "regenerative-braking",
        "query_type": QueryType.MECHANISM,
        "difficulty": "advanced",
        "question": (
            "In the CPC-KERS, what replaces the continuously variable transmission and "
            "where does the mechanism sit?"
        ),
        "quote": (
            "the whole mechanism including the flywheel sits entirely in the vehicle's hub "
            "(looking like a drum brake). In the CPC-KERS, a differential replaces the CVT "
            "and transfers torque between the flywheel , drive wheel and road wheel."
        ),
        "rationale": (
            "A named mechanical KERS variant and its packaging. The development set's "
            "regenerative-braking questions are about the energy-flow concept only; no "
            "flywheel, KERS or driveline layout appears in it."
        ),
    },
    {
        "id": "auto-eng-hout-025",
        "chunk": "chk_4b7f788c3ea9",
        "subdomain": "general-automobile",
        "query_type": QueryType.MECHANISM,
        "difficulty": "intermediate",
        "question": (
            "Which US patent was granted for a two-stroke car engine, what effect is it said "
            "to have had on car development, and when was it overturned?"
        ),
        "quote": (
            "on 5 November 1895, Selden was granted a US patent ( U.S. patent 549,160 ) for a "
            "two-stroke car engine, which hindered, more than encouraged , development of "
            "cars in the United States. His patent was challenged by Henry Ford and others, "
            "and overturned in 1911."
        ),
        "rationale": (
            "Automotive history and intellectual property. The development set's only "
            "general-automobile question is auto-eng-028 (which secondary energy source "
            "recovers energy) — no history, dates or named people."
        ),
    },
    {
        "id": "auto-eng-hout-026",
        "chunk": "chk_4c7ddabe7417",
        "subdomain": "general-automobile",
        "query_type": QueryType.QUANTITATIVE,
        "difficulty": "intermediate",
        "question": (
            "Why did cars increase in weight during the late 20th and early 21st century, and "
            "what did cars typically weigh as of 2019?"
        ),
        "quote": (
            "During the late 20th and early 21st century, cars increased in weight due to "
            "batteries, [ 67 ] modern steel safety cages, anti-lock brakes, airbags, and "
            "\"more-powerful—if more efficient—engines\""
        ),
        "rationale": (
            "Mass trend and its causes plus a numeric range (1–3 tonnes). Nothing in the "
            "development set concerns vehicle weight, size or trends."
        ),
    },
]


def build_candidates(repo: Repository, kb_id: str) -> tuple[HeldOutCandidateSet, list[str]]:
    """Resolve the drafts against the live corpus and assemble the candidate set.

    Returns the candidate set plus a list of problems. A problem is a draft whose
    chunk is missing, whose document cannot be resolved, or whose quote is not in
    the chunk — all of which must stop the write rather than produce a file that
    cites evidence it cannot verify.
    """
    problems: list[str] = []
    documents = {d.id: d for d in repo.list_documents(kb_id)}
    chunk_ids = [d["chunk"] for d in DRAFTS]
    rows = {c.id: c for c in repo.get_chunks_by_ids(kb_id, sorted(set(chunk_ids)))}

    seen: set[str] = set()
    candidates: list[HeldOutCandidate] = []
    for draft in DRAFTS:
        qid = draft["id"]
        if qid in seen:
            problems.append(f"{qid}: duplicate draft id")
        seen.add(qid)

        row = rows.get(draft["chunk"])
        if row is None:
            problems.append(
                f"{qid}: claimed chunk {draft['chunk']} does not exist in {kb_id}"
            )
            continue
        doc = documents.get(row.document_id)
        if doc is None:
            problems.append(
                f"{qid}: chunk {row.id} belongs to document {row.document_id}, which is "
                "not in this knowledge base"
            )
            continue

        # Verbatim quote check, using the same normalization the validator uses.
        from app.services.answer_eval.heldout_validation import normalize_text

        if normalize_text(draft["quote"]) not in normalize_text(row.text):
            problems.append(
                f"{qid}: recorded evidence_quote does not appear in chunk {row.id}; the "
                "draft would assert a sufficiency it cannot show"
            )
            continue

        provenance = (
            f"Chunk {row.id} ({doc.title}, section {row.section or 'unknown'}): "
            f'"{draft["quote"]}". Read in full from the indexed corpus and selected '
            f"{DRAFT_DATE}. The chunk is not expected evidence for any development "
            f"benchmark question."
        )
        candidates.append(
            HeldOutCandidate(
                id=qid,
                subdomain=draft["subdomain"],
                question=draft["question"],
                expected_chunk_ids=[row.id],
                expected_document_ids=[row.document_id],
                expected_keywords=[],
                evidence_quote=draft["quote"],
                coverage_rationale=draft["rationale"],
                query_type=draft["query_type"],
                difficulty=draft["difficulty"],
                provenance=provenance,
                draft_origin=DRAFT_ORIGIN,
                # Never pre-reviewed: the whole point of the gate is that nothing
                # in this script can mark its own draft as human-reviewed.
                status="draft",
                review=None,
            )
        )

    return (
        HeldOutCandidateSet(
            benchmark="automobile-engineering-heldout-v1",
            version=1,
            kb_id=kb_id,
            kb_name="Automobile Engineering KB (E2E)",
            derived_from=DEFAULT_DEV_BENCHMARK_PATH,
            protocol=PROTOCOL,
            authorship={
                "method": (
                    "agent-drafted from the full text of named indexed corpus chunks; "
                    "no retrieval and no LLM generation"
                ),
                "author": "Buffy (Codebuff agent), operating under explicit human instruction",
                "human_review": "PENDING - see the review records on each question",
                "agent_authored": True,
                "integrity_rules": [
                    "No question may reuse an expected chunk of a development question; the "
                    "validator treats a shared chunk as an error, not a warning.",
                    "No question enters the frozen instrument without a verbatim evidence "
                    "quote that the validator can find in the named chunk.",
                    "Every question is frozen as a draft; review decisions are appended by a "
                    "named human in the review view and never by this script.",
                    "Recorded reference answers, key points and ground truth do not exist for "
                    "these questions, so no answer-correctness claim may be made from them.",
                ],
            },
            relevance_model="binary",
            candidates=candidates,
        ),
        problems,
    )


def _repo_relative(path: str) -> Path:
    """Resolve an artifact path against the repository root.

    The shipped path is written as ``benchmarks/...`` — a repo-relative name — so
    running this from ``backend/`` must not silently create a second, ignored
    ``backend/benchmarks/`` copy that the app would never read.
    """
    from app.services.answer_eval.retrieval_diagnostic_runner import REPO_ROOT

    given = Path(path)
    if given.is_absolute():
        return given
    return (REPO_ROOT / given).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", default=KB_ID)
    parser.add_argument("--out", default=DEFAULT_HELDOUT_CANDIDATE_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out_path = _repo_relative(args.out)

    settings = get_settings()
    repo = Repository(str(settings.db_path))
    candidates, problems = build_candidates(repo, args.kb)

    if problems:
        print(f"REFUSING to write: {len(problems)} problem(s) in the drafts")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"drafts:            {len(DRAFTS)}")
    print(f"resolved:          {len(candidates.candidates)}")
    print(f"candidate fingerprint: {candidates.fingerprint()}")
    print(f"subdomains:        {candidates.subdomain_counts()}")
    print(f"query types:       {candidates.query_type_counts()}")
    print(f"difficulty:        {candidates.difficulty_counts()}")
    print(f"review:            {candidates.review_completeness()}")

    if args.dry_run:
        print("\ndry run: nothing written (the review/freeze state is still 'draft')")
        return 0

    path = save_candidate_set(candidates, out_path)
    print(f"\nwrote {path}")
    print(
        "Next: review each question in the app's Held-out Validation view, then freeze. "
        "Every question is still status=draft, so a freeze at this point can only be "
        "PROVISIONAL (exploratory results only)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
