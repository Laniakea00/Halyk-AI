"""Resolve which ledger counterparties are related parties — in code.

The KYC dossier gives an ownership/voting percentage per counterparty and,
usually, its own numeric threshold for what counts as a related party
(confirmed on the public dataset: "20.0% и более голосующих прав"). The
comparison between the two is pure arithmetic and belongs in code, never in
an LLM call — Block 2's KycExtractionResult deliberately never asks the
model to compute this (see schemas.py), only to report the percentage and
the threshold as separately-extracted facts.

Matching a KYC-disclosed name to the ledger's `counterparty` column reuses
fuzzy_match.py — confirmed necessary on real data: the KYC dossier says
"Ertis Capital, LLP" (comma) while the ledger says "Ertis Capital LLP" (no
comma), and separately "Aktau Holdings LLP" vs "Aktau Holdings L.L.P.".
"""

from __future__ import annotations

import logging

from covenant_agent.linking.fuzzy_match import match_counterparty
from covenant_agent.models import RelatedPartyMatch
from covenant_agent.schemas import KycExtractionResult

logger = logging.getLogger(__name__)


def resolve_related_parties(
    scenario_id: str,
    kyc: KycExtractionResult | None,
    ledger_counterparties: list[str],
) -> dict[str, RelatedPartyMatch]:
    """Map ledger counterparty string -> RelatedPartyMatch, for entities the
    KYC dossier discloses AND that have at least one transaction in this
    scenario's ledger. Absence from the returned dict means "not known to
    be a related party" — not proof of the opposite.
    """
    if kyc is None or not kyc.disclosures:
        logger.info(
            "Scenario %s: no KYC disclosures available — no related parties can be "
            "positively identified (treated as none, not as 'confirmed zero').",
            scenario_id,
        )
        return {}

    threshold = kyc.related_party_threshold_pct
    if threshold is None:
        logger.warning(
            "Scenario %s: KYC dossier states no numeric related-party threshold — "
            "falling back to the document's own explicit 'related party' labels only "
            "(explicitly_labeled_related_party), since there's no percentage to compare.",
            scenario_id,
        )

    matches: dict[str, RelatedPartyMatch] = {}
    for disclosure in kyc.disclosures:
        pct = disclosure.ownership_or_voting_pct
        if threshold is not None and pct is not None:
            is_related = pct >= threshold
            basis = f"{pct}% {'>=' if is_related else '<'} threshold {threshold}%"
        elif disclosure.explicitly_labeled_related_party:
            is_related = True
            basis = "no numeric threshold available; document explicitly labels this entity a related party"
        else:
            is_related = False
            basis = "no numeric threshold and not explicitly labeled — insufficient basis to call this a related party"

        match = match_counterparty(disclosure.counterparty_name, ledger_counterparties)
        if match is None:
            logger.info(
                "Scenario %s: KYC-disclosed entity %r has no matching ledger counterparty "
                "(0 transactions with them this period).",
                scenario_id,
                disclosure.counterparty_name,
            )
            continue

        if not match.is_exact:
            logger.warning(
                "Scenario %s: fuzzy-matched KYC entity %r to ledger counterparty %r "
                "(similarity %.2f) — not an exact string match.",
                scenario_id,
                disclosure.counterparty_name,
                match.candidate,
                match.score,
            )

        if match.candidate in matches:
            logger.warning(
                "Scenario %s: ledger counterparty %r already matched to KYC entity %r — "
                "skipping second match from %r.",
                scenario_id,
                match.candidate,
                matches[match.candidate].kyc_name,
                disclosure.counterparty_name,
            )
            continue

        matches[match.candidate] = RelatedPartyMatch(
            ledger_counterparty=match.candidate,
            kyc_name=disclosure.counterparty_name,
            ownership_pct=pct,
            threshold_pct=threshold,
            is_related=is_related,
            basis=basis,
        )
        logger.info(
            "Scenario %s: %r -> related=%s (%s)",
            scenario_id,
            match.candidate,
            is_related,
            basis,
        )

    return matches
