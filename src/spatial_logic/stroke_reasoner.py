from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .events import StrokeEvent


@dataclass
class FinalStroke:
    type: str
    classifier_label: Optional[str] = None
    event_type: Optional[str] = None
    confidence: float = 0.5
    hitter_role: Optional[str] = None
    landing_region: Optional[str] = None


def combine_classifier_and_event(
    classifier_label: Optional[str],
    event: StrokeEvent,
) -> FinalStroke:
    """
    Simple fusion:
    - If classifier and event agree -> high confidence.
    - If classifier only -> use it with medium confidence.
    - If event only -> use it with medium-low confidence.
    - If mismatch -> prefer classifier but lower confidence.
    """
    event_type = event.type if event is not None else None
    hitter_role = event.hitter_role if event is not None else None
    landing_region = event.landing_region if event is not None else None

    if classifier_label and event_type:
        if classifier_label.lower().endswith(event_type):
            return FinalStroke(
                type=classifier_label,
                classifier_label=classifier_label,
                event_type=event_type,
                confidence=0.9,
                hitter_role=hitter_role,
                landing_region=landing_region,
            )
        else:
            return FinalStroke(
                type=classifier_label,
                classifier_label=classifier_label,
                event_type=event_type,
                confidence=0.65,
                hitter_role=hitter_role,
                landing_region=landing_region,
            )

    if classifier_label and not event_type:
        return FinalStroke(
            type=classifier_label,
            classifier_label=classifier_label,
            event_type=None,
            confidence=0.7,
            hitter_role=hitter_role,
            landing_region=landing_region,
        )

    if event_type and not classifier_label:
        return FinalStroke(
            type=event_type,
            classifier_label=None,
            event_type=event_type,
            confidence=0.6,
            hitter_role=hitter_role,
            landing_region=landing_region,
        )

    return FinalStroke(
        type="unknown",
        classifier_label=classifier_label,
        event_type=event_type,
        confidence=0.3,
        hitter_role=hitter_role,
        landing_region=landing_region,
    )
