"""Training-label definitions.

Target definitions (identical meaning for synthetic and external data):

``target_irrigation_needed``
    1 if, **without irrigation**, root-zone depletion reaches the readily
    available water threshold (RAW, the onset of crop water stress) at the
    decision time or within the next ``HORIZON_HOURS`` hours, taking the rain
    that actually fell into account.

``target_irrigation_mm``
    Net irrigation depth (mm) recommended when irrigation is needed, from
    :func:`agronomy.water_balance.net_irrigation_requirement`; 0 otherwise.

Labels necessarily look into the future (that is what is being predicted);
features never do. The training split embargo (see ``ml.splitting``) is at
least ``HORIZON_HOURS`` long so no label window crosses a split boundary.

For synthetic data the simulator computes labels from the *true* soil state.
For external datasets :func:`derive_labels_from_observations` estimates them
from sensor depletion, observed rain and Hargreaves ETc ("weak labels").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import effective_rain_array

HORIZON_HOURS = 24
TARGET_CLASS = "target_irrigation_needed"
TARGET_REG = "target_irrigation_mm"


def derive_labels_from_observations(features: pd.DataFrame) -> pd.DataFrame:
    """Add weak labels to a feature frame built from observed (external) data.

    Uses, per field, the observed rain and estimated crop ET of the *next*
    24 hours (``shift(-24)`` of the trailing 24 h sums). The last 24 hours of
    each field have no complete future window and receive NaN labels.
    """
    out = features.copy()
    needed_parts, depth_parts = [], []
    for _, g in out.groupby("field_id", sort=False):
        etc_next = g["etc_24h_mm"].shift(-HORIZON_HOURS)
        rain_next = g["rain_sum_24h"].shift(-HORIZON_HOURS)
        eff_rain_next = pd.Series(effective_rain_array(rain_next.fillna(0).to_numpy()), index=g.index)
        dr = g["depletion_mm"].clip(lower=0)
        raw, taw = g["raw_mm"], g["taw_mm"]
        projected = dr + etc_next - eff_rain_next
        needed = ((dr >= raw) | (projected >= raw)).astype(float)
        need_mm = np.minimum(np.maximum(dr - eff_rain_next, etc_next), taw)
        depth = np.where(needed == 1, need_mm, 0.0)
        valid = etc_next.notna() & g["depletion_mm"].notna()
        needed_parts.append(needed.where(valid))
        depth_parts.append(pd.Series(depth, index=g.index).where(valid))
    out[TARGET_CLASS] = pd.concat(needed_parts)
    out[TARGET_REG] = pd.concat(depth_parts)
    return out
