"""
Pydantic schema models for validating ACN and UrbanEV data.

Provides strict validation schemas and a ``validate_dataframe`` helper
that checks every row of a :class:`pandas.DataFrame` against a given
Pydantic model.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)


# =========================================================================
# ACN Session Schema
# =========================================================================
class ACNSessionSchema(BaseModel):
    """Validation schema for a single ACN charging-session record.

    Fields
    ------
    connectionTime : datetime
        When the vehicle was plugged in.
    disconnectTime : datetime
        When the vehicle was unplugged.
    doneChargingTime : datetime | None
        When charging finished (may be null if session was interrupted).
    kWhDelivered : float
        Total energy delivered during the session (kWh).
    stationID : str
        Identifier for the charging station.
    siteID : str
        Identifier for the site / location.
    """

    connectionTime: datetime
    disconnectTime: datetime
    doneChargingTime: Optional[datetime] = None
    kWhDelivered: float = Field(ge=0.0)
    stationID: str
    siteID: str

    @field_validator("disconnectTime")
    @classmethod
    def disconnect_after_connect(
        cls, v: datetime, info: Any
    ) -> datetime:
        """Ensure disconnectTime is not earlier than connectionTime."""
        conn = info.data.get("connectionTime")
        if conn is not None and v < conn:
            raise ValueError(
                f"disconnectTime ({v}) is before connectionTime ({conn})"
            )
        return v

    model_config = {"str_strip_whitespace": True}


# =========================================================================
# UrbanEV Information Schema
# =========================================================================
class UrbanEVInfoSchema(BaseModel):
    """Validation schema for UrbanEV ``information.csv`` rows.

    Fields
    ------
    num : int
        District / node index.
    grid : str | int
        Grid reference identifier.
    count : int
        Total number of charging stations in the district.
    fast_count : int
        Number of fast-charging stations.
    slow_count : int
        Number of slow-charging stations.
    area : float
        District area (km²).
    lon : float
        Longitude.
    la : float
        Latitude (column name is ``la`` in the raw CSV).
    CBD : float
        Distance to Central Business District (km).
    dynamic_pricing : int
        Whether the district uses dynamic pricing (0 or 1).
    """

    num: int = Field(ge=0)
    grid: Any  # sometimes int, sometimes str in raw data
    count: int = Field(ge=0)
    fast_count: int = Field(ge=0)
    slow_count: int = Field(ge=0)
    area: float = Field(gt=0.0)
    lon: float
    la: float
    CBD: float = Field(ge=0.0)
    dynamic_pricing: int = Field(ge=0, le=1)

    model_config = {"str_strip_whitespace": True}


# =========================================================================
# DataFrame validation helper
# =========================================================================
def validate_dataframe(
    df: pd.DataFrame,
    schema: Type[BaseModel],
    *,
    sample_size: Optional[int] = None,
    raise_on_error: bool = False,
) -> Dict[str, Any]:
    """Validate rows of *df* against a Pydantic *schema*.

    Parameters
    ----------
    df : pd.DataFrame
        The data to validate.
    schema : Type[BaseModel]
        A Pydantic model class whose fields correspond to DataFrame columns.
    sample_size : int | None
        If given, validate only a random sample of this many rows
        (useful for large DataFrames).
    raise_on_error : bool
        If ``True``, raise ``ValueError`` when any row fails validation.

    Returns
    -------
    dict
        ``{"total": int, "valid": int, "invalid": int, "errors": list}``
    """
    target = df if sample_size is None else df.sample(min(sample_size, len(df)))
    errors: List[Dict[str, Any]] = []
    valid_count = 0

    for idx, row in target.iterrows():
        try:
            schema.model_validate(row.to_dict())
            valid_count += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"index": idx, "error": str(exc)})

    result = {
        "total": len(target),
        "valid": valid_count,
        "invalid": len(errors),
        "errors": errors[:20],  # cap for readability
    }

    if errors:
        logger.warning(
            "Schema validation: %d / %d rows invalid (showing first %d errors).",
            len(errors),
            len(target),
            min(len(errors), 20),
        )
        if raise_on_error:
            raise ValueError(
                f"{len(errors)} rows failed {schema.__name__} validation. "
                f"First error: {errors[0]}"
            )
    else:
        logger.info(
            "Schema validation passed: all %d rows valid.", len(target)
        )

    return result
