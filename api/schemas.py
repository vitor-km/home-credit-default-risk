"""
Pydantic schemas defining the API's request and response contracts.
Field names match the 17 scorecard variables exactly, so the request
maps directly to the dict that score_client() expects.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ClientFeatures(BaseModel):
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None
    NAME_EDUCATION_TYPE_BINNED: str
    AMT_CREDIT: float = Field(..., gt=0)
    CODE_GENDER: str
    YEARS_EMPLOYED: Optional[float] = None
    ORGANIZATION_TYPE_BINNED: str
    FLOORSMAX_AVG: Optional[float] = None
    BUREAU_CREDIT_ACTIVE_ACTIVE_COUNT: Optional[float] = None
    BUREAU_AMT_CREDIT_SUM_LIMIT_MEAN: Optional[float] = None
    OCCUPATION_TYPE_BINNED: Optional[str] = None
    REGION_POPULATION_RELATIVE: float
    DAYS_ID_PUBLISH: int
    BUREAU_AMT_CREDIT_SUM_DEBT_MEAN: Optional[float] = None
    DAYS_LAST_PHONE_CHANGE: Optional[float] = None
    BUREAU_DAYS_CREDIT_MEAN: Optional[float] = None


class VariableBreakdown(BaseModel):
    raw_value: Optional[float | str] = None
    woe: float
    points: float


class ScoreResponse(BaseModel):
    score: float
    probability_default: float
    breakdown: dict[str, VariableBreakdown]
    unmapped_variables: list[str]
    review_flag: bool