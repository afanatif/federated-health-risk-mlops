from pydantic import BaseModel, Field, model_validator
from typing import List, Optional


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str = "pothole"

    @model_validator(mode="before")
    def check_box_geometry(cls, values):
        # If any coordinate is missing / None, skip reorder logic
        x1 = values.get("x1")
        x2 = values.get("x2")
        y1 = values.get("y1")
        y2 = values.get("y2")

        # If coordinates are not present yet or are None — just return (validation will catch types later).
        if x1 is None or x2 is None or y1 is None or y2 is None:
            return values

        # Ensure numeric types (pydantic will still validate afterwards)
        try:
            if x1 > x2:
                values["x1"], values["x2"] = x2, x1
            if y1 > y2:
                values["y1"], values["y2"] = y2, y1
        except TypeError:
            # If comparison fails, leave values unchanged and let pydantic report type errors
            return values

        return values


class InferenceResponse(BaseModel):
    detections: List[Box] = Field(default_factory=list)
    inference_time_ms: float


class InferenceRequest(BaseModel):
    # Accept base64-encoded images for API usage
    image_base64: Optional[str] = Field(
        None,
        description="Base64 encoded image. Use multipart upload in future versions."
    )

    # helper: allow passing an image path locally for CLI/main usage
    image_path: Optional[str] = Field(
        None,
        description="Local image path (used by local CLI / tests)."
    )
