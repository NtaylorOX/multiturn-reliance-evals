"""Light adversarial pressure conditions, appended to the simulator's persona prompt.

Filled in at the red-team step; `none` is the unpressured baseline.
"""

PRESSURES: dict[str, str] = {
    "none": "",
}
