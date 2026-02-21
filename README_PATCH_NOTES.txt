Bottom-only ROI dilation/band patch
==================================

This patch adds optional controls to prevent floor ROI expansion from polluting the upper half of the frame:

  - BADC_FLOOR_ROI_DILATE_BOTTOM_ONLY=1
  - BADC_FLOOR_ROI_DILATE_Y_FRAC=0.55
  - BADC_FLOOR_ROI_BAND_BOTTOM_ONLY=1

How to use:

  export BADC_ADV_POSTBLOB=1
  export BADC_FLOOR_ROI_GREEN_CLOSE=5
  export BADC_FLOOR_ROI_GREEN_DILATE=10
  export BADC_FLOOR_ROI_GREEN_DILATE_ITERS=1
  export BADC_FLOOR_ROI_FLOOR_BAND_PX=30

  export BADC_FLOOR_ROI_DILATE_BOTTOM_ONLY=1
  export BADC_FLOOR_ROI_DILATE_Y_FRAC=0.55
  export BADC_FLOOR_ROI_BAND_BOTTOM_ONLY=1

  python scripts/demo_friend.py

Tuning:
  - If the ROI still misses the "knee-behind" area, decrease Y_FRAC (e.g., 0.55 -> 0.50 -> 0.45).
  - If the ROI starts leaking upward again, increase Y_FRAC.
