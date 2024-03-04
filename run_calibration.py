#!/usr/bin/env python
import numpy as np

from hardware.calibrate_camera import Calibration

if __name__ == '__main__':
    calibration = Calibration(
        cam_id=830112070066,
        calib_grid_step=0.03,
        checkerboard_offset_from_tool=[0.0, 0.0215, 0.0115],
        workspace_limits=np.asarray([[0.50, 0.65], [-0.25, -0.1], [0.0, 0.24]])
    )
    calibration.run()
