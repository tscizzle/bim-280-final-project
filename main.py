import glob
from datetime import datetime
from dateutil import tz

import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBFile, TimeSeries, NWBHDF5IO
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject
from pynwb.behavior import SpatialSeries, Position


def main():
    data_dir = "data"
    nwb_extenstion = ".nwb"
    nwb_filenames = glob.glob(f"{data_dir}/*{nwb_extenstion}")

    with NWBHDF5IO(nwb_filenames[0], "r") as io:
        read_nwbfile = io.read()
        print(read_nwbfile)


if __name__ == "__main__":
    main()
