from __future__ import annotations
import numpy as np

from sipkit.util import _read_effective_index

neff_data, width_data, wav_data = _read_effective_index(
    "neff_fitted_Si_fitted_SiO2_width_240_5_700_wav_1200_0p1_1700.csv"
)

wav_size = wav_data.shape[0]
wav_min = np.min(wav_data)
wav_max = np.max(wav_data)

width_size = width_data.shape[0]
width_min = np.min(width_data)
width_max = np.max(width_data)


effective_index_te0, _, _ = _read_effective_index(
    "neff_te0_fitted_Si_fitted_SiO2_width_240_5_700_wav_1200_0p1_1700.csv"
)

effective_index_tm0, _, _ = _read_effective_index(
    "neff_tm0_fitted_Si_fitted_SiO2_width_240_5_700_wav_1200_0p1_1700.csv"
)

effective_index_te1, _, _ = _read_effective_index(
    "neff_te1_fitted_Si_fitted_SiO2_width_240_5_700_wav_1200_0p1_1700.csv"
)

effective_index_tm1, _, _ = _read_effective_index(
    "neff_tm1_fitted_Si_fitted_SiO2_width_240_5_700_wav_1200_0p1_1700.csv"
)

effective_index_te2, _, _ = _read_effective_index(
    "neff_te2_fitted_Si_fitted_SiO2_width_240_5_700_wav_1200_0p1_1700.csv"
)
