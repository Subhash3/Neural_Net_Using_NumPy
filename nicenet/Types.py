import typing
import numpy as np

T_Features = np.ndarray
T_Targets = np.ndarray

T_DataSample = typing.Tuple[T_Features, T_Targets]
T_Dataset = typing.List[T_DataSample]
