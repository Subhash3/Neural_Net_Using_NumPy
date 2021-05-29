import typing
import numpy as np

T_Feature_Array = np.ndarray
T_Target_Array = np.ndarray

T_Data_Sample = typing.Tuple[T_Feature_Array, T_Target_Array]
T_Dataset = typing.List[T_Data_Sample]
