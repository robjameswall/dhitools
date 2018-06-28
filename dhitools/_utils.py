import numpy as np
import clr

# Import .NET libraries

clr.AddReference('System')

import System
from System import Array
from System.Runtime.InteropServices import GCHandle, GCHandleType

import ctypes


def dotnet_arr_to_ndarr(dotnet_arr):
    """
    Converts data read in with dotNet (i.e. reading dfsu through DHI .NET
    libraries) and efficiently converts to numpy ndarray

    Parameters
    ----------
    dotnet_array : either: System.Double, System.Single, System.Int32
        .NET array read in using pythonnet

    Returns
    -------
    ndarray : dotnet_array converted to ndarray

    """

    src_hndl = GCHandle.Alloc(dotnet_arr, GCHandleType.Pinned)

    try:
        src_ptr = src_hndl.AddrOfPinnedObject().ToInt64()
        dtype = dotnet_arr.GetType()

        if dtype == Array.CreateInstance(System.Single,0).GetType():
            bufType = ctypes.c_float*len(dotnet_arr)
        elif dtype == Array.CreateInstance(System.Int32,0).GetType():
            bufType = ctypes.c_int*len(dotnet_arr)
        elif dtype == Array.CreateInstance(System.Double,0).GetType():
            bufType = ctypes.c_double*len(dotnet_arr)
        cbuf = bufType.from_address(src_ptr)
        ndarray = np.frombuffer(cbuf, dtype=cbuf._type_)
    finally:
        if src_hndl.IsAllocated:
            src_hndl.Free()
    return ndarray
