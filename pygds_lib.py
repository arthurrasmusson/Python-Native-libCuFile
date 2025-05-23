import os
import time
import ctypes
from cuda import cuda

class CUfileError(ctypes.Structure):
    _fields_ = [("err", ctypes.c_int), ("cu_err", ctypes.c_int)]

class _DescrUnion(ctypes.Union):
    _fields_ = [("fd", ctypes.c_int)]

class CUfileDescr(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("handle", _DescrUnion)]

class PYGDS:
    """GPUDirect Storage I/O library for direct GPU-NVMe data transfer."""
    
    def __init__(self, buffer_size=4*1024, pattern_byte=0xAB):
        """Initialize CUDA context and cuFile driver."""
        self.buffer_size = buffer_size
        self.pattern_byte = pattern_byte
        self._initialize_cuda()
        self._initialize_cufile()
        self.device = None
        self.ctx = None
        self.dptr_w = None
        self.dptr_r = None
        self.hptr = None
        self.cf_handle = None
        self.fd = None

    def _check_cuda(self, err, msg=""):
        """Check CUDA API call return value."""
        if err != 0:
            raise RuntimeError(f"{msg} (CUDA err={err})")

    def _check_cufile(self, status: CUfileError, name: str):
        """Check cuFile API call return value."""
        if status.err != 0:
            raise RuntimeError(f"{name} failed (cuFile err={status.err}, cuda_err={status.cu_err})")

    def _initialize_cuda(self):
        """Initialize CUDA driver and context."""
        self._check_cuda(cuda.cuInit(0)[0], "cuInit failed")
        self.device = cuda.cuDeviceGet(0)[1]
        self.ctx = cuda.cuCtxCreate(0, self.device)[1]

    def _initialize_cufile(self):
        """Load and initialize cuFile library."""
        self.libcufile = ctypes.CDLL("libcufile.so")
        
        # Define cuFile API signatures
        self.libcufile.cuFileDriverOpen.restype = CUfileError
        self.libcufile.cuFileDriverClose.restype = CUfileError
        self.libcufile.cuFileHandleRegister.restype = CUfileError
        self.libcufile.cuFileHandleDeregister.restype = CUfileError
        self.libcufile.cuFileBufRegister.restype = CUfileError
        self.libcufile.cuFileBufDeregister.restype = CUfileError
        self.libcufile.cuFileHandleRegister.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(CUfileDescr)]
        self.libcufile.cuFileBufRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.libcufile.cuFileBufDeregister.argtypes = [ctypes.c_void_p]
        self.libcufile.cuFileRead.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                            ctypes.c_longlong, ctypes.c_longlong]
        self.libcufile.cuFileWrite.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                             ctypes.c_longlong, ctypes.c_longlong]
        
        self._check_cufile(self.libcufile.cuFileDriverOpen(), "cuFileDriverOpen")

    def setup_buffers(self):
        """Allocate and initialize GPU buffers."""
        self.dptr_w = cuda.cuMemAlloc(self.buffer_size)[1]
        self.dptr_r = cuda.cuMemAlloc(self.buffer_size)[1]
        cuda.cuMemsetD8(self.dptr_w, ctypes.c_ubyte(self.pattern_byte), self.buffer_size)
        
        for ptr in (self.dptr_w, self.dptr_r):
            self._check_cufile(
                self.libcufile.cuFileBufRegister(ctypes.c_void_p(int(ptr)), self.buffer_size, 0),
                "cuFileBufRegister"
            )

    def open_file(self, filepath, flags=os.O_CREAT | os.O_RDWR | os.O_TRUNC | os.O_DIRECT, mode=0o644):
        """Open file and register cuFile handle."""
        self.fd = os.open(filepath, flags, mode)
        descr = CUfileDescr()
        descr.type = 1  # CU_FILE_HANDLE_TYPE_OPAQUE_FD
        descr.handle.fd = self.fd
        self.cf_handle = ctypes.c_void_p()
        self._check_cufile(
            self.libcufile.cuFileHandleRegister(ctypes.byref(self.cf_handle), ctypes.byref(descr)),
            "cuFileHandleRegister"
        )

    def write_buffer(self, file_offset=0, device_offset=0):
        """Write GPU buffer to file and return bandwidth."""
        t0 = time.perf_counter()
        n = self.libcufile.cuFileWrite(
            self.cf_handle, ctypes.c_void_p(int(self.dptr_w)),
            self.buffer_size, file_offset, device_offset
        )
        dt = time.perf_counter() - t0
        if n != self.buffer_size:
            raise RuntimeError(f"cuFileWrite returned {n}")
        bandwidth = (n / dt) / 1e9  # GB/s
        return n / 1e6, dt * 1e3, bandwidth  # MB, ms, GB/s

    def read_buffer(self, file_offset=0, device_offset=0):
        """Read from file to GPU buffer and return bandwidth."""
        t0 = time.perf_counter()
        n = self.libcufile.cuFileRead(
            self.cf_handle, ctypes.c_void_p(int(self.dptr_r)),
            self.buffer_size, file_offset, device_offset
        )
        dt = time.perf_counter() - t0
        if n != self.buffer_size:
            raise RuntimeError(f"cuFileRead returned {n}")
        bandwidth = (n / dt) / 1e9  # GB/s
        return n / 1e6, dt * 1e3, bandwidth  # MB, ms, GB/s

    def verify_buffer(self):
        """Copy GPU buffer to host and verify pattern."""
        self.hptr = cuda.cuMemAllocHost(self.buffer_size)[1]
        self._check_cuda(
            cuda.cuMemcpyDtoH(self.hptr, self.dptr_r, self.buffer_size)[0],
            "cuMemcpyDtoH"
        )
        host_buf = (ctypes.c_ubyte * self.buffer_size).from_address(self.hptr)
        if any(b != self.pattern_byte for b in host_buf):
            raise RuntimeError("Verification FAILED: data mismatch!")
        return True

    def cleanup(self):
        """Release all resources."""
        if self.cf_handle:
            self.libcufile.cuFileHandleDeregister(self.cf_handle)
        if self.fd is not None:
            os.close(self.fd)
        if self.dptr_w:
            self.libcufile.cuFileBufDeregister(ctypes.c_void_p(int(self.dptr_w)))
            cuda.cuMemFree(self.dptr_w)
        if self.dptr_r:
            self.libcufile.cuFileBufDeregister(ctypes.c_void_p(int(self.dptr_r)))
            cuda.cuMemFree(self.dptr_r)
        if self.hptr:
            cuda.cuMemFreeHost(self.hptr)
        if self.ctx:
            cuda.cuCtxDestroy(self.ctx)
        self.libcufile.cuFileDriverClose()

    def run_demo(self, filepath="test_gds.bin"):
        """Run complete GPUDirect Storage demo."""
        try:
            self.setup_buffers()
            self.open_file(filepath)
            
            # Write operation
            mb_written, ms_taken, bandwidth = self.write_buffer()
            print(f"WRITE  {mb_written:.2f} MB in {ms_taken:.2f} ms  ({bandwidth:.2f} GB/s)")
            
            # Read operation
            mb_read, ms_taken, bandwidth = self.read_buffer()
            print(f"READ   {mb_read:.2f} MB in {ms_taken:.2f} ms  ({bandwidth:.2f} GB/s)")
            
            # Verify
            if self.verify_buffer():
                print(f"Verification PASSED (all bytes 0x{self.pattern_byte:02X}).")
                
        finally:
            self.cleanup()

if __name__ == "__main__":
    gds = PYGDS()
    gds.run_demo()
