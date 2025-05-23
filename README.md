Full Python implementation of GDS as a library.

Usage example:

```
from pygds_lib import PYGDS

# Basic usage
gds = PYGDS(buffer_size=4*1024, pattern_byte=0xAB)
gds.run_demo("test_gds.bin")

# Custom usage
gds = PYGDS(buffer_size=8*1024, pattern_byte=0xCD)
gds.setup_buffers()
gds.open_file("custom_file.bin")
size_mb, time_ms, bandwidth = gds.write_buffer()
print(f"Custom write: {size_mb:.2f} MB in {time_ms:.2f} ms ({bandwidth:.2f} GB/s)")
gds.cleanup()
```
