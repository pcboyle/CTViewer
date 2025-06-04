import cupy as cp
import numpy as np # Need numpy for _compute_bounds helper

# ==============================================================================
# Helper Functions (Refactored CuPy Versions)
# ==============================================================================

def _cupy_extrapolate1d_x(padded_data: cp.ndarray, interpolation_order: int, padding_offset: int):
    """Extrapolates data along the first axis (axis 0). (Implementation as above)"""
    nx = padded_data.shape[0]
    k = interpolation_order # Use shorter name internally for formula clarity
    o = padding_offset
    for ix in range(o):
        il = o - ix - 1
        ih = nx - (o - ix)
        if k == 1:
            padded_data[il] = 2 * padded_data[il + 1] - padded_data[il + 2]
            padded_data[ih] = 2 * padded_data[ih - 1] - padded_data[ih - 2]
        elif k == 3:
            padded_data[il] = 4 * padded_data[il + 1] - 6 * padded_data[il + 2] + 4 * padded_data[il + 3] - padded_data[il + 4]
            padded_data[ih] = 4 * padded_data[ih - 1] - 6 * padded_data[ih - 2] + 4 * padded_data[ih - 3] - padded_data[ih - 4]
        elif k == 5:
            padded_data[il] = 6*padded_data[il+1] - 15*padded_data[il+2] + 20*padded_data[il+3] - 15*padded_data[il+4] + 6*padded_data[il+5] - padded_data[il+6]
            padded_data[ih] = 6*padded_data[ih-1] - 15*padded_data[ih-2] + 20*padded_data[ih-3] - 15*padded_data[ih-4] + 6*padded_data[ih-5] - padded_data[ih-6]
        elif k == 7:
            padded_data[il] = 8*padded_data[il+1] - 28*padded_data[il+2] + 56*padded_data[il+3] - 70*padded_data[il+4] + 56*padded_data[il+5] - 28*padded_data[il+6] + 8*padded_data[il+7] - padded_data[il+8]
            padded_data[ih] = 8*padded_data[ih-1] - 28*padded_data[ih-2] + 56*padded_data[ih-3] - 70*padded_data[ih-4] + 56*padded_data[ih-5] - 28*padded_data[ih-6] + 8*padded_data[ih-7] - padded_data[ih-8]
        elif k == 9:
            padded_data[il] = 10*padded_data[il+1] - 45*padded_data[il+2] + 120*padded_data[il+3] - 210*padded_data[il+4] + 252*padded_data[il+5] - 210*padded_data[il+6] + 120*padded_data[il+7] - 45*padded_data[il+8] + 10*padded_data[il+9] - padded_data[il+10]
            padded_data[ih] = 10*padded_data[ih-1] - 45*padded_data[ih-2] + 120*padded_data[ih-3] - 210*padded_data[ih-4] + 252*padded_data[ih-5] - 210*padded_data[ih-6] + 120*padded_data[ih-7] - 45*padded_data[ih-8] + 10*padded_data[ih-9] - padded_data[ih-10]

def _cupy_extrapolate1d_y(padded_data: cp.ndarray, interpolation_order: int, padding_offset: int):
    """Extrapolates data along the second axis (axis 1). (Implementation as above)"""
    ny = padded_data.shape[1]
    k = interpolation_order
    o = padding_offset
    for ix in range(o):
        il = o - ix - 1
        ih = ny - (o - ix)
        if k == 1:
            padded_data[:, il] = 2 * padded_data[:, il + 1] - padded_data[:, il + 2]
            padded_data[:, ih] = 2 * padded_data[:, ih - 1] - padded_data[:, ih - 2]
        elif k == 3:
            padded_data[:, il] = 4 * padded_data[:, il + 1] - 6 * padded_data[:, il + 2] + 4 * padded_data[:, il + 3] - padded_data[:, il + 4]
            padded_data[:, ih] = 4 * padded_data[:, ih - 1] - 6 * padded_data[:, ih - 2] + 4 * padded_data[:, ih - 3] - padded_data[:, ih - 4]
        elif k == 5:
            padded_data[:, il] = 6*padded_data[:, il+1] - 15*padded_data[:, il+2] + 20*padded_data[:, il+3] - 15*padded_data[:, il+4] + 6*padded_data[:, il+5] - padded_data[:, il+6]
            padded_data[:, ih] = 6*padded_data[:, ih-1] - 15*padded_data[:, ih-2] + 20*padded_data[:, ih-3] - 15*padded_data[:, ih-4] + 6*padded_data[:, ih-5] - padded_data[:, ih-6]
        elif k == 7:
            padded_data[:, il] = 8*padded_data[:, il+1] - 28*padded_data[:, il+2] + 56*padded_data[:, il+3] - 70*padded_data[:, il+4] + 56*padded_data[:, il+5] - 28*padded_data[:, il+6] + 8*padded_data[:, il+7] - padded_data[:, il+8]
            padded_data[:, ih] = 8*padded_data[:, ih-1] - 28*padded_data[:, ih-2] + 56*padded_data[:, ih-3] - 70*padded_data[:, ih-4] + 56*padded_data[:, ih-5] - 28*padded_data[:, ih-6] + 8*padded_data[:, ih-7] - padded_data[:, ih-8]
        elif k == 9:
            padded_data[:, il] = 10*padded_data[:, il+1] - 45*padded_data[:, il+2] + 120*padded_data[:, il+3] - 210*padded_data[:, il+4] + 252*padded_data[:, il+5] - 210*padded_data[:, il+6] + 120*padded_data[:, il+7] - 45*padded_data[:, il+8] + 10*padded_data[:, il+9] - padded_data[:, il+10]
            padded_data[:, ih] = 10*padded_data[:, ih-1] - 45*padded_data[:, ih-2] + 120*padded_data[:, ih-3] - 210*padded_data[:, ih-4] + 252*padded_data[:, ih-5] - 210*padded_data[:, ih-6] + 120*padded_data[:, ih-7] - 45*padded_data[:, ih-8] + 10*padded_data[:, ih-9] - padded_data[:, ih-10]

def _cupy_extrapolate1d_z(padded_data: cp.ndarray, interpolation_order: int, padding_offset: int):
    """Extrapolates data along the third axis (axis 2). (Implementation as above)"""
    nz = padded_data.shape[2]
    k = interpolation_order
    o = padding_offset
    for ix in range(o):
        il = o - ix - 1
        ih = nz - (o - ix)
        if k == 1:
            padded_data[:, :, il] = 2 * padded_data[:, :, il + 1] - padded_data[:, :, il + 2]
            padded_data[:, :, ih] = 2 * padded_data[:, :, ih - 1] - padded_data[:, :, ih - 2]
        elif k == 3:
            padded_data[:, :, il] = 4 * padded_data[:, :, il + 1] - 6 * padded_data[:, :, il + 2] + 4 * padded_data[:, :, il + 3] - padded_data[:, :, il + 4]
            padded_data[:, :, ih] = 4 * padded_data[:, :, ih - 1] - 6 * padded_data[:, :, ih - 2] + 4 * padded_data[:, :, ih - 3] - padded_data[:, :, ih - 4]
        elif k == 5:
            padded_data[:, :, il] = 6*padded_data[:, :, il+1] - 15*padded_data[:, :, il+2] + 20*padded_data[:, :, il+3] - 15*padded_data[:, :, il+4] + 6*padded_data[:, :, il+5] - padded_data[:, :, il+6]
            padded_data[:, :, ih] = 6*padded_data[:, :, ih-1] - 15*padded_data[:, ih-2] + 20*padded_data[:, ih-3] - 15*padded_data[:, ih-4] + 6*padded_data[:, ih-5] - padded_data[:, ih-6]
        elif k == 7:
            padded_data[:, :, il] = 8*padded_data[:, :, il+1] - 28*padded_data[:, :, il+2] + 56*padded_data[:, :, il+3] - 70*padded_data[:, :, il+4] + 56*padded_data[:, :, il+5] - 28*padded_data[:, :, il+6] + 8*padded_data[:, :, il+7] - padded_data[:, :, il+8]
            padded_data[:, :, ih] = 8*padded_data[:, :, ih-1] - 28*padded_data[:, :, ih-2] + 56*padded_data[:, :, ih-3] - 70*padded_data[:, ih-4] + 56*padded_data[:, ih-5] - 28*padded_data[:, ih-6] + 8*padded_data[:, ih-7] - padded_data[:, ih-8]
        elif k == 9:
            padded_data[:, :, il] = 10*padded_data[:, :, il+1] - 45*padded_data[:, :, il+2] + 120*padded_data[:, :, il+3] - 210*padded_data[:, :, il+4] + 252*padded_data[:, :, il+5] - 210*padded_data[:, :, il+6] + 120*padded_data[:, :, il+7] - 45*padded_data[:, :, il+8] + 10*padded_data[:, :, il+9] - padded_data[:, :, il+10]
            padded_data[:, :, ih] = 10*padded_data[:, :, ih-1] - 45*padded_data[:, ih-2] + 120*padded_data[:, :, ih-3] - 210*padded_data[:, :, ih-4] + 252*padded_data[:, :, ih-5] - 210*padded_data[:, ih-6] + 120*padded_data[:, ih-7] - 45*padded_data[:, ih-8] + 10*padded_data[:, :, ih-9] - padded_data[:, :, ih-10]


def _cupy_fill3(input_data: cp.ndarray, padded_data: cp.ndarray, offset_x: int, offset_y: int, offset_z: int):
    """Copies input_data into the center of padded_data. (Implementation as above)"""
    nx, ny, nz = input_data.shape
    padded_data[offset_x : offset_x + nx, offset_y : offset_y + ny, offset_z : offset_z + nz] = input_data

def _cupy_extrapolate3d(input_data: cp.ndarray, interpolation_order: int, is_periodic: list[bool], use_padding: list[bool], extrapolation_distance: list[int]):
    """Pads and extrapolates a 3D CuPy array. (Implementation as above)"""
    padx = (not is_periodic[0]) and use_padding[0]
    pady = (not is_periodic[1]) and use_padding[1]
    padz = (not is_periodic[2]) and use_padding[2]
    if padx or pady or padz:
        k = interpolation_order
        offset_x = (k // 2) + extrapolation_distance[0] if padx else 0
        offset_y = (k // 2) + extrapolation_distance[1] if pady else 0
        offset_z = (k // 2) + extrapolation_distance[2] if padz else 0
        offsets = [offset_x, offset_y, offset_z]
        padded_shape = (input_data.shape[0] + 2 * offset_x, input_data.shape[1] + 2 * offset_y, input_data.shape[2] + 2 * offset_z)
        padded_data = cp.zeros(padded_shape, dtype=input_data.dtype)
        _cupy_fill3(input_data, padded_data, offset_x, offset_y, offset_z)
        if padx: _cupy_extrapolate1d_x(padded_data, k, offset_x)
        if pady: _cupy_extrapolate1d_y(padded_data, k, offset_y)
        if padz: _cupy_extrapolate1d_z(padded_data, k, offset_z)
        return padded_data, offsets
    else:
        return input_data, [0, 0, 0]

# Helper from fast_interp.py to calculate bounds (runs on CPU during init)
def _compute_bounds1(a, b, h, p, c, e, k):
    if p:
        return -1e100, 1e100
    elif not c:
        d = h*(k//2)
        return a+d, b-d
    else:
        d = e*h
        u = b+d
        # subtracting eps protects against floating point errors if xr==ub
        # maybe not needed?
        u -= u*1e-15
        return a-d, u

def _compute_bounds(a, b, h, p, c, e, k):
    m = len(a)
    bounds = [_compute_bounds1(a[i], b[i], h[i], p[i], c[i], e[i], k) for i in range(m)]
    # Returns [[lb_x, lb_y, lb_z], [ub_x, ub_y, ub_z]]
    return [list(x) for x in zip(*bounds)]

# ==============================================================================
# Elementwise Kernel Definition (k=1)
# ==============================================================================
_interp3d_k1_kernel = cp.ElementwiseKernel(
    '''
    raw T input_data,
    float64 xout, float64 yout, float64 zout,
    raw float64 lower_bounds, raw float64 grid_spacings,
    raw int32 data_shape, raw bool is_periodic, raw int32 padding_offsets,
    raw float64 processed_lower_bounds, raw float64 processed_upper_bounds
    ''',
    'T fout', # Output argument
    '''
    typedef double T; // Assuming data type is float64 for T
    // Bounds checking
    T xr = min(max(xout, processed_lower_bounds[0]), processed_upper_bounds[0]);
    T yr = min(max(yout, processed_lower_bounds[1]), processed_upper_bounds[1]);
    T zr = min(max(zout, processed_lower_bounds[2]), processed_upper_bounds[2]);
    // Calculate indices and ratios
    T xx = xr - lower_bounds[0];
    T yy = yr - lower_bounds[1];
    T zz = zr - lower_bounds[2];
    int ix = floor(xx / grid_spacings[0]);
    int iy = floor(yy / grid_spacings[1]);
    int iz = floor(zz / grid_spacings[2]);
    T ratx = xx / grid_spacings[0] - (ix + 0.5);
    T raty = yy / grid_spacings[1] - (iy + 0.5);
    T ratz = zz / grid_spacings[2] - (iz + 0.5);
    // Weights
    T asx[2]; 
    T asy[2]; 
    T asz[2];
    asx[0] = 0.5 - ratx; 
    asx[1] = 0.5 + ratx;
    asy[0] = 0.5 - raty; 
    asy[1] = 0.5 + raty;
    asz[0] = 0.5 - ratz; 
    asz[1] = 0.5 + ratz;
    // Add offsets
    ix += padding_offsets[0]; 
    iy += padding_offsets[1]; 
    iz += padding_offsets[2];
    // Interpolation sum
    fout = 0.0;
    for (int i = 0; i < 2; ++i) {
        int ixi = ix + i;
        if (is_periodic[0]) { ixi %= data_shape[0]; if (ixi < 0) ixi += data_shape[0]; }
        for (int j = 0; j < 2; ++j) {
            int iyj = iy + j;
            if (is_periodic[1]) { iyj %= data_shape[1]; if (iyj < 0) iyj += data_shape[1]; }
            for (int k = 0; k < 2; ++k) {
                int izk = iz + k;
                if (is_periodic[2]) { izk %= data_shape[2]; if (izk < 0) izk += data_shape[2]; }
                // Access padded data using multi-dimensional index helper input_data(x,y,z)
                fout += input_data(ixi, iyj, izk) * asx[i] * asy[j] * asz[k];
            }
        }
    }
    ''',
    name='interp3d_k1_kernel'
)

# Wrapper function for k=1 kernel
def _interp3d_k1_cupy(
    padded_data: cp.ndarray,
    xout: cp.ndarray, yout: cp.ndarray, zout: cp.ndarray, fout: cp.ndarray,
    lower_bounds: cp.ndarray, grid_spacings: cp.ndarray,
    data_shape: cp.ndarray, is_periodic: cp.ndarray, padding_offsets: cp.ndarray,
    processed_lower_bounds: cp.ndarray, processed_upper_bounds: cp.ndarray):
    """Calls the compiled ElementwiseKernel for k=1 interpolation."""
    # Ensure correct types for kernel arguments
    xout_k = xout.astype(cp.float64, copy=False)
    yout_k = yout.astype(cp.float64, copy=False)
    zout_k = zout.astype(cp.float64, copy=False)
    lb_k = lower_bounds.astype(cp.float64, copy=False)
    gs_k = grid_spacings.astype(cp.float64, copy=False)
    ds_k = data_shape.astype(cp.int32, copy=False)
    ip_k = is_periodic.astype(cp.bool_, copy=False)
    po_k = padding_offsets.astype(cp.int32, copy=False)
    plb_k = processed_lower_bounds.astype(cp.float64, copy=False)
    pub_k = processed_upper_bounds.astype(cp.float64, copy=False)

    # Call the kernel
    _interp3d_k1_kernel(
        padded_data, xout_k, yout_k, zout_k,
        lb_k, gs_k, ds_k, ip_k, po_k, plb_k, pub_k,
        fout # Output array is modified in-place
    )

# ==============================================================================
# interp3d_cupy Class Definition
# ==============================================================================

# Placeholder for higher-order kernels (implement later if needed)
# _interp3d_k3_kernel = ...
# _interp3d_k3_cupy = ...
# ... etc

INTERP_3D_CUPY = {
    1: _interp3d_k1_cupy,
    # 3: _interp3d_k3_cupy, # Add once implemented
    # 5: _interp3d_k5_cupy,
    # 7: _interp3d_k7_cupy,
    # 9: _interp3d_k9_cupy,
}

class interp3d_cupy(object):
    """
    3D Interpolator class using CuPy for GPU acceleration.

    Mirrors the interface of fast_interp.interp3d but operates on the GPU.
    Currently only supports interpolation_order k=1 (linear).
    """
    def __init__(self, lower_bounds: list[float], upper_bounds: list[float], grid_spacings: list[float],
                 input_data: np.ndarray, interpolation_order: int = 1,
                 is_periodic: list[bool] = [False]*3, use_padding: list[bool] = [True]*3,
                 extrapolation_distance: list[int] = [0]*3):
        """
        Initializes the CuPy 3D interpolator.

        Args:
            lower_bounds: List of lower bounds for each dimension [ax, ay, az].
            upper_bounds: List of upper bounds for each dimension [bx, by, bz].
            grid_spacings: List of grid spacings for each dimension [hx, hy, hz].
            input_data: The 3D data array (NumPy array).
            interpolation_order: Order of interpolation (default 1, currently only 1 supported).
            is_periodic: List of booleans indicating periodicity [px, py, pz].
            use_padding: List of booleans indicating whether to pad boundaries [cx, cy, cz].
            extrapolation_distance: List of integers for extrapolation distance [ex, ey, ez].
        """
        if interpolation_order not in INTERP_3D_CUPY:
             raise NotImplementedError(f"CuPy interpolation order k={interpolation_order} is not yet implemented.")

        self.interpolation_order = interpolation_order
        self.dtype = input_data.dtype # Preserve original dtype if possible? Or force float32/64? Let's use input dtype for now.
        if not np.issubdtype(self.dtype, np.floating):
            print(f"Warning: Input data type {self.dtype} is not float. Casting to float64 for interpolation.")
            self.dtype = np.float64


        # Store parameters as CuPy arrays (use get() for helpers needing lists)
        self.lower_bounds_cp = cp.asarray(lower_bounds, dtype=cp.float64)
        self.upper_bounds_cp = cp.asarray(upper_bounds, dtype=cp.float64)
        self.grid_spacings_cp = cp.asarray(grid_spacings, dtype=cp.float64)
        self.is_periodic_cp = cp.asarray(is_periodic, dtype=cp.bool_)
        self.use_padding_cp = cp.asarray(use_padding, dtype=cp.bool_)
        self.extrapolation_distance_cp = cp.asarray(extrapolation_distance, dtype=cp.int32)
        self.data_shape_cp = cp.asarray(input_data.shape, dtype=cp.int32)

        # Transfer input data to GPU and ensure correct float type
        _input_data_cp = cp.asarray(input_data, dtype=self.dtype)

        # Handle extrapolation and padding using CuPy versions
        # Pass lists/ints to the helper, which expects Python types for logic
        self._padded_data, _offsets = _cupy_extrapolate3d(
            _input_data_cp,
            self.interpolation_order,
            is_periodic, # Pass Python list
            use_padding, # Pass Python list
            extrapolation_distance # Pass Python list
        )
        self._padding_offsets_cp = cp.asarray(_offsets, dtype=cp.int32)

        # Compute processed bounds (CPU calculation during init is fine)
        lb, ub = _compute_bounds(
            lower_bounds, upper_bounds, grid_spacings,
            is_periodic, use_padding, extrapolation_distance,
            self.interpolation_order
        )
        self.processed_lower_bounds_cp = cp.asarray(lb, dtype=cp.float64)
        self.processed_upper_bounds_cp = cp.asarray(ub, dtype=cp.float64)

        # Select the appropriate interpolation function (kernel wrapper)
        self._interp_func = INTERP_3D_CUPY[self.interpolation_order]


    def __call__(self, xout, yout, zout, fout=None):
        """
        Performs interpolation on the GPU.

        Args:
            xout, yout, zout: Coordinate arrays (NumPy or CuPy) at which to interpolate.
                              Must be broadcastable to the same shape.
            fout: Optional CuPy array to store the output. If None, a new CuPy
                  array is created.

        Returns:
            cp.ndarray: The interpolated values in a CuPy array with the same
                        shape as the broadcasted coordinate arrays.
        """
        # Ensure coordinates are CuPy arrays
        _xout = cp.asarray(xout)
        _yout = cp.asarray(yout)
        _zout = cp.asarray(zout)

        # Ensure coordinates can be broadcast together and find output shape
        try:
            output_shape = np.broadcast(_xout.get(), _yout.get(), _zout.get()).shape
        except ValueError:
            raise ValueError("Coordinate arrays (xout, yout, zout) could not be broadcast together.")

        # Handle output array
        if fout is None:
            _fout = cp.empty(output_shape, dtype=self.dtype)
        else:
            if not isinstance(fout, cp.ndarray):
                 raise TypeError("fout must be a CuPy ndarray if provided.")
            if fout.shape != output_shape:
                 raise ValueError(f"fout shape {fout.shape} does not match broadcast coordinate shape {output_shape}.")
            if fout.dtype != self.dtype:
                 raise TypeError(f"fout dtype {fout.dtype} does not match interpolator dtype {self.dtype}.")
            _fout = fout # Use the provided array

        # Ravel inputs for the kernel wrapper, which expects flat arrays
        # Use broadcast_to for coordinate arrays if they aren't full shape
        x_flat = cp.broadcast_to(_xout, output_shape).ravel()
        y_flat = cp.broadcast_to(_yout, output_shape).ravel()
        z_flat = cp.broadcast_to(_zout, output_shape).ravel()
        out_flat = _fout.ravel() # Operate on the raveled view

        # Call the selected GPU interpolation function (kernel wrapper)
        self._interp_func(
            self._padded_data,
            x_flat, y_flat, z_flat, out_flat,
            self.lower_bounds_cp, self.grid_spacings_cp,
            self.data_shape_cp, self.is_periodic_cp, self._padding_offsets_cp,
            self.processed_lower_bounds_cp, self.processed_upper_bounds_cp
        )

        # Return the output array (already shaped correctly if fout was None,
        # or the modified input fout array)
        return _fout