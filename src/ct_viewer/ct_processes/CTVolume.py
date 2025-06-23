from .Globals import *

# np.set_printoptions(
#     formatter={
#         'float': lambda x: (
#                             f"{x:10.0f}" if abs(x) < 1e-4 else
#                             f"{x:10,.0f}"
#                             )
#                 }
#     )

#########################################################

# Initialize interp functions

def _initialize_interps():
    
    x = np.random.random(10)
    y = np.random.random(100).reshape(10, 10)
    z = np.random.random(1000).reshape(10, 10, 10)
    
    interp1d(0, 10, 1, x, k=1)(5)
    interp2d([0, 0], [10, 10], [1, 1], y, k=1)(5, 5)
    interp3d([0, 0, 0], [10, 10, 10], [1, 1, 1], z, taylor_order=1)(5, 5, 5)
    
    return

#########################################################

class CTVolume(object):

    mode_return_type = lambda x, y: x.astype(getattr(cp, y)) if G.GPU_MODE else lambda x, y: x.astype(getattr(np, y))
    mode_create_array = lambda x: cp.array(x) if G.GPU_MODE else lambda x: x
    mode_function = lambda x: getattr(cp, x.__name__) if G.GPU_MODE else getattr(np, x.__name__)
    mode_value = lambda x: getattr(cp, x.__str__()) if G.GPU_MODE else getattr(np, x.__str__())
    cp_to_np = lambda x: cp.asnumpy(x) if G.GPU_MODE else lambda x: x
    np_to_cp = lambda x: cp.asarray(x) if G.GPU_MODE else lambda x: x

    def __init__(self, 
                 name:str, 
                 file:Path, 
                 volume:np.ndarray, 
                 mask:np.ndarray = None, 
                 mask_fill:int|float = -10000, 
                 affine:np.ndarray = None,  
                 force_isotropic: bool = True,
                 start_pixel:np.ndarray|list[float]|tuple[float] = None,
                 dim_order:tuple[int, int, int] = (0, 1, 2)):
        """
        Can add fill value for mask, eg np.nan, -10000, etc. If None, no mask applied. 

        Parameters:
        ----------
        name: Name of the volume. 
        file: Source file of the volume.
        volume: data comprising the volume.
            Should be oriented in a way such that: 
                [d1, d2, d3]    -> [Y,  X,  Z], [j_hat, i_hat, k_hat]
                                -> [AP, LR, SI]
                                -> [+,  -,  -]
            So it should have the Y dimension first, X second, and Z third. 
            The physical dimensions are as follows: 
                Y dimension:    Anterior-Posterior  Increases towards Posterior
                X dimension:    Left-Right          Increases towards Left
                Z dimension:    Superior-Inferior   Increases towards Superior

            Due to the way arrays are read in and displayed, we need to
            flip the Y-axis in the interpolator. 
                
        """
        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()

        self.volume:np.ndarray|cp.ndarray = CTVolume.mode_function(np.moveaxis)(CTVolume.mode_return_type(CTVolume.mode_create_array(volume), 
                                                                                                          'float32'), 
                                                                                                (0, 1, 2), 
                                                                                                dim_order)
        self.force_isotropic = force_isotropic

        print(f'CTVolume Message: Initial Shape: {self.volume.shape}')

        self.unique_values = self.determine_unique_values(self.volume, difference_threshold = 500)
        self.name = name
        self.file = Path(file)
        self.orientation = ['AP', 'RL', 'IS'] # Anterior-Posterior, Right-Left, Inferior-Superior
        self.affine:np.ndarray|cp.ndarray = np.eye(4, dtype = np.float32)

        self.set_geometry(volume, 
                          affine, 
                          dim_order)

        self.affine = CTVolume.np_to_cp(self.affine)

        # if type(pixel_dims) == type(None):
        #     pixel_dims = np.abs(affine.diagonal()[:3])
        
        # if not isinstance(pixel_dims, np.ndarray):
        #     pixel_dims = np.array(pixel_dims)

        print(f'CTVolume Message: Affine: {self.affine}')

        # self.pixel_dims:np.ndarray = np.array(np.abs([pixel_dims[dim] for dim in dim_order]), dtype = np.float32)
        # self.pixel_steps:list = [x.item() for x in affine.diagonal()[:3]]
        # self.step_directions:np.ndarray = np.sign(self.pixel_steps, dtype = np.float32)
        # self.physical_shape:np.ndarray = np.array([x.item() for x in np.abs(self.pixel_steps)*np.array(self.volume.shape)], dtype = np.float32)
        # self.physical_start:np.ndarray = np.array([x.item() for x in self.affine[:3,3]], dtype = np.float32)
        # self.physical_stop:np.ndarray = self.physical_start + self.step_directions * (self.physical_shape - self.pixel_dims)
        # self.physical_center:np.ndarray = (self.physical_start + self.physical_stop)/2.0
        # self.physical_steps:np.ndarray = np.array(self.pixel_steps, dtype = np.float32)
        # self.shape:np.ndarray = np.array(self.volume.shape)
        self.corners = {'Right-Anterior-Inferior': [],
                        'Right-Anterior-Superior': [],
                        'Right-Posterior-Inferior': [],
                        'Right-Posterior-Superior': [],
                        'Left-Anterior-Inferior': [],
                        'Left-Anterior-Superior': [],
                        'Left-Posterior-Inferior': [],
                        'Left-Posterior-Superior': []}
        self.slice_direction = self.step_directions[2]

        print(f'CTVolume Message: {self.name} Physical Parameters:')
        print(f'\tPixel Dimensions [mm/pix]    {self.pixel_dims}')
        print(f'\tPixel Steps [mm/pix]         {self.pixel_steps}')
        print(f'\tPixel Directions             {self.step_directions}')
        print(f'\tPhysical Dimensions [mm]     {self.physical_shape}')
        print(f'\tPhysical Start               {self.physical_start}')
        print(f'\tPhysical Stop                {self.physical_stop}')
        print(f'\tPhysical Center [mm]         {self.physical_center}')
        print(f'\tPhysical Steps [mm]          {self.physical_steps}')

        self.mask = CTVolume.mode_function(np.zeros)(self.volume.shape, dtype = np.int8)
        self.mask[CTVolume.mode_function(np.isnan)(self.volume) | (self.volume < self.unique_values[0])] += 1
        self.volume[self.mask > 0] = CTVolume.mode_value(np.nan)
        self.original_shape = self.volume.shape

        self.texture_dim = int(np.ceil(np.sqrt(np.sum(self.physical_shape*self.physical_shape))))
        self.texture_center = float(self.texture_dim)/2.0
        
        G.TEXTURE_DIM = int(np.max([G.TEXTURE_DIM, self.texture_dim]))
            
        G.TEXTURE_CENTER = int(G.TEXTURE_DIM/2)

        if G.TEXTURE_CENTER > dpg.get_value('global_texture_center'):
            dpg.set_value('global_texture_center', 1*G.TEXTURE_CENTER)

        # These are also the centers for the same dimensions. 
        self.volume_center = CTVolume.np_to_cp((self.physical_shape.reshape(3, 1)/2.0).round(decimals = 2))

        print(f'CTVolume Message: {self.volume_center = }')

        self.volume_min = self.unique_values[0]
        self.volume_max = self.unique_values[-1]
        # Get range to set initial view parameters
        self.volume_range = self.volume_max - self.volume_min
        if self.volume_range == 0:
            self.volume_range = 1

        if len(self.unique_values) < 100:
            self.histogram_step = self.volume_range/100
        else:
            self.histogram_step = cp.median(cp.diff(self.unique_values)) if G.GPU_MODE else np.median(np.diff(self.unique_values))

        if self.volume_range/self.histogram_step > 5000:
            self.histogram_step = self.volume_range/5000.0

        # These need to be accessed by dpg. 
        if G.GPU_MODE:
            self.unique_values = cp.asnumpy(self.unique_values)
            self.histogram_step = cp.asnumpy(self.histogram_step)

        self.rounded_min_max = [(self.volume_min + self.volume_range/10).round(decimals = 0), 
                                (self.volume_max - self.volume_range/10).round(decimals = 0)]
    
        
        self.dim_shifts = np.array([0, 0, 0])
        # This contains lower limit and upper limit for use in interpolator.
        self.trans_coords_min_max = np.zeros((3, 2), dtype = int)
        self.trans_coords_shift = [0, 0, 0]
        # This is the entire grid for use as coordinates. 
        self.trans_coord_grid_list = []

        self.histogram = None
        self.bin_edges = None
        
        self.initialize_image_coords()

        print(f'CTVolume Message: {name = }, {self.shape = }, {G.TEXTURE_DIM = }, {G.TEXTURE_CENTER = }')


    def set_geometry(self, 
                     volume:np.ndarray, 
                     affine:np.ndarray,
                     dim_order:tuple[int, int, int]):
        
        """
        Sets geometry of the volume using the affine matrix and dim_order. Assigns values for the following: 

            self.pixel_dims
            self.pixel_steps
            self.step_directions
            self.physical_shape
            self.physical_start
            self.phsyical_stop
            self.physical_center
            self.physical_steps
            self.shape

        """
        
        for dim in dim_order:
            self.affine[dim] = affine[dim]

        pixel_dims = np.diag(self.affine)[:3]

        self.pixel_dims:np.ndarray = np.array(np.abs([pixel_dims[dim] for dim in dim_order]), dtype = np.float32)
        self.pixel_steps:list = [x.item() for x in self.affine.diagonal()[:3]]
        self.step_directions:np.ndarray = np.sign(self.pixel_steps, dtype = np.float32)
        self.physical_shape:np.ndarray = np.array([x.item() for x in np.abs(self.pixel_steps)*np.array(self.volume.shape)], dtype = np.float32)
        self.physical_start:np.ndarray = np.array([x.item() for x in self.affine[:3,3]], dtype = np.float32)
        self.physical_stop:np.ndarray = self.physical_start + self.step_directions * (self.physical_shape - self.pixel_dims)
        self.physical_center:np.ndarray = (self.physical_start + self.physical_stop)/2.0
        self.physical_steps:np.ndarray = np.array(self.pixel_steps, dtype = np.float32)
        self.shape:np.ndarray = np.array(self.volume.shape)

        flip_axes = tuple([])
        if self.step_directions[0] < 0: # Y
            flip_axes += (0,)

        if self.step_directions[1] < 0: # X
            flip_axes += (1,)

        if self.step_directions[2] > 0: # Z
            flip_axes += (2,)

        print(f'FLIP AXES: {flip_axes}')

        if len(flip_axes):
            self.volume = CTVolume.mode_function(np.flip)(self.volume, axis = flip_axes)


    def determine_unique_values(self, volume, difference_threshold = 500):
        # Get rounded unique values. Currently rounding to 0 decimals. 
        # Then get rid of nan values in unique values. 
        # Then get rid of mask values, which will be large outliers. 
        # cp.diff is a[i + 1] - a[i]
        rounded_vol = volume.round(decimals = 0)
        unique_values = CTVolume.mode_function(np.unique)(rounded_vol)
        unique_values = unique_values[~CTVolume.mode_function(np.isnan)(unique_values)]

        mask_indices = CTVolume.mode_function(np.full)(unique_values.shape, True, dtype=np.bool_)
        mask_indices[:-1] = CTVolume.mode_function(np.diff)(unique_values) < difference_threshold
        if len(unique_values) > 10:
            if unique_values[-1] - unique_values[-2] >= difference_threshold:
                mask_indices[-1] = False

        return CTVolume.mode_function(np.sort)(unique_values[mask_indices])

    def initialize_image_coords(self):
        for dim, shape in enumerate(self.shape):
            min_coord = int(-shape/2)
            max_coord = int(shape/2)
            self.trans_coords_shift[dim] = shape/2 # Add this to the transformed coordinates to undo the transformation.
            
            self.trans_coords_min_max[dim, 0] = min_coord
            self.trans_coords_min_max[dim, 1] = max_coord - 1
            self.trans_coord_grid_list.append(list(range(min_coord, max_coord)))

    def interpolate_volume(self, coords, order = 1, out_array = None):
        """
        Coords are in the shape of: 

            [3, xdim, ydim, zdim]

        """

        if type(out_array) == type(None):
            return ndi.map_coordinates(self.volume, coords, order = order, cval=cp.nan)
        
        else:
            ndi.map_coordinates(self.volume, coords, order = order, output=out_array, cval=cp.nan)


    def interpolate_mask(self, coords, order = 1, out_array = None):
        """
        Coords are in the shape of: 

            [3, xdim, ydim, zdim]

        """

        if type(out_array) == type(None):
            return ndi.map_coordinates(self.mask, coords, order = order, cval=cp.nan)
        
        else:
            ndi.map_coordinates(self.mask, coords, order = order, output=out_array, cval=cp.nan)
            

    def _cleanup_(self):
        dict_keys = list(self.__dict__.keys())
        while len(dict_keys) > 0:
            attrib_key = dict_keys.pop()
            setattr(self, attrib_key, None)
            delattr(self, attrib_key)
            cp._default_memory_pool.free_all_blocks()