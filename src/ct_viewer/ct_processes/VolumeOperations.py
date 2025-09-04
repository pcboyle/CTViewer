from .Globals import *

class VolumeOperations(object):
    def __init__(self, 
                 texture_dim: int = G.TEXTURE_DIM,
                 max_volume_size: int = 256):
        """
        self.texture_dim            -> Int      Texture edge length
        self.enabled                -> Bool     When an operation is being done
        self.start                  -> Int      Start index
        self.stop                   -> Int      Stop index
        self.operation              -> Str      Operation choice: Add, Multiply, Difference, Mean, Max, Min, Std Dev
        self.weighted               -> ndarray  Weights for each index
        self.interpolation_slab     -> ndarray  Array of indices to interpolate. Shape (3, max_volume_size, texture_dim, texture_dim)
        self.volume_slab            -> ndarray  Interpolated volume. Shape (max_volume_size, texture_dim, texture_dim)
        self.texture_content        -> ndarray  Final texture content. Shape (texture_dim, texture_dim)
        
        The max_volume_size parameter determines the size, and therefore GPU memory footprint, of the interpolation volume. 
        A volume of size (512, 860, 860) of float32's is ~1150 MB. 
        The GPU uses ~6 GB of VRAM between self.interpolation (4.5 GB) slab and self.volume (1.5 GB) slab.
        
        """

        self.texture_dim = texture_dim
        self.enabled = False
        self.start = 0
        self.stop = 0
        self.operation = ''
        self.weighted = False
        self.interpolation_slab:np.ndarray|cp.ndarray = cp.full((3, max_volume_size, self.texture_dim, self.texture_dim), fill_value = cp.nan, dtype = cp.float32)
        self.volume_slab: np.ndarray|cp.ndarray = cp.full((max_volume_size, self.texture_dim, self.texture_dim), fill_value = cp.nan, dtype = cp.float32)
        self.texture_content:np.ndarray|cp.ndarray = cp.full((self.texture_dim, self.texture_dim), fill_value = cp.nan, dtype = cp.float32)
        self.norm_array = cp.zeros((3, max_volume_size), dtype = cp.float32) # norm_vector * cp.linspace(start, stop, steps, dtype = cp.float32)
        
    
    def set_operation(self, 
                      enabled: bool,
                      start: int, 
                      stop: int,
                      operation: str,
                      weighted: np.ndarray | cp.ndarray) -> None:
        
        self.enabled = enabled
        self.start = start
        self.stop = stop
        self.operation = operation
        self.weighted = weighted
    
    def format_interpolation_slab(self,
                                  norm_vector: np.ndarray | cp.ndarray, # (3, 1)
                                  volume_view_plane: np.ndarray | cp.ndarray, # (3, N) Points
                                  start: int,
                                  stop: int) -> cp.ndarray:
        """
        Output array is of shape (start - stop + 1, texture_dim, texture_dim)
        
        """
        steps = stop - start + 1
        self.norm_array[:, :steps] = norm_vector * cp.linspace(start, stop, steps, dtype = cp.float32)
        self.interpolation_slab[:, :steps, :, :] = volume_view_plane.reshape((3, 1, self.texture_dim, self.texture_dim)).repeat(steps, axis = 1)
        # view_slab = volume_view_plane.reshape((3, 1, volume_view_plane.shape[1])).repeat(steps, axis = 1)
        self.interpolation_slab[:, :steps] = cp.moveaxis(cp.moveaxis(self.interpolation_slab[:, :steps], (0, 1, 2, 3), (2, 3, 0, 1)) 
                                                         + self.norm_array[:, :steps], (2, 3, 0, 1), (0, 1, 2, 3))
        # del view_slab
        # del norm_array
        # cp.get_default_memory_pool().free_all_blocks()
        # cp.get_default_pinned_memory_pool().free_all_blocks()
    

    def interpolate_volume(self, 
                           ctvolume, 
                           steps,
                           view_slab, 
                           order = 1) -> cp.ndarray:
        """
        
        Returns interpolated result of shape (view_slab[0], texture_dim, texture_dim)

        """
        self.volume_slab.fill(cp.nan)
        
        self.volume_slab[:steps] = ctvolume.interpolate_volume(self.interpolation_slab[:, :steps], order = order)

    def get_operation_volume(self, 
                             ctvolume, 
                             norm_vector, 
                             volume_view_plane, 
                             start, 
                             stop,
                             order = 1) -> cp.ndarray:
        
        if start > stop: 
            start = 1.0*stop

        steps = stop - start + 1
        self.format_interpolation_slab(norm_vector, 
                                       volume_view_plane, 
                                       start,
                                       stop)
        
        self.interpolate_volume(ctvolume, 
                                steps,
                                self.interpolation_slab,
                                order = order)
    
    def perform_operation(self, 
                          operation, 
                          norm_vector_1,
                          view_plane_1,
                          ctvolume_1,
                          norm_vector_2 = None,
                          view_plane_2 = None,
                          ctvolume_2 = None,
                          start = 0, 
                          stop = 0,
                          order = 1) -> cp.ndarray:
        
        if start > stop: 
            return
        # Set arithmetic operations to use only 1 slice, the current location. 
        # Later set up compound operations so we can examine the mean of the additions, 
        # or the addition of means. 

        self.texture_content.fill(cp.nan)
        self.get_operation_volume(ctvolume_1, 
                                  norm_vector_1,
                                  view_plane_1, 
                                  start, 
                                  stop, 
                                  order = order)
        
        steps = stop - start + 1

        if operation == 'add':
            return self.add_volumes(self.get_operation_volume(ctvolume_1, norm_vector_1, view_plane_1, 0, 0, order = order), 
                                    self.get_operation_volume(ctvolume_2, norm_vector_2, view_plane_2, 0, 0, order = order))

        if operation == 'multiply':
            return self.multiply_volumes(self.get_operation_volume(ctvolume_1, norm_vector_1, view_plane_1, 0, 0, order = order), 
                                         self.get_operation_volume(ctvolume_2, norm_vector_2, view_plane_2, 0, 0, order = order))

        if operation == 'difference':
            return self.difference_volumes(self.get_operation_volume(ctvolume_1, norm_vector_1, view_plane_1, 0, 0, order = order), 
                                           self.get_operation_volume(ctvolume_2, norm_vector_2, view_plane_2, 0, 0, order = order))
        
        if operation == 'Mean':
            self.volume_mean(self.volume_slab[:steps])

        if operation == 'Max':
            self.volume_max(self.volume_slab[:steps])

        if operation == 'Min':
            self.volume_min(self.volume_slab[:steps])

        if operation == 'Standard Deviation':
            self.volume_std_dev(self.volume_slab[:steps])

        
        

    def add_volumes(self, volume_1, volume_2) -> cp.ndarray:
        return volume_1 + volume_2

    def multiply_volumes(self, volume_1, volume_2) -> cp.ndarray:
        return volume_1 * volume_2

    def difference_volumes(self, volume_1, volume_2) -> cp.ndarray:
        return volume_1 - volume_2

    def volume_mean(self, 
                    volume, 
                    # start, 
                    # end, 
                    axis = 2) -> cp.ndarray:
        self.texture_content = cp.mean(volume, axis = 0)

    def volume_min(self, 
                   volume,
                #    start, 
                #    end, 
                   axis = 2) -> cp.ndarray:
        self.texture_content = cp.min(volume, axis = 0)

    def volume_max(self, 
                    volume, 
                    # start, 
                    # end, 
                    axis = 2) -> cp.ndarray:
        self.texture_content = cp.max(volume, axis = 0)

    def volume_std_dev(self, 
                    volume, 
                    # start, 
                    # end, 
                    axis = 2) -> cp.ndarray:
        self.texture_content = cp.std(volume, axis = 0)