from .Globals import *
from . import CTVolume
from . import Themes


class Texture(object):
    """
    This class holds the actual texture information for each VolumeLayer object. 
    
    """

    mode_return_type = lambda x, y: x.astype(getattr(cp, y)) if G.GPU_MODE else lambda x, y: x.astype(getattr(np, y))
    mode_create_array = lambda x: cp.array(x) if G.GPU_MODE else lambda x: x
    mode_function = lambda x: getattr(cp, x.__name__) if G.GPU_MODE else getattr(np, x.__name__)
    mode_value = lambda x: getattr(cp, x.__str__()) if G.GPU_MODE else getattr(np, x.__str__())
    cp_to_np = lambda x: cp.asnumpy(x) if G.GPU_MODE else lambda x: x
    np_to_cp = lambda x: cp.asarray(x) if G.GPU_MODE else lambda x: x

    def __init__(self, 
                 ct_volume: CTVolume.CTVolume, 
                 drawlist_tag: str,
                 instantiate_texture: bool = False):
        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()
            print(f'Textures Message: Using CUDA device {G.DEVICE}.')

        self.ct_volume = ct_volume
        self.texture_dim = G.TEXTURE_DIM # ct_volume.texture_dim
        self.shape = [self.texture_dim, self.texture_dim]
        self.texture_content:np.ndarray | cp.ndarray = Texture.mode_function(np.full)(self.shape, Texture.mode_value(np.nan), dtype = np.float32)
        self.texture = np.zeros(self.texture_dim * self.texture_dim * 4, dtype = np.float32)
        self.colormap: list[RegularGridInterpolator, RegularGridInterpolator, RegularGridInterpolator] = [None, None, None] #G.DEFAULT_IMAGE_SETTINGS['colormap']
        self.zoom_level = 100 # final_zoom = self.zoom_level*self.zoom_constant
        self.zoom_constant = 0.01 # Percent of original image
        self.x_shift = 0
        self.y_shift = 0
        self.pixel_start = [0, 0]
        self.pixel_end = [self.texture_dim, self.texture_dim]
        self.uv_min = [0.0, 0.0]
        self.uv_max = [1.0, 1.0]
        self.min_value = 0.0 # 1.0 * self.ct_volume.volume_min
        self.max_value = 1200.0 # 1.0 * self.ct_volume.volume_max
        self.colormap_min = 1.0 * self.min_value
        self.colormap_max = 1.0 * self.max_value
        self.colormap_scale_tag: str = 'COLORMAP_SCALE_NOT_INITIALIZED'
        self.texture_tag = create_tag('Textures', 'Texture', ct_volume.name)
        self.drawlist_tag = drawlist_tag
        self.drawimage_tag = create_tag('Textures', 'DrawImage', ct_volume.name)

        if instantiate_texture:
            self.assign_texture()
            self.create_static_texture()

            print(f'Textures Message: Added {self.texture_tag} to texture registry')


    def get_texture_value_at_pos(self, 
                                 draw_x: int, 
                                 draw_y: int, 
                                 rescale: bool = True,
                                 as_numpy: bool = True):


        if rescale:
            texture_value = (self.texture_content[draw_y, draw_x] * (self.max_value - self.min_value)) + self.min_value
            if dpg.get_value('colormap_scale_combo') == 'Log':
                texture_value = cp.power(10, texture_value)
        else:
            texture_value =  self.texture_content[draw_y, draw_x]

        if as_numpy:
            return Texture.cp_to_np(texture_value)
        
        return texture_value
    

    def get_texture_patch(self, 
                          patch_x: list[int],
                          patch_y: list[int],
                          exclude_nan:bool = True, 
                          rescale: bool = True,
                          return_as_numpy: bool = True) -> np.ndarray:
        
        texture_patch = self.texture_content[patch_x[0]:patch_x[1], 
                                             patch_y[0]:patch_y[1]]

        if rescale: 
            texture_patch = texture_patch[~Texture.mode_function(np.isnan)(texture_patch)]

        if exclude_nan:
            texture_patch = (texture_patch[~Texture.mode_function(np.isnan)(texture_patch)]
                             * (self.max_value - self.min_value)) + self.min_value
        if return_as_numpy:
            return Texture.cp_to_np(texture_patch)
        
        return texture_patch
    

    def colorize_texture_patch(self,
                               texture_patch: np.ndarray,
                               patch_x: list[int],
                               patch_y: list[int],
                               colormap_name: str) -> np.ndarray:

        red_interp, green_interp, blue_interp = self.colormap
        alpha_patch = self.get_alpha_mask()[patch_x[0]:patch_x[1], 
                                            patch_y[0]:patch_y[1]]

        patch_length = texture_patch.size

        colorized_texture = np.zeros(patch_length*4)
        
        print('Texture Message: colorize_texture_patch')
        print(f'\t{patch_length = }')

        colorized_texture[0::4] = Texture.mode_return_type(Texture.cp_to_np(red_interp(texture_patch.flatten()[:])), 'float32')[:]
        colorized_texture[1::4] = Texture.mode_return_type(Texture.cp_to_np(green_interp(texture_patch.flatten()[:])), 'float32')[:]
        colorized_texture[2::4] = Texture.mode_return_type(Texture.cp_to_np(blue_interp(texture_patch.flatten()[:])), 'float32')[:]
        colorized_texture[3::4] = Texture.mode_return_type(Texture.cp_to_np(alpha_patch.flatten()), 'float32')[:]
        
        return colorized_texture
    
    
    def get_texture_content(self, 
                            exclude_nan:bool = True, 
                            rescale: bool = True):
        if exclude_nan:
            if rescale:
                return (self.texture_content[~Texture.mode_function(np.isnan)(self.texture_content)]
                        * (self.max_value - self.min_value)) + self.min_value
            return self.texture_content[~Texture.mode_function(np.isnan)(self.texture_content)]
        
        if rescale:
            return (self.texture_content[:] * (self.max_value - self.min_value)) + self.min_value


    def set_texture_value(self, value):
        self.texture_content[:] = Texture.np_to_cp(value)
        

    def create_static_texture(self, 
                              parent = None):
        if parent == None:
            parent = G.TEX_REG_TAG

        if dpg.does_item_exist(self.texture_tag):
            print(f'Texture Message: create_static_texture(): {self.texture_tag} already exists.')
            return
        
        dpg.add_static_texture(width = self.texture_dim,
                               height = self.texture_dim,
                               tag = self.texture_tag, 
                               default_value = self.texture,
                               parent = G.TEX_REG_TAG)
        print(f'Textures Message: Added {self.texture_tag} to texture registry')

    def delete_texture(self):
        if dpg.does_item_exist(self.texture_tag):
            dpg.delete_item(self.texture_tag)
            
        if dpg.does_alias_exist(self.texture_tag):
            dpg.remove_alias(self.texture_tag)

        if dpg.does_item_exist(self.drawimage_tag):
            dpg.delete_item(self.drawimage_tag)
            
        if dpg.does_alias_exist(self.drawimage_tag):
            dpg.remove_alias(self.drawimage_tag)
    

    def window_and_normalize(self, 
                             colormap_scale_type = dpg.get_value('colormap_scale_combo')):
        
        match colormap_scale_type:
            case 'Log':
                self.texture_content = Texture.mode_function(np.log10)(self.texture_content + 1e-6)

            case 'Absolute':
                self.texture_content = Texture.mode_function(np.abs)(self.texture_content)

            case 'Abs-Log':
                self.texture_content = Texture.mode_function(np.log10)(Texture.mode_function(np.abs)(self.texture_content) + 1e-6)

            case _:
                pass
        
        self.texture_content = (self.texture_content - self.min_value)/(self.max_value - self.min_value)

        return

    
    def get_alpha_mask(self) -> np.ndarray | cp.ndarray:
        # self.alpha_mask is a boolean array where we don't have nans. 
        return ~Texture.mode_function(np.isnan)(self.texture_content)


    def assign_texture(self, colormap = None):
        if colormap == None:
            colormap = self.colormap

        red_interp, green_interp, blue_interp = colormap

        self.texture[0::4] = Texture.mode_return_type(Texture.cp_to_np(red_interp(self.texture_content.flatten()[:])), 'float32')[:]
        self.texture[1::4] = Texture.mode_return_type(Texture.cp_to_np(green_interp(self.texture_content.flatten()[:])), 'float32')[:]
        self.texture[2::4] = Texture.mode_return_type(Texture.cp_to_np(blue_interp(self.texture_content.flatten()[:])), 'float32')[:]
        self.texture[3::4] = Texture.mode_return_type(Texture.cp_to_np(self.get_alpha_mask().flatten()), 'float32')[:]
        

    def set_colormap(self, 
                     colormap: list[RegularGridInterpolator]):
        
        for cmap_index in range(len(colormap)):
            self.colormap[cmap_index] = colormap[cmap_index]

    def set_colormap_scale_tag(self,
                               colormap_scale_tag: str):
        self.colormap_scale_tag = colormap_scale_tag

    def set_colormap_info(self, 
                          colormap: list[RegularGridInterpolator] = None, 
                          colormap_scale_type: str = None,
                          colormap_scale_tag: str = None):
        
        self.set_colormap_scale_tag(colormap_scale_tag)
        self.set_colormap(colormap)
        self.set_colormap_scale(colormap_scale_type)
    

    def set_colormap_scale(self, 
                           colormap_scale_type: str):
        """
        
        colormap_scale_type: string
            Options are: ['Log', 'Absolute', 'Abs-Log', 'Linear']

        """

        self.min_value = dpg.get_value(G.OPTION_TAG_DICT['min_intensity'])
        self.max_value = dpg.get_value(G.OPTION_TAG_DICT['max_intensity'])

        match colormap_scale_type:
            case 'Log':
                self.min_value = np.log10(self.min_value + 1e-6)
                self.max_value = np.log10(self.max_value)

            case 'Absolute':
                self.min_value = np.abs(self.min_value)
                self.max_value = np.abs(self.max_value)

            case 'Abs-Log':
                self.min_value = np.log10(np.abs(self.min_value) + 1e-6)
                self.max_value = np.log10(np.abs(self.max_value))

            case _:
                pass

        dpg.configure_item(self.colormap_scale_tag, 
                           min_scale = self.min_value, 
                           max_scale = self.max_value)

    def update_texture(self,
                       colormap: list[RegularGridInterpolator] = None,
                       colormap_scale_type: str = None,
                       colormap_scale_tag: str = None, 
                       x_shift: float = 0.0,
                       y_shift: float = 0.0,
                       pixel_start: list[float|int, float|int] = [None, None], 
                       pixel_end: list[float|int, float|int] = [None, None], 
                       uv_min: list[float|int, float|int] = [0, 0],
                       uv_max: list[float|int, float|int] = [1, 1],
                       drawlist: str = ''):
        # self.texture_content has been filled with the 
        # interpolated value at this point. 
        # Now we do our windowing and assigning. 
        self.set_colormap_info(colormap = colormap, 
                               colormap_scale_type = colormap_scale_type,
                               colormap_scale_tag = colormap_scale_tag)
        self.window_and_normalize(colormap_scale_type = colormap_scale_type)
        self.assign_texture(colormap = colormap)
        self.delete_texture()
        self.create_static_texture()
        self.update_draw_image(x_shift = x_shift, 
                               y_shift = y_shift)
        self.add_texture_to_drawlist(pixel_start = pixel_start,
                                     pixel_end = pixel_end,
                                     uv_min = uv_min,
                                     uv_max = uv_max,
                                     drawlist = drawlist)

    def update_draw_image(self, 
                          x_shift:float = 0.0,
                          y_shift:float = 0.0):
        
        # if x_shift == None:
        #     x_shift = dpg.get_value(G.ORIGIN_X_SLIDER)

        # if y_shift == None: 
        #     y_shift = dpg.get_value(G.ORIGIN_Y_SLIDER)

        self.x_shift = 1.0*x_shift
        self.y_shift = 1.0*y_shift
        # print(f'Texture Message: update_draw_image(): {self.x_shift = :<5}, {self.y_shift = :<5}, {self.drawlist_tag = }')

    def delete_drawn_texture(self):
        print(f'Textures Message: Deleting {self.drawimage_tag}')
        if dpg.does_item_exist(self.drawimage_tag):
            dpg.delete_item(self.drawimage_tag)
        if dpg.does_alias_exist(self.drawimage_tag):
            dpg.remove_alias(self.drawimage_tag)

    def redraw_texture(self, 
                       drawlist:str = None, 
                       pmin: list = None,
                       pmax: list = None,
                       uv_min: list = None, 
                       uv_max: list = None,
                       user_data = None):
        
        dpg.draw_image(self.texture_tag,
                       tag = self.drawimage_tag,
                       pmin = [self.pixel_start[0] - self.x_shift, self.pixel_start[1] - self.y_shift] if type(pmin) == type(None) else pmin,
                       pmax = [self.pixel_end[0] - self.x_shift, self.pixel_end[1] - self.y_shift] if type(pmax) == type(None) else pmax,
                       uv_min = self.uv_min if type(uv_min) == type(None) else uv_min,
                       uv_max = self.uv_max if type(uv_max) == type(None) else uv_max,
                       parent = self.drawlist_tag if type(drawlist) == type(None) else drawlist,
                       user_data = [self.x_shift, self.y_shift] if type(user_data) == type(None) else user_data)

    def add_texture_to_drawlist(self, 
                                pixel_start:list[float|int, float|int] = [None, None], 
                                pixel_end:list[float|int, float|int] = [None, None], 
                                uv_min:list[float|int, float|int] = [0, 0],
                                uv_max:list[float|int, float|int] = [1, 1],
                                drawlist:str = ''):
        
        if drawlist != '':
            self.drawlist_tag = drawlist

        
        
        pixel_start = self.pixel_start if pixel_start == [None, None] else pixel_start
        pixel_end = self.pixel_end if pixel_end == [None, None] else pixel_end

        print(f'Textures Message: Drawing {self.texture_tag} on {self.drawlist_tag} from {pixel_start} to {pixel_end}.')

        pmin = [pixel_start[0] - self.x_shift, pixel_start[1] + self.y_shift]
        pmax = [pixel_end[0] - self.x_shift, pixel_end[1] + self.y_shift]

        # pmin = [pixel_start[0] + 0.0, pixel_start[1] + 0.0]
        # pmax = [pixel_end[0] + 0.0, pixel_end[1] + 0.0]

        uv_min = self.uv_min if type(uv_min) == type(None) else uv_min
        uv_max = self.uv_max if type(uv_max) == type(None) else uv_max

        dpg.draw_image(self.texture_tag, 
                       tag = self.drawimage_tag, 
                       pmin = pmin,
                       pmax = pmax, 
                       uv_min = uv_min, 
                       uv_max = uv_max, 
                       parent = self.drawlist_tag,
                       user_data = [self.x_shift, self.y_shift])
    
    def change_zoom_level(self, sender, app_data):
        x_shift, y_shift = dpg.get_item_user_data(self.drawimage_tag)
        dpg.configure_item
        
        dpg.delete_item(self.drawimage_tag)
        x_zoom, y_zoom, _, _ = app_data
        dpg.draw_image(self.texture_tag, 
                       tag = self.drawimage_tag,
                       pmin = [(self.pixel_start[0] - x_shift) - x_zoom*self.zoom_constant, 
                               (self.pixel_start[1] - y_shift) - y_zoom*self.zoom_constant],
                       pmax = [(self.pixel_end[0] + x_shift) - x_zoom*self.zoom_constant, 
                               (self.pixel_end[1] + y_shift) - y_zoom*self.zoom_constant],
                       )
        
    def _cleanup_(self):
        dict_keys = list(self.__dict__.keys())
        while len(dict_keys) > 0:
            attrib_key = dict_keys.pop()
            setattr(self, attrib_key, None)
            delattr(self, attrib_key)
            cp._default_memory_pool.free_all_blocks()