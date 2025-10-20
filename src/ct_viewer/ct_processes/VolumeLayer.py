from .Globals import *
from . import OptionValue
from . import CTVolume
from . import Textures
from . import Landmarks
from . import VolumeOperations

class VolumeLayer(object):

    mode_return_type = lambda x, y: x.astype(getattr(cp, y)) if G.GPU_MODE else lambda x, y: x.astype(getattr(np, y))
    mode_create_array = lambda x: cp.array(x) if G.GPU_MODE else lambda x: x
    mode_function = lambda x: getattr(cp, x.__name__) if G.GPU_MODE else getattr(np, x.__name__)
    mode_value = lambda x: getattr(cp, x.__str__()) if G.GPU_MODE else getattr(np, x.__str__())
    cp_to_np = lambda x: cp.asnumpy(x) if G.GPU_MODE else lambda x: x
    np_to_cp = lambda x: cp.asarray(x) if G.GPU_MODE else lambda x: x

    def __init__(self, 
                 group: "VolumeLayerGroup", 
                 index: int,
                 ctvolume: CTVolume.CTVolume,
                 volume_name: str = None, 
                 default_origin = [0, 0, 0],
                 default_angle = [0, 0, 0],
                 default_norm = 0.0,
                 default_quaternion = G.QTN_DICT[G.VIEW], 
                 default_colormap_info = {'min_intensity': {'tag': G.OPTION_TAG_DICT['min_intensity'],
                                                            'default_value': 0.0, 
                                                            'default_limits': [-1e6, 1e6 - 1]},
                                          'max_intensity': {'tag': G.OPTION_TAG_DICT['max_intensity'],
                                                            'default_value': 1200.0, 
                                                            'default_limits': [-1e6 + 1, 1e6]},
                                          'colormap': G.DEFAULT_IMAGE_SETTINGS['colormap'],
                                          'colormap_reversed': False,
                                          'colormap_name': G.DEFAULT_IMAGE_SETTINGS['colormap_combo'],
                                          'colormap_scale_type': 'Linear',
                                          'colormap_scale_tag': 'COLORMAP_SCALE_NOT_INITIALIZED'},
                 default_histogram = [0, 1200, 1, 'Linear'], # Bins min, max, step, scale
                 default_limits_origin = [-500, 500],
                 default_limits_norm = [-500, 500],
                 default_texture_drawlayers:list[str] = ['', ''],
                 default_landmark_drawlayers:list[str] = ['', '']):
        
        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()

        # This let's us use the group when necessary
        self.Group:VolumeLayerGroup = group
        # Store the CTVolume itself here
        self.CTVolume:CTVolume.CTVolume = ctvolume
        self.pixel_steps:list[float,float,float] = [1.0*p_dim for p_dim in ctvolume.pixel_steps]
        self.name:str = self.CTVolume.name if volume_name is not None else volume_name
        self.file:Path = self.CTVolume.file
        self.group_name:str = self.Group.group_name
        self.index: int = index

        self.intensity_limits = [default_colormap_info['min_intensity']['default_limits'],
                                 default_colormap_info['max_intensity']['default_limits']]
        
        self.control_list: list[str] = ['origin_x', 'origin_y', 'origin_z', 'norm', 
                                        'pitch', 'yaw', 'roll', 
                                        'pixel_spacing_x', 'pixel_spacing_y', 'slice_thickness', 
                                        'min_intensity', 'max_intensity', 'colormap_name']
        self.orientation_control_list: list[str] = ['origin_x', 'origin_y', 'origin_z', 'norm', 
                                                    'pitch', 'yaw', 'roll', 
                                                    'pixel_spacing_x', 'pixel_spacing_y', 'slice_thickness']
        
        self.intensity_control_list: list[str] = ['min_intensity', 'max_intensity', 'colormap_name']
        self.geometry_control_list: list[str] = ['pixel_spacing_x', 'pixel_spacing_y', 'slice_thickness']
        self.histogram_control_list: list[str] = ['min_value', 'max_value', 'step', 'scale']
        self.orientation_control: str = G.DEFAULT_GROUP_LAYER_CONTROL
        self.intensity_control: str = G.DEFAULT_GROUP_LAYER_CONTROL
        self.geometry_control: str = 'Layer'
        self.is_volume_viewed: bool = False
        self.current_volume: bool = False
        self.info_box = None
        
        self.layer_control = []
        self.group_control = [control_option for control_option in self.control_list]
        
        self.Orientation: OptionValue.OrientationInfo = OptionValue.OrientationInfo(tag = f'{self.name}|VolumeLayer|Orientation',
                                                                                    default_origin = default_origin,
                                                                                    default_angle = default_angle, 
                                                                                    default_norm = default_norm,
                                                                                    default_norm_sign = self.CTVolume.slice_direction,
                                                                                    default_quaternion = default_quaternion, 
                                                                                    default_geometry = [1.0, 1.0, 1.0],
                                                                                    default_limits_origin = default_limits_origin,
                                                                                    default_limits_norm = default_limits_norm,
                                                                                    default_limits_geometry = [0.10, 50.0],
                                                                                    default_texture_dim = G.TEXTURE_DIM, #self.CTVolume.texture_dim,
                                                                                    default_texture_center = G.TEXTURE_CENTER, #self.CTVolume.texture_center)
                                                                                    voxel_start = self.CTVolume.physical_start,
                                                                                    voxel_center = self.CTVolume.physical_center,
                                                                                    voxel_steps = self.CTVolume.physical_steps,
                                                                                    default_drawlayer_tags = default_texture_drawlayers)

        self.Intensity: OptionValue.IntensityInfo = OptionValue.IntensityInfo(tag = f'{self.name}|VolumeLayer|Intensity',
                                                                              default_colormap_info = default_colormap_info)
        
        self.VolumeHistogram: OptionValue.HistogramInfo = OptionValue.HistogramInfo(default_histogram = default_histogram)
        self.TextureHistogram: OptionValue.HistogramInfo = OptionValue.HistogramInfo(default_histogram = default_histogram)

        self.ViewPlane: OptionValue.ViewPlane = self.Orientation.view_plane
        self.volume_center = self.CTVolume.volume_center
        self.volume_shape = self.CTVolume.shape
        self.texture_dim = G.TEXTURE_DIM # self.CTVolume.texture_dim
        self.texture_center = G.TEXTURE_CENTER

        self.colormap: OptionValue.ColormapValue = self.Intensity.colormap
        self.colormap_name: str = self.Intensity.colormap_name
        self.colormap_string: str = self.Intensity.get_colormap_string()
        self.colormap_reversed: bool = self.Intensity.colormap_reversed
        self.colormap_log: bool = self.Intensity.colormap_log
        self.colormap_rescaled: bool = self.Intensity.colormap_rescaled

        self.Landmarks:Landmarks.Landmarks = Landmarks.Landmarks(self.name,
                                                                 1.0 * self.CTVolume.physical_center,
                                                                 1.0 * self.CTVolume.physical_start,
                                                                 1.0 * self.CTVolume.physical_steps,
                                                                 texture_center = self.texture_center,
                                                                 group_name = self.group_name,
                                                                 volume_file = self.file,
                                                                 max_landmarks = 2**16)


        self.Texture: Textures.Texture = Textures.Texture(self.name,
                                                        #   self.CTVolume, 
                                                          default_texture_drawlayers[0],
                                                          [self.Group.window_dict[list(self.Group.window_dict.keys())[0]]['height'], 
                                                           self.Group.window_dict[list(self.Group.window_dict.keys())[0]]['width']],
                                                          pixel_start = [0, 0],
                                                          pixel_end = [self.texture_dim, self.texture_dim])
        
        ortho_start = round(-(1/3) * self.texture_dim)
        ortho_end = round((2/3) * self.texture_dim)
        self.TextureOrtho: Textures.Texture = Textures.Texture(self.name,
                                                            #    self.CTVolume, 
                                                               default_texture_drawlayers[1],
                                                               [self.Group.window_dict[list(self.Group.window_dict.keys())[1]]['height'], 
                                                                self.Group.window_dict[list(self.Group.window_dict.keys())[1]]['width']],
                                                                pixel_start = [ortho_start, ortho_start],
                                                                pixel_end = [ortho_end, ortho_end],
                                                                tag_suffix = 'Ortho')

        self.initialize_texture(self.Texture, 
                                self.Orientation.view_plane,
                                self.Intensity,
                                dpg.get_value('interpolation_combo_box'))
        

        self.initialize_texture(self.TextureOrtho, 
                                self.Orientation.view_plane_ortho,
                                self.Intensity,
                                dpg.get_value('interpolation_combo_box'))

    
    def initialize_texture(self, 
                           Texture:Textures.Texture,
                           ViewPlane:OptionValue.ViewPlane,
                           Intensity:OptionValue.IntensityInfo,
                           interpolation_method:str):
        
        self.interpolate_texture(Texture.texture_content,
                                 ViewPlane.get_voxel_view(),
                                 interpolation_method)
        Texture.window_and_normalize()
        Texture.update_draw_image()
        Texture.assign_texture(Intensity.colormap.current_value)
        # Texture.create_static_texture()
        Texture.create_raw_texture()


    def get_drawing_pos_texture_value(self, 
                                      draw_x: int,
                                      draw_y: int,
                                      window_type = 'view_plane',
                                      decimals = 3) -> float:
        
        if window_type == 'view_plane':
            return np.round(self.Texture.get_texture_value_at_pos(draw_x, draw_y), decimals = decimals)
        
        elif window_type == 'view_plane_ortho':
            return np.round(self.TextureOrtho.get_texture_value_at_pos(draw_x, draw_y), decimals = decimals)
    
        else:
            print(f'VolumeLayerMessage: Unknown window type {window_type} in get_drawing_pos_texture_value')

    def get_physical_voxel_coords(self, 
                                  draw_x: int,
                                  draw_y: int,
                                  window_type: str = 'view_plane') -> np.ndarray:
    
        physical_coords = self.get_physical_pos_coords(draw_x, draw_y, window_type = window_type)
        physical_voxel_coords = (physical_coords - self.CTVolume.physical_start) / self.CTVolume.pixel_steps
        
        return physical_voxel_coords.round(2)


    def get_physical_pos_coords(self, 
                                draw_x: int, 
                                draw_y: int,
                                window_type: str = 'view_plane') -> np.ndarray:
        
        image_pos_coords = self.get_drawing_pos_coords(draw_x, draw_y, window_type = window_type)
        physical_coords = image_pos_coords + self.CTVolume.physical_center

        return physical_coords


    def get_drawing_pos_coords(self, 
                               draw_x: int,
                               draw_y: int,
                               window_type: str = 'view_plane') -> np.ndarray:
        if window_type == 'view_plane':
            return self.get_orientation().view_plane.get_coords(int(draw_x), 
                                                                int(draw_y)).get()
        
        elif window_type == 'view_plane_ortho':
            return self.get_orientation().view_plane_ortho.get_coords(int(draw_x), 
                                                                      int(draw_y)).get()
        
        else:
            print(f'VolumeLayerMessage: Unknown window type {window_type} in get_drawing_pos_coords')


    def get_mouse_coords(self, window_type:str = 'view_plane') -> np.ndarray:
        mouse_x, mouse_y = dpg.get_drawing_mouse_pos()
        return self.get_drawing_pos_coords(int(mouse_x), int(mouse_y), window_type = window_type)


    def get_crosshair_coords(self, window_type:str = 'view_plane') -> np.ndarray:
        return self.get_drawing_pos_coords(int(self.texture_center), int(self.texture_center), window_type = window_type)


    def add_tag(self, tag_name, tag_value):
        setattr(self, tag_name, tag_value)


    def set_current(self, state: bool):
        if state == False:
            self.hide_drawimage()
            # self.remove_texture_from_drawlayer()
        self.current_volume = state


    def delete_landmark(self, 
                        landmark_id):
        self.Landmarks.delete_landmark(landmark_id)


    def save_landmarks(self, 
                       file_path: Path = None):
        
        if file_path == None:
            file_path = self.Landmarks.file_path

        volume_affine = self.CTVolume.affine

        self.Landmarks.save_landmarks(file_path, 
                                      self.name,
                                      volume_affine)


    def load_landmark_data(self, 
                           current_origin,
                           current_quaternion,
                           current_geometry,
                           file_path: Path = None) -> dict:
        """
        
        Returns a dictionary containing: 
            image_coords, voxel_coords, norms, quaternions, geometries
        
        """

        if file_path == None:
            file_path = self.Landmarks.file_path
        
        loaded_data = self.Landmarks.load_landmark_data(current_origin,
                                                        current_quaternion, 
                                                        current_geometry, 
                                                        file_path)

        return loaded_data


    def get_landmark_patches(self, 
                             loaded_landmark_data,
                             volume_operations):
        """
        loaded_landmark_data is of form: 

            loaded_data = {'image_coords': ..., 
                           'voxel_coords': ...,
                           'drawing_coords': ...,
                           'norms': ...,
                           'quaternions': ...,
                           'geometries': ...}

            drawing_pos_x, drawing_pos_y = 
        """

        patches = []
        patch_widths = []

        orientation_info = {'origin_x': 0.0,
                            'origin_y': 0.0, 
                            'origin_z': 0.0,
                            'norm': 0.0,
                            'pitch': 0.0,
                            'yaw': 0.0,
                            'roll': 0.0,
                            'pixel_spacing_x': 1.0,
                            'pixel_spacing_y': 1.0,
                            'slice_thickness': 1.0,
                            'drawlayer_tags': self.get_orientation().drawlayer.current_value}
        
        for landmark_index in range(loaded_landmark_data['image_coords'].shape[0]):
            quaternion = qtn.array(loaded_landmark_data['quaternions'][landmark_index])
            yaw, pitch, roll = quaternion.to_axis_angle
            orientation_info['origin_x'] = loaded_landmark_data['image_coords'][landmark_index, 1].item()
            orientation_info['origin_y'] = -1.0*loaded_landmark_data['image_coords'][landmark_index, 0].item()
            orientation_info['origin_z'] = loaded_landmark_data['image_coords'][landmark_index, 2].item()
            orientation_info['norm'] = 0.0
            orientation_info['pitch'] = pitch.item()
            orientation_info['yaw'] = yaw.item()
            orientation_info['roll'] = roll.item()
            orientation_info['quaternion'] = quaternion
            orientation_info['pixel_spacing_x'] = loaded_landmark_data['geometries'][landmark_index, 1].item()
            orientation_info['pixel_spacing_y'] = loaded_landmark_data['geometries'][landmark_index, 0].item()
            orientation_info['slice_thickness'] = loaded_landmark_data['geometries'][landmark_index, 2].item()
            orientation_info['drawlayer_tags'] = [self.get_orientation().drawlayer.current_value, self.get_orientation().drawlayer_ortho.current_value]
            

            # print(f'\tget_landmark_patches(): Updating Orientation')
            self.reset_orientation()
            self.update_orientation(orientation_info, update_type = 'Update')

            # print(f'\tget_landmark_patches(): Updating Texture')
            self.update_texture(volume_operations, 
                                loading_landmarks = True)

            geometry = self.get_orientation().geometry_vector.current_value.get().reshape((3, ))
            patch_width = int(10.0 * np.mean(geometry))

            drawing_coords = [self.texture_center, self.texture_center] # loaded_landmark_data['drawing_coords'][landmark_index]

            patch_x = [int(drawing_coords[0] - patch_width), int(drawing_coords[0] + patch_width + 1)]
            patch_y = [int(drawing_coords[1] - patch_width), int(drawing_coords[1] + patch_width + 1)]
            # print(f'\tget_landmark_patches(): Getting Patch about {drawing_coords}')
            patches.append(self.get_texture_patch(patch_x,
                                                  patch_y,
                                                  exclude_nan = False,
                                                  rescale = False,
                                                  return_as_numpy = True,
                                                  colorize = True))
            patch_widths.append(patch_width)
        
        self.reset_orientation()
        return patches, patch_widths
    

    def add_landmark(self, 
                     drawing_coords: tuple[int],
                     image_coords: np.ndarray, 
                     image_coords_hu: float,
                     voxel_coords: np.ndarray,
                     quaternion: qtn.QuaternionicArray, 
                     viewplane_norm: np.ndarray, 
                     draw_layer: str,
                     color: tuple[int] = dpg.get_value('landmark_color_picker'),
                     size: float = 5.0, 
                     geometry: np.ndarray = np.ones(3, dtype = np.float32), 
                     affine: np.ndarray = np.array([np.eye(4, dtype = np.float32)]*4),
                     landmark_patch: np.ndarray = np.zeros((11, 11), dtype = np.float32),
                     patch_size:int = 11,
                     show_landmark: bool = True):
        
        print('VolumeLayer Message: add_landmark')
        print(f'\tlandmarks.add_landmark()')
        self.Landmarks.add_landmark(drawing_coords, 
                                    image_coords, 
                                    image_coords_hu,
                                    voxel_coords,
                                    quaternion,
                                    viewplane_norm,
                                    draw_layer,
                                    color = color, 
                                    size = size, 
                                    geometry = geometry,
                                    affine = affine,
                                    landmark_patch = landmark_patch, 
                                    patch_size = patch_size,
                                    show_landmark = show_landmark)
        
    def remove_landmark(self, landmark_id):
        pass


    def update_landmark_colors(self, landmark_id: int|None = None):
        self.Landmarks.update_landmark_colors(landmark_id = landmark_id)

    
    def update_landmarks(self, 
                         update_type: str = 'Orientation',
                         origin_vector: cp.ndarray = None,
                         quaternion: qtn.QuaternionicArray = None,
                         geometry_vector: cp.ndarray = None,
                         landmark_id: int|None = None):
        
        if self.Landmarks.number_of_landmarks == 0:
            return
        
        with dpg.mutex(): 
        
            if update_type == 'Colors':
                self.Landmarks.update_landmark_colors(landmark_id = landmark_id)

            elif update_type == 'Orientation':
                self.Landmarks.update_landmarks(origin_vector, 
                                                quaternion,
                                                geometry_vector)
                
            elif update_type == 'All':

                self.Landmarks.update_landmarks(origin_vector, 
                                                quaternion,
                                                geometry_vector)
                
                self.Landmarks.update_landmark_colors(landmark_id = landmark_id)
                
            else:
                print(f'VolumeLayers Message: update_landmarks Error.')
                print(f'\t{update_type = } Not Recognized')
                return
        

    def set_texture(self):
        self.Texture = Textures.Texture(self.CTVolume)


    def set_ctvolume(self, volume_name:str, volume_data:CTVolume.CTVolume):
        self.CTVolume = volume_data
        self.name = self.CTVolume.name if volume_name is not None else volume_name
        self.file = self.CTVolume.file


    def set_is_volume_viewed(self, volume_viewed_bool:bool = False):
        self.is_volume_viewed = volume_viewed_bool


    def add_to_group(self, group: "VolumeLayerGroup"):
        if self.name not in group.volume_names:
            group.set_volume(self)


    def update_all(self, 
                   orientation_info: dict, 
                   intensity_info: dict,
                   changed_volume: bool,
                   update_type:str,
                   operation_instance: VolumeOperations.VolumeOperations):
        
        for key, value in orientation_info.items():
            print(f'{key:<25}: {value}')
        
        with dpg.mutex():
            print(f'VolumeLayer Message: Updating Orientation')
            self.update_orientation(orientation_info, update_type = update_type)
            # self.get_orientation().print_values()

            print(f'VolumeLayer Message: Updating Intensity')
            self.update_intensity(intensity_info)

            print(f'VolumeLayer Message: Updating Texture')
            self.update_texture(operation_instance)
            # self.update_texture(self.get_orientation().view_plane.get_voxel_view())

            print(f'VolumeLayer Message: Updating Landmarks')
            self.update_landmarks(update_type = 'All',
                                  origin_vector = self.get_crosshair_coords(), #self.get_orientation_value('origin_vector', 'current_value').reshape((3, )).get(),
                                  quaternion = self.get_orientation_value('quaternion', 'current_value'),
                                  geometry_vector = self.get_orientation_value('geometry_vector', 'current_value').reshape((3, )).get())


    def set_control_options(self, 
                            option_class:str, 
                            changing_volumes:bool = False):
        
        """
        We need to set the control options for any new volumes/control sources we change to and from. 
        So we run this when changing volumes or between Layer and Group control. 
        """
        
        if option_class not in ['geometry', 'Orientation', 'Intensity']:
            print(f'VolumeLayer Message: Option class {option_class} not recognized. Only "Orientation" or "Intensity" accepted.')
            return

        for control_option_name in getattr(self, f'{option_class.lower()}_control_list'):
            option_tag = G.OPTION_TAG_DICT[control_option_name]
            if getattr(self, f'{option_class.lower()}_control') == 'Group':
                control_option = getattr(getattr(self.Group, option_class), control_option_name)
            else:
                control_option = getattr(getattr(self, option_class), control_option_name)
            # We need to configure the option limits when changing volumes, 
            # but not when changing control sources.
            # We do this so we can keep track to the limits associated with 
            # each image parameter. 
            if changing_volumes:
                if control_option_name == 'colormap_name':
                    dpg.configure_item(option_tag, 
                                       default_value = control_option.default_value)
                else:
                    dpg.configure_item(option_tag, 
                                       min_value = control_option.limit_low, 
                                       max_value = control_option.limit_high, 
                                       default_value = control_option.default_value)
                
            print(f'{option_tag:<25}: {control_option_name}: {control_option.current_value}')
            dpg.set_value(f'{option_tag}_current_value', 
                          control_option.current_value)
            
            if control_option_name != 'colormap_name':
                dpg.set_value(f'{option_tag}_step_value',
                            control_option.step_value)
                dpg.set_value(f'{option_tag}_step_fast_value',
                            control_option.step_fast_value)


    def get_attribute(self, attribute_name: str):
        if attribute_name not in list(self.__dict__.keys()):
            print(f'VolumeLayer Message: {attribute_name} not in {self.name}.')
            return
        return getattr(self, attribute_name)
    
    
    def set_attribute(self, attribute_name, attribute_value, make_copy = True):
        setattr(self, attribute_name, attribute_value)
    
    
    def reset_orientation(self):
        self.get_orientation().reset_orientation()


    def reset_orientation_origin(self):
        self.get_orientation().reset_origin()


    def reset_orientation_angle(self):
        self.get_orientation().reset_angle()


    def reset_orientation_zoom(self):
        self.get_orientation().reset_geometry()


    def update_intensity(self, 
                         intensity_info: dict):
        
        print(f'VolumeLayer Message: Updating Colormap')
        self.update_colormap(colormap_name = intensity_info['colormap_name'],
                             colormap_reversed = intensity_info['colormap_reversed'],
                             colormap_log = intensity_info['colormap_log'],
                             colormap_scale_type = intensity_info['colormap_scale_type'],
                             colormap_scale_tag = intensity_info['colormap_scale_tag'],
                             colormap_rescaled = intensity_info['colormap_rescaled'])
        
        print(f'VolumeLayer Message: Updating Window')                     
        self.update_window(min_intensity = intensity_info['min_intensity'],
                           max_intensity = intensity_info['max_intensity'],
                           window_size = intensity_info['window_size'])


    def update_orientation(self, orientation_info:dict, update_type:str = 'Update'):
        if update_type == 'Set':
            self.get_orientation().set_orientation(**orientation_info)

        else:
            self.get_orientation().update_orientation(**orientation_info)


        print(f'VolumeLayer Message: Updating Values')
        value_tag_dict = {'OptionPanel_quaternion_display': self.print_orientation_value('quaternion'),
                          'OptionPanel_global_quaternion_display': self.print_orientation_value('global_quaternion'),
                            'OptionPanel_origin_vector_display': self.print_orientation_value('origin_vector'),
                            'OptionPanel_norm_vector_display': self.print_orientation_value('norm_vector'),
                            'origin_x_slider_step_value': 1.0/orientation_info['pixel_spacing_x'],
                            'origin_y_slider_step_value': 1.0/orientation_info['pixel_spacing_y'],
                            'origin_z_slider_step_value': 1.0/orientation_info['slice_thickness'],
                            'origin_x_slider_step_fast_value': 5.0/orientation_info['pixel_spacing_x'],
                            'origin_y_slider_step_fast_value': 5.0/orientation_info['pixel_spacing_y'],
                            'origin_z_slider_step_fast_value': 5.0/orientation_info['slice_thickness']}
        
        for value_tag, value in value_tag_dict.items():
            dpg.set_value(value_tag, value)

        dpg.set_item_user_data('OptionPanel_quaternion_display', self.get_orientation().quaternion.current_value)

        dpg.configure_item('origin_x_slider', 
                            step = dpg.get_value('origin_x_slider_step_value'),
                            step_fast = dpg.get_value('origin_x_slider_step_fast_value'))
        dpg.configure_item('origin_y_slider', 
                            step = dpg.get_value('origin_y_slider_step_value'),
                            step_fast = dpg.get_value('origin_y_slider_step_fast_value'))
        dpg.configure_item('origin_z_slider', 
                            step = dpg.get_value('origin_z_slider_step_value'),
                            step_fast = dpg.get_value('origin_z_slider_step_fast_value'))


    def update_colormap(self, 
                        colormap_name: str = None,
                        colormap_reversed: bool = None,
                        colormap_log: bool = None,
                        colormap_scale_tag: str = None,
                        colormap_scale_type: str = None,
                        colormap_rescaled: bool = None):
        
        if self.intensity_control == 'Group':
            self.Group.Intensity.update_colormap(colormap_name = colormap_name,
                                                 colormap_reversed = colormap_reversed, 
                                                 colormap_log = colormap_log,
                                                 colormap_scale_type = colormap_scale_type,
                                                 colormap_scale_tag = colormap_scale_tag,
                                                 colormap_rescaled = colormap_rescaled)
        else:
            self.Intensity.update_colormap(colormap_name = colormap_name,
                                           colormap_reversed = colormap_reversed, 
                                           colormap_log = colormap_log,
                                           colormap_scale_type = colormap_scale_type,
                                           colormap_scale_tag = colormap_scale_tag,
                                           colormap_rescaled = colormap_rescaled)
        
        self.colormap = self.Intensity.colormap
        self.colormap_reversed = self.Intensity.colormap_reversed
        self.colormap_log = self.Intensity.colormap_log
        self.colormap_name = self.Intensity.colormap_name
        self.colormap_string = self.Intensity.get_colormap_string()
        self.colormap_rescaled = self.Intensity.colormap_rescaled


    def update_window(self, 
                      min_intensity: float|int = None, 
                      max_intensity: float|int = None, 
                      window_size: float|int = 1): #np.min([dpg.get_value(f'{G.OPTIONS_DICT["min_intensity_slider"]["slider_tag"]}_increment_input_float'),
                                       #     dpg.get_value(f'{G.OPTIONS_DICT["max_intensity_slider"]["slider_tag"]}_increment_input_float')])):

        if min_intensity == None:
            min_intensity = dpg.get_value(G.OPTION_TAG_DICT['min_intensity'])
        if max_intensity == None:
            max_intensity = dpg.get_value(G.OPTION_TAG_DICT['max_intensity'])

        if self.intensity_control == 'Group':
            self.Group.Intensity.update_window(min_intensity, max_intensity, window_size)
        else:
            self.Intensity.update_window(min_intensity, max_intensity, window_size)


    def set_dpg_window(self):
        if self.intensity_control == 'Group':
            dpg.set_value(G.OPTION_TAG_DICT['min_intensity'], self.Group.Intensity.min_intensity.current_value)
            dpg.set_value(G.OPTION_TAG_DICT['max_intensity'], self.Group.Intensity.max_intensity.current_value)
            
        else:
            dpg.set_value(G.OPTION_TAG_DICT['min_intensity'], self.Intensity.min_intensity.current_value)
            dpg.set_value(G.OPTION_TAG_DICT['max_intensity'], self.Intensity.max_intensity.current_value)


    def set_colormap_scale_tag(self, 
                               colormap_scale_tag: str):
        self.Intensity.set_colormap_scale_tag(colormap_scale_tag)
        self.Texture.set_colormap_scale_tag(colormap_scale_tag)


    def set_drawlayer_info(self,
                           drawlayer_tags,
                           drawlayer_info):
        
        self.Orientation.set_drawlayer_tags(drawlayer_tags)


    def set_drawlayer_tags(self, 
                           drawlayer_tags: list[str]):
        self.Orientation.set_drawlayer_tags(drawlayer_tags)


    def update_texture(self,
                       operation_instance: VolumeOperations.VolumeOperations,
                       loading_landmarks: bool = False):

        # pixel_start = [0, 0]
        # pixel_end = [G.TEXTURE_DIM, G.TEXTURE_DIM]

        x_shift = 0.0
        y_shift = 0.0
        drawlayer = self.get_orientation().drawlayer.current_value
        drawlayer_ortho = self.get_orientation().drawlayer_ortho.current_value

        colormap = self.get_intensity_value('colormap', '')
        colormap_scale_tag = self.get_intensity_value('colormap_scale_tag', 'current_value')
        colormap_scale_type = self.get_intensity_value('colormap_scale_type', 'current_value')
        colormap_rescale = self.get_intensity_value('colormap_rescaled', '')

        uv_min = [0, 0]
        uv_max = [1, 1]

        order_dict = {'Nearest Neighbor': 0,
                      'Linear': 1}
        
        if operation_instance.enabled: 
            operation_instance.perform_operation(
                operation_instance.operation, 
                self.get_orientation().norm_vector.current_value,
                self.get_orientation().view_plane.get_voxel_view(),
                self.CTVolume,
                start = operation_instance.start,
                stop = operation_instance.stop,
                order = order_dict[dpg.get_value('interpolation_combo_box')])
            self.Texture.set_texture_value(operation_instance.texture_content)
            self.interpolate_texture(self.TextureOrtho.texture_content, 
                                     self.get_orientation().view_plane_ortho.get_voxel_view(),
                                     dpg.get_value('interpolation_combo_box'))

        else:
            self.interpolate_texture(self.Texture.texture_content, 
                                     self.get_orientation().view_plane.get_voxel_view(),
                                     dpg.get_value('interpolation_combo_box'))
            
            self.interpolate_texture(self.TextureOrtho.texture_content, 
                                     self.get_orientation().view_plane_ortho.get_voxel_view(),
                                     dpg.get_value('interpolation_combo_box'))

        self.Texture.update_texture(colormap = colormap,
                                    colormap_scale_type = colormap_scale_type,
                                    colormap_scale_tag = colormap_scale_tag,
                                    x_shift = x_shift,
                                    y_shift = y_shift,
                                    # pixel_start = pixel_start,
                                    # pixel_end = pixel_end,
                                    uv_min = uv_min,
                                    uv_max = uv_max,
                                    drawlayer = drawlayer,
                                    loading_landmarks = loading_landmarks)
        
        self.TextureOrtho.update_texture(colormap = colormap,
                                         colormap_scale_type = colormap_scale_type,
                                         colormap_scale_tag = colormap_scale_tag,
                                         x_shift = x_shift,
                                         y_shift = y_shift,
                                        #  pixel_start = pixel_start,
                                        #  pixel_end = pixel_end,
                                         uv_min = uv_min,
                                         uv_max = uv_max,
                                         drawlayer = drawlayer_ortho,
                                         loading_landmarks = loading_landmarks)


    def get_texture_patch(self, 
                          patch_x: list[int],
                          patch_y: list[int],
                          exclude_nan: bool = False,
                          rescale:bool = True,
                          return_as_numpy: bool = True,
                          colorize: bool = True) -> np.ndarray:
        
        texture_patch = self.Texture.get_texture_patch(patch_x, 
                                                       patch_y,
                                                       exclude_nan = exclude_nan,
                                                       rescale = rescale,
                                                       return_as_numpy = return_as_numpy)

        if colorize: 
            texture_patch = self.Texture.colorize_texture_patch(texture_patch, 
                                                                patch_x, 
                                                                patch_y,
                                                                self.Intensity.get_colormap_string())
        
        return texture_patch


    def update_control(self, 
                       orientation_control: str = 'Group',
                       intensity_control: str = 'Group'):
        
        # This will make it so we copy the group intensity to the current volume when 
        # we go from Layer -> Group, aligning the layer to the Group it, and from Group -> Layer, 
        # making the Layer independent of the Group. 
        # if G.FILE_LOADED:
        #     self.Orientation.copy_orientation(self.Group.Orientation)
        #     self.Intensity.copy_intensity(self.Group.Intensity)
            # self.VolumeHistogram.copy_histogram(self.Group.histogram)
        
        self.orientation_control = orientation_control
        self.intensity_control = intensity_control
        # self.histogram_control = intensity_control

    def window_and_normalize_texture(self):
        self.Texture.window_and_normalize()


    def assign_texture(self):
        self.Texture.assign_texture(self.get_intensity().colormap.current_value)


    def interpolate_texture(self, 
                            texture_content: np.ndarray | cp.ndarray,
                            view_plane: np.ndarray | cp.ndarray,
                            interpolation_method: str,
                            fill_nan: bool = True,
                            rescale: bool = False):
        
        if fill_nan:
            texture_content.fill(VolumeLayer.mode_value(cp.nan))

        if G.GPU_MODE:
            self.interpolate_view(out_array = texture_content, 
                                  view_plane = view_plane,
                                  interpolation_method = interpolation_method)
            
        else:
            print('Must use GPU!')
        
        return
    
    def delete_texture(self):
        if 'Texture' in dir(self):
            self.Texture.delete_texture()
            self.TextureOrtho.delete_texture()

    def delete_landmarks(self):
        if 'Landmarks' in dir(self):
            self.Landmarks.delete_all_landmarks()

    def remove_texture_from_drawlayer(self):
        self.Texture.delete_drawn_texture()
        self.TextureOrtho.delete_drawn_texture()

    def add_textures_to_drawlayers(self,
                                   drawlayers):
        
        self.add_texture_to_drawlayer(drawlayer = drawlayers[0],
                                      Texture = self.Texture)
        
        self.add_texture_to_drawlayer(drawlayer = drawlayers[1],
                                      Texture = self.TextureOrtho)

    def add_texture_to_drawlayer(self, 
                                pixel_start:list[float|int, float|int] = [None, None], 
                                pixel_end:list[float|int, float|int] = [None, None],
                                uv_min:list[float|int, float|int] = [0, 0],
                                uv_max:list[float|int, float|int] = [1, 1], 
                                drawlayer:str = '',
                                Texture:Textures.Texture = None):
        
        if pixel_start == [None, None]:
            pixel_start = [0, 0]

        if pixel_end == [None, None]:
            pixel_end = [self.texture_dim, self.texture_dim]
        
        Texture.add_texture_to_drawlayer(pixel_start = pixel_start,
                                         pixel_end = pixel_end,
                                         uv_min = uv_min,
                                         uv_max = uv_max,
                                         drawlayer = drawlayer)
       
    def interpolate_view(self, 
                         out_array = None, 
                         out_mask = None, 
                         view_plane = None, 
                         interpolation_method: str = dpg.get_value('interpolation_combo_box')):
        """
        
        """

        order_dict = {'Nearest Neighbor': 0,
                      'Linear': 1}
        
        order = order_dict[interpolation_method]

        if type(out_array) == type(None):

            return self.CTVolume.interpolate_volume(view_plane,
                                                    order = order)
        
        else:
            if type(out_mask) == type(None):
                out_array[:] = self.CTVolume.interpolate_volume(view_plane,
                                                                order = order).reshape(out_array.shape)[:]
                
            else:
                out_array[out_mask] = self.CTVolume.interpolate_volume(view_plane,
                                                                       order = order).reshape(out_array.shape)[out_mask]

    def interpolate_mask(self, 
                         out_array = None, 
                         out_mask = None, 
                         view_plane = None, 
                         interpolation_method: str = dpg.get_value('interpolation_combo_box')):
        """
        
        """

        order_dict = {'Nearest Neighbor': 0,
                      'Linear': 1}
        
        order = order_dict[interpolation_method]

        if type(out_array) == type(None):

            return self.CTVolume.interpolate_mask(view_plane, 
                                                  order = order)

         
        else:
            if type(out_mask) == type(None):
                out_array[:] = self.CTVolume.interpolate_mask(view_plane, 
                                                              order = order).reshape(out_array.shape)[:]
                
            else:
                out_array[out_mask] = self.CTVolume.interpolate_mask(view_plane, 
                                                                     order = order).reshape(out_array.shape)[out_mask]
                
    def print_orientation_value(self, value: str, check_control = True):
        """
        value: string
            Can be any of the OptionValue.OrientationInfo attributes. 
        """
        if check_control and (self.orientation_control == 'Group'):
            return f"{getattr(self.Group.Orientation, f'{value}')}"
        
        return f"{getattr(self.Orientation, f'{value}')}"


    def get_orientation(self) -> OptionValue.OrientationInfo:
        if self.orientation_control == 'Group':
            return self.Group.Orientation
        
        return self.Orientation
    
    def get_intensity(self):
        if self.intensity_control == 'Group':
            return self.Group.Intensity
        
        return self.Intensity

    
    def get_orientation_value(self, 
                              value: str, 
                              modifier: str, 
                              check_control = True):
        """
        value:
        modifier: string
            Can be ['default_value', 'current_value', 'previous_value', 'difference_value']
            Default is 'current_value'
        """

        if check_control and (self.orientation_control == 'Group'):
            value_object = getattr(self.Group.Orientation, f'{value}')

        else:
            value_object = getattr(self.Orientation, f'{value}')

        if modifier == '':
            return value_object
        
        else:
            return getattr(value_object, f'{modifier}')
    
    def get_intensity_value(self, value: str, 
                            modifier:str, 
                            check_control = True):
        """
        value:
        modifier: string
            Can be ['default_value', 'current_value', 'previous_value', 'difference_value']
            Default is 'current_value'
        """
        if check_control and (self.intensity_control == 'Group'):
            value_object = getattr(self.Group.Intensity, f'{value}')

        else:
            value_object = getattr(self.Intensity, f'{value}')

        if modifier == '':
            return value_object
        
        else:
            return getattr(value_object, f'{modifier}')
        
    def update_histogram(self, 
                         histogram_type:str,
                         histogram_info:dict):
        print('VolumeLayer Message: Updating Histogram')
        with dpg.mutex():
            if histogram_type == 'volume':
                # histogram_name = 'VolumeHistogram'
                info_prefix = 'InfoBoxTab_histogram_volume'
                bin_min = histogram_info[f'{info_prefix}_bins_min']
                bin_max = histogram_info[f'{info_prefix}_bins_max']
                bin_step = histogram_info[f'{info_prefix}_bin_step']
                target_line_series = histogram_info[f'{info_prefix}_plot_line_series']
                self.VolumeHistogram.update_bins(min_value = bin_min,
                                                 max_value = bin_max,
                                                 bin_step = bin_step)
                self.VolumeHistogram.update_histogram_counts(self.CTVolume.volume[~self.CTVolume.mask.astype(bool)])
                dpg.set_value(target_line_series, self.VolumeHistogram.get_histogram(return_order='reversed'))


            elif histogram_type == 'texture':
                # histogram_name = 'TextureHistogram'
                info_prefix = 'InfoBoxTab_histogram_texture'
                bin_min = histogram_info[f'{info_prefix}_bins_min']
                bin_max = histogram_info[f'{info_prefix}_bins_max']
                bin_step = (bin_max - bin_min) / histogram_info[f'{info_prefix}_n_bins']
                target_line_series = histogram_info[f'{info_prefix}_plot_line_series']
                self.TextureHistogram.update_bins(min_value = bin_min,
                                                 max_value = bin_max,
                                                 bin_step = bin_step)
                self.TextureHistogram.update_histogram_counts(self.CTVolume.volume[~self.CTVolume.mask.astype(bool)])
                dpg.set_value(target_line_series, self.TextureHistogram.get_histogram(return_order='reversed'))
                
            else:
                print(f'Histogram type {histogram_type} not recognized!')
                return
        
            # self.VolumeHistogram
            # self.TextureHistogram
            # histogram:OptionValue.HistogramInfo = getattr(self, histogram_name)
            
            # getattr(self, histogram_name).update_bins(min_value = bin_min,
            #                                           max_value = bin_max,
            #                                           bin_step = bin_step)
            
            # if histogram_type == 'volume':
            #     getattr(self, histogram_name).update_histogram_counts(self.CTVolume.volume[~self.CTVolume.mask.astype(bool)])

            # else: 
            #     getattr(self, histogram_name).update_histogram_counts(self.Texture.get_texture_content(exclude_nan = True, rescale = True))

            # dpg.set_value(target_line_series, getattr(self, histogram_name).get_histogram(return_order='reversed'))
        
        
    def set_dpg_histogram_values(self):
        dpg.set_value('InfoBoxTab_histogram_volume_bins_min', self.VolumeHistogram.bins_min())
        dpg.set_value('InfoBoxTab_histogram_volume_bins_max', self.VolumeHistogram.bins_max())
        dpg.set_value('InfoBoxTab_histogram_volume_bin_step', self.VolumeHistogram.bin_step())


    def get_histogram(self, 
                      update_histogram = True, 
                      return_order = 'reversed', 
                      asnumpy = True):

        min_enabled = dpg.get_value('InfoBoxTab_histogram_volume_bins_min_checkbox')
        max_enabled = dpg.get_value('InfoBoxTab_histogram_volume_bins_max_checkbox')
        step_enabled = dpg.get_value('InfoBoxTab_histogram_volume_bin_step_checkbox')

        bin_min = dpg.get_value('InfoBoxTab_histogram_volume_bins_min') if min_enabled else None
        bin_max = dpg.get_value('InfoBoxTab_histogram_volume_bins_max') if max_enabled else None
        bin_step = dpg.get_value('InfoBoxTab_histogram_volume_bin_step') if step_enabled else None

        if update_histogram:
            self.VolumeHistogram.update_histogram_params(min_value = bin_min, 
                                                          max_value = bin_max, 
                                                          bin_step = bin_step)
            self.VolumeHistogram.update_histogram_counts(self.CTVolume.volume[~self.CTVolume.mask.astype(bool)])
        
        return self.VolumeHistogram.get_histogram(return_order = return_order, asnumpy = asnumpy)


    def set_dpg_histogram_limits(self, 
                                 x_axis, y_axis, unlock_axes = True,
                                 min_x = None, max_x = None, 
                                 min_y = None, max_y = None):
        
        dpg.set_axis_limits(x_axis, 
                            self.VolumeHistogram.bins_min() if min_x is None else min_x,
                            self.VolumeHistogram.bins_max() if max_x is None else max_x)
        
        dpg.set_axis_limits(y_axis, 
                            self.VolumeHistogram.bin_counts().min() if min_y is None else min_y,
                            self.VolumeHistogram.bin_counts().max() if max_y is None else max_y)
        
        if unlock_axes:
            dpg.set_axis_limits_auto(x_axis)
            dpg.set_axis_limits_auto(y_axis)


    def show_landmarks(self):
        self.Landmarks.show_landmarks()


    def hide_landmarks(self):
        self.Landmarks.hide_landmarks()

    def show_drawimage(self):
        dpg.show_item(self.Texture.drawimage_tag)
        dpg.show_item(self.TextureOrtho.drawimage_tag)


    def hide_drawimage(self):
        dpg.hide_item(self.Texture.drawimage_tag)
        dpg.hide_item(self.TextureOrtho.drawimage_tag)

    def _cleanup_(self):
        attrib_list = list(self.__dict__.keys())
        while len(attrib_list) > 0:
            attrib_name = attrib_list.pop()
            # if attrib_name == 'group':
            #     setattr(self, attrib_name, None)
            #     delattr(self, attrib_name)
            if attrib_name == 'Group':
                continue

            elif '_cleanup_' in dir(getattr(self, attrib_name)):
                try:
                    getattr(self, attrib_name)._cleanup_()
                finally:
                    setattr(self, attrib_name, None)
                    delattr(self, attrib_name)
            else:
                setattr(self, attrib_name, None)
                delattr(self, attrib_name)


class VolumeLayerGroup(object):

    """
    Each VolumeLayerGroup can hold volume information, but does not hold any volumes themselves. 
    
    Instead, it holds the orientation and rendering information for each volume in the group:
        
        Origin X
        Origin Y
        Origin Z
        Norm location
        Pitch
        Yaw
        Roll
        Min Intensity
        Max Intensity
        Colormap
        Quatnernion (Current, Previous, Default)
        Drag Points
        
    It can also hold global values for each of the above. The global values do not influence the local values, 
    except when adding a new volume to the group. When selecting between the two, the MainView will simply render 
    the appropriate image and send it to the texture. 
    
    """
    def __init__(self, 
                 group_name, 
                 default_origin = [0, 0, 0],
                 default_angle = [0, 0, 0],
                 default_norm = 0.0,
                 default_quaternion = G.QTN_DICT[G.VIEW], 
                 default_intensity = [0, 1200, G.DEFAULT_IMAGE_SETTINGS['colormap'], 'Fire', 'Linear'],
                 default_histogram = [-0.5, 1200.5, 1, 'Linear'], # Bins min, max, step, scale
                 default_limits_origin = [-500, 500],
                 default_limits_norm = [-500, 500],
                 default_limits_intensity = [[-1e6, 1e6 - 1], [-1e6 + 1, 1e6]],
                 default_drawlayer_tag = ''):
        
        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()
        
        self.volume_names:list[str] = []
        self.n_volumes:int = 0
        self.group_name:str = group_name

        self.control_list: list[str] = ['origin_x', 'origin_y', 'origin_z', 'norm', 
                                        'pitch', 'yaw', 'roll', 
                                        'pixel_spacing_x', 'pixel_spacing_y', 'slice_thickness', 
                                        'min_intensity', 'max_intensity', 'colormap_name']
        self.orientation_control_list: list[str] = ['origin_x', 'origin_y', 'origin_z', 'norm', 
                                                    'pitch', 'yaw', 'roll', 
                                                    'pixel_spacing_x', 'pixel_spacing_y', 'slice_thickness']
        self.intensity_control_list:list[str] = ['min_intensity', 'max_intensity', 'colormap_name']
        self.histogram_control_list:list[str] = ['min_value', 'max_value', 'step', 'scale']
        self.current_volume: VolumeLayer = None
        self.volume_reference_dict: dict = {}
        self.text_info_tag: str = None
        self.texture_drawlayer_tags: list[str] = []
        self.landmark_drawlayer_tags: list[str] = []
        self.landmark_draw_layer_tag: str = None
        self.window_dict = {}

    def set_texture_drawlayer_tags(self, 
                                   drawlayer_tags: list[str]):
        
        self.texture_drawlayer_tags.extend([tag for tag in drawlayer_tags])


    def set_landmark_drawlayer_tags(self, 
                                   drawlayer_tags: list[str]):
        
        self.landmark_drawlayer_tags.extend([tag for tag in drawlayer_tags])
        self.landmark_draw_layer_tag = self.landmark_drawlayer_tags[0]


    def configure_orientation_options(self,
                                      default_origin,
                                      default_angle,
                                      default_norm,
                                      default_norm_sign,
                                      default_quaternion,
                                      default_geometry,
                                      default_limits_origin, 
                                      default_limits_norm,
                                      default_limits_geometry,
                                      default_texture_dim,
                                      default_texture_center, 
                                      voxel_start,
                                      voxel_center,
                                      voxel_steps,
                                      default_drawlayer_tags): 
        
        self.Orientation: OptionValue.OrientationInfo = OptionValue.OrientationInfo(tag = f'{self.group_name}|VolumeLayerGroup|Orientation',
                                                                                    default_origin = default_origin,
                                                                                    default_angle = default_angle, 
                                                                                    default_norm = default_norm,
                                                                                    default_norm_sign = default_norm_sign,
                                                                                    default_quaternion = default_quaternion, 
                                                                                    default_geometry = default_geometry,
                                                                                    default_limits_origin = default_limits_origin,
                                                                                    default_limits_norm = default_limits_norm,
                                                                                    default_limits_geometry = default_limits_geometry,
                                                                                    default_texture_dim = default_texture_dim, #self.CTVolume.texture_dim,
                                                                                    default_texture_center = default_texture_center, #self.CTVolume.texture_center)
                                                                                    voxel_start = voxel_start,
                                                                                    voxel_center = voxel_center,
                                                                                    voxel_steps = voxel_steps,
                                                                                    default_drawlayer_tags = default_drawlayer_tags)
        
    def configure_intensity_options(self, 
                                    default_colormap_info):
        self.Intensity: OptionValue.IntensityInfo = OptionValue.IntensityInfo(tag = f'{self.group_name}|VolumeLayerGroup|Intensity',
                                                                              default_colormap_info = default_colormap_info)
        
    def configure_histogram_options(self, 
                                    default_histogram):
        self.VolumeHistogram: OptionValue.HistogramInfo = OptionValue.HistogramInfo(default_histogram = default_histogram)
        self.TextureHistogram: OptionValue.HistogramInfo = OptionValue.HistogramInfo(default_histogram = default_histogram)


    def set_text_info_tag(self, drawlayer_tag:str):
        self.text_info_tag = drawlayer_tag


    def set_landmark_draw_layer_tag(self, drawlayer_tag:str):
        self.landmark_draw_layer_tag = drawlayer_tag


    def set_draw_window_dict(self, 
                            window_dict: dict):
        self.window_dict = window_dict

    def get_volume_by_name(self, volume_name: str) -> VolumeLayer:
        if volume_name not in self.volume_names:
            print(f'VolumeLayer Message: {volume_name} not in {self.group_name}')
            return
        return getattr(self, volume_name)


    def get_volume_by_index(self, volume_index: int) -> VolumeLayer:
        if volume_index < 0:
            print(f'VolumeLayer Message: {volume_index} is a negative value!')
            return getattr(self, self.volume_names[0])
        
        if volume_index > self.n_volumes:
            print(f'VolumeLayer Message: {volume_index} larger than number of volumes ({self.n_volumes}) in {self.group_name}!')
            return getattr(self, self.volume_names[-1])
        
        return getattr(self, self.volume_names[volume_index - 1])


    def get_last_volume(self) -> VolumeLayer:
        return getattr(self, self.volume_names[-1])


    def get_volume_attribute(self, volume_name: str, attribute_name: str):
        return self.get_volume_by_name(volume_name).get_attribute(attribute_name)
    

    def set_volume_attribute(self, volume_name: str, attribute_name: str, attribute_value):
        self.get_volume_by_name(volume_name).set_attribute(attribute_name, attribute_value)


    def set_colormap_scale_tag(self, 
                               colormap_scale_tag: str):
        self.Intensity.set_colormap_scale_tag(colormap_scale_tag)


    def set_drawlayer_info(self,
                           drawlayer_tags,
                           drawlayer_info):
        
        self.Orientation.set_drawlayer_tags(drawlayer_tags)


    def set_drawlayer_tags(self, 
                           drawlayer_tags: list[str]):
        self.Orientation.set_drawlayer_tags(drawlayer_tags)


    def set_volume(self, volumeLayer: VolumeLayer):
        if volumeLayer.name not in self.volume_names:
            print(f'VolumeLayer Message: Adding Volume {volumeLayer.name} to Group {self.group_name}')
            self.volume_names.append(volumeLayer.name)
            setattr(self, volumeLayer.name, volumeLayer)
        else:
            print(f'VolumeLayer Message: Volume {volumeLayer.name} already in Group {self.group_name}')


    def add_volume(self, 
                   ctvolume:CTVolume.CTVolume,
                   volume_name = None, 
                   default_origin = [0, 0, 0],
                   default_angle = [0, 0, 0],
                   default_norm = 0.0,
                   default_quaternion = G.QTN_DICT[G.VIEW], 
                   default_colormap_info = {'min_intensity': {'tag': G.OPTION_TAG_DICT['min_intensity'],
                                                              'default_value': 0.0, 
                                                              'default_limits': [-1e6, 1e6 - 1]},
                                            'max_intensity': {'tag': G.OPTION_TAG_DICT['max_intensity'],
                                                              'default_value': 1200.0, 
                                                              'default_limits': [-1e6 + 1, 1e6]},
                                            'colormap': G.DEFAULT_IMAGE_SETTINGS['colormap'],
                                            'colormap_reversed': False,
                                            'colormap_name': G.DEFAULT_IMAGE_SETTINGS['colormap_combo'],
                                            'colormap_scale_type': 'Linear',
                                            'colormap_scale_tag': 'COLORMAP_SCALE_NOT_INITIALIZED'},
                   default_limits_origin = [-G.TEXTURE_CENTER, G.TEXTURE_CENTER],
                   default_limits_norm = [-G.TEXTURE_CENTER, G.TEXTURE_CENTER],  # [[min_low, min_high], [max_low, max_high]]
                   ):

        temp_name = ctvolume.name if volume_name is None else volume_name
        if temp_name in self.volume_names:
            print(f'VolumeLayer Message: Volume {temp_name} already in Group {self.group_name}.')
            return
        
        print(f'VolumeLayer Message: Adding Volume {temp_name} to Group {self.group_name}.')
        
        self.volume_names.append(temp_name)
        temp_index = int(self.n_volumes)

        if self.n_volumes == 0:

            default_norm_sign = ctvolume.slice_direction
            default_geometry = [1.0, 1.0, 1.0]
            default_limits_geometry = [0.10, 50.0]
            default_texture_dim = G.TEXTURE_DIM
            default_texture_center = G.TEXTURE_CENTER
            voxel_start = 1.0*ctvolume.physical_start
            voxel_center = 1.0*ctvolume.physical_center
            voxel_steps = 1.0*ctvolume.physical_steps
            default_histogram = [ctvolume.unique_values[0] - ctvolume.histogram_step/2, 
                                 ctvolume.unique_values[-1] + ctvolume.histogram_step/2,
                                 ctvolume.histogram_step, 
                                 'Linear']
            
            self.configure_orientation_options(default_origin,
                                               default_angle,
                                               default_norm,
                                               default_norm_sign,
                                               default_quaternion, 
                                               default_geometry,
                                               default_limits_origin,
                                               default_limits_norm,
                                               default_limits_geometry,
                                               default_texture_dim,
                                               default_texture_center,
                                               voxel_start,
                                               voxel_center,
                                               voxel_steps,
                                               self.texture_drawlayer_tags)
            
            self.configure_intensity_options(default_colormap_info)
            self.configure_histogram_options(default_histogram)

        setattr(self, temp_name, VolumeLayer(self, # This Group
                                             temp_index,
                                             ctvolume,
                                             volume_name = temp_name,
                                             default_origin = default_origin,
                                             default_angle = default_angle,
                                             default_norm = default_norm, 
                                             default_quaternion = default_quaternion, 
                                             default_colormap_info = default_colormap_info,
                                             default_histogram = [ctvolume.unique_values[0] - ctvolume.histogram_step/2, 
                                                                  ctvolume.unique_values[-1] + ctvolume.histogram_step/2,
                                                                  ctvolume.histogram_step, 
                                                                  'Linear'],
                                             default_limits_origin = default_limits_origin, 
                                             default_limits_norm = default_limits_norm,
                                             default_texture_drawlayers = self.texture_drawlayer_tags,
                                             default_landmark_drawlayers = self.landmark_drawlayer_tags))
        
        if self.n_volumes == 0:
            self.current_volume: VolumeLayer = self.get_volume_by_index(temp_index)
        self.n_volumes += 1
    

    def update_view(self):
        pass


    def set_current_volume(self, volume_index):
        self.current_volume.set_current(False)
        self.current_volume: VolumeLayer = self.get_volume_by_index(volume_index)
        self.current_volume.set_current(True)


    def get_orientation_info(self, control_mode = 'Group'):
        if control_mode == 'Group':
            return self.Orientation.view_plane(), self.Orientation.norm_vector(), self.Orientation.norm()
        
        else:
            pass


    def update_current_volume(self,
                              orientation_info: dict, 
                              intensity_info: dict,
                              changed_volume: bool,
                              update_type: str,
                              operation_instance: VolumeOperations.VolumeOperations):
        
        self.current_volume.update_all(orientation_info, 
                                       intensity_info,
                                       changed_volume,
                                       update_type,
                                       operation_instance)
        

    def add_landmark(self, 
                     drawing_coords: tuple[int],
                     image_coords: np.ndarray, 
                     image_coords_hu: float,
                     voxel_coords: np.ndarray,
                     quaternion: qtn.QuaternionicArray, 
                     viewplane_norm: np.ndarray, 
                     draw_layer: str,
                     color: tuple[int] = dpg.get_value('landmark_color_picker'),
                     size: float = 5.0, 
                     geometry: np.ndarray = np.ones(3, dtype = np.float32),
                     affine: np.ndarray = np.array([np.eye(4, dtype = np.float32)]*4),
                     landmark_patch: np.ndarray = np.zeros((11, 11), dtype = np.float32),
                     patch_size = 11,
                     show_landmark: bool = True):
        print('VolumeLayerGroup Message: add_landmark')
        print(f'\tcurrent_volume().add_landmark()')
        
        self.current_volume.add_landmark(drawing_coords,
                                         image_coords,
                                         image_coords_hu,
                                         voxel_coords,
                                         quaternion,
                                         viewplane_norm,
                                         draw_layer,
                                         color = color,
                                         size = size,
                                         geometry = geometry,
                                         affine = affine,
                                         landmark_patch = landmark_patch,
                                         patch_size = patch_size,
                                         show_landmark = show_landmark)
        
        
    def get_intensity_info(self, control_mode = 'Group'):
        pass


    def remove_volume(self, 
                      volume_name, 
                      clean_up = True):
        print(f'VolumeLayer Message: Removing {volume_name} from Group {self.group_name}.')
        # Doing this so we can simply use this function in remove_all_volumes below.
        if volume_name in self.volume_names:

            self.get_volume_by_name(volume_name).delete_texture()
            self.get_volume_by_name(volume_name).delete_landmarks()
            self.get_volume_by_name(volume_name)._cleanup_()
            self.volume_names.remove(volume_name)

        if clean_up:
            getattr(self, volume_name)._cleanup_()

        setattr(self, volume_name, None)
        delattr(self, volume_name)

        self.n_volumes -= 1

    def remove_all_volumes(self,
                           clean_up = False):
        while len(self.volume_names):
            volume_name = self.volume_names[-1]
            self.remove_volume(volume_name, clean_up = clean_up)

    def _cleanup_(self):
        self.remove_all_volumes(clean_up = True)
        dict_keys = list(self.__dict__.keys())
        while len(dict_keys):
            attrib_key = dict_keys[-1]
            # print(attrib_key)
            setattr(self, attrib_key, None)
            delattr(self, attrib_key)
            dict_keys.remove(attrib_key)
                
    def reorder_volumes(self):
        pass
    

class VolumeLayerGroups(object):
    def __init__(self):
        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()
        self.group_dict = {}
        self.group_names = []
        self.volume_names = {}  # {Group_1: [Volume_1, Volume_2, ...], 
                                #  Group_2: [Volume_1, Volume_2, ...], 
                                #  ...}
        self.current_group_and_volume: tuple[int, int] = (None, None)
        self.active = False
        self.volume_operations = VolumeOperations.VolumeOperations(texture_dim = G.TEXTURE_DIM)
        self.texture_drawlayer_tags = []
        self.landmark_drawlayer_tags = []
        self.window_dict = {}


    def get_group_by_name(self, group_name) -> VolumeLayerGroup:
        return getattr(self, group_name)

    def get_group_by_index(self, group_index) -> VolumeLayerGroup:
        return getattr(self, self.group_names[group_index])
    
    def get_volume_by_index(self, group_index, volume_index) -> VolumeLayer:
        return self.get_group_by_index(group_index).get_volume_by_index(volume_index)

    def get_volume_by_name(self, group_name, volume_name) -> VolumeLayer:
        return self.get_group_by_name(group_name).get_volume_by_name(volume_name)

    def reset_current_group_and_volume(self):
        setattr(self, 'current_group_and_volume', tuple([None, None]))

    def set_current_volume_by_index(self, group_index, volume_index):
        if group_index == None:
            setattr(self, 'current_group_and_volume', tuple([None, None]))
            return
        self.get_group_by_index(group_index).set_current_volume(volume_index)
        setattr(self, 'current_group_and_volume', tuple([group_index, volume_index]))

    def set_current_volume_by_name(self, group_name, volume_name):
        group_index = self.group_names.index(group_name)
        volume_index = self.get_group_by_index(group_index).volume_names.index(volume_name)

        self.set_current_volume_by_index(group_index, volume_index)

    def get_current_group(self) -> VolumeLayerGroup:
        if self.current_group_and_volume[0] == None:
            return
        return self.get_group_by_index(self.current_group_and_volume[0])
    
    def get_current_volume(self) -> VolumeLayer:
        if self.current_group_and_volume[0] == None:
            return
        return self.get_volume_by_index(*self.current_group_and_volume)
    
    def get_current_volume_index(self) -> int:
        pass

    def set_options_panel(self, **options_panel_info):
        """
        
        """
        pass


    def set_draw_window_dict(self, 
                            window_dict: dict):
        self.window_dict = window_dict


    def set_texture_drawlayer_tags(self, 
                                   drawlayer_tags: list[str]):
        
        self.texture_drawlayer_tags.extend([tag for tag in drawlayer_tags])


    def set_landmark_drawlayer_tags(self, 
                                   drawlayer_tags: list[str]):
        
        self.landmark_drawlayer_tags.extend([tag for tag in drawlayer_tags])


    def set_control_options(self, 
                            changing_volumes: bool = False) -> None:
        self.get_current_volume().set_control_options('Orientation', changing_volumes = changing_volumes)
        self.get_current_volume().set_control_options('Intensity', changing_volumes = changing_volumes)


    def change_current_group_and_volume(self, 
                                        img_index: int = 0):
        self.get_current_volume().hide_drawimage()
        self.get_current_volume().hide_landmarks()
        self.set_current_volume_by_index(0, img_index)

        self.set_control_options(changing_volumes = True)
        self.get_current_volume().show_landmarks()
        self.get_current_volume().show_drawimage()

        dpg.set_item_label(f'orientation_group_layer_control_button', self.get_current_volume().orientation_control)
        dpg.set_item_label(f'intensity_group_layer_control_button', self.get_current_volume().intensity_control)
        dpg.set_value('colormap_combo_current_value', self.get_current_volume().get_intensity().colormap_name.current_value)
        dpg.set_value('reverse_colormap_checkbox', self.get_current_volume().get_intensity().colormap_reversed)
        dpg.set_item_label(G.VOLUME_TAB_TAG, f'Volume Tab: {self.get_current_volume().name}')


    def update_frame_of_reference(self,
                                  orientation_control: str = 'Group',
                                  intensity_control: str = 'Group'):
        
                self.get_current_volume().update_control(orientation_control = orientation_control,
                                                         intensity_control = intensity_control)
                self.set_control_options(changing_volumes = False)

                dpg.set_item_user_data('orientation_group_layer_control_button', False)
                dpg.set_item_user_data('intensity_group_layer_control_button', False)

    def update_control(self, 
                       img_index: int = 0,
                       orientation_control: str = 'Group',
                       intensity_control: str = 'Group',
                       update_orientation_group_control: bool = False,
                       update_intensity_group_control: bool = False) -> bool:
        
        changed_volume = False
        with dpg.mutex():
            if self.current_group_and_volume != (0, img_index):
                self.change_current_group_and_volume(img_index = img_index)
                changed_volume = True

            elif (update_orientation_group_control or update_intensity_group_control):
                self.update_frame_of_reference(orientation_control = orientation_control,
                                               intensity_control = intensity_control)
            else: 
                self.set_control_options(changing_volumes = changed_volume)

        return changed_volume
    
    def update_operation(self, 
                         operation_info: dict):
        self.volume_operations.set_operation(operation_info['enabled'],
                                             operation_info['start'],
                                             operation_info['stop'],
                                             operation_info['operation'],
                                             operation_info['weighted'])

            
    def update_current_volume(self, 
                              changed_volume: bool|None,
                              control_info: dict,
                              update_type: str, 
                              orientation_info: dict,
                              intensity_info: dict,
                              text_info: dict,
                              operation_info: dict):
        """
        We need to align the OptionPanel with the Volume. 
        We already ensured that the OptionPanel reflects the current volume in update_control, which
        handles switching between volumes and between Group-Layer control. 
        """
        # orientation_info['apply_scaling'] = not changed_volume
        # Control update is handled in OptionsPanel.update_image_index and update_frame_of_reference
        self.update_operation(operation_info)
        print(f'VolumeLayerGroups Message:')
        print(f'\t{orientation_info = }')
        self.get_current_group().update_current_volume(orientation_info,
                                                       intensity_info,
                                                       changed_volume,
                                                       update_type,
                                                       self.volume_operations)
        
        crosshair_draw_pos = self.get_crosshair_drawing_pos()
        mouse_draw_pos = dpg.get_drawing_mouse_pos()

        mouse_text_tag = text_info['mouse_info_text_tag']
        crosshair_text_tag = text_info['crosshair_info_text_tag']
        window_type = text_info['window_type']

        for position, text_tag, position_id in zip([crosshair_draw_pos, mouse_draw_pos], 
                                                   [crosshair_text_tag, mouse_text_tag],
                                                   ['Crosshair', 'Mouse']):
            self.set_drawing_position_info_text(
                                text_tag,
                                self.get_drawing_position_info_text(*position, position_id, window_type),
                )
            
        if changed_volume:
            self.update_histogram('volume')

        self.update_histogram('texture')
        

    def get_histogram_info(self, 
                           histogram_type: str):
        
        histogram_info_volume = {'InfoBoxTab_histogram_volume_plot_line_series': 'InfoBoxTab_histogram_volume_plot_line_series',
                                 'InfoBoxTab_histogram_volume_bins_min': dpg.get_value('InfoBoxTab_histogram_volume_bins_min'),
                                 'InfoBoxTab_histogram_volume_bins_min_checkbox': dpg.get_value('InfoBoxTab_histogram_volume_bins_min_checkbox'),
                                 'InfoBoxTab_histogram_volume_bins_max': dpg.get_value('InfoBoxTab_histogram_volume_bins_max'),
                                 'InfoBoxTab_histogram_volume_bins_max_checkbox': dpg.get_value('InfoBoxTab_histogram_volume_bins_max_checkbox'),
                                 'InfoBoxTab_histogram_volume_bin_step': dpg.get_value('InfoBoxTab_histogram_volume_bin_step'),
                                 'InfoBoxTab_histogram_volume_bin_step_checkbox': dpg.get_value('InfoBoxTab_histogram_volume_bin_step_checkbox')}
        
        histogram_info_current_view = {'InfoBoxTab_histogram_texture_plot_line_series': 'InfoBoxTab_histogram_texture_plot_line_series',
                                       'InfoBoxTab_histogram_texture_bins_min': dpg.get_value('InfoBoxTab_histogram_texture_bins_min'),
                                       'InfoBoxTab_histogram_texture_bins_min_checkbox': dpg.get_value('InfoBoxTab_histogram_texture_bins_min_checkbox'),
                                       'InfoBoxTab_histogram_texture_bins_max': dpg.get_value('InfoBoxTab_histogram_texture_bins_max'),
                                       'InfoBoxTab_histogram_texture_bins_max_checkbox': dpg.get_value('InfoBoxTab_histogram_texture_bins_max_checkbox'),
                                       'InfoBoxTab_histogram_texture_n_bins': dpg.get_value('InfoBoxTab_histogram_texture_n_bins'),
                                       'InfoBoxTab_histogram_texture_n_bins_checkbox': dpg.get_value('InfoBoxTab_histogram_texture_n_bins_checkbox')}

        return histogram_info_volume if histogram_type == 'volume' else histogram_info_current_view


    def update_histogram(self,
                         histogram_type: str, 
                         histogram_info: dict = None):
        
        if histogram_info == None:
            histogram_info = self.get_histogram_info(histogram_type)
        
        self.get_current_volume().update_histogram(histogram_type,
                                                   histogram_info)


    def get_last_group(self) -> VolumeLayerGroup:
        return self.get_group_by_name(self.group_names[-1])
    

    def get_last_volume(self) -> VolumeLayer:
        return self.get_last_group().get_last_volume()


    def add_group(self, group_name = 'Group_1', **kwargs):
        if group_name in self.group_names:
            print(f'VolumeLayer Message: {group_name} already added.')
            return
        
        print(f'VolumeLayer Message: Adding {group_name} to groups.')
        setattr(self, group_name, VolumeLayerGroup(group_name, **kwargs))
        self.group_dict[group_name] = []
        self.group_names.append(group_name)

        self.get_last_group().set_draw_window_dict(self.window_dict)

        if self.current_group_and_volume[0] == None:
            self.current_group_and_volume = (0, None)


    def add_volume_to_group(self, 
                            group_name, 
                            ctvolume: CTVolume.CTVolume,
                            **kwargs):
        if group_name not in self.group_names:
            self.add_group(group_name = group_name)
            self.group_dict[group_name] = []

        if ctvolume.name not in self.group_dict[group_name]:
            self.get_group_by_name(group_name).add_volume(ctvolume,
                                                          **kwargs)
            self.group_dict[group_name].append(ctvolume.name)

            self.get_last_volume().add_textures_to_drawlayers(self.texture_drawlayer_tags)
            self.get_last_volume().hide_drawimage()
        
    def add_volumes_to_group(self, group_name, list_of_ctvolumes, **kwargs):
        for ctvolume in list_of_ctvolumes:
            self.add_volume_to_group(group_name, 
                                     ctvolume,
                                     **kwargs)


    def set_tags(self, tags_dict: dict):
        print('VolumeLayerGroups Message: NOT Adding Tags')
        for tag_name, tag_value in tags_dict.items():
            print(f'\t: {tag_name:<25}: {tag_value}')
            # self.get_current_volume().add_tags(tags_dict)


    def get_current_orientation(self) -> OptionValue.OrientationInfo:
        return self.get_current_volume().get_orientation()


    def get_norm_vector(self, print = False) -> np.ndarray:
        modifier = 'current_value'
        if print: 
            modifier = ''
        return self.get_current_volume().get_orientation_value('norm_vector', modifier)


    def get_origin_vector(self, print = False) -> np.ndarray:
        modifier = 'current_value'
        if print: 
            modifier = ''
        return self.get_current_volume().get_orientation_value('origin_vector', modifier)           
    

    def get_viewport_origin_vector(self, print = False) -> np.ndarray:
        modifier = 'current_value'
        if print: 
            modifier = ''
        return self.get_current_volume().get_orientation_value('viewport_origin_vector', modifier)
    

    def add_landmark(self, 
                     drawing_coords: tuple[int],
                     window_type: str = 'view_plane') -> None:
        """

        drawing_coords: tuple[int, int]
            Coordinates on the drawing canvas with format (x, y).

        """

        image_coords = self.get_drawing_pos_coords(drawing_coords[0], 
                                                   drawing_coords[1],
                                                   window_type = window_type)
        
        image_coords_hu = self.get_drawing_pos_texture_value(drawing_coords[0], 
                                                             drawing_coords[1],
                                                             window_type = window_type)
        
        voxel_coords = self.get_physical_voxel_coords(drawing_coords[0], 
                                                      drawing_coords[1],
                                                      window_type = window_type)
        
        quaternion = self.get_current_orientation().quaternion.current_value
        viewplane_norm = self.get_current_orientation().norm_vector.current_value.get().reshape((3, ))
        draw_layer = self.get_current_group().landmark_draw_layer_tag
        color = dpg.get_value('landmark_color_picker')
        geometry = self.get_current_orientation().geometry_vector.current_value.get().reshape((3, ))
        affine = self.get_current_orientation().affine.matrices.get()
        size = 5.0
        patch_width = int(10 * np.mean(geometry))
        patch_x = [int(drawing_coords[0] - patch_width), int(drawing_coords[0] + patch_width + 1)]
        patch_y = [int(drawing_coords[1] - patch_width), int(drawing_coords[1] + patch_width + 1)]
        landmark_patch = self.get_texture_patch(patch_x,
                                                patch_y,
                                                exclude_nan = False,
                                                rescale = False,
                                                return_as_numpy = True,
                                                colorize = True)

        show_landmark = True
        
        print('VolumeLayerGroups Message: add_landmark')
        print(f'\tget_current_group().add_landmark()')
        self.get_current_group().add_landmark(drawing_coords,
                                              image_coords,
                                              image_coords_hu,
                                              voxel_coords,
                                              quaternion,
                                              viewplane_norm,
                                              draw_layer,
                                              color = color, 
                                              size = size, 
                                              geometry = geometry, 
                                              affine = affine,
                                              landmark_patch = landmark_patch,
                                              patch_size = 2*patch_width + 1,
                                              show_landmark = show_landmark)
        

    def delete_landmark(self, volume_name, landmark_id):
        self.get_volume_by_name(self.group_names[0], volume_name).delete_landmark(landmark_id)


    def save_landmarks(self):
        if self.active:
            print(f'VolumeLayerGroups Message: Saving landmarks for {self.get_current_volume().name}.')
            self.get_current_volume().save_landmarks()
            # for group_name in self.group_dict.keys():
            #     for volume_name in self.group_dict[group_name]:
            #         self.get_volume_by_name(group_name, volume_name).save_landmarks()


    def load_landmarks(self):
        if self.active:
            with dpg.mutex():
                print(f'VolumeLayerGroups Message: Loading landmarks for {self.get_current_volume().name}.')

                landmark_data = self.get_current_volume().load_landmark_data(self.get_current_orientation().origin_vector.current_value.reshape((3, )).get(),
                                                                             self.get_current_orientation().quaternion.current_value,
                                                                             self.get_current_orientation().geometry_vector.current_value.reshape((3, )).get())
                
                landmark_draw_layer_tag = self.get_current_group().landmark_draw_layer_tag
                if landmark_data == False:
                    return
                
                self.get_current_volume().reset_orientation()
                # print(f'\tget_current_volume().get_landmark_patches')
                patches, patch_widths = self.get_current_volume().get_landmark_patches(landmark_data, self.volume_operations)

                last_landmark_info = []
                
                for landmark_index in range(len(patches)):
                    drawing_coords = tuple([landmark_data['drawing_coords'][landmark_index, :2].astype(int)[0].item(), 
                                            landmark_data['drawing_coords'][landmark_index, :2].astype(int)[1].item()])
                    

                    self.get_current_group().add_landmark(drawing_coords,
                                                         landmark_data['image_coords'][landmark_index, :3],
                                                         landmark_data['image_coords'][landmark_index, 3],
                                                         landmark_data['voxel_coords'][landmark_index, :3], 
                                                         landmark_data['quaternions'][landmark_index],
                                                         landmark_data['norms'][landmark_index],
                                                         landmark_draw_layer_tag,
                                                         color = dpg.get_value('landmark_color_picker'),
                                                         size = 5.0, 
                                                         geometry = landmark_data['geometries'][landmark_index],
                                                         affine = landmark_data['affines'][landmark_index],
                                                         landmark_patch = patches[landmark_index],
                                                         patch_size = 2*patch_widths[landmark_index] + 1, 
                                                         show_landmark = True)
                    
                    print(f'\tCrosshair Coords: {self.get_crosshair_coords().round(decimals=3)}')
                    # print(f'\t{self.get_last_landmark() = }')
                    last_landmark_info.append(self.get_last_landmark())

            return last_landmark_info


    def get_last_landmark(self) -> list:
        return self.get_current_volume().Landmarks.get_last_landmark()


    def update_mouse_volume_coord_info(self, 
                                       sender, 
                                       app_data, 
                                       user_data: dict):
        """
        sender:

        app_data:

        user_data: dict
            Dictionary with keys: 
                'mouse_info_text_tag'
                'window_type' 
        """
        if self.active:
            self.set_drawing_position_info_text(user_data['mouse_info_text_tag'], 
                                                self.get_drawing_position_info_text(*dpg.get_drawing_mouse_pos(),
                                                                                    'Mouse',
                                                                                     user_data['window_type']))


    def get_drawing_pos_coords(self, 
                               draw_x: int,
                               draw_y: int,
                               window_type: str = 'view_plane') -> np.ndarray:
        if self.active:
            if window_type == 'view_plane':
                return self.get_current_orientation().view_plane.get_coords(int(draw_x), 
                                                                            int(draw_y)).get()
        
            elif window_type == 'view_plane_ortho':
                return self.get_current_orientation().view_plane_ortho.get_coords(int(draw_x), 
                                                                                  int(draw_y)).get()
            else:
                print(f'VolumeLayerGroups Message: Unknown window type {window_type} in get_drawing_pos_coords')


    def get_mouse_coords(self) -> np.ndarray:
        if self.active:
            mouse_x, mouse_y = dpg.get_drawing_mouse_pos()

            return self.get_drawing_pos_coords(int(mouse_x), int(mouse_y))


    def get_crosshair_drawing_pos(self, window_type:str = 'view_plane'):
        if self.active:
            if window_type == 'view_plane':
                return int(G.TEXTURE_CENTER), int(G.TEXTURE_CENTER)
            elif window_type == 'view_plane_ortho':
                return int(G.TEXTURE_CENTER), int(G.TEXTURE_CENTER)


    def get_crosshair_coords(self) -> np.ndarray:
        if self.active:
            return self.get_current_volume().get_crosshair_coords()
        

    def get_physical_pos_coords(self, 
                                draw_x: int, 
                                draw_y: int,
                                window_type: str = 'view_plane') -> np.ndarray:
        

        return self.get_current_volume().get_physical_pos_coords(draw_x, draw_y, window_type = window_type)
    

    def get_physical_voxel_coords(self, 
                                  draw_x: int,
                                  draw_y: int,
                                  window_type: str = 'view_plane') -> np.ndarray:
            
        return self.get_current_volume().get_physical_voxel_coords(draw_x, draw_y, window_type = window_type)
    

    def get_drawing_pos_texture_value(self, 
                                      draw_x: int,
                                      draw_y: int,
                                      window_type: str = 'view_plane',
                                      decimals: int = 3) -> float|int:
        
        return self.get_current_volume().get_drawing_pos_texture_value(draw_x, draw_y, window_type = window_type, decimals = decimals)
    

    def get_texture_patch(self, 
                          patch_x: list[int],
                          patch_y: list[int],
                          exclude_nan: bool = False,
                          rescale: bool = True,
                          return_as_numpy: bool = True,
                          colorize: bool = True):
        
        return self.get_current_volume().get_texture_patch(patch_x, 
                                                           patch_y,
                                                           exclude_nan = exclude_nan,
                                                           rescale = rescale,
                                                           return_as_numpy = return_as_numpy,
                                                           colorize = colorize)


    def get_drawing_position_info_text(self,
                                       draw_x: int,
                                       draw_y: int,
                                       position_identifier: str, 
                                       window_type: str):
        
        if self.current_group_and_volume[0] == None:
            return
        
        if self.get_current_group().n_volumes < 1:
            return


        if window_type == 'view_plane_ortho':
            pixel_start = self.get_current_volume().TextureOrtho.pixel_start
            draw_x -= pixel_start[0]
            draw_y -= pixel_start[1]
        
        coords = self.get_drawing_pos_coords(draw_x, draw_y, window_type = window_type)
        coords.round(3) + 0.000

        coords_hu = self.get_drawing_pos_texture_value(draw_x, draw_y, window_type = window_type, decimals = 2)

        physical_coords = self.get_physical_pos_coords(draw_x, draw_y, window_type = window_type).round(3)
        physical_voxel_coords = self.get_physical_voxel_coords(draw_x, draw_y, window_type = window_type).round(3)

        spacing = ' '*18
        position_id_string = f'{position_identifier} Position'

        display_text =                  f'{spacing}  (X    , Y    , Z    , HU   )'
        texture_position_string =       f'Texture Position  : ({draw_x:.2f}, {draw_y:.2f})'
        physical_coords_string =        f'Physical Position : ({physical_coords[1]:.2f}, {physical_coords[0]:.2f}, {physical_coords[2]:.2f})'
        physical_voxel_coords_string =  f'Voxel Position    : ({physical_voxel_coords[1]:.2f}, {physical_voxel_coords[0]:.2f}, {physical_voxel_coords[2]:.2f})'
        position_coords_string =        f'{position_id_string:<18}: ({coords[1]:.2f}, {-1.0*coords[0] + 0.0:.2f}, {coords[2]:.2f}, {coords_hu:.2f})'
        display_text = f'{display_text}\n{texture_position_string}\n{physical_coords_string}\n{physical_voxel_coords_string}\n{position_coords_string}'

        return display_text

    
    def set_drawing_position_info_text(self, 
                                       text_tag: str|int,
                                       display_text: str):
    
        text_drawlayer = dpg.get_item_parent(text_tag)
        text_pos = dpg.get_item_user_data(text_tag)
        if dpg.does_item_exist(text_tag):
            dpg.delete_item(text_tag)
        if dpg.does_alias_exist(text_tag):
            dpg.remove_alias(text_tag)

        dpg.draw_text(text_pos, 
                      display_text, 
                      tag = text_tag, 
                      user_data = text_pos,
                      size = 14, 
                      parent = text_drawlayer)


    def update_landmark_colors(self):
        self.get_current_volume().update_landmarks(update_type = 'Colors')


    def set_active(self):
        self.active = True


    def set_inactive(self):
        self.active = False

    
    def rescale_volume(self, rescale_bool):
        self.get_current_volume().CTVolume.rescale_volume(rescale_bool)


    def get_orientation_info(self):
        pass


    def get_intensity_info(self):
        pass


    def get_control_info(self):
        pass


    def get_text_info(self):
        pass


    def remove_group(self, group_name):
        print(f'VolumeLayer Message: Removing {group_name}')
        self.get_group_by_name(group_name).remove_all_volumes()
        setattr(self, group_name, None)
        delattr(self, group_name)
        self.group_names.remove(group_name)
        del self.group_dict[group_name]


    def remove_all_groups(self):
        while len(self.group_names):
            self.remove_group(self.group_names[-1])
        self.reset_current_group_and_volume()
        self.group_dict.clear()
        
        self.set_inactive()

        print(f'VolumeLayerGroups Message: Removed all Volumes: ')
        print(f'\t{self.active = }')
        print(f'\t{self.group_dict = }')
        print(f'\t{self.group_names = }')
        print(f'\t{self.current_group_and_volume = }')


    def _cleanup_(self):
        self.remove_all_groups()
        dict_keys = list(self.__dict__.keys())
        while len(dict_keys):
            attrib_key = dict_keys.pop()
            setattr(self, attrib_key, None)