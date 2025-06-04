from .Globals import *
from . import VolumeLayer
# Holds the primary viewing window for the chosen image. 
# Works closely with ImageTools

class MainView:
    def __init__(self):
        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()
            print(f'MainView Message: Using CUDA device {G.DEVICE}.')

        self.mode_return_type = lambda x, y: x.astype(getattr(cp, y)) if G.GPU_MODE else lambda x, y: x.astype(getattr(np, y))
        self.mode_return_array = lambda x: x.get() if G.GPU_MODE else lambda x: x
        self.mode_create_array = lambda x: cp.array(x) if G.GPU_MODE else lambda x: np.array(x)
        self.mode_function = lambda x: getattr(cp, x.__name__) if G.GPU_MODE else getattr(np, x.__name__)
        self.mode_number = lambda x: getattr(cp, x.__str__()) if G.GPU_MODE else getattr(np, x.__str__())
        
        self.plot_x_axis = 'main_view_plot_x_axis'
        self.plot_y_axis = 'main_view_plot_y_axis'
        self.colormap_scale = 'main_view_colormap_scale'
        self.mouse_hover_handler = 'main_view_mouse_hover_handler'
        self.information_box = 'main_view_information_box'
        self.mouse_position_text = 'mouse_position_text'
        self.crosshair_position_text = 'crosshair_position_text'
        self.mouse_position_information = [0, 0, 0, 0]
        self.crosshair_position_information = [0, 0, 0, 0]
        
        self.plot_x_line = 'main_view_x_crosshair'
        self.plot_y_line = 'main_view_y_crosshair'
        
        with dpg.group(horizontal = True):
            with dpg.child_window(width = G.CONFIG_DICT['app_settings']['main_pane_width'], # G.MAIN_PLOT_VIEW_WINDOW_DEFAULTS['WINDOW_WIDTH'], 
                                  height =G.CONFIG_DICT['app_settings']['main_pane_height']): # G.MAIN_PLOT_VIEW_WINDOW_DEFAULTS['WINDOW_HEIGHT']
                with dpg.group(horizontal = True):
                    with dpg.plot(label = '', tag = G.MAIN_PLOT_VIEW, equal_aspects = True,
                                  width = G.CONFIG_DICT['app_settings']['main_plot_width'], 
                                  height = G.CONFIG_DICT['app_settings']['main_plot_width']
                                ):
                        dpg.add_plot_axis(dpg.mvXAxis, tag=self.plot_x_axis, 
                                              no_gridlines = True, no_tick_marks = True, 
                                              no_tick_labels = True)
                        dpg.add_plot_axis(dpg.mvYAxis, tag=self.plot_y_axis,
                                              no_gridlines = True, no_tick_marks = True, 
                                              no_tick_labels = True)
                        
                    
                    dpg.add_colormap_scale(min_scale=0, 
                                           max_scale=1200, 
                                           width=65, 
                                           height=-1, 
                                           tag = self.colormap_scale, 
                                           colormap=G.DEFAULT_IMAGE_SETTINGS['colormap_scale'])
        
        with dpg.window(label="Loading Volume", modal=False, show=False, tag=G.LOADING_WINDOW_TAG,
                        width = 800, height = 400, pos = [250, 250]):
            dpg.add_text('', tag=G.LOADING_WINDOW_TEXT)
        
        # Set up mouse handler registry. 
        dpg.add_item_hover_handler(parent = G.ITEM_HANDLER_REG_TAG, callback = self.main_view_hovered)
        dpg.bind_item_handler_registry(G.MAIN_PLOT_VIEW, G.ITEM_HANDLER_REG_TAG)
        # dpg.add_mouse_move_handler(callback=self.check_mouse_hover_position, tag = self.mouse_hover_handler, parent = G.HANDLER_REG_TAG)
        dpg.add_mouse_wheel_handler(callback=self.main_view_scroll, tag = 'main_view_mouse_scroll_handler', parent=G.HANDLER_REG_TAG)
        dpg.add_key_down_handler(dpg.mvKey_LControl, callback = self.main_view_key_down, tag = 'main_view_key_down_handler_Control', parent = G.HANDLER_REG_TAG)
        dpg.add_key_down_handler(dpg.mvKey_LShift, callback = self.main_view_key_down, tag = 'main_view_key_down_handler_Shift', parent = G.HANDLER_REG_TAG)
        dpg.add_key_press_handler(key = dpg.mvKey_None, callback = self.main_view_keyboard_navigation, tag = 'main_view_keyboard_navigation_handler', parent = G.HANDLER_REG_TAG)
        # dpg.add_key_down_handler(dpg.mvKey_Alt, callback = self.main_view_key_down, tag = 'main_view_key_down_handler_Alt', parent = G.HANDLER_REG_TAG)
        # dpg.add_key_down_handler(dpg.mvKey_Spacebar, callback = self.main_view_key_down, tag = 'main_view_key_down_handler_Spacebar', parent = G.HANDLER_REG_TAG)

        self.current_group_info:VolumeLayer.VolumeLayerGroup = None
        self.current_volume_info:VolumeLayer.VolumeLayer = None
        self.current_group_name:str = ''
        self.current_volume_name:str = ''
        self.current_volume_index:int = 0
        self.landmark_view_coords = self.mode_function(np.zeros)((3, 11, 11))
        self.landmark_steps = self.mode_function(np.array)(range(-5, 6))
        self.last_mouse_position = np.array([0, 0, 0])

        self.refresh_mouse = 0 

    
    def initialize_view(self):
        self.colormap = G.DEFAULT_IMAGE_SETTINGS['colormap']
        self.default_orientation = G.DEFAULT_IMAGE_SETTINGS['view'] # axial, coronal, sagittal
        
        self.min_value = self.current_volume_info.intensity.min_intensity.default_value
        self.max_value = self.current_volume_info.intensity.max_intensity.default_value
        
        # view_plane is the texture plane, not the interpolated location plane. 

        self.initial_view_plane = self.current_volume_info.orientation.view_plane.default_value
        self.view_plane_shape = (G.TEXTURE_DIM, G.TEXTURE_DIM)
        
        self.current_view = self.mode_function(np.full)(self.view_plane_shape, self.mode_number(np.nan))
        self.windowed_view = self.mode_function(np.zeros)(self.view_plane_shape)
        self.current_mask_view = self.mode_function(np.zeros)(self.view_plane_shape)
        self.color_rgb = self.mode_function(np.zeros)(G.TEXTURE_DIM*G.TEXTURE_DIM)
        self.alpha_mask = self.mode_function(np.zeros)(self.view_plane_shape, dtype = bool)
        self.operation_view = self.mode_function(np.zeros)(self.view_plane_shape)
        self.operation_count = self.mode_function(np.zeros)(self.view_plane_shape)
        self.operation_volume = self.mode_function(np.full)((G.TEXTURE_DIM, G.TEXTURE_DIM, G.TEXTURE_DIM), 
                                                            self.mode_number(np.nan))

        self.initial_volume_index = 0
        self.initial_quaternion = 1.0*self.current_volume_info.orientation.quaternion.default_value
        self.initial_norm_vector = 1.0*self.current_volume_info.orientation.norm_vector.default_value
        self.initial_origin_vector = 1.0*self.current_volume_info.orientation.origin_vector.default_value
        self.current_view_plane = 1.0*self.current_volume_info.orientation.view_plane.current_value 
        self.current_norm_vector = 1.0*self.current_volume_info.orientation.norm_vector.current_value
        self.current_norm = 1.0*self.current_volume_info.orientation.norm.difference_value
        self.current_origin_vector = 1.0*self.current_volume_info.orientation.origin_vector.current_value
        self.current_origin_difference = self.current_origin_vector - self.initial_origin_vector
        self.current_quaternion = 1.0*self.current_volume_info.orientation.quaternion.current_value
        self.current_view_slice = 0
        self.view_scale = 1
        self.geometry_vector = cp.array([1.0, 1.0, 1.0])
        print(f'MainView Message: Initalize: {self.geometry_vector = }')
        load_message = dpg.get_value(G.LOADING_WINDOW_TEXT)
        load_message = f'{load_message}\n\nViewing window initialized!'
        dpg.set_value(G.LOADING_WINDOW_TEXT, load_message)
        
        return
    

    def delete_static_texture(self, texture_tag:str = ''):
        if texture_tag == '':
            texture_tag = G.TEXTURE_TAG

        if dpg.does_item_exist(texture_tag):
            dpg.delete_item(texture_tag)
            
        if dpg.does_alias_exist(texture_tag):
            dpg.remove_alias(texture_tag)


    def add_static_texture(self, 
                           width:int = 0, 
                           height:int = 0, 
                        #    format = dpg.mvFormat_Float_rgba, 
                           parent:str = '',
                           texture_tag:str = '', 
                           texture_array: np.ndarray = None):
        
        if type(texture_array) != type(None):
            texture_array = self.texture

        if texture_tag == '':
            texture_tag = G.TEXTURE_TAG

        if parent == '':
            parent = G.APP.texture_registry

        if (width == 0) or (height == 0):
            height, width = self.view_plane_shape

        dpg.add_static_texture(width = width, 
                               height = height, 
                               default_value = texture_array, 
                               tag = texture_tag, 
                               parent = parent)
        

    def initialize_volume_texture(self):
        
        print(f'MainView Message: {G.TEXTURE_DIM = }, {G.TEXTURE_DIM**2 = }')

        self.texture = np.zeros(G.TEXTURE_DIM * G.TEXTURE_DIM * 4, dtype = np.float32)
        self.mask_texture = np.zeros(self.texture.shape, dtype = np.float32)
        
        load_message = dpg.get_value(G.LOADING_WINDOW_TEXT)
        load_message = f'{load_message}\nTextures initialized!'
        dpg.set_value(G.LOADING_WINDOW_TEXT, load_message)
        
        return


    def interpolate_view(self):
        self.current_view[:] = self.mode_number(np.nan)

        if (dpg.get_value('ImageTools_operation_enabled') == True) or (dpg.get_value('ImageTools_compare_two_images_enabled') == True):
            self.current_view[:] = self.operation_view[:]
            return
        
        if G.GPU_MODE:
            self.current_volume_info.interpolate_view(out_array = self.current_view, 
                                                      # out_mask = self.alpha_mask.reshape(self.view_plane_shape), #self.alpha_mask.reshape(self.view_plane_shape), 
                                                      view_plane = (self.current_view_plane).round(decimals = 2))
            
        else:
            self.current_view[self.alpha_mask.reshape(self.view_plane_shape)] = getattr(self.current_volume, 'interpolator')(self.current_view_plane[0][self.alpha_mask],
                                                                                                                             self.current_view_plane[1][self.alpha_mask],
                                                                                                                             self.current_view_plane[2][self.alpha_mask])
        
        if dpg.get_value('rescale_colormap_checkbox'):
            self.current_view = self.current_view - self.mode_function(np.nanmin)(self.current_view)

        return
    

    def interpolate_mask(self):

        if G.GPU_MODE:
            self.current_volume_info.interpolate_mask(out_array = self.current_mask_view, 
                                                      # out_mask = self.alpha_mask.reshape(self.view_plane_shape), 
                                                      view_plane = self.current_view_plane)
                        
        else:
            self.current_mask_view[self.alpha_mask.reshape(self.view_plane_shape)] = getattr(self.current_volume, 'mask_interpolator')(self.current_view_plane[0][self.alpha_mask],
                                                                                                                                       self.current_view_plane[1][self.alpha_mask],
                                                                                                                                       self.current_view_plane[2][self.alpha_mask])
    

    def window_and_normalize_image(self):

        colormap_scale_type = dpg.get_value('colormap_scale_combo')

        match colormap_scale_type:
            case 'Log':
                self.current_view = self.mode_function(np.log10)(self.current_view + 1e-6)

            case 'Absolute':
                self.current_view = self.mode_function(np.abs)(self.current_view)

            case 'Abs-Log':
                self.current_view = self.mode_function(np.log10)(self.mode_function(np.abs)(self.current_view) + 1e-6)

            case _:
                pass
        
        self.windowed_view = (self.current_view - self.min_value)/(self.max_value - self.min_value)

        return
    
    
    def set_rgba_view(self):

        self.red_interp, self.green_interp, self.blue_interp = self.current_volume_info.colormap
        self.color_rgb[:] = self.windowed_view.clip(0, 1).flatten()[:]

        self.texture[0::4] = self.mode_return_type(self.mode_return_array(self.red_interp(self.color_rgb)), 'float32')[:]
        self.texture[1::4] = self.mode_return_type(self.mode_return_array(self.green_interp(self.color_rgb)), 'float32')[:]
        self.texture[2::4] = self.mode_return_type(self.mode_return_array(self.blue_interp(self.color_rgb)), 'float32')[:]
        self.texture[3::4] = self.mode_return_type(self.mode_return_array(self.alpha_mask.flatten()), 'float32')[:]


    def set_rgba_mask(self):
        r, g, b, a = dpg.get_value('mask_color_picker')

        if dpg.get_value('enable_mask_low_exclude_checkbox') == True:
            self.current_mask_view[self.mode_function(np.nan_to_num)(self.current_view, nan = -1e9) < dpg.get_value('mask_exclude_low_float')] = 1.0
        
        if dpg.get_value('enable_mask_high_exclude_checkbox') == True:
            self.current_mask_view[self.mode_function(np.nan_to_num)(self.current_view, nan = -1e9) > dpg.get_value('mask_exclude_high_float')] = 1.0

        self.mask_texture[0::4] = self.mode_return_type(self.mode_return_array(self.current_mask_view.flatten() * (r/255.0)), 'float32')[:]
        self.mask_texture[1::4] = self.mode_return_type(self.mode_return_array(self.current_mask_view.flatten() * (g/255.0)), 'float32')[:]
        self.mask_texture[2::4] = self.mode_return_type(self.mode_return_array(self.current_mask_view.flatten() * (b/255.0)), 'float32')[:]
        self.mask_texture[3::4] = self.mode_return_type(self.mode_return_array(self.current_mask_view.flatten() * self.alpha_mask.flatten() * dpg.get_value('mask_opacity_slider')), 'float32')[:]            
    

    def update_alpha_mask(self):
        # self.alpha_mask is a boolean array where we don't have nans. 
        self.alpha_mask[:] = ~self.mode_function(np.isnan)(self.current_view)


    def load_image(self):
        """
        Function that loads the volume and volume information as:

        self.current_group_info
            VolumeLayerGroup object
            Can access each volume in group via:
                self.current_group_info.<volume_layer_name>

        self.current_volume_info
            VolumeLayer object
        
        We want both of these so we can manipulate the entire group via self.current_group_info
        or the individual volume via self.current_volume_info

        """
        print('MainView Message: MainView Message: Loading image.')
        self.current_group_name = 'AllVolumes'
        self.current_group_info = getattr(G.APP.VolumeLayerGroups, self.current_group_name) # Returns VolumeLayerGroups.VolumeLayerGroup
        self.current_volume_name = self.current_group_info.volume_names[self.current_volume_index]
        self.current_volume_info = getattr(self.current_group_info, self.current_volume_name) # Returns VolumeLayerGroup.VolumeLayer
        print('MainView Message: MainView Message: Initializing view')
        self.initialize_view()
        self.initialize_volume_texture()
        height, width = self.view_plane_shape

        dpg.set_value('img_index_slider_current_value', 1)
        dpg.set_value('img_index_slider', 1)
        
        # Load options
        self.current_volume_info.set_control_options('orientation')
        self.current_volume_info.set_control_options('intensity')
        # self.current_volume_info.set_control_options('geometry')
        
        self.image_settings = {'colormap': self.current_volume_info.intensity.colormap, 
                               'view': G.DEFAULT_IMAGE_SETTINGS['view'], 
                               'min_value': self.current_group_info.intensity.min_intensity.limit_low, 
                               'max_value': self.current_group_info.intensity.max_intensity.limit_low}
        
        print('MainView Message: MainView Message: Adding Textures.')
        self.interpolate_view()
        self.update_alpha_mask()
        self.set_rgba_view()

        # self.add_static_texture(width = width, height = height, texture_array = self.texture,
        #                         texture_tag = G.TEXTURE_TAG, parent = G.APP.texture_registry)
        
        dpg.add_raw_texture(width = width, 
                            height = height, 
                            default_value = self.texture, 
                            format = dpg.mvFormat_Float_rgba, 
                            tag = G.TEXTURE_TAG, 
                            parent = G.APP.texture_registry)
        
        self.interpolate_mask()
        self.set_rgba_mask()

        # self.add_static_texture(width = width, height = height, texture_array = self.mask_texture, 
        #                         texture_tag = G.MASK_TEXTURE_TAG, parent = G.APP.texture_registry)
        
        dpg.add_raw_texture(width = width, 
                            height = height, 
                            default_value = self.mask_texture, 
                            format = dpg.mvFormat_Float_rgba, 
                            tag = G.MASK_TEXTURE_TAG, 
                            parent = G.APP.texture_registry)


        # Height and Width are the same for these textures. 
        image_size = G.DEFAULT_IMAGE_SETTINGS['image_size_multiplier']*height
        image_extent = int(np.round(image_size/2))
        
        print('MainView Message: Adding image series.')
        dpg.add_image_series(G.TEXTURE_TAG, 
                             tag = G.TEXTURE_IMAGE_TAG, 
                             bounds_min = [-image_extent, -image_extent], 
                             bounds_max = [image_extent, image_extent], 
                             parent = self.plot_y_axis)

        dpg.add_image_series(G.MASK_TEXTURE_TAG, 
                             tag = G.TEXTURE_MASK_TAG, 
                             bounds_min = [-image_extent, -image_extent], 
                             bounds_max = [image_extent, image_extent], 
                             parent = self.plot_y_axis,
                             show = dpg.get_value('enable_mask_checkbox'))
        
        print('MainView Message: Initial update')
        self.update_view()
        self.plot_x_line_value = 0
        self.plot_y_line_value = 0

        dpg.add_inf_line_series([self.plot_x_line_value], tag = self.plot_x_line, parent = self.plot_y_axis)
        dpg.add_inf_line_series([self.plot_y_line_value], horizontal=True, tag = self.plot_y_line, parent = self.plot_y_axis)

        # dpg.add_vline_series([self.plot_x_line_value], tag = self.plot_x_line, parent = self.plot_y_axis)
        # dpg.add_hline_series([self.plot_y_line_value], tag = self.plot_y_line, parent = self.plot_y_axis)

        dpg.bind_item_theme(self.plot_x_line, G.LINE_THEME)
        dpg.bind_item_theme(self.plot_y_line, G.LINE_THEME)

        dpg.fit_axis_data(self.plot_x_axis)
        dpg.fit_axis_data(self.plot_y_axis)
        
        dpg.set_axis_limits(self.plot_x_axis, -image_extent, image_extent)
        dpg.set_axis_limits(self.plot_y_axis, -image_extent, image_extent)

        dpg.configure_item(self.plot_x_axis, **{'lock_min': True, 'lock_max': True})
        dpg.configure_item(self.plot_y_axis, **{'lock_min': True, 'lock_max': True})

        # Configure Options
        # G.OPTIONS_DICT['img_index_slider']['max_value'] = G.N_VOLUMES
        G.OPTIONS_DICT['norm_slider']['min_value'] = int(-1*height/2)
        G.OPTIONS_DICT['norm_slider']['max_value'] = int(height/2)
        
        # dpg.configure_item(G.OPTIONS_DICT['img_index_slider']['slider_tag'], max_value = G.OPTIONS_DICT['img_index_slider']['max_value'])
        # dpg.set_item_label(G.OPTIONS_DICT['img_index_slider']['slider_tag'], self.current_volume_name)
        dpg.set_item_label(G.VOLUME_TAB_TAG, f'Volume Tab: {self.current_volume_name}')
        dpg.configure_item(G.OPTIONS_DICT['norm_slider']['slider_tag'], min_value = self.current_group_info.orientation.norm.limit_low)
        dpg.configure_item(G.OPTIONS_DICT['norm_slider']['slider_tag'], max_value = self.current_group_info.orientation.norm.limit_high)

        self.update_view()
        print('MainView Message: Volume loaded!')
        
        
    def close_image(self):
        # Reset all options to their default on close. 
        for option_key in G.OPTIONS_DICT.keys():
            slider_tag = G.OPTIONS_DICT[option_key]['slider_tag']
            default_value = G.OPTIONS_DICT[option_key]['default_value']*1
            
            G.OPTIONS_DICT[option_key]['current_value'] = default_value*1
            dpg.set_value(f'{slider_tag}_current_value', default_value)
            dpg.set_value(slider_tag, default_value)
        
        dpg.set_value('colormap_combo', G.DEFAULT_IMAGE_SETTINGS['colormap_combo'])
        dpg.configure_item(self.colormap_scale, min_scale = G.DEFAULT_IMAGE_SETTINGS['min_value'])
        dpg.configure_item(self.colormap_scale, max_scale = G.DEFAULT_IMAGE_SETTINGS['max_value'])
        dpg.configure_item(self.colormap_scale, colormap = G.DEFAULT_IMAGE_SETTINGS['colormap_scale'])

        dpg.set_item_label(f'orientation_{G.GROUP_LAYER_CONTROL_BUTTON}', G.DEFAULT_GROUP_LAYER_CONTROL)
        dpg.set_item_label(f'intensity_{G.GROUP_LAYER_CONTROL_BUTTON}', G.DEFAULT_GROUP_LAYER_CONTROL)
        
        G.OPTIONS_DICT['img_index_slider']['max_value'] = 1*1
        G.OPTIONS_DICT['norm_slider']['min_value'] = 0*1
        G.OPTIONS_DICT['norm_slider']['max_value'] = 0*1
        
        dpg.configure_item(G.OPTIONS_DICT['img_index_slider']['slider_tag'], max_value = G.OPTIONS_DICT['img_index_slider']['max_value'])
        dpg.configure_item(G.OPTIONS_DICT['norm_slider']['slider_tag'], min_value = G.OPTIONS_DICT['norm_slider']['min_value'])
        dpg.configure_item(G.OPTIONS_DICT['norm_slider']['slider_tag'], max_value = G.OPTIONS_DICT['norm_slider']['max_value'])
        
        # Performed this way to ensure defaults are restored. 
        setattr(self, 'current_volume_index', 0)
        setattr(self, 'current_volume_name', '')
        setattr(self, 'current_group_name', '')
        setattr(self, 'current_volume_info', None)
        setattr(self, 'current_group_info', None)
        setattr(self, 'current_volume', None)
        
        # dpg.set_item_label(G.OPTIONS_DICT['img_index_slider']['slider_tag'], self.current_volume_name)
        dpg.set_item_label(G.VOLUME_TAB_TAG, 'Volume Tab')
        print('MainView Message: Closing image!')
        if dpg.does_item_exist(G.TEXTURE_TAG):
            dpg.delete_item(G.TEXTURE_TAG)
            
        if dpg.does_alias_exist(G.TEXTURE_TAG):
            dpg.remove_alias(G.TEXTURE_TAG)
            
        if dpg.does_item_exist(G.MASK_TEXTURE_TAG):
            dpg.delete_item(G.MASK_TEXTURE_TAG)
            
        if dpg.does_alias_exist(G.MASK_TEXTURE_TAG):
            dpg.remove_alias(G.MASK_TEXTURE_TAG)
        
        if dpg.does_item_exist(self.plot_x_line):
            dpg.delete_item(self.plot_x_line)

        if dpg.does_alias_exist(self.plot_x_line):
            dpg.remove_alias(self.plot_x_line)

        if dpg.does_item_exist(self.plot_y_line):
            dpg.delete_item(self.plot_y_line)

        if dpg.does_alias_exist(self.plot_y_line):
            dpg.remove_alias(self.plot_y_line)
            
        dpg.delete_item('Texture_Image')
        dpg.delete_item('Mask_Texture_Image')
    
    def update_interpolator(self):
        self.geometry_vector[:] = self.current_volume_info.orientation.geometry_vector.current_value.reshape(3)[:]
        print(f'MainView Message: Update Interpolator: {self.geometry_vector = }')
        print(f'MainView Message: Update Interpolator: {self.current_volume_info.orientation.geometry_vector.current_value = }')
        geometry_list = self.geometry_vector.get().tolist()
        update_interpolator = False

        print(f'{geometry_list = }')
        print(f'{self.current_volume_info.ctvolume.interpolation_steps = }')

        for geo_index in range(len(geometry_list)):
            if geometry_list[geo_index] != self.current_volume_info.ctvolume.interpolation_steps[geo_index]:
                update_interpolator = True
                self.current_volume_info.ctvolume.interpolation_steps[geo_index] = 1.0*geometry_list[geo_index]
        
        if update_interpolator:
            self.current_volume_info.ctvolume.set_volume_interpolator(self.current_volume_info.ctvolume.volume, 
                                                                      step_sizes = self.current_volume_info.pixel_steps,
                                                                      geometry_scale = geometry_list)
            
            self.current_volume_info.ctvolume.set_mask_interpolator(step_sizes = self.current_volume_info.pixel_steps,
                                                                    geometry_scale = geometry_list)
    
    def update_view(self):
        """
        Central function for updating view_plane and the textures. 

        Requires functions to be in a certain order. 

        TODO: Add data checks. Make unit tests. 
        """
        with dpg.mutex():
            self.update_current_volume()
            self.update_colormap()
            self.update_window()
            self.update_volume_orientation() # Transformation handled in each VolumeLayer object. 
            self.update_interpolator()
            self.update_current_view()
            self.update_geometry_display()

            self.update_volume_operation()
            self.update_comparison_operation()
            self.interpolate_view()
            self.update_alpha_mask()
            self.window_and_normalize_image()
            self.update_histogram_current_view()
            self.set_rgba_view()
            # self.delete_static_texture(texture_tag = G.TEXTURE_TAG)
            # self.add_static_texture(texture_tag = G.TEXTURE_TAG, 
            #                         texture_array = self.texture)
            self.interpolate_mask()
            self.set_rgba_mask()
            # self.delete_static_texture(texture_tag = G.MASK_TEXTURE_TAG)
            # self.add_static_texture(texture_tag = G.MASK_TEXTURE_TAG, 
            #                         texture_array = self.mask_texture)
            self.update_histogram()
            self.update_landmark_opacity()
            self.main_view_hovered()
            self.check_crosshair_position()

    def update_geometry_display(self):
        for update_display in ['quaternion', 'origin_vector', 'norm_vector']:
            dpg.set_value(f'OptionPanel_{update_display}_display', 
                          self.current_volume_info.print_orientation_value(f'{update_display}', 'print'))
        
        # quaternion_string = f"{self.current_volume_info.get_orientation_value('quaternion', 'current_value')}"
        # origin_vector_string = f"{self.current_volume_info.get_orientation_value('origin_vector', 'current_value')}"
        # norm_vector_string = f"{self.current_volume_info.get_orientation_value('norm_vector', 'current_value')}"


        # dpg.set_value('OptionPanel_quaternion_display', quaternion_string)
        # dpg.set_value('OptionPanel_origin_vector_display', origin_vector_string)
        # dpg.set_value('OptionPanel_norm_vector_display', norm_vector_string)

    def update_current_view(self):
        self.current_view_plane = self.current_volume_info.get_orientation_value('view_plane', 'current_value')
        self.current_norm_vector = self.current_volume_info.get_orientation_value('norm_vector', 'current_value')
        self.current_norm = self.current_volume_info.get_orientation_value('norm', 'difference_value')
        

    def update_current_volume(self):
        
        if self.current_volume_index != int(dpg.get_value('img_index_slider') - 1): #This means we've changed images
            print(f'MainView Message: {self.current_volume_index} CHANGED VOLUMES')
            
            # Hide all landmarks on previous volume
            for landmark_index in self.current_volume_info.landmark_points.keys():
                landmark_circle = self.current_volume_info.landmark_points[landmark_index]['viewport_circle']
                dpg.configure_item(landmark_circle, show = False)

            # Change to new index. 
            self.current_volume_index = int(dpg.get_value('img_index_slider') - 1)
            self.current_volume_name = G.APP.Volumes[self.current_volume_index].name
            self.current_volume_info = getattr(getattr(G.APP.VolumeLayerGroups, self.current_group_name), self.current_volume_name)

            # Show all landmarks on updated volume
            for landmark_index in self.current_volume_info.landmark_points.keys():
                landmark_circle = self.current_volume_info.landmark_points[landmark_index]['viewport_circle']
                dpg.configure_item(landmark_circle, show = True)

            # Update volume name next to slider. 
            # dpg.set_item_label(G.OPTIONS_DICT['img_index_slider']['slider_tag'], self.current_volume_name)
            dpg.set_item_label(G.VOLUME_TAB_TAG, f'Volume Tab: {self.current_volume_name}')
            
            # This switches the Layer/Group button to the correct one. 
            dpg.set_item_label(f'orientation_{G.GROUP_LAYER_CONTROL_BUTTON}', self.current_volume_info.orientation_control)
            dpg.set_item_label(f'intensity_{G.GROUP_LAYER_CONTROL_BUTTON}', self.current_volume_info.intensity_control)
            
            # Update options
            self.current_volume_info.set_control_options('orientation', update = True)
            self.current_volume_info.set_control_options('intensity', update = True)
            # self.current_volume_info.set_control_options('geometry', update = True)

            dpg.set_value('colormap_combo', self.current_volume_info.colormap_name)
            dpg.set_value('reverse_colormap_checkbox', self.current_volume_info.colormap_reversed)
    

    def update_histogram(self):
        self.current_volume_info.set_dpg_histogram_values()
        dpg.set_value('InfoBoxTab_histogram_volume_plot_line_series', 
                      self.current_volume_info.get_histogram(update_histogram = True, 
                                                             return_order = 'reversed',
                                                             asnumpy = True))
        
        self.current_volume_info.set_dpg_histogram_limits('InfoBoxTab_histogram_volume_plot_xaxis',
                                                          'InfoBoxTab_histogram_volume_plot_yaxis')
    

    def update_histogram_current_view(self, bins_min = None, bins_max = None, n_bins = None):
        bins_min = self.mode_function(np.nanmin)(self.current_view[self.alpha_mask]) if bins_min == None else bins_min
        bins_max = self.mode_function(np.nanmax)(self.current_view[self.alpha_mask]) if bins_max == None else bins_max
        n_bins = 1000 if n_bins == None else n_bins

        print(f'MainView Message: Update Histogram: {bins_min = }, {bins_max = }, {n_bins = }')

        # Check if min and max are integers. 
        if (bins_min % 1.0 <1e-6) and (bins_max % 1.0 < 1e-6):
            bins_min -= 0.50
            bins_max += 0.50
            n_bins = int(bins_max - bins_min)
        
        # bin_edges, bin_step = cp.linspace(bins_min, bins_max, n_bins, 
                                        #   retstep = True, endpoint = True, dtype = cp.float32)

        bin_counts, bin_edges = cp.histogram(self.current_view[self.alpha_mask],
                                             range = (bins_min, bins_max), 
                                             bins = self.current_volume_info.histogram.bin_edges())

        bin_centers = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1])/2.0

        bin_counts_range = bin_counts.max() - bin_counts.min()
        
        dpg.set_value('InfoBoxTab_histogram_current_view_plot_line_series', 
                      [1.0*bin_centers.get(), (1.0*bin_counts).get()])
        # dpg.set_axis_limits('InfoBoxTab_histogram_current_view_plot_xaxis',
        #                     self.current_volume_info.histogram.bins_min(),
        #                     self.current_volume_info.histogram.bins_max())
        # dpg.set_axis_limits('InfoBoxTab_histogram_current_view_plot_yaxis',
        #                     0.0,
        #                     50000.0)
                            #cp.asnumpy(bin_counts.min() - 0.05*bin_counts_range).item(), 
                            #cp.asnumpy(bin_counts.max() + 0.05*bin_counts_range).item())


    def update_window(self):
        min_intensity = dpg.get_value(G.OPTION_TAG_DICT['min_intensity'])
        max_intensity = dpg.get_value(G.OPTION_TAG_DICT['max_intensity'])
        window_size = np.min([dpg.get_value(f'{G.OPTIONS_DICT["min_intensity_slider"]["slider_tag"]}_increment_input_float'),
                              dpg.get_value(f'{G.OPTIONS_DICT["max_intensity_slider"]["slider_tag"]}_increment_input_float')])
        
        self.current_volume_info.update_window(min_intensity, max_intensity, window_size)
        self.current_volume_info.set_dpg_window()
            
        colormap_scale_type = dpg.get_value('colormap_scale_combo')

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

        dpg.configure_item(self.colormap_scale, min_scale = self.min_value)
        dpg.configure_item(self.colormap_scale, max_scale = self.max_value)
        
        
    def update_volume_orientation(self):
        self.current_volume_info.update_orientation()

    
    def reset_volume_orientation(self):
        self.current_volume_info.reset_orientation()
        self.update_view()


    def update_crosshair(self):

        diff_value_x = dpg.get_value(G.ORIGIN_X_SLIDER) - dpg.get_value(self.plot_x_line)
        diff_value_y = dpg.get_value(G.ORIGIN_Y_SLIDER) - dpg.get_value(self.plot_y_line)

        self.plot_x_line_value += diff_value_x
        self.plot_y_line_value += diff_value_y

        dpg.set_value(self.plot_x_line, [[self.plot_x_line_value], [], [], [], []])
        dpg.set_value(self.plot_y_line, [[self.plot_y_line_value], [], [], [], []])

        print(f'MainView Message: {dpg.get_value(self.plot_x_line) = }, {dpg.get_value(self.plot_y_line) = }')

    def update_colormap(self):
        self.current_volume_info.update_colormap()
        dpg.configure_item(self.colormap_scale, 
                           colormap = f'colormap_{self.current_volume_info.colormap_string}')


    def update_mask(self):
        
        dpg.configure_item('Mask_Texture_Image', show = dpg.get_value('enable_mask_checkbox'))
        self.interpolate_mask()
        self.set_rgba_mask()


    def update_landmarks(self):
        self.current_volume_info.landmarks.update_landmarks()
    
    def update_landmark_opacity(self):
        if len(self.current_volume_info.landmark_points.keys()) == 0:
            return

        for landmark_index in self.current_volume_info.landmark_points.keys():
            # landmark = self.current_volume_info.landmark_points[landmark_index]['Viewport Coords'] # list like [x, y, z]
            landmark = []
            landmark_circle = self.current_volume_info.landmark_points[landmark_index]['viewport_circle']
            
            # Calculate landmark projection on viewing plane. 
            for lmark in self.current_volume_info.landmark_points[landmark_index]['Viewport Coords']:
                # print(f'MainView Message: {lmark = }, {type(lmark) = }')
                landmark.append(lmark)

            # print(f'MainView Message: {landmark = }, {type(landmark) = }')
            if self.current_volume_info.orientation_control == 'Group':
                # landmark_projection = self.current_group_info.orientation.calculate_point_projection_on_viewplane(self.mode_return_array(self.mode_create_array([landmark[2], landmark[1], landmark[0]])))
                landmark_projection = self.current_group_info.orientation.calculate_point_projection_on_viewplane(self.mode_create_array([landmark[2], landmark[1], landmark[0]]))

            else:
                # landmark_projection = self.current_volume_info.orientation.calculate_point_projection_on_viewplane(self.mode_return_array(self.mode_create_array([landmark[2], landmark[1], landmark[0]])))
                landmark_projection = self.current_volume_info.orientation.calculate_point_projection_on_viewplane(self.mode_create_array([landmark[2], landmark[1], landmark[0]]))
            
            distance, parallel_proj, orthogonal_proj, point_proj_plane, viewport_x, viewport_y = landmark_projection
            landmark_fade = 2.0 / G.APP.image_tools.landmark_radius
            landmark_color = dpg.get_item_configuration(landmark_circle)['color']
            landmark_color[-1] = np.exp(-landmark_fade * distance / np.round(dpg.get_value('landmark_opacity_factor_input'), decimals = 1) + 0) # Set alpha
            new_color = [landmark_color[0] * 255.0, landmark_color[1] * 255.0, landmark_color[2] * 255.0, landmark_color[3] * 255.0]

            dpg.configure_item(landmark_circle, center = (viewport_x, viewport_y), color = new_color)


    def update_comparison_operation(self):
        with dpg.mutex():
            operation_enabled = dpg.get_value('ImageTools_operation_enabled')
            comparison_enabled = dpg.get_value('ImageTools_compare_two_images_enabled')

            comparison = dpg.get_value('ImageTools_compare_two_images_combobox')

            if dpg.get_value('ImageTools_compare_two_images_make_image_1_current'):
                dpg.set_value('ImageTools_compare_two_images_selector_1', self.current_volume_info.name)

            if dpg.get_value('ImageTools_compare_two_images_make_image_2_current'):
                dpg.set_value('ImageTools_compare_two_images_selector_2', self.current_volume_info.name)

            vol_1 = dpg.get_value('ImageTools_compare_two_images_selector_1')
            vol_2 = dpg.get_value('ImageTools_compare_two_images_selector_2')

            if not comparison_enabled or operation_enabled:
                return

            if (vol_1 not in self.current_group_info.volume_names) or (vol_2 not in self.current_group_info.volume_names):
                return
            
        self.operation_volume[:] = self.mode_number(np.nan)
        
        getattr(self.current_group_info, vol_1).interpolate_view(out_array = self.operation_volume[0],
                                                                 # out_mask = self.alpha_mask.reshape(self.view_plane_shape), 
                                                                 view_plane = self.current_view_plane)

        getattr(self.current_group_info, vol_2).interpolate_view(out_array = self.operation_volume[1],
                                                                 # out_mask = self.alpha_mask.reshape(self.view_plane_shape), 
                                                                 view_plane = self.current_view_plane)

        match comparison:
            case 'Multiply':
                self.operation_view[:] = self.mode_function(np.prod)(self.operation_volume[:2], axis = 0)[:]
            case 'Add':
                self.operation_view[:] = self.mode_function(np.sum)(self.operation_volume[:2], axis = 0)[:]
            case 'Difference':
                self.operation_volume[1] *= -1.0
                self.operation_view[:] = self.mode_function(np.sum)(self.operation_volume[:2], axis = 0)[:]


    def update_volume_operation(self):
        operation_enabled = dpg.get_value('ImageTools_operation_enabled')
        comparison_enabled = dpg.get_value('ImageTools_compare_two_images_enabled')
        if not operation_enabled or comparison_enabled:
            return
        
        start_value = dpg.get_value('ImageTools_perform_operation_start')
        end_value = dpg.get_value('ImageTools_perform_operation_end') + 1
        operation = dpg.get_value('ImageTools_perform_operation_combobox')
        
        step_range = range(start_value, end_value)
        n_steps = len(list(step_range))

        if n_steps < 2:
            return

        increment = dpg.get_value(f'{G.OPTIONS_DICT["norm_slider"]["slider_tag"]}_increment_input_float')
        weights = [1.0 for value in step_range]
    
        self.operation_volume[:] = self.mode_number(np.nan)

        for step_index, step in enumerate(step_range):
            self.operation_volume[step_index] = self.mode_number(np.nan)
            self.current_volume_info.interpolate_view(out_array = self.operation_volume[step_index], 
                                                      # out_mask = self.alpha_mask.reshape(self.view_plane_shape), 
                                                      view_plane = (step*increment*cp.array(np.abs(self.current_norm_vector)) + 
                                                                    self.current_view_plane))
            if dpg.get_value('ImageTools_operation_weighted_checkbox') == True:
                weights[step_index] = 1.0/(1.0 + self.mode_function(np.abs)(step))

        w_string = ''
        for w in weights:
            w_formatted = f'{np.round(w, decimals = 2):.2f}'
            w_string = f'{w_string}{w_formatted:<5}'
        
        dpg.set_value('ImageTools_operation_weighted_text', w_string)

        nan_array = self.mode_return_type(~self.mode_function(np.isnan)(self.operation_volume[0:n_steps]), 'float32')
        for index, weight in enumerate(weights):
            nan_array[index] *= weight
            
        match operation:
            case 'Mean':
                self.operation_view[:] = self.mode_function(np.sum)(self.operation_volume[0:n_steps]*nan_array, axis = 0)/self.mode_function(np.sum)(nan_array, axis = 0)
                # self.operation_view[:] = self.mode_function(np.nanmean)(self.operation_volume[0:n_steps + 1], axis = 0) #[self.alpha_mask.reshape(self.view_plane_shape)]

            case 'Max': 
                self.operation_view[:] = self.mode_function(np.nanmax)(self.operation_volume[0:n_steps + 1], axis = 0)[:]
            
            case 'Min': 
                self.operation_view[:] = self.mode_function(np.nanmin)(self.operation_volume[0:n_steps + 1], axis = 0)[:]
            
            case 'Standard Deviation': 
                w_mean = (self.mode_function(np.sum)(self.operation_volume[0:n_steps]*nan_array, axis = 0))/(self.mode_function(np.sum)(nan_array, axis = 0))
                w_mean_2 = (self.mode_function(np.sum)((self.operation_volume[0:n_steps]**2)*nan_array, axis = 0))/(self.mode_function(np.sum)(nan_array, axis = 0))
                self.operation_view[:] = self.mode_function(np.sqrt)(w_mean_2 - w_mean**2)
                # self.operation_view[:] = self.mode_function(np.nanstd)(self.operation_volume[0:n_steps + 1], axis = 0)[:]


    def main_view_key_down(self, sender, app_data, user_data):
        
        # TODO: Check that shift key is released
        if dpg.is_item_hovered(G.MAIN_PLOT_VIEW) and G.FILE_LOADED:
            if sender == 'main_view_key_down_handler_Shift':
                dpg.configure_item(self.plot_x_axis, **{'lock_min': False, 'lock_max': False})
                dpg.configure_item(self.plot_y_axis, **{'lock_min': False, 'lock_max': False})
                dpg.set_axis_limits_auto(self.plot_x_axis)
                dpg.set_axis_limits_auto(self.plot_y_axis)
            
            if not dpg.is_key_down(dpg.mvKey_LShift):
                x_axis_limits = dpg.get_axis_limits(self.plot_x_axis)
                y_axis_limits = dpg.get_axis_limits(self.plot_y_axis)
                dpg.set_axis_limits(self.plot_x_axis, *x_axis_limits)
                dpg.set_axis_limits(self.plot_y_axis, *y_axis_limits)        
                dpg.configure_item(self.plot_x_axis, **{'lock_min': True, 'lock_max': True})
                dpg.configure_item(self.plot_y_axis, **{'lock_min': True, 'lock_max': True})
                
            self.update_view()
            
            
    def main_view_scroll(self, sender, app_data, user_data):
        """ 
        main_view_key_press_handler_LControl
        main_view_key_press_handler_LShift
        main_view_key_press_handler_LAlt
        """
        
        if dpg.is_key_down(dpg.mvKey_LShift):
            return
        
        if dpg.is_item_hovered(G.MAIN_PLOT_VIEW) and G.FILE_LOADED:
            option_key = 'norm_slider'
            step = dpg.get_value(f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float')
            
            if dpg.is_key_down(dpg.mvKey_LControl):
                option_key = 'img_index_slider'
                step = 1.0
                
            current_value_tag = '{0}_current_value'.format(G.OPTIONS_DICT[option_key]['slider_tag'])
            current_value = dpg.get_value(option_key)
            new_value = current_value + app_data * step
            
            clipped_new_value = G.APP.options_panel.clamp_option_value(option_key, new_value)

            dpg.set_value(current_value_tag, clipped_new_value)
            dpg.set_value(G.OPTIONS_DICT[option_key]['slider_tag'], clipped_new_value)
            
            self.update_view()
            

    def main_view_keyboard_navigation(self, sender, app_data):
        if G.FILE_LOADED:
            match app_data:
                case dpg.mvKey_W:
                    option_key = 'origin_y_slider'
                    step = 1.0 * dpg.get_value(f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float')
                case dpg.mvKey_S:
                    option_key = 'origin_y_slider'
                    step = -1.0 * dpg.get_value(f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float')
                case dpg.mvKey_A:
                    option_key = 'origin_x_slider'
                    step = -1.0 * dpg.get_value(f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float')
                case dpg.mvKey_D:
                    option_key = 'origin_x_slider'
                    step = 1.0 * dpg.get_value(f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float')
                case dpg.mvKey_Up:
                    option_key = 'norm_slider'
                    step = 1.0 * dpg.get_value(f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float')
                case dpg.mvKey_Down:
                    option_key = 'norm_slider'
                    step = -1.0 * dpg.get_value(f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float')

                case dpg.mvKey_N:
                    option_key = 'pitch_slider'
                    step = -1.0 * dpg.get_value(f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float')
                case dpg.mvKey_M:
                    option_key = 'pitch_slider'
                    step = 1.0 * dpg.get_value(f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float')
                case dpg.mvKey_K:
                    option_key = 'yaw_slider'
                    step = -1.0 * dpg.get_value(f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float')
                case dpg.mvKey_L:
                    option_key = 'yaw_slider'
                    step = 1.0 * dpg.get_value(f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float')
                case dpg.mvKey_O:
                    option_key = 'roll_slider'
                    step = -1.0 * dpg.get_value(f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float')
                case dpg.mvKey_P:
                    option_key = 'roll_slider'
                    step = 1.0 * dpg.get_value(f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float')

                case dpg.mvKey_Left:
                    return
                case dpg.mvKey_Right:
                    return

                case dpg.mvKey_Minus:
                    option_key = 'min_intensity_slider'
                    step = -1.0 * dpg.get_value(f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float')
                case dpg.mvKey_Plus:
                    option_key = 'min_intensity_slider'
                    step = 1.0 * dpg.get_value(f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float')

                case dpg.mvKey_Open_Brace:
                    option_key = 'max_intensity_slider'
                    step = -1.0 * dpg.get_value(f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float')
                case dpg.mvKey_Close_Brace:
                    option_key = 'max_intensity_slider'
                    step = 1.0 * dpg.get_value(f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float')

                case dpg.mvKey_Q:
                    step = -1
                    option_key = 'img_index_slider'
                case dpg.mvKey_E:
                    step = 1
                    option_key = 'img_index_slider'

                case dpg.mvKey_Tab:
                    tag = f'orientation_{G.GROUP_LAYER_CONTROL_BUTTON}'
                    G.APP.options_panel.update_frame_of_reference(tag)
                    self.update_view()
                    return
                
                case _:
                    return

            current_value_tag = '{0}_current_value'.format(G.OPTIONS_DICT[option_key]['slider_tag'])
            current_value = dpg.get_value(option_key)
            new_value = current_value + step
            
            clipped_new_value = G.APP.options_panel.clamp_option_value(option_key, new_value)

            dpg.set_value(current_value_tag, clipped_new_value)
            dpg.set_value(G.OPTIONS_DICT[option_key]['slider_tag'], clipped_new_value)
            
            self.update_view()           


    def check_crosshair_position(self):
        
        if not G.FILE_LOADED:
            return

        crosshair_x = 0 + G.TEXTURE_CENTER # These are always zero
        crosshair_y = 0 + G.TEXTURE_CENTER # Always zero
        
        # Dimensions go (z, x, y)

        raveled_index = np.ravel_multi_index((crosshair_y, crosshair_x), self.view_plane_shape)
        view_plane_coords_interp = self.current_view_plane[:, raveled_index].round(decimals = 3)
        view_plane_coords = self.current_view_plane[:, raveled_index].round(decimals = 2)
        vol_coords = (view_plane_coords + self.current_volume_info.volume_center.reshape(3)).round(decimals = 2)
        
        if G.GPU_MODE:
            view_plane_coord_value = interpn((cp.arange(0, G.TEXTURE_DIM, 1), cp.arange(0, G.TEXTURE_DIM, 1)), self.current_view, [crosshair_y, crosshair_x], bounds_error = False)[0] + 0

            vol_coords = cp.asnumpy(vol_coords.squeeze()) + 0
            view_plane_coords = cp.asnumpy(view_plane_coords.squeeze()) + 0
            
        else:
            view_plane_coord_value = np.round(getattr(self.current_volume, "interpolator")(view_plane_coords_interp[0],
                                                                                           view_plane_coords_interp[1],
                                                                                           view_plane_coords_interp[2]), decimals = 2)
        view_plane_coord_value = np.round(view_plane_coord_value, decimals = 2)
        self.crosshair_position_information = [vol_coords[0], vol_coords[1], vol_coords[2], view_plane_coord_value, 
                                                0, 0,
                                                view_plane_coords[0], view_plane_coords[1], view_plane_coords[2],
                                                self.current_quaternion,
                                                self.current_view[crosshair_y - 5:crosshair_y + 5, crosshair_x - 5:crosshair_x + 5].round(decimals = 1)]
    
        line_1 = f'{vol_coords[2]:7}, {vol_coords[1]:7}, {vol_coords[0]:7}, {view_plane_coord_value:7}'

        dpg.set_value(self.crosshair_position_text, f'{line_1}')


    def main_view_hovered(self):
        """
        This checks if the mouse is hovering over the main view and 
        updates the information panel with positions and intensity. 
        
        It also enables usage of hotkeys while hovering in the main view. 

        Seems to be less efficient on the GPU than just using mouse_move_handler. 
        """

        if not G.FILE_LOADED:
            return
        
        abs_mouse_x, abs_mouse_y = dpg.get_plot_mouse_pos()
        abs_mouse_x += 0
        abs_mouse_y += 0
        rel_mouse_z = dpg.get_value('norm_slider') + 0

        if np.linalg.norm(np.array([abs_mouse_x, abs_mouse_y, rel_mouse_z]) - self.last_mouse_position) > 0.01:
            
            self.last_mouse_position = np.array([abs_mouse_x, abs_mouse_y, rel_mouse_z])
            
            # Dimensions go (z, x, y)
            rel_mouse_x = abs_mouse_x + G.TEXTURE_CENTER + 0
            rel_mouse_y = G.TEXTURE_CENTER - abs_mouse_y + 0

            # Need to invert this because the add_plot_axis already inverts the image and puts the origin in the lower left.             
            rounded_rel_mouse_x = int(np.round(rel_mouse_x, decimals = 0))
            rounded_rel_mouse_y = int(np.round(rel_mouse_y, decimals = 0))
            rounded_rel_mouse_y_inverse = int(G.TEXTURE_DIM - rounded_rel_mouse_y)

            # print(inverse_mouse_y, rel_mouse_x, raveled_index)
            
            if (rel_mouse_x >= 0) and (rel_mouse_y >= 0) and (rel_mouse_x < G.TEXTURE_DIM) and (rel_mouse_y < G.TEXTURE_DIM):
                raveled_index = np.ravel_multi_index((rounded_rel_mouse_y,
                                                      rounded_rel_mouse_x),
                                                      self.view_plane_shape)
                
                raveled_index_x_advanced = np.ravel_multi_index((rounded_rel_mouse_y,
                                                                rounded_rel_mouse_x + 1), 
                                                                self.view_plane_shape, mode = 'clip')
                raveled_index_y_advanced = np.ravel_multi_index((rounded_rel_mouse_y - 1,
                                                                rounded_rel_mouse_x),
                                                                self.view_plane_shape, mode = 'clip')
                
                # Calculate linear interpolation of the image locations using the current raveled index and next raveled index
                view_plane_coords_rounded = self.current_view_plane[:, raveled_index]
                view_plane_coords_x_advanced = self.current_view_plane[:, raveled_index_x_advanced]
                view_plane_coords_y_advanced = self.current_view_plane[:, raveled_index_y_advanced]

                x_slopes = view_plane_coords_x_advanced - view_plane_coords_rounded # Slopes of d0, d1, d2 when we move in the x direction on the plot
                y_slopes = view_plane_coords_y_advanced - view_plane_coords_rounded
                
                view_plane_coords_interp = view_plane_coords_rounded - x_slopes * (float(rounded_rel_mouse_x) - rel_mouse_x)
                view_plane_coords_interp += y_slopes * (float(rounded_rel_mouse_y) - rel_mouse_y)
                view_plane_coords_interp = view_plane_coords_interp.round(decimals = 2)

                view_plane_alpha = self.alpha_mask[rounded_rel_mouse_y, rounded_rel_mouse_x]
                vol_coords = (view_plane_coords_interp + self.current_volume_info.volume_center.reshape(3)).round(decimals = 2)

                for row_index, y_step in enumerate(self.landmark_steps):
                    for col_index, x_step in enumerate(self.landmark_steps):
                        self.landmark_view_coords[:, row_index, col_index] = x_slopes*x_step + y_slopes*y_step + view_plane_coords_interp

                if ~view_plane_alpha:
                    view_plane_coord_value = np.nan
                    landmark_view = None
                    
                else:
                    if G.GPU_MODE:
                        view_plane_coord_value = interpn((cp.arange(0, G.TEXTURE_DIM, 1), cp.arange(0, G.TEXTURE_DIM, 1)), self.current_view, [rel_mouse_y, rel_mouse_x], bounds_error = False)[0] + 0

                        vol_coords = cp.asnumpy(vol_coords) + 0
                        view_plane_coord_value = cp.asnumpy(view_plane_coord_value)
                        landmark_view = self.current_volume_info.interpolator(self.landmark_view_coords.reshape(3, 121).T).round(decimals = 2).reshape((11, 11))
                        # landmark_view = getattr(self.current_volume, "interpolator")(self.landmark_view_coords.reshape(3, 121).T).round(decimals = 2).reshape((11, 11))
                        landmark_view = cp.asnumpy(landmark_view) + 0

                        view_plane_coords_interp = cp.asnumpy(view_plane_coords_interp.squeeze()) + 0
                        
                    else:

                        view_plane_coord_value = self.current_volume_info.interpolator(view_plane_coords_interp[0],
                                                                                              view_plane_coords_interp[1],
                                                                                              view_plane_coords_interp[2]).round(decimals = 2)
                        landmark_view = self.current_volume_info.interpolator(self.landmark_view_coords).round(decimals = 2).reshape((11, 11))
                view_plane_coord_value = np.round(view_plane_coord_value, decimals = 2)
                    # print(f'MainView Message: {landmark_view.shape = }')
                line_1 = f'{np.round(abs_mouse_x, decimals = 2):7}, {np.round(abs_mouse_y, decimals = 2):7}, {np.round(rel_mouse_z, decimals = 2):7}, {view_plane_coord_value:7}'
                line_2 = f'{view_plane_coords_interp[2]:7}, {view_plane_coords_interp[1]:7}, {view_plane_coords_interp[0]:7}, {view_plane_coord_value:7}'
                line_3 = f'{vol_coords[2]:7}, {vol_coords[1]:7}, {vol_coords[0]:7}, {view_plane_coord_value:7}'
                # line_3 = f'{np.round(self.current_quaternion, decimals = 3)}'
                self.mouse_position_information = [vol_coords[2], vol_coords[1], vol_coords[0], view_plane_coord_value, 
                                                    abs_mouse_x, abs_mouse_y,
                                                    view_plane_coords_interp[2], view_plane_coords_interp[1], view_plane_coords_interp[0],
                                                    self.current_quaternion,
                                                    landmark_view] #landmark_view]

                dpg.set_value(self.mouse_position_text, f'{line_1}\n{line_2}\n{line_3}')

    def _cleanup_(self):
        pass