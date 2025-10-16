from .Globals import *
from . import VolumeLayer

class ImageTools:
    
    """
    ImageTools class. Holds various options like drawing or annotating the image. Requires MainView to be initialized to run. 
    """
    
    def __init__(self, 
                 VolumeLayerGroups:VolumeLayer.VolumeLayerGroups, 
                 tab = ''):

        self.operation_dict = {'Mean': 'mean',
                               'Standard Deviation': 'std',
                               'Min': 'min',
                               'Max': 'max'}

        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()
            
        self.tag_list = []
        self.VolumeLayerGroups:VolumeLayer.VolumeLayerGroups = VolumeLayerGroups
        self.options_panel = None
        self.landmark_radius = 1*float(G.CONFIG_DICT['landmark_settings']['circle_radius'])
        self.operation_info = {'operation': 'Mean',
                               'enabled': False,
                               'start': -5,
                               'stop': 5,
                               'weighted': False}

        with dpg.child_window(width = G.CONFIG_DICT['app_settings']['option_panel_width'], # G.OPTIONS_PANEL_WINDOW_DEFAULTS['WINDOW_WIDTH'], 
                              height = G.CONFIG_DICT['app_settings']['option_panel_height'],
                              autosize_y = True): 
            with dpg.group(tag = 'ImageTools_perform_operation_group'):
                with dpg.group(horizontal = True):
                    tag = 'ImageTools_perform_operation_combobox_label'
                    dpg.add_text(f'{"Operation:":<14}', tag = tag)

                    tag = 'ImageTools_perform_operation_combobox'
                    dpg.add_combo(label = '', 
                                  items = ['Mean', 'Max', 'Min', 'Standard Deviation'], 
                                  width = 150,
                                  default_value = 'Mean', 
                                  tag = tag, 
                                  callback = self.update_operation)
                    self.tag_list.append(tag)

                    tag = 'ImageTools_operation_enabled'
                    dpg.add_checkbox(label = 'Enable', 
                                     callback = self.update_operation, 
                                     tag = tag, 
                                     default_value = False)
                    self.tag_list.append(tag)

                with dpg.group(horizontal = True):
                    tag = 'ImageTools_perform_operation_start_label'
                    dpg.add_text(f'{"Start":<14}', tag = tag)

                    tag = 'ImageTools_perform_operation_start'
                    dpg.add_input_int(label = '', 
                                      callback = self.update_operation, 
                                      width = 100,
                                      step = 1, 
                                      step_fast = 5, 
                                      default_value = -5,
                                      min_value = -100, 
                                      max_value = 100,
                                      min_clamped = True, 
                                      max_clamped = True,
                                      on_enter = True,
                                      tag = tag)
                    self.tag_list.append(tag)

                with dpg.group(horizontal = True):
                    tag = 'ImageTools_perform_operation_end_label'
                    dpg.add_text(f'{"End":<14}', tag = tag)
                    # self.tag_list.append(tag)
                    tag = 'ImageTools_perform_operation_end'
                    dpg.add_input_int(label = '', 
                                      callback = self.update_operation, 
                                      width = 100,
                                      step = 1, 
                                      step_fast = 5, 
                                      default_value = 5, 
                                      min_value = -100, 
                                      max_value = 100, 
                                      min_clamped = True, 
                                      max_clamped = True, 
                                      on_enter = True,
                                      tag = tag)
                    self.tag_list.append(tag)
                with dpg.group(horizontal = True):
                    dpg.add_checkbox(label = 'Weighted', 
                                     callback = self.update_operation, 
                                     tag = 'ImageTools_operation_weighted_checkbox', 
                                     default_value = False)
                    self.tag_list.append(dpg.get_item_alias(dpg.last_item()))
                    dpg.add_text(default_value = '', 
                                 label = '', 
                                 tag = 'ImageTools_operation_weighted_text')
                    # self.tag_list.append(dpg.get_item_alias(dpg.last_item()))

            with dpg.group(tag = 'ImageTools_compare_two_images'):
                with dpg.group(horizontal = True):
                    tag = 'ImageTools_compare_two_images_combobox_label'
                    dpg.add_text(f'{"Comparison:":<14}', tag = tag)

                    tag = 'ImageTools_compare_two_images_combobox'
                    dpg.add_combo(label = '', items = ['Multiply', 'Add', 'Difference'], width = 150,
                                default_value = 'Add', tag = tag, callback = self.update_comparison)
                    self.tag_list.append(tag)

                    tag = 'ImageTools_compare_two_images_enabled'
                    dpg.add_checkbox(label = 'Enable', callback = self.update_comparison, tag = tag, default_value = False)
                    self.tag_list.append(tag)
                
                with dpg.group(horizontal=True):
                    tag = 'ImageTools_compare_two_images_image_1_label'
                    dpg.add_text(f'{"Image 1:":<14}', tag = tag)

                    tag = 'ImageTools_compare_two_images_selector_1'
                    dpg.add_combo(label = '', items = [''], width = 175,
                                  tag = tag, callback = self.update_comparison)
                    self.tag_list.append(tag)
                    
                    tag = 'ImageTools_compare_two_images_make_image_1_current'
                    dpg.add_checkbox(label = 'Current', callback = self.update_comparison, 
                                     tag = tag, default_value = False)
                    self.tag_list.append(tag)

                with dpg.group(horizontal=True):
                    tag = 'ImageTools_compare_two_images_image_2_label'
                    dpg.add_text(f'{"Image 2:":<14}', tag = tag)

                    tag = 'ImageTools_compare_two_images_selector_2'
                    dpg.add_combo(label = '', items = [''], width = 175,
                                  tag = tag, callback = self.update_comparison)
                    self.tag_list.append(tag)

                    tag = 'ImageTools_compare_two_images_make_image_2_current'
                    dpg.add_checkbox(label = 'Current', callback = self.update_comparison, 
                                     tag = tag, default_value = False)
                    self.tag_list.append(tag)

            with dpg.group(horizontal=True):
                dpg.add_text(f'{"Landmark Radius:":18}', tag = 'landmark_radius_float_label')
                dpg.add_input_float(label = '', 
                                    tag = 'landmark_radius_float_input',
                                  min_value = 0.5, max_value = 100.0, default_value=float(self.landmark_radius), 
                                  min_clamped=True, max_clamped=True, width = 100,
                                  format = '%.1f',
                                  step = 0.5, step_fast=2.5, 
                                  callback = self.update_landmark_radius)
                self.tag_list.append('landmark_radius_float_input')
                dpg.add_color_edit(label = 'Landmark Color', tag = 'landmark_color_picker', 
                                default_value=G.CONFIG_DICT['default_colors']['default_landmark_color'], 
                                no_inputs = True, 
                                callback = self.update_landmark_colors)
                self.tag_list.append('landmark_color_picker')
                
            with dpg.group(horizontal = True):
                dpg.add_text(f'{"Landmark Opacity:":<18}', tag = 'landmark_opacity_factor_label')
                dpg.add_input_float(label = '', 
                                    tag = 'landmark_opacity_factor_input', width = 100,
                                    step = 0.1, step_fast = 1.0,
                                    default_value = 1, min_value = 0.5, max_value = 50,
                                    min_clamped = True, max_clamped = True, 
                                    callback = self.update_landmark_colors)
                self.tag_list.append('landmark_opacity_factor_input')
            
            dpg.add_color_edit(label = 'Crosshair Color', tag = 'crosshair_color_picker', 
                               default_value=G.CONFIG_DICT['default_colors']['default_crosshair_color'], 
                               no_inputs = True,
                               callback = self.update_crosshair_color)
            self.tag_list.append('crosshair_color_picker')

            with dpg.group(horizontal=True):
                tag = 'ImageTools_save_image'
                dpg.add_button(label = 'Save Image', width = 100, callback = self.save_image, tag = tag)
                self.tag_list.append(tag)
                dpg.add_input_text(label = 'Prefix', tag = 'ImageTools_save_image_prefix', width = 100)
                self.tag_list.append(dpg.get_item_alias(dpg.last_item()))

        dpg.add_mouse_double_click_handler(callback=self.new_landmark_double_click, tag = 'image_tools_new_landmark_double_click_handler', parent = G.HANDLER_REG_TAG)
        dpg.add_key_press_handler(key=dpg.mvKey_Spacebar, callback=self.new_landmark_spacebar, tag='image_tools_new_landmark_spacebar_press_handler', parent=G.HANDLER_REG_TAG)
        
        self.disable_options()


    def set_options_panel(self, options_panel):
        self.options_panel = options_panel


    def save_image(self):
        file_name = G.APP.main_view.current_volume_name.replace('/', '-').replace('\\', '-')
        file_prefix = dpg.get_value('ImageTools_save_image_prefix')
        file_path = G.APP.main_view.current_volume_info.file.with_name(f'{file_prefix}{file_name}.png')
        print(f'ImageTools Message: Saving image at {file_path}.')
        colormap_dict = G.COLORMAP_DICT[G.APP.main_view.current_volume_info.intensity.colormap_name]
        cmap = getattr(colormap_dict['module'], colormap_dict['name'])
        good_values = np.where(G.APP.main_view.alpha_mask.reshape(G.APP.main_view.view_plane_shape).get())
        min_points = np.min(good_values, axis = 1)
        max_points = np.max(good_values, axis = 1)
        
        plt.close('all')

        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 6), dpi = 300)

        ax.imshow(G.APP.main_view.windowed_view.get()[min_points[0]:max_points[0], min_points[1]:max_points[1]], 
                  cmap = cmap, origin = 'upper', vmin = 0.0, vmax = 1.0)
        relative_file_loc = ''
        file_parts = G.APP.main_view.current_volume_info.file.parent.parts
        if len(file_parts) > 3:
            for part in file_parts[-3:]:
                relative_file_loc = f'{relative_file_loc}{os.sep}{part}'
        else:
            for part in file_parts:
                relative_file_loc = f'{relative_file_loc}{os.sep}{part}'

        ax.text(0.05, 0.95, f'{relative_file_loc}\n{G.APP.main_view.current_volume_name}', 
                horizontalalignment= 'left', verticalalignment = 'top', 
                transform = ax.transAxes,
                bbox = {'linewidth': 0, 'fc': 'white', 'alpha': 0.75})
        ax.set_axis_off()

        ax.set_facecolor('white')

        fig.tight_layout()

        fig.savefig(file_path, bbox_inches = 'tight', facecolor = 'white')


    def update_crosshair_color(self):
        # print(dpg.get_value('crosshair_color_theme'), dpg.get_value('crosshair_color_picker'))
        dpg.set_value('crosshair_color_theme', dpg.get_value('crosshair_color_picker'))
        

    def update_landmark_colors(self):
        # print(f'ImageTools Message: {dpg.get_value("landmark_color_picker") = }, {dpg.get_value("landmark_opacity_factor_input") = }')
        self.VolumeLayerGroups.update_landmark_colors()

    def update_landmark_radius(self, sender, app_data):
        self.landmark_radius = 1*app_data
        G.APP.info_box.update_landmark_radius(self.landmark_radius)


    def new_landmark_double_click(self):
        return
        if dpg.is_item_hovered(G.MAIN_PLOT_VIEW) and G.FILE_LOADED:
            G.APP.info_box.add_landmark(option='mouse')

    
    def new_landmark_spacebar(self):
        return
        if G.FILE_LOADED:
            G.APP.info_box.add_landmark(option='spacebar')


    def update_comparison(self):
        return
        # G.APP.main_view.update_view()


    def get_operation_info(self):
        
        self.operation_info['operation'] = dpg.get_value('ImageTools_perform_operation_combobox')
        self.operation_info['enabled'] = dpg.get_value('ImageTools_operation_enabled')
        self.operation_info['start'] = dpg.get_value('ImageTools_perform_operation_start')
        self.operation_info['stop'] = dpg.get_value('ImageTools_perform_operation_end')
        self.operation_info['weighted'] = dpg.get_value('ImageTools_operation_weighted_checkbox')

        return self.operation_info


    def update_operation(self):
        self.options_panel.update_volume(None, None, False)
        # self.VolumeLayerGroups.update_operation(self.get_operation_info())


    def update_selector_lists(self, volume_names: list[str]):
        dpg.configure_item('ImageTools_compare_two_images_selector_1', 
                           items=[f'{name}' for name in volume_names])
        
        dpg.configure_item('ImageTools_compare_two_images_selector_2', 
                           items=[f'{name}' for name in volume_names])


    def enable_options(self):
        
        for image_tool_tag in self.tag_list:
            dpg.enable_item(image_tool_tag)


    def disable_options(self):
        dpg.configure_item('ImageTools_compare_two_images_selector_1', 
                           items=[''])
        
        dpg.configure_item('ImageTools_compare_two_images_selector_2', 
                           items=[''])
        for image_tool_tag in self.tag_list:
            print(f'ImageTools Message: {image_tool_tag} disabled')
            dpg.disable_item(image_tool_tag)

    def close_image(self):
        self.disable_options()

    def _cleanup_(self):
        pass