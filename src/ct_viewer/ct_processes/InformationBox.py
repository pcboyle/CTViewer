from .Globals import *
from . import VolumeLayer

class InformationBox(object):
    def __init__(self, 
                 VolumeLayerGroups: VolumeLayer.VolumeLayerGroups):
        
        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()

        self.group_text = ''
        self.items = []
        self.aliases = []
        self.infobox_options = []
        self.landmark_volumes = []
        self.handler_list = []
        self.VolumeLayerGroups = VolumeLayerGroups
        self.OptionsPanel = None
        self.item_handler_registry = dpg.add_item_handler_registry(tag = create_tag('InformationBox', 'ItemHandlerRegistry', 'DoubleClick'))

        dpg.add_item_double_clicked_handler(callback=self.landmark_double_clicked, 
                                            user_data = None,
                                            tag = 'infomation_box_mouse_double_click_handler',
                                            parent = self.item_handler_registry)
        self.handler_list.append(dpg.last_item())

        dpg.add_item_clicked_handler(button=dpg.mvMouseButton_Right,
                                     callback = self.item_right_clicked,
                                     tag = 'infomation_box_mouse_right_click_handler',
                                     parent = self.item_handler_registry)
        self.handler_list.append(dpg.last_item())
        
        with dpg.child_window(tag = 'InformationBox_Window',
                              width = G.CONFIG_DICT['app_settings']['info_box_width'], # G.INFORMATION_BOX_WINDOW_DEFAULTS['WINDOW_WIDTH'], 
                              height = G.CONFIG_DICT['app_settings']['info_box_height']): #G.INFORMATION_BOX_WINDOW_DEFAULTS['WINDOW_HEIGHT']):
            with dpg.tab_bar(tag = 'InfoBox_TabBar'):
                dpg.add_tab(label = 'Landmarks', tag = 'landmark_tab')

                with dpg.tab(label = 'Group Tab', tag = 'InfoBoxTab_groups'):
                    dpg.add_text(self.group_text, 
                                 tag = 'group_tab_text')

                with dpg.tab(label = 'Layer Tab', tag = 'InfoBoxTab_layers'):
                    dpg.add_text('', tag = 'InfoBoxTab_layers_text')

                with dpg.tab(label = 'Histograms', tag = 'InfoBoxTab_volume_histograms'):
                    with dpg.group(tag = 'InfoBox_histogram_volume'):
                        with dpg.plot(label = '', 
                                      tag = 'InfoBoxTab_histogram_volume_plot', 
                                      width = -1, 
                                      height = 250):
                            dpg.add_plot_axis(dpg.mvXAxis, 
                                              tag = 'InfoBoxTab_histogram_volume_plot_xaxis')
                            dpg.add_plot_axis(dpg.mvYAxis, 
                                              tag = 'InfoBoxTab_histogram_volume_plot_yaxis')
                            dpg.add_line_series(np.zeros(5), 
                                                np.zeros(5), 
                                                parent = 'InfoBoxTab_histogram_volume_plot_yaxis', 
                                                show = False, 
                                                tag = 'InfoBoxTab_histogram_volume_plot_line_series')
                        with dpg.group(horizontal=True):
                            dpg.add_text('Bins Min:')
                            dpg.add_input_float(label = '', 
                                                callback = self.update_histogram_volume, 
                                                width = 100, 
                                                step = 1, 
                                                step_fast = 5,
                                                default_value = -3500, 
                                                tag = 'InfoBoxTab_histogram_volume_bins_min',
                                                on_enter = True)
                            self.infobox_options.append(dpg.get_item_alias(dpg.last_item()))
                            dpg.add_checkbox(label = 'Enable', 
                                             tag = 'InfoBoxTab_histogram_volume_bins_min_checkbox', 
                                             default_value=True, 
                                             callback = self.update_histogram_volume)
                            self.infobox_options.append(dpg.get_item_alias(dpg.last_item()))
                        with dpg.group(horizontal=True):
                            dpg.add_text('Bins Max:')
                            dpg.add_input_float(label = '', 
                                                callback = self.update_histogram_volume, 
                                                width = 100, 
                                                step = 1, 
                                                step_fast = 5,
                                                default_value = 3500, 
                                                tag = 'InfoBoxTab_histogram_volume_bins_max',
                                                on_enter = True)
                            self.infobox_options.append(dpg.get_item_alias(dpg.last_item()))
                            dpg.add_checkbox(label = 'Enable', 
                                             tag = 'InfoBoxTab_histogram_volume_bins_max_checkbox', 
                                             default_value=True, 
                                             callback = self.update_histogram_volume)
                            self.infobox_options.append(dpg.get_item_alias(dpg.last_item()))
                        with dpg.group(horizontal=True):
                            dpg.add_text('Bin Step:')
                            dpg.add_input_float(label = '', 
                                                callback = self.update_histogram_volume, 
                                                width = 100, 
                                                step = 1, 
                                                step_fast = 5,
                                                default_value = 1, 
                                                tag = 'InfoBoxTab_histogram_volume_bin_step',
                                                on_enter = True)
                            self.infobox_options.append(dpg.get_item_alias(dpg.last_item()))
                            dpg.add_checkbox(label = 'Enable', 
                                             tag = 'InfoBoxTab_histogram_volume_bin_step_checkbox', 
                                             default_value=True, 
                                             callback = self.update_histogram_volume)
                            self.infobox_options.append(dpg.get_item_alias(dpg.last_item()))

                    with dpg.group(tag = 'InfoBox_histogram_current_view'):
                        with dpg.plot(label = '', 
                                      tag = 'InfoBoxTab_histogram_texture_plot', 
                                      width = -1, 
                                      height = 250):
                            dpg.add_plot_axis(dpg.mvXAxis, 
                                              tag = 'InfoBoxTab_histogram_texture_plot_xaxis')
                            dpg.add_plot_axis(dpg.mvYAxis, 
                                              tag = 'InfoBoxTab_histogram_texture_plot_yaxis')
                            dpg.add_line_series(np.zeros(5), 
                                                np.zeros(5), 
                                                parent = 'InfoBoxTab_histogram_texture_plot_yaxis', 
                                                show = False, 
                                                tag = 'InfoBoxTab_histogram_texture_plot_line_series')
                        with dpg.group(horizontal=True):
                            dpg.add_text('Bins Min:')
                            dpg.add_input_float(label = '', 
                                                callback = self.update_histogram_current_view, 
                                                width = 100, 
                                                step = 1, 
                                                step_fast = 5,
                                                default_value = -3500, 
                                                tag = 'InfoBoxTab_histogram_texture_bins_min',
                                                on_enter = True)
                            self.infobox_options.append(dpg.get_item_alias(dpg.last_item()))
                            dpg.add_checkbox(label = 'Enable', 
                                             tag = 'InfoBoxTab_histogram_texture_bins_min_checkbox', 
                                             default_value=True, 
                                             callback = self.update_histogram_current_view)
                            self.infobox_options.append(dpg.get_item_alias(dpg.last_item()))
                        with dpg.group(horizontal=True):
                            dpg.add_text('Bins Max:')
                            dpg.add_input_float(label = '', 
                                                callback = self.update_histogram_current_view, 
                                                width = 100, 
                                                step = 1, 
                                                step_fast = 5,
                                                default_value = 3500, 
                                                tag = 'InfoBoxTab_histogram_texture_bins_max',
                                                on_enter = True)
                            self.infobox_options.append(dpg.get_item_alias(dpg.last_item()))
                            dpg.add_checkbox(label = 'Enable', 
                                             tag = 'InfoBoxTab_histogram_texture_bins_max_checkbox', 
                                             default_value=True, 
                                             callback = self.update_histogram_current_view)
                            self.infobox_options.append(dpg.get_item_alias(dpg.last_item()))
                        with dpg.group(horizontal=True):
                            dpg.add_text('N Bins  :')
                            dpg.add_input_int(label = '', 
                                              callback = self.update_histogram_current_view, 
                                              width = 100, 
                                              step = 1, 
                                              step_fast = 5,
                                              default_value = 100, 
                                              tag = 'InfoBoxTab_histogram_texture_n_bins',
                                              on_enter = True)
                            self.infobox_options.append(dpg.get_item_alias(dpg.last_item()))
                            dpg.add_checkbox(label = '', 
                                             tag = 'InfoBoxTab_histogram_texture_n_bins_checkbox', 
                                             default_value=True, 
                                             callback = self.update_histogram_current_view)
                            self.infobox_options.append(dpg.get_item_alias(dpg.last_item()))

                if G.DEBUG_MODE:
                    with dpg.tab(label = 'Debug Tab'):
                        with dpg.table(header_row = False, tag = 'DEBUG_TABLE', resizable=True):
                            dpg.add_table_column()
                            dpg.add_table_column()
                            for debug_tag in G.DEBUG_INFO_TAGS:
                                with dpg.table_row():
                                    dpg.add_text(debug_tag, tag = f'{debug_tag}_debug_label')
                                    if G.FILE_LOADED:
                                        dpg.add_text(default_value = f'{getattr(G.APP.main_view, debug_tag)}', 
                                                     tag = f'{debug_tag}_debug_info')
                                    else:
                                        dpg.add_text(default_value = 'Test', tag = f'{debug_tag}_debug_info')

    
    def get_histogram_info(self, 
                           histogram_type):
        
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


    def set_options_panel(self, 
                          OptionsPanel):
        self.OptionsPanel = OptionsPanel


    def update_histogram_volume(self):
        histogram_info = self.get_histogram_info('volume')
        self.VolumeLayerGroups.update_histogram('volume', 
                                                  histogram_info)

    def update_histogram_current_view(self):
        histogram_info = self.get_histogram_info('texture')
        self.VolumeLayerGroups.update_histogram('texture', 
                                                  histogram_info)


    def update_layer_tab(self):
        pass

        
    def initialize_landmark_tables(self, volume_names):
        with dpg.mutex():
            for volume_name in volume_names:
                if volume_name not in self.landmark_volumes:
                    self.landmark_volumes.append(volume_name)

                    print(f'InformationBox Message: Adding Landmark Table: {volume_name}_landmarks_table')

                    with dpg.tree_node(label = volume_name, 
                                    tag = f'{volume_name}_landmarks_node', 
                                    parent = 'landmark_tab'):
                        self.items.append(f'{volume_name}_landmarks_node')
                        with dpg.table(header_row = True, 
                                    tag = f'{volume_name}_landmarks_table',
                                    height = 250,
                                    clipper = True,
                                    scrollY = True):
                            self.items.append(f'{volume_name}_landmarks_table')
                            dpg.add_table_column(label = ' ', width = 10, width_fixed = True)
                            dpg.add_table_column(label = f'{"X":^7}') # X
                            dpg.add_table_column(label = f'{"Y":^7}') # Y
                            dpg.add_table_column(label = f'{"Z":^7}') # Z
                            dpg.add_table_column(label = f'{"I":^7}') # I

                        dpg.show_item(f'{volume_name}_landmarks_table')

    def add_landmark(self, 
                     volume_name, 
                     landmark_index,
                     landmark_coords,
                     landmark_geometry,
                     landmark_quaternion,
                     landmark_id,
                     landmark_patch_id,
                     landmark_patch,
                     texture_registry = 'main_texture_registry'):
        
        print('InformationBox Message: add_landmark')
        # print(f'\tvolume_name           : {volume_name}')
        # print(f'\tlandmark_index        : {landmark_index}')
        # print(f'\tlandmark_coords       : {landmark_coords}')
        # print(f'\tlandmark_quaternion   : {tlandmark_quaternion}')
        # print(f'\tlandmark_geometry     : {landmark_geometry}')
        # print(f'\tlandmark_id           : {landmark_id}')
        # print(f'\tlandmark_patch_id     : {landmark_patch_id}')
        # print(f'\tlandmark_patch        : {landmark_patch}')
        
        current_row = len(dpg.get_item_children(f'{volume_name}_landmarks_table', slot = 1)) + 1
        with dpg.mutex():
            with dpg.table_row(parent=f'{volume_name}_landmarks_table', 
                               tag = f'{volume_name}_landmark_{landmark_index}_row', 
                               user_data = [landmark_coords, 
                                            landmark_geometry, 
                                            landmark_quaternion]):
                self.items.append(f'{volume_name}_landmark_{landmark_index}_row')

                dpg.add_selectable(label = f'{current_row}', 
                                   span_columns=True, 
                                   user_data = f'{volume_name}_landmark_{landmark_index}_popup',
                                   tag = f'{volume_name}_landmark_{landmark_index}_row_index')
                self.items.append(f'{volume_name}_landmark_{landmark_index}_row_index')

                dpg.add_selectable(label = f'{landmark_coords[1]:>7.2f}',
                                   span_columns = True,
                                   user_data = f'{volume_name}_landmark_{landmark_index}_popup',
                                   tag = f'{volume_name}_landmark_{landmark_index}_row_x')
                self.items.append(f'{volume_name}_landmark_{landmark_index}_row_x')

                dpg.add_selectable(label = f'{-1.0 * landmark_coords[0] + 0.0:>7.2f}',
                                   span_columns = True,
                                   user_data = f'{volume_name}_landmark_{landmark_index}_popup',
                                   tag = f'{volume_name}_landmark_{landmark_index}_row_y')
                self.items.append(f'{volume_name}_landmark_{landmark_index}_row_y')

                dpg.add_selectable(label = f'{landmark_coords[2]:>7.2f}',
                                   span_columns = True,
                                   user_data = f'{volume_name}_landmark_{landmark_index}_popup',
                                   tag = f'{volume_name}_landmark_{landmark_index}_row_z')
                self.items.append(f'{volume_name}_landmark_{landmark_index}_row_z')

                dpg.add_selectable(label = f'{landmark_coords[3]:>7.2f}',
                                   span_columns = True,
                                   user_data = f'{volume_name}_landmark_{landmark_index}_popup',
                                   tag = f'{volume_name}_landmark_{landmark_index}_row_i')
                self.items.append(f'{volume_name}_landmark_{landmark_index}_row_i')
                
                with dpg.popup(dpg.last_item(), 
                               mousebutton = dpg.mvMouseButton_Right, 
                               min_size = [15, 15], 
                               tag = f'{volume_name}_landmark_{landmark_index}_popup'):
                    with dpg.drawlist(height = 115, 
                                    width = 115, 
                                    tag = f'{landmark_id}||PatchDrawList'):
                        dpg.add_static_texture(width = landmark_patch[0], 
                                               height = landmark_patch[0],
                                               default_value = landmark_patch[1],
                                               tag = landmark_patch_id,
                                               parent = texture_registry)
                        dpg.draw_image(landmark_patch_id,
                                    pmin = [0, 0], 
                                    pmax = [115, 115])
                        
                    dpg.add_button(label = 'Remove Landmark', 
                                user_data = [volume_name, landmark_id],
                                tag = f'remove++{volume_name}++landmark++{landmark_index}', 
                                callback = self.delete_landmark)
                    self.items.append(f'remove++{volume_name}++landmark++{landmark_index}')

            for row_child in dpg.get_item_children(f'{volume_name}_landmark_{landmark_index}_row', 1):
                dpg.bind_item_handler_registry(row_child, self.item_handler_registry)

    def delete_landmark(self, sender, app_data, user_data):
        volume_name, landmark_id = user_data
        self.VolumeLayerGroups.delete_landmark(volume_name, landmark_id)
        
        popup_id = dpg.get_item_parent(sender)
        volume_name = sender.split('++')[1]
        landmark_index = int(sender.split('++')[-1])

        dpg.configure_item(popup_id, show=False)

        dpg.delete_item(f'{volume_name}_landmark_{landmark_index}_row')

        dpg.delete_item(f'{volume_name}_landmark_{landmark_index}_circle')
        dpg.delete_item(popup_id)

        print(f'InformationBox Message: Deleted {volume_name}_landmark_{landmark_index}_circle')

    def item_right_clicked(self, sender, app_data, user_data):
        if self.VolumeLayerGroups.active:
            _, row_child = app_data
        
            popup_id = dpg.get_item_user_data(row_child)
            with dpg.mutex():
                if dpg.get_item_pos(popup_id) != dpg.get_mouse_pos(local=False):
                    dpg.set_item_pos(popup_id, dpg.get_mouse_pos(local=False))
                if dpg.is_item_shown(popup_id):
                    dpg.hide_item(popup_id)
                if not dpg.is_item_shown(popup_id):
                    dpg.show_item(popup_id)

    def landmark_double_clicked(self, sender, app_data, user_data):
        if self.VolumeLayerGroups.active:
            _, row_child = app_data
            location, geometry, quaternion = dpg.get_item_user_data(dpg.get_item_parent(row_child))

            origin_x = np.round(location[1], decimals = 2) + 0.0
            origin_y = np.round(-1.0*location[0], decimals = 2) + 0.0
            origin_z = np.round(location[2], decimals = 2) + 0.0

            dpg.set_value('origin_x_slider_current_value', origin_x)
            dpg.set_value('origin_y_slider_current_value', origin_y)
            dpg.set_value('origin_z_slider_current_value', origin_z)

            pixel_spacing_x = np.round(geometry[1], decimals = 2) + 0.0
            pixel_spacing_y = np.round(geometry[0], decimals = 2) + 0.0
            slice_thickness = np.round(geometry[2], decimals = 2) + 0.0

            dpg.set_value('pixel_spacing_x_input_current_value', pixel_spacing_x)
            dpg.set_value('pixel_spacing_y_input_current_value', pixel_spacing_y)
            dpg.set_value('slice_thickness_input_current_value', slice_thickness)

            quaternion = qtn.array(quaternion)
            yaw, pitch, roll = quaternion.to_axis_angle

            dpg.set_value('yaw_slider_current_value', yaw)
            dpg.set_value('pitch_slider_current_value', pitch)
            dpg.set_value('roll_slider_current_value', roll)

            self.OptionsPanel.update_volume('InformationBox.landmark_double_clicked', None, None)


    def reset_histogram_plot(self):
        dpg.hide_item('InfoBoxTab_histogram_volume_plot_line_series')
        dpg.set_value('InfoBoxTab_histogram_volume_plot_line_series', 
                      [np.zeros(5), np.zeros(5)])
        dpg.hide_item('InfoBoxTab_histogram_texture_plot_line_series')
        dpg.set_value('InfoBoxTab_histogram_texture_plot_line_series', 
                      [np.zeros(5), np.zeros(5)])


    def load_image(self, VolumeLayerGroups:VolumeLayer.VolumeLayerGroups):
        self.update_group_names(VolumeLayerGroups)
        self.enable_options()
        dpg.show_item('InfoBoxTab_histogram_volume_plot_line_series')
        dpg.show_item('InfoBoxTab_histogram_texture_plot_line_series')


    def close_image(self):
        for item in self.items:
            if dpg.does_item_exist(item):
                dpg.delete_item(item)
            
            if dpg.does_alias_exist(item):
                dpg.remove_alias(item)

        self.landmark_volumes.clear()
        self.reset_histogram_plot()
        self.disable_options()


    def update_group_names(self, VolumeLayerGroups: VolumeLayer.VolumeLayerGroups = None):
        self.group_text = ''
        for group in VolumeLayerGroups.group_names: #G.APP.VolumeLayerGroups.group_names:
            self.group_text = f'{self.group_text}\n{group}'
        dpg.set_value('group_tab_text', self.group_text)

    
    def enable_options(self):
        for option_tag in self.infobox_options:
            dpg.enable_item(option_tag)

        for handler in self.handler_list:
            dpg.show_item(handler)

    def disable_options(self):
        for option_tag in self.infobox_options:
            dpg.disable_item(option_tag)

        for handler in self.handler_list:
            dpg.hide_item(handler)

    def _cleanup_(self):
        pass