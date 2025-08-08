from .Globals import *
from . import VolumeLayer
from . import NewMainView
from . import InformationBox
from . import ImageTools

class OptionsPanel:
    """
    Options panel class. Holds the various functional, geometric, and color options in the leftmost panel. 
    Each option here needs to be added to the G.OPTIONS_DICT and associated lists. 

    This also defines the keyboard controls for navigating through the volumes. 
    """
    def __init__(self,
                 tab = '',
                 debug = False):
        
        # These are set in set_volume_and_draw_objects
        self.VolumeLayerGroups = None
        self.DrawWindow = None
        self.InformationBox = None
        self.orientation_tags = []
        self.intensity_tags = []
        self.debug = debug
        self.ImageTools = None
        self.tag_list = []
        self.handler_list = []
        self.reset_list = []
        self.hover_info = {'mouse_info_text_tag': '',
                           'window_type': 'view_plane',
                           'window_hovered': ''}
        self.drawing_moved_registry = dpg.add_item_handler_registry(tag = create_tag('OptionsPanel', 'ItemHandlerRegistry', 'DrawingMoved'))

        # dpg.mvKey: [no mods | mod shift | mod alt], key_alias. 
        self.mouse_and_keyboard_controls = {

            dpg.mvKey_A: [[['origin_x',  1.0, self.update_volume], 
                           ['roll',   1.0, self.update_volume],
                           ['pixel_spacing_x',   -1.0, self.update_volume]], 
                          'mvKey_A'],
            dpg.mvKey_D: [[['origin_x', -1.0, self.update_volume], 
                           ['roll',  -1.0, self.update_volume],
                           ['pixel_spacing_x',   1.0, self.update_volume]], 
                          'mvKey_D'],
            dpg.mvKey_W: [[['origin_y', -1.0, self.update_volume],
                           ['pitch',  1.0, self.update_volume],
                           ['pixel_spacing_y',  1.0, self.update_volume]], 
                          'mvKey_W'],
            dpg.mvKey_S: [[['origin_y',  1.0, self.update_volume], 
                           ['pitch',  -1.0, self.update_volume],
                           ['pixel_spacing_y',  -1.0, self.update_volume]], 
                          'mvKey_S'],
            dpg.mvKey_Q: [[['origin_z',  -1.0, self.update_volume], 
                           ['yaw',    -1.0, self.update_volume], 
                           ['slice_thickness',   -1.0, self.update_volume]], 
                          'mvKey_Q'],
            dpg.mvKey_E: [[['origin_z', 1.0, self.update_volume], 
                           ['yaw',    1.0, self.update_volume], 
                           ['slice_thickness',    1.0, self.update_volume]], 
                          'mvKey_E'],
            dpg.mvKey_Spacebar: [[['landmark', None, self.add_landmark], 
                                  [None, None, None], 
                                  [None, None, None]], 
                                 'mvKey_Spacebar'],
            dpg.mvKey_Z: [[['img_index', int(-1), self.update_image_index], 
                           ['orientation_group_layer_control_button', None, self.update_frame_of_reference],
                           [None, None, None]], 
                          'mvKey_Z'],
            dpg.mvKey_C: [[['img_index', int(1), self.update_image_index], 
                           ['orientation_group_layer_control_button', None, self.update_frame_of_reference],
                           [None, None, None]], 
                          'mvKey_C'],
            dpg.mvKey_Tab: [[['orientation_group_layer_control_button', None, self.update_frame_of_reference], 
                             [None, None, None],
                             [None, None, None]], 
                            'mvKey_Tab']
                            }
        
        dpg.add_mouse_wheel_handler(callback=self.mouse_wheel_navigation, 
                                    user_data = None, 
                                    tag = 'option_panel_mouse_wheel_handler', 
                                    parent = G.HANDLER_REG_TAG)
        self.handler_list.append(dpg.last_item())

        dpg.add_item_clicked_handler(button=dpg.mvMouseButton_Right, 
                                     show = False,
                                     callback=self.item_right_clicked, 
                                     tag = 'option_panel_mouse_click_handler',
                                     parent = G.ITEM_HANDLER_REG_TAG)
        self.handler_list.append(dpg.last_item())


        dpg.add_mouse_double_click_handler(button = dpg.mvMouseButton_Left,
                                            show = False,
                                            callback = lambda sender, app_data, user_data: self.drawing_double_clicked('OptionPanel_drawing_double_clicked',
                                                                                                                       self.DrawWindow.get_texture_drawlist_tags(),
                                                                                                                       [dpg.is_item_hovered(self.DrawWindow.get_texture_drawlist_tags()[0]),  
                                                                                                                        dpg.is_item_hovered(self.DrawWindow.get_texture_drawlist_tags()[1])]),
                                            tag = 'OptionPanel_drawing_double_clicked',
                                            parent = G.HANDLER_REG_TAG)
        
        self.handler_list.append(dpg.last_item())
        
        dpg.add_mouse_down_handler(button = dpg.mvMouseButton_Middle,
                                   show = False, 
                                   user_data = [0, 0],
                                   tag = 'option_panel_mouse_middle_handler',
                                   parent = G.HANDLER_REG_TAG,
                                   callback = lambda sender, app_data, user_data: self.image_mouse_move('option_panel_mouse_middle_handler',
                                                                                                        dpg.get_drawing_mouse_pos(),
                                                                                                        dpg.is_mouse_button_clicked(dpg.mvMouseButton_Middle)))
        self.handler_list.append(dpg.last_item())

        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()


    def create_option_slider_group(self, 
                                   slider_func,
                                   group_kwargs: dict,
                                   text_kwargs: dict,
                                   slider_kwargs: dict):
        
        with dpg.group(**group_kwargs):
            dpg.add_text(**text_kwargs)
            slider_func(**slider_kwargs)

        self.tag_list.append(dpg.last_item())
        self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])

        pass

    def create_options_panel(self, 
                             tag = 'OptionsPanelWindow', 
                             width = G.CONFIG_DICT['app_settings']['option_panel_width'], 
                             height = G.CONFIG_DICT['app_settings']['option_panel_height']):
        
        with dpg.child_window(tag = tag,
                              width = width, 
                              height = height): 
            
            option_key = 'img_index_slider'
            self.create_option_slider_group(dpg.add_slider_int,
                                            group_kwargs = {'horizontal': True, 'tag': G.OPTIONS_DICT[option_key]['group_tag']},
                                            text_kwargs = {'label': G.OPTIONS_DICT[option_key]['label'], 
                                                           'tag': G.OPTIONS_DICT[option_key]['label_tag']},
                                            slider_kwargs = {'label': '', 
                                                             'width': 150, 
                                                             'height': 25, 
                                                             'user_data': (0, None), 
                                                             'callback': self.update_image_index, 
                                                             'source': f"{G.OPTIONS_DICT[option_key]['slider_tag']}_current_value",
                                                             'default_value': G.OPTIONS_DICT[option_key]['default_value'], 
                                                             'min_value': G.OPTIONS_DICT[option_key]['min_value'],
                                                             'max_value': G.OPTIONS_DICT[option_key]['max_value'],
                                                             'tag': G.OPTIONS_DICT[option_key]['slider_tag']})
            # with dpg.group(horizontal = True, tag = G.OPTIONS_DICT[option_key]['group_tag']):
            #     dpg.add_text(label = G.OPTIONS_DICT[option_key]['label'], 
            #                  tag = G.OPTIONS_DICT[option_key]['label_tag'])
            #     dpg.add_slider_int(label = '', 
            #                        width = 150, 
            #                        height = 25, 
            #                        user_data = (0, None), # update_user_data, previous_volume_index
            #                        callback = self.update_image_index,
            #                        source = f"{G.OPTIONS_DICT[option_key]['slider_tag']}_current_value",
            #                        default_value = G.OPTIONS_DICT[option_key]['default_value'],
            #                        min_value = G.OPTIONS_DICT[option_key]['min_value'], 
            #                        max_value = G.OPTIONS_DICT[option_key]['max_value'],
            #                        tag = G.OPTIONS_DICT[option_key]['slider_tag'])
            #     self.tag_list.append(dpg.last_item())
            #     self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])
                
            
            # Orientation Group
            with dpg.group(horizontal = True, tag = 'orientation_group_options'):
                b_height = 0
                b_increment = 22.5
                with dpg.group(tag = 'orientation_group_options_slider_group'):
                    for option_key in G.ORIENTATION_OPTIONS: 
                        b_height += b_increment
                        with dpg.group(horizontal = True, 
                                       user_data = f'{G.OPTIONS_DICT[option_key]["group_tag"]}_increment_popup',
                                       tag = G.OPTIONS_DICT[option_key]['group_tag']):
                            with dpg.window(popup=True, 
                                            min_size = [15, 15], 
                                            show = False,
                                            tag = f'{G.OPTIONS_DICT[option_key]["group_tag"]}_increment_popup'):
                                dpg.add_input_float(label = 'Increment', 
                                                    user_data = G.OPTIONS_DICT[option_key]['slider_tag'],
                                                    step = 0.05, 
                                                    step_fast = 0.5, 
                                                    default_value = 1, 
                                                    source = f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_step_value',
                                                    tag = f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float',
                                                    min_value = 0.01, 
                                                    max_value = 10.0, 
                                                    min_clamped = True,
                                                    max_clamped = True, 
                                                    on_enter = True,
                                                    callback = self.set_increment_value)
                                self.tag_list.append(dpg.last_item())
                                self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])
                                
                            dpg.add_text(G.OPTIONS_DICT[option_key]['label'], 
                                         tag = G.OPTIONS_DICT[option_key]['label_tag'])
                            dpg.add_input_float(label = '',
                                                tag = G.OPTIONS_DICT[option_key]['slider_tag'],
                                                width = 100, 
                                                format = '%.2f',
                                                step = G.OPTIONS_DICT[option_key]['step_value'], 
                                                step_fast = G.OPTIONS_DICT[option_key]['step_fast_value'], 
                                                source = f"{G.OPTIONS_DICT[option_key]['slider_tag']}_current_value",
                                                default_value = G.OPTIONS_DICT[option_key]['default_value'],
                                                min_value = G.OPTIONS_DICT[option_key]['min_value'], 
                                                max_value = G.OPTIONS_DICT[option_key]['max_value'],
                                                min_clamped = True, 
                                                max_clamped = True,
                                                on_enter = True,
                                                callback = self.update_volume)
                            self.tag_list.append(dpg.last_item())
                            self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])
                            self.orientation_tags.append(dpg.last_item())
                        
                        dpg.bind_item_handler_registry(G.OPTIONS_DICT[option_key]['group_tag'], 
                                                       G.ITEM_HANDLER_REG_TAG)
                
                # b_height = 135
                tag = f'orientation_{G.GROUP_LAYER_CONTROL_BUTTON}'
                dpg.add_button(label = G.DEFAULT_GROUP_LAYER_CONTROL, 
                               tag = tag, 
                               height = b_height, 
                               user_data = False,
                               callback = self.update_frame_of_reference)
                self.tag_list.append(dpg.last_item())
                
                b_increment = 22
                with dpg.group():
                    tag = f'orientation_{G.GROUP_LAYER_RESET_BUTTON}_origin'
                    dpg.add_button(label = 'Reset \nOrigin', 
                                    tag = tag, 
                                    height = 21.5 * 4, 
                                    callback = self.reset_orientation_origin)
                    self.tag_list.append(dpg.last_item())

                    tag = f'orientation_{G.GROUP_LAYER_RESET_BUTTON}_angle'
                    dpg.add_button(label = 'Reset \nAngle', 
                                    tag = tag, 
                                    height = b_increment * 3, 
                                    callback = self.reset_orientation_angle)
                    self.tag_list.append(dpg.last_item())

                    tag = f'orientation_{G.GROUP_LAYER_RESET_BUTTON}_geometry'
                    dpg.add_button(label = 'Reset \nZoom', 
                                    tag = tag, 
                                    height = b_increment * 3, 
                                    callback = self.reset_orientation_zoom)
                    self.tag_list.append(dpg.last_item())
            
            with dpg.group(horizontal = True, tag = 'intensity_group_options'):
                with dpg.group(tag = 'intensity_group_options_slider_group'):
                    for option_key in ['min_intensity_slider', 'max_intensity_slider']:
                        with dpg.group(horizontal = True, 
                                       user_data = f'{G.OPTIONS_DICT[option_key]["group_tag"]}_increment_popup',
                                       tag = G.OPTIONS_DICT[option_key]['group_tag']):
                            with dpg.window(popup=True, 
                                            min_size = [15, 15], 
                                            show = False,
                                            tag = f'{G.OPTIONS_DICT[option_key]["group_tag"]}_increment_popup'):
                                dpg.add_input_float(label = 'Increment', 
                                                    user_data = G.OPTIONS_DICT[option_key]['slider_tag'],
                                                    step = 0.25, 
                                                    step_fast = 1, 
                                                    default_value = 1, 
                                                    source = f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_step_value',
                                                    tag = f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float',
                                                    min_value = 0.01, 
                                                    max_value = 10.0, 
                                                    min_clamped = True, 
                                                    max_clamped = True, 
                                                    on_enter = True,
                                                    callback = self.set_increment_value)
                                self.tag_list.append(dpg.last_item())
                                self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])
                                
                            dpg.add_text(G.OPTIONS_DICT[option_key]['label'], 
                                         tag = G.OPTIONS_DICT[option_key]['label_tag'])
                            dpg.add_input_float(label = '', 
                                                tag = G.OPTIONS_DICT[option_key]['slider_tag'],
                                                width = 150, 
                                                format = '%.2f', 
                                                step = G.OPTIONS_DICT[option_key]['step_value'], 
                                                step_fast = G.OPTIONS_DICT[option_key]['step_fast_value'], 
                                                source = f"{G.OPTIONS_DICT[option_key]['slider_tag']}_current_value",
                                                default_value = G.OPTIONS_DICT[option_key]['default_value'],
                                                min_value = G.OPTIONS_DICT[option_key]['min_value'], 
                                                max_value = G.OPTIONS_DICT[option_key]['max_value'],
                                                min_clamped = True, 
                                                max_clamped = True,
                                                on_enter = True,
                                                callback = self.update_volume)
                            self.tag_list.append(dpg.last_item())
                            self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])
                            self.intensity_tags.append(dpg.last_item())
                                
                        dpg.bind_item_handler_registry(G.OPTIONS_DICT[option_key]['group_tag'], G.ITEM_HANDLER_REG_TAG)

                tag = f'intensity_{G.GROUP_LAYER_CONTROL_BUTTON}'
                dpg.add_button(label = G.DEFAULT_GROUP_LAYER_CONTROL, 
                               tag = tag, 
                               height = 42, 
                               user_data = False,
                               callback = self.update_frame_of_reference)
                self.tag_list.append(dpg.last_item())
            
            with dpg.group(tag = 'colormap_and_colorscale'):
                with dpg.group(horizontal = True):
                    dpg.add_text('Colormap  :')
                    dpg.add_combo(items = list(G.COLORMAP_DICT.keys()), 
                                  user_data = G.COLORMAP_DICT,
                                  source = 'colormap_combo_current_value',
                                  width = 225, 
                                  callback = self.update_volume, 
                                  default_value = 'Fire',
                                  tag = 'colormap_combo')
                    self.tag_list.append(dpg.last_item())
                    self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])

                    dpg.add_checkbox(label = 'Reverse', 
                                     tag = 'reverse_colormap_checkbox', 
                                     callback = self.update_volume)
                    dpg.set_value('colormap_combo', 'Fire')
                    
                    self.tag_list.append(dpg.last_item())
                    self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])

                with dpg.group(horizontal = True):
                    dpg.add_text('Colorscale:')
                    dpg.add_combo(items = list(G.COLORMAP_SCALES), 
                                  width = 225,
                                  callback = self.update_volume, 
                                  default_value = 'Linear',
                                  tag = 'colormap_scale_combo')
                    self.tag_list.append(dpg.last_item())
                    self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])

                    dpg.add_checkbox(label = 'Rescale', 
                                     tag = 'rescale_colormap_checkbox', 
                                     callback = self.update_volume)
                    self.tag_list.append(dpg.last_item())
                    self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])
                
            with dpg.group(tag = 'mask_segmentation_options'):
                with dpg.group(horizontal = True, 
                               tag = 'mask_segmentation_group'):
                    dpg.add_text('Mask      :')
                    dpg.add_combo(items = list(G.MASK_DICT.keys()), 
                                  width = 225, 
                                  callback = self.update_volume, 
                                  default_value = 'Body', 
                                  tag = 'mask_combo')
                    self.tag_list.append(dpg.last_item())
                    self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])

                    dpg.set_value('mask_combo', 'Body')
                    dpg.add_checkbox(label = 'Enable', 
                                     tag = 'enable_mask_checkbox', 
                                     default_value=False, 
                                     callback = self.update_volume)
                    self.tag_list.append(dpg.last_item())
                    self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])

                with dpg.group(horizontal = True, 
                               tag = 'mask_exclude_low_group'):
                    dpg.add_text('Mask Low  :')
                    dpg.add_input_float(label = '', 
                                        callback = self.update_volume, 
                                        width = 150, 
                                        step = 1, 
                                        step_fast = 5,
                                        default_value = -3500, 
                                        tag = 'mask_exclude_low_float')
                    self.tag_list.append(dpg.last_item())
                    self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])

                    dpg.add_checkbox(label = 'Enable', 
                                     tag = 'enable_mask_low_exclude_checkbox', 
                                     default_value=False, 
                                     callback = self.update_volume)
                    self.tag_list.append(dpg.last_item())
                    self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])
                
                    with dpg.popup('mask_exclude_low_group', 
                                   min_size = [15, 15], 
                                   tag = f'mask_exclude_low_increment_popup'):
                        dpg.add_input_float(label = 'Increment', 
                                            step = 0.05, 
                                            step_fast = 0.5, 
                                            default_value = 1, 
                                            tag = f'mask_exclude_low_float_increment_input_float',
                                            min_value = 0.01, 
                                            max_value = 10.0, 
                                            min_clamped = True, 
                                            max_clamped = True, 
                                            on_enter = True,
                                            callback = self.set_increment_value)
                        self.tag_list.append(dpg.last_item())
                        self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])

                with dpg.group(horizontal = True, 
                               tag = 'mask_exclude_high_group'):
                    dpg.add_text('Mask High :')
                    dpg.add_input_float(label = '', 
                                        callback = self.update_volume, 
                                        width = 150, 
                                        step = 1, 
                                        step_fast = 5,
                                        default_value = 3500, 
                                        tag = 'mask_exclude_high_float')
                    self.tag_list.append(dpg.last_item())
                    self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])

                    dpg.add_checkbox(label = 'Enable', 
                                     tag = 'enable_mask_high_exclude_checkbox', 
                                     default_value=False, 
                                     callback = self.update_volume)
                    self.tag_list.append(dpg.last_item())
                    self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])

                    with dpg.popup('mask_exclude_high_group', 
                                   min_size = [15, 15], 
                                   tag = f'mask_exclude_high_increment_popup'):
                        dpg.add_input_float(label = 'Increment', 
                                            step = 0.05, 
                                            step_fast = 0.5, 
                                            default_value = 1, 
                                            tag = f'mask_exclude_high_float_increment_input_float',
                                            min_value = 0.01, 
                                            max_value = 10.0, 
                                            min_clamped = True, 
                                            max_clamped = True, 
                                            on_enter = True,
                                            callback = self.set_increment_value)
                        self.tag_list.append(dpg.last_item())
                        self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])
                    
                with dpg.group(horizontal = True, 
                               tag = 'mask_opacity_group'):
                    dpg.add_text('Opacity   :', 
                                 tag = 'mask_opacity_label')
                    dpg.add_input_float(label = '', 
                                        callback = self.update_volume, 
                                        width = 150, 
                                        step = 0.01, 
                                        step_fast = 0.1, 
                                        format = '%.2f', 
                                        default_value = 1.0, 
                                        min_value = 0.0, 
                                        max_value = 1.0, 
                                        min_clamped = True, 
                                        max_clamped = True, 
                                        tag = 'mask_opacity_slider')
                    self.tag_list.append(dpg.last_item())
                    self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])
                    
                    dpg.add_color_edit(label = 'Mask Color', 
                                       tag = 'mask_color_picker', 
                                       default_value=(20, 20, 230, 255), 
                                       no_inputs = True, 
                                       callback = self.update_volume)
                    self.tag_list.append(dpg.last_item())
                    self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])
            
            with dpg.group(tag = 'interpolation_options', 
                           horizontal=True):
                dpg.add_text('Interpolation:')
                dpg.add_combo(items = G.INTERPOLATION_OPTIONS, 
                              width = 225,
                              callback = self.update_volume, 
                              default_value = G.INTERPOLATION_OPTIONS[0], 
                              tag = 'interpolation_combo_box')
                self.tag_list.append(dpg.last_item())
                self.reset_list.append([dpg.last_item(), dpg.get_value(dpg.last_item())])
            
            with dpg.group(horizontal=True):
                dpg.add_text('Quaternion   ')
                dpg.add_text('(0, 0, 0, 0)', 
                             tag = 'OptionPanel_quaternion_display',
                             user_data = qtn.array([1.0, 0.0, 0.0, 0.0]))
            with dpg.group(horizontal=True):
                dpg.add_text('G Quaternion ')
                dpg.add_text('(0, 0, 0, 0)', 
                             tag = 'OptionPanel_global_quaternion_display',
                             user_data = qtn.array([1.0, 0.0, 0.0, 0.0]))
            with dpg.group(horizontal=True):
                dpg.add_text('Origin       ')
                dpg.add_text('(0, 0, 0)', 
                             tag = 'OptionPanel_origin_vector_display')
            with dpg.group(horizontal=True):
                dpg.add_text('Norm         ')
                dpg.add_text('(0, 0, 0)', 
                             tag = 'OptionPanel_norm_vector_display')

        self.disable_options()


    def set_info_box(self, 
                     InformationBox: InformationBox.InformationBox):
        self.InformationBox = InformationBox

    def set_image_tools(self, ImageTools: ImageTools.ImageTools):
        self.ImageTools = ImageTools

    def set_volume_and_draw_objects(self,
                                    VolumeLayerGroups: VolumeLayer.VolumeLayerGroups,
                                    DrawWindow: NewMainView.MainView):
        
        self.VolumeLayerGroups = VolumeLayerGroups
        self.VolumeLayerGroups.get_orientation_info = self.get_orientation_info
        self.VolumeLayerGroups.get_intensity_info = self.get_intensity_info
        self.VolumeLayerGroups.get_control_info = self.get_control_info
        self.VolumeLayerGroups.get_text_info = self.get_text_info

        self.DrawWindow = DrawWindow

    def image_mouse_move(self, sender, app_data, user_data):
        if self.VolumeLayerGroups.active:
            with dpg.mutex():
                mouse_pos = dpg.get_drawing_mouse_pos()

                if user_data: #dpg.is_mouse_button_clicked(dpg.mvMouseButton_Middle):
                    dpg.set_item_user_data(sender, mouse_pos)

                previous_pos = dpg.get_item_user_data(sender)

                delta_mouse_x = mouse_pos[0] - previous_pos[0]
                delta_mouse_y = mouse_pos[1] - previous_pos[1]
                
                change_position = False
                if delta_mouse_x != 0:
                    change_position = True

                if delta_mouse_y != 0:
                    change_position = True

                if change_position or user_data:
                    dpg.set_item_user_data(sender,
                                           mouse_pos)
                    
                    previous_mouse_image_coords = self.VolumeLayerGroups.get_drawing_pos_coords(*previous_pos)
                    current_mouse_image_coords = self.VolumeLayerGroups.get_drawing_pos_coords(*mouse_pos)
                    coord_delta = np.round(current_mouse_image_coords - previous_mouse_image_coords, decimals = 0)
                    
                    dpg.set_value("origin_x_slider_current_value", dpg.get_value("origin_x_slider_current_value") - coord_delta[1])
                    dpg.set_value("origin_y_slider_current_value", dpg.get_value("origin_y_slider_current_value") + coord_delta[0])
                    dpg.set_value("origin_z_slider_current_value", dpg.get_value("origin_z_slider_current_value") - coord_delta[2])

            self.update_volume('image_mouse_move', 'Update', None)


    def drawing_double_clicked(self, sender, app_data, user_data):
        if self.VolumeLayerGroups.active and sum(user_data):
            with dpg.mutex():
                window_type = 'view_plane' if user_data[0] else 'view_plane_ortho'
                print('OptionsPanel Message: drawing_double_clicked')
                print(f'\t{window_type = }')
                print(f'\t{user_data = }')
                mouse_pos = dpg.get_drawing_mouse_pos()
                if window_type == 'view_plane_ortho':
                    pixel_start = self.VolumeLayerGroups.get_current_volume().TextureOrtho.pixel_start
                    mouse_pos[0] -= pixel_start[0]
                    mouse_pos[1] -= pixel_start[1]

                coords = self.VolumeLayerGroups.get_drawing_pos_coords(*mouse_pos, window_type = window_type)

                dpg.set_value("origin_x_slider_current_value", coords[1])
                dpg.set_value("origin_y_slider_current_value", -1.0*coords[0])
                dpg.set_value("origin_z_slider_current_value", coords[2])
            
            self.update_volume(sender, 'Update', None)


    def item_right_clicked(self, sender, app_data):
        
        popup_id = dpg.get_item_user_data(app_data[1])
        with dpg.mutex():
            if dpg.get_item_pos(popup_id) != dpg.get_mouse_pos():
                dpg.set_item_pos(popup_id, dpg.get_mouse_pos())
            if dpg.is_item_shown(popup_id):
                dpg.hide_item(popup_id)
            if not dpg.is_item_shown(popup_id):
                dpg.show_item(popup_id)

  
    def set_increment_value(self, sender, app_data, user_data):
        
        dpg.set_value(f'{user_data}_step_value', app_data)
        dpg.set_value(f'{user_data}_step_fast_value', 5*app_data)
        dpg.configure_item(user_data, step = app_data)
        dpg.configure_item(user_data, step_fast = 5*app_data)


    def update_image_index(self, 
                           sender, 
                           app_data, 
                           user_data):
        
        # print(f'OptionsPanel Message: Setting image index {sender}: \n\tUpdated Index: (0, {app_data - 1})\n\tPrevious Index: {user_data}')
        # This needs to be done here to ensure the correct information is reflected 
        # when we send the orientation and intensity info over in self.update_volume()
        changed_volumes = self.VolumeLayerGroups.update_control(**self.get_control_info())

        self.update_volume('Update Image Index', 'Update', changed_volumes)


    def update_frame_of_reference(self, sender, app_data, user_data):
        # print(sender, app_data)
        # current_label = dpg.get_item_label(sender)
        frame_dict = {'Group': 'Layer', 'Layer': 'Group'}
        
        dpg.set_item_label(sender, frame_dict[dpg.get_item_label(sender)])
        dpg.set_item_user_data(sender, True)
        # This needs to be done here to ensure the correct information is reflected 
        # when we send the orientation and intensity info over in self.update_volume()
        
        self.VolumeLayerGroups.update_control(**self.get_control_info())
        self.update_volume('Update Frame of Reference', 'Update', None)


    def get_control_info(self):
        control_info = {'img_index': dpg.get_value('img_index_slider') - 1,
                        'orientation_control': dpg.get_item_label('orientation_group_layer_control_button'),
                        'intensity_control': dpg.get_item_label('intensity_group_layer_control_button'),
                        'update_orientation_group_control': dpg.get_item_user_data('orientation_group_layer_control_button'),
                        'update_intensity_group_control': dpg.get_item_user_data('intensity_group_layer_control_button')}
        
        return control_info
        

    def get_orientation_info(self):
        
        drawwindow_tags = self.DrawWindow.get_window_tags()
        drawlayer_tags = [self.DrawWindow.return_texture_drawlayer_tag(tag) for tag in drawwindow_tags]

        orientation_info = {'origin_x': dpg.get_value('origin_x_slider'),
                            'origin_y': dpg.get_value('origin_y_slider'), 
                            'origin_z': dpg.get_value('origin_z_slider'),
                            'norm': dpg.get_value('norm_slider'),
                            'pitch': dpg.get_value('pitch_slider'),
                            'yaw': dpg.get_value('yaw_slider'),
                            'roll': dpg.get_value('roll_slider'),
                            'quaternion': dpg.get_item_user_data('OptionPanel_quaternion_display'),
                            'pixel_spacing_x': dpg.get_value('pixel_spacing_x_input'),
                            'pixel_spacing_y': dpg.get_value('pixel_spacing_y_input'),
                            'slice_thickness': dpg.get_value('slice_thickness_input'),
                            'drawlayer_tags': drawlayer_tags}

        return orientation_info
    
    def get_intensity_info(self):
        intensity_info = {'colormap_name': dpg.get_value('colormap_combo'),
                          'colormap_reversed': dpg.get_value('reverse_colormap_checkbox'),
                          'colormap_log': dpg.get_value('colormap_scale_combo') == 'Log',
                          'min_intensity': dpg.get_value('min_intensity_slider'),
                          'max_intensity': dpg.get_value('max_intensity_slider'),
                          'window_size': 1.0,
                          'colormap_rescaled': dpg.get_value('rescale_colormap_checkbox'),
                          'colormap_scale_type': dpg.get_value('colormap_scale_combo'),
                          'colormap_scale_tag': self.DrawWindow.return_colormap_tag(self.DrawWindow.get_window_tags()[0])}
        
        return intensity_info
    

    def get_text_info(self):
        text_info = {'window_type': self.hover_info['window_type'],
                     'mouse_info_text_tag': self.DrawWindow.return_mouse_pos_texture_info_text_tag(self.DrawWindow.get_window_tags()[0]),
                     'crosshair_info_text_tag': self.DrawWindow.return_crosshair_pos_texture_info_text_tag(self.DrawWindow.get_window_tags()[0])}
        return text_info
    

    def add_landmark(self, sender, app_data, user_data):
        #TODO Add mouse landmarking via double click. 
        # Currently only uses spacebar. 
        self.VolumeLayerGroups.add_landmark(app_data)
        self.InformationBox.add_landmark(*self.VolumeLayerGroups.get_last_landmark())


    def update_hover_info(self, sender, app_data, user_data: dict):
        if self.VolumeLayerGroups.active:
            for window_key in user_data['Windows']:
                if dpg.is_item_hovered(window_key):
                    self.hover_info['mouse_info_text_tag'] = user_data['mouse_info_text_tag']
                    self.hover_info['window_type'] = user_data['Windows'][window_key]
                    self.hover_info['window_hovered'] = window_key
            
            self.VolumeLayerGroups.update_mouse_volume_coord_info('OptionsPanel.update_hover_info',
                                                                None,
                                                                self.hover_info)


    def update_volume(self, sender, app_data, user_data):

        changed_volume:bool = user_data

        update_type:str = app_data

        colorbar_config = dpg.get_item_configuration(self.DrawWindow.return_colormap_tag(self.DrawWindow.get_window_tags()[0]))

        control_info = self.get_control_info()

        orientation_info = self.get_orientation_info()
        
        intensity_info = self.get_intensity_info()

        text_info = self.get_text_info()

        operation_info = self.ImageTools.get_operation_info()

        # print(f'OptionsPanel Message: ')
        # for key, tag in G.OPTION_TAG_DICT.items():
        #     print(f'\t{f"{key} value":<30}: {dpg.get_value(tag)}')
        self.VolumeLayerGroups.update_current_volume(changed_volume,
                                                     control_info,
                                                     update_type,
                                                     orientation_info, 
                                                     intensity_info,
                                                     text_info,
                                                     operation_info)


    def reset_orientation(self, sender, app_data, user_data):
        print(f"OptionsPanel Message: Resetting orientation.")
        for orientation_option_tag in G.ORIENTATION_OPTIONS:
            dpg.set_value(f'{orientation_option_tag}_current_value', 
                          G.OPTIONS_DICT[orientation_option_tag]['default_value'])
            dpg.set_value(G.OPTIONS_DICT[orientation_option_tag]['slider_tag'], 
                          G.OPTIONS_DICT[orientation_option_tag]['default_value'])
            
        self.VolumeLayerGroups.get_current_volume().reset_orientation()
        if not user_data:
            self.update_volume('Reset Volume', 'Update', None)


    def reset_orientation_origin(self, sender, app_data, user_data):
        print(f"OptionsPanel Message: Resetting origin.")
        with dpg.mutex():
            for orientation_option_tag in ['origin_x_slider',
                                           'origin_y_slider',
                                           'origin_z_slider',
                                           'norm_slider']:
                dpg.set_value(f'{orientation_option_tag}_current_value', 
                              G.OPTIONS_DICT[orientation_option_tag]['default_value'])
                dpg.set_value(G.OPTIONS_DICT[orientation_option_tag]['slider_tag'], 
                              G.OPTIONS_DICT[orientation_option_tag]['default_value'])
                
                print(f"\t{orientation_option_tag}: {dpg.get_value(G.OPTIONS_DICT[orientation_option_tag]['slider_tag']):.2f}")
                
        self.VolumeLayerGroups.get_current_volume().reset_orientation_origin()
        self.update_volume('Reset Volume Origin', 'Update', None)


    def reset_orientation_angle(self, sender, app_data, user_data):
        print(f"OptionsPanel Message: Resetting angles.")
        with dpg.mutex():
            for orientation_option_tag in ['pitch_slider', 
                                           'yaw_slider', 
                                           'roll_slider']:
                dpg.set_value(f'{orientation_option_tag}_current_value', 
                              G.OPTIONS_DICT[orientation_option_tag]['default_value'])
                dpg.set_value(G.OPTIONS_DICT[orientation_option_tag]['slider_tag'], 
                              G.OPTIONS_DICT[orientation_option_tag]['default_value'])
            
            dpg.set_item_user_data('OptionPanel_quaternion_display', qtn.array([1.0, 0.0, 0.0, 0.0]))
        
        self.VolumeLayerGroups.get_current_volume().reset_orientation_angle()
        self.update_volume('Reset Volume Angle', 'Update', None)


    def reset_orientation_zoom(self, sender, app_data, user_data):
        print(f"OptionsPanel Message: Resetting zoom.")
        with dpg.mutex():
            for orientation_option_tag in ['pixel_spacing_x_input',
                                           'pixel_spacing_y_input',
                                           'slice_thickness_input']:
                dpg.set_value(f'{orientation_option_tag}_current_value', 
                              G.OPTIONS_DICT[orientation_option_tag]['default_value'])
                dpg.set_value(G.OPTIONS_DICT[orientation_option_tag]['slider_tag'], 
                              G.OPTIONS_DICT[orientation_option_tag]['default_value'])
        
        self.VolumeLayerGroups.get_current_volume().reset_orientation_zoom()
        self.update_volume('Reset Volume Zoom', 'Update', None)


    def update_option_values_from_volume(self):
        for control_category in ['orientation', 'intensity']:
            for control_list in getattr(self.VolumeLayerGroups.get_current_volume(), 
                                        f'{control_category}_control_list'):
                for control_option_name in control_list:
                    option_tag = G.OPTION_TAG_DICT[control_option_name]
                    if control_category == 'orientation':
                        option_value = self.VolumeLayerGroups.get_current_volume().get_orientation_value(control_option_name, modifier = 'current_value')
                    elif control_category == 'intensity':
                        option_value = self.VolumeLayerGroups.get_current_volume().get_orientation_value(control_option_name, modifier = 'current_value')
                    else:
                        print(f'OptionsPanel Message: \n\tupdate_option_values_from_volume: Control category {control_category} not recognized')
                        return
                    # Set value registry value 
                    dpg.set_value(f'{option_tag}_current_value', 
                                  option_value)
                    # Set slider option 
                    dpg.set_value(option_tag, 
                                  option_value)


    def update_option_values(self, sender:str):
        control_category = sender.split(f'_{G.GROUP_LAYER_CONTROL_BUTTON}')[0]
        control_list = getattr(self.VolumeLayerGroups.get_current_volume(), 
                               f'{control_category}_control_list')
        for control_option_name in control_list: 
            option_tag = G.OPTION_TAG_DICT[control_option_name]
            if dpg.get_item_label(sender) == 'Group': # Indicates we've changed from Layer -> Group, so retrieve info from group_info
                control_option = getattr(getattr(self.VolumeLayerGroups.get_current_group(), 
                                                 control_category), 
                                                 control_option_name)
            else:# Indicates we've changed from Group -> Layer, so retrieve info from volume_info
                control_option = getattr(getattr(self.VolumeLayerGroups.get_current_volume(), 
                                                 control_category), 
                                                 control_option_name)
            dpg.set_value(f'{option_tag}_current_value', 
                          control_option.current_value)
            dpg.set_value(option_tag, 
                          control_option.current_value)


    def mouse_and_keyboard_navigation(self, sender, app_data, user_data):
        if self.VolumeLayerGroups.active:

            if app_data not in self.mouse_and_keyboard_controls.keys():
                return
            
            with dpg.mutex():
                debug_string = 'OptionsPanel DEBUG: mouse_and_keyboard_navigation'
                try:
                    key_options, key_alias = self.mouse_and_keyboard_controls[app_data]

                    if dpg.is_key_down(dpg.mvKey_LShift):
                        if dpg.is_key_down(dpg.mvKey_LAlt):
                            return
                        tag, increment_sign, update_function = key_options[1]
                        if tag == None:
                            return
                        key_alias_string = f'\n\tShift Key Press   : {key_alias}'
                        
                    elif dpg.is_key_down(dpg.mvKey_LAlt):
                        if dpg.is_key_down(dpg.mvKey_LShift):
                            return
                        
                        tag, increment_sign, update_function = key_options[2]

                        if tag == None:
                            return
                        key_alias_string = f'\n\tAlt Key Press     : {key_alias}'
                        
                    else:
                        tag, increment_sign, update_function = key_options[0]
                        key_alias_string = f'\n\tStandard Key Press: {key_alias}'

                    
                    tag_string = f'\n\ttag               : {tag}'
                    increment_sign_string = f'\n\tincrement_sign    : {increment_sign}'
                    update_function_string = f'\n\tupdate_function   : {update_function}'
                    debug_string = f'{debug_string}{key_alias_string}{tag_string}{increment_sign_string}{update_function_string}'

                    if self.debug:
                        print(debug_string)

                    if tag == 'landmark': 
                        update_function('add_landmark', tuple([G.TEXTURE_CENTER, G.TEXTURE_CENTER]), None)
                        return

                    if tag == 'orientation_group_layer_control_button':
                        update_function(tag, None, None)
                        return
                    
                    if tag in ['pixel_spacing_x', 'pixel_spacing_y', 'slice_thickness']:
                        slider_tag = f'{tag}_input'

                    elif tag in ['origin_x', 'origin_y', 'origin_z', 'roll', 'pitch', 'yaw', 'img_index']:
                        slider_tag = f'{tag}_slider'

                    else:
                        print(f'OptionsPanel Message: Option {tag} not recognized.')
                        return

                    if dpg.does_alias_exist(f'{slider_tag}_increment_input_float'):
                        if dpg.is_key_down(dpg.mvKey_LControl):
                            increment = increment_sign * dpg.get_value(f'{slider_tag}_step_fast_value')
                        else:
                            increment = increment_sign * dpg.get_value(f'{slider_tag}_step_value')
                    else:
                        increment = increment_sign

                except Exception as e:
                    # print(f'OptionsPanel Message:\n{e}')
                    return
                
                current_value_tag = f'{slider_tag}_current_value'
                current_value = dpg.get_value(current_value_tag)
                new_value = current_value + increment

                if tag == 'img_index':
                    clipped_new_value = self.clamp_option_value(slider_tag, new_value)

                else:
                    clipped_new_value = self.clamp_option_value(tag, new_value)

                dpg.set_value(current_value_tag, clipped_new_value)

            update_function(f'mouse_and_keyboard_control', tag, None)

    def mouse_wheel_navigation(self, sender, app_data, user_data):
        with dpg.mutex():
            if dpg.is_item_hovered('OptionsPanelWindow'):
                for option_key in G.OPTIONS_DICT.keys():
                    slider_tag = G.OPTIONS_DICT[option_key]['slider_tag']
                    if dpg.is_item_hovered(G.OPTIONS_DICT[option_key]['group_tag']) and dpg.is_item_enabled(slider_tag):
                        current_value_tag = f'{slider_tag}_current_value'
                        current_value = dpg.get_value(option_key)

                        if option_key != 'img_index_slider':
                            step = 1.0*dpg.get_value(f'{slider_tag}_step_value')
                            if dpg.is_key_down(dpg.mvKey_LControl):
                                step = 1.0*dpg.get_value(f'{slider_tag}_step_fast_value')

                        else:
                            step = 1

                        new_value = current_value + app_data * step
                        if option_key == 'img_index_slider':
                            clipped_new_value = self.clamp_option_value(option_key, new_value)

                        elif option_key in ['pixel_spacing_x_input', 'pixel_spacing_y_input', 'slice_thickness_input']:
                            clipped_new_value = self.clamp_option_value(option_key.split('_input')[0], new_value)

                        else:
                            clipped_new_value = self.clamp_option_value(option_key.split('_slider')[0], new_value)
                        
                        dpg.set_value(current_value_tag, clipped_new_value)
                        dpg.set_value(slider_tag, clipped_new_value)
                        
                        self.update_volume('Mouse Wheel', 'Update', None)
                        return

    def clamp_option_value(self, option_key, new_value):
        print(f'OptionPanel Message: clamp_option_value: {option_key = }, {new_value = }')
        if option_key in self.VolumeLayerGroups.get_current_volume().orientation_control_list:
            min_value = getattr(self.VolumeLayerGroups.get_current_volume().Orientation, 
                                option_key).limit_low
            max_value = getattr(self.VolumeLayerGroups.get_current_volume().Orientation, 
                                option_key).limit_high
                    
        elif option_key in self.VolumeLayerGroups.get_current_volume().intensity_control_list:
            min_value = getattr(self.VolumeLayerGroups.get_current_volume().Intensity, 
                                option_key).limit_low
            max_value = getattr(self.VolumeLayerGroups.get_current_volume().Intensity, 
                                option_key).limit_high
        
        elif option_key == 'img_index_slider':
            min_value = 1
            max_value = self.VolumeLayerGroups.get_current_group().n_volumes

        if new_value <= min_value:
            clipped_new_value = min_value
                    
        elif new_value >= max_value:
            clipped_new_value = max_value

        else:
            clipped_new_value = new_value

        return clipped_new_value 
    

    def key_press_handler(self, sender, app_data):
        
        """
        app_data is an int corresponding to the key pressed. 
        
        """
        # print(sender, app_data)
        
        if app_data == dpg.mvKey_Right:
            value = 1
                    
        elif app_data == dpg.mvKey_Left:
            value = -1
            
        else:
            value = 0
        
        for option_key in G.OPTIONS_DICT.keys():
            if dpg.is_item_hovered(G.OPTIONS_DICT[option_key]['group_tag']) and dpg.is_item_enabled(G.OPTIONS_DICT[option_key]['slider_tag']):
                current_value_tag = f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_current_value'
                current_value = dpg.get_value(option_key)
                
                if option_key != 'img_index_slider':
                    step = 1*dpg.get_value(f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float')
                    step_fast = 5*dpg.get_value(f'{G.OPTIONS_DICT[option_key]["slider_tag"]}_increment_input_float')
                    
                else:
                    step = 1
                    step_fast = 5
                    
                if dpg.is_key_down(dpg.mvKey_LControl):
                    new_value = current_value + value * step_fast
                else:
                    new_value = current_value + value * step

                clipped_new_value = self.clamp_option_value(option_key, new_value)
                
                # G.OPTIONS_DICT[option_key]['current_value'] = clipped_new_value
                dpg.set_value(current_value_tag, clipped_new_value)
                dpg.set_value(G.OPTIONS_DICT[option_key]['slider_tag'], clipped_new_value)
                
                self.update_volume('Key Press', 'Update', None)
            
                
    def enable_options(self):
        for tag in self.tag_list:
            dpg.enable_item(tag)

        for handler in self.handler_list:
            dpg.show_item(handler)


    def disable_options(self):
        for tag in self.tag_list:
            print(f'OptionsPanel Message: {tag} disabled' )
            dpg.disable_item(tag)
        
        for handler in self.handler_list:
            dpg.hide_item(handler)


    def reset_all(self):
        print(f'OptionsPanel Message: ')
        for (tag, default_value) in self.reset_list:
            print(f'\tResetting {tag}')
            dpg.set_value(tag, default_value)


    def close_image(self):
        self.reset_all()
        self.disable_options()


    def _cleanup_(self):
        pass


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

        return {'operation': dpg.get_value('ImageTools_perform_operation_combobox'),
                'enabled': dpg.get_value('ImageTools_operation_enabled'),
                'start': dpg.get_value('ImageTools_perform_operation_start'),
                'stop': dpg.get_value('ImageTools_perform_operation_end'),
                'weighted': dpg.get_value('ImageTools_operation_weighted_checkbox')}


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


    def reset_all(self):
        print(f'OptionsPanel Message: ')
        for tag in self.reset_list:
            print(f'\tResetting {tag}')
            dpg.set_value(tag, dpg.get_item_configuration(tag)['default_value'])


    def close_image(self):
        self.reset_all()
        self.disable_options()


    def _cleanup_(self):
        pass