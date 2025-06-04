from .Globals import *

class MenuBar:
    def __init__(self, value_registry = None):
        if G.GPU_MODE:
            cp.cuda.Device(G.DEVICE).use()

        with dpg.menu_bar(tag = 'MenuBar'):
            with dpg.menu(label = 'File', tag = 'MenuBarFile'):
                dpg.add_menu_item(label = 'Open Volumes', 
                                  tag = 'MenuBarFile_open', 
                                  callback = self.open_files)
                dpg.add_menu_item(label = 'Add Volumes', 
                                  tag = 'MenuBarFile_add_volume', 
                                  callback = self.open_files)
                dpg.add_menu_item(label = 'Close All', 
                                  tag = 'MenuBarFile_close', 
                                  callback = self.close_all)
                dpg.add_menu_item(label = 'Exit', 
                                  tag = 'MenuBarFile_exit', 
                                  callback = self.exit_app)
                
            with dpg.menu(label = 'Layers', tag = 'MenuBarLayers'):
                dpg.add_menu_item(label = 'New Group',
                                  tag = 'MenuBarLayers_create_new_group', 
                                  callback = self.create_new_group)
            
            with dpg.menu(label = 'Analysis', tag = 'MenuBarAnalysis'):
                dpg.add_menu_item(label = 'Open Analysis Window', 
                                  tag = 'MenuBarAnalysis_open_analysis_window', 
                                  callback = self.open_analysis_window)
                dpg.add_menu_item(label = 'Open Node Editor', 
                                  tag = create_tag('MenuBar', 'MenuItem', 'NodeEditor'), 
                                  callback = self.open_node_editor)

            with dpg.menu(label = 'Settings', tag = 'MenuBarSettings'):
                dpg.add_menu_item(label='Configuration', 
                                  tag = 'MenuBarSettings_open_config_menu', 
                                  user_data = False,
                                  callback = self.open_configuration)
                with dpg.menu(label='Debug Options', 
                              tag = 'MenuBarSettings_open_debug_menu'):
                    dpg.add_menu_item(label = 'Show Debug', 
                                      tag = 'MenuBarDebugOptions_show_debug', 
                                      user_data = False, 
                                      callback = self.show_debug)
                    dpg.add_menu_item(label = 'Show Metrics', 
                                      tag = 'MenuBarDebugOptions_show_metrics',
                                      user_data = False, 
                                      callback = self.show_metrics)
                    dpg.add_menu_item(label = 'Show Item Registry', 
                                      tag = 'MenuBarDebugOptions_show_item_registry', 
                                      user_data = False, 
                                      callback = self.show_item_registry)
                    dpg.add_menu_item(label = 'Show Texture Registry', 
                                      tag = 'MenuBarDebugOptions_show_texture_registry', 
                                      user_data = False, 
                                      callback = self.show_texture_registry)
                    dpg.add_menu_item(label = 'Show About', 
                                      tag = 'MenuBarDebugOptions_show_about', 
                                      user_data = False, 
                                      callback = self.show_about)
                    dpg.add_menu_item(label = 'Show Documentation', 
                                      tag = 'MenuBarDebugOptions_show_documentation', 
                                      user_data = False, 
                                      callback = self.show_documentation)
                    dpg.add_menu_item(label = 'Show Value Registry',
                                      tag = 'MenuBarDebugOptions_show_value_registry',
                                      user_data = {'registry_hidden': False, 
                                                   'registry_tag': value_registry},
                                      callback = self.open_value_registry)


    def open_node_editor(self):
        G.APP.NodeEditor.open_node_editor()

    def show_about(self):
        dpg.show_about()

    def show_documentation(self):
        dpg.show_documentation()

    def show_metrics(self, sender, app_data, user_data):
        if G.SHOW_METRICS:
            return
        dpg.show_metrics()

    def show_debug(self, sender, app_data, user_data):
        if G.SHOW_DEBUG:
            return
        dpg.show_debug()

    def show_item_registry(self, sender, app_data, user_data):
        if G.SHOW_ITEM_REGISTRY:
            return
        dpg.show_item_registry()

    def show_texture_registry(self, sender, app_data, user_data):
        if user_data:
            return
        dpg.show_item(G.TEX_REG_TAG)

    def open_files(self, sender, app_data):
        print(f'MenuBar Message: {G.VIEW = }')
        G.APP.FileDialog.show()

    def close_all(self):
        print('MenuBar Message: Closing All Volumes')
        print(f'MenuBar Message: {G.FILE_LOADED = }')
        if G.FILE_LOADED:
            dpg.set_value('InfoBoxTab_layers_text', '')
            
            # TODO Finish closing logic for new Texture methods. 
            # G.APP.main_view.close_image() # This no longer needs to close any images. 
            G.APP.info_box.close_image()
            G.APP.options_panel.disable_options()
            G.APP.image_tools.disable_options()
            dpg.configure_item('save_landmarks_button', enabled = False)

            G.APP.VolumeLayerGroups.remove_all_groups()
            # G.APP.Volumes.clear()

            setattr(self, 'VOLUME_LOADED', False)
            for GLOBAL_KEY in G.GLOBAL_DEFAULTS.keys():
                setattr(G, GLOBAL_KEY, G.GLOBAL_DEFAULTS[GLOBAL_KEY])
        
            print(f'MenuBar Message: {G.FILE_LOADED}, False')
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            print(f'MenuBar Message: {G.FILE_LOADED = }')
    
    def exit_app(self):
        print('MenuBar Message: Exiting App')
        dpg.stop_dearpygui()

    def create_new_group(self):
        pass
    
    def open_analysis_window(self):
        if G.FILE_LOADED:
            pass
            
        else:
            pass
        

    def open_value_registry(self, sender, app_data, user_data):

        if user_data['registry_hidden']:
            dpg.show_item('MenuBar_ValueRegistryWindow')
            return
        
        else:
            user_data['registry_hidden'] = True

        value_label_tags = []

        with dpg.window(label='CTViewer Value Registry', 
                        tag = 'MenuBar_ValueRegistryWindow',
                        pos = [500, 100],
                        max_size=[950, 800],
                        horizontal_scrollbar = True,
                        autosize = True,
                        show = True):
            
            value_tag_label_length = 0

            for dpg_index in dpg.get_item_children(user_data['registry_tag'], slot=1):
                value_tag_alias = dpg.get_item_alias(dpg_index)
                value_tag_label_length = max(value_tag_label_length, len(f'{value_tag_alias}'))

                with dpg.group(horizontal = True):
                    dpg.add_text(value_tag_alias)
                    value_label_tags.append(dpg.last_item())
                    if isinstance(dpg.get_value(dpg_index), dict):
                        dpg.add_text('DICTIONARY_PLACEHOLDER')
                    else:
                        dpg.add_text(f'\t{dpg.get_value(dpg_index)}')
            for value_label_tag in value_label_tags:
                dpg.set_value(value_label_tag, f'{dpg.get_value(value_label_tag):<{value_tag_label_length}} :')

    def open_configuration(self, sender, app_data, user_data):

        if user_data:
            dpg.show_item('MenuBar_Configuration_Window')
            return
        
        else:
            dpg.set_item_user_data(sender, True)

        with dpg.window(label='CTViewer Configuration', 
                        tag = 'MenuBar_Configuration_Window',
                        pos = [500, 300],
                        autosize = True,
                        show = True):
            
            config_section_label_length = 0
            config_value_label_length = 0

            for section in G.CONFIG_DICT.keys():
                config_section_label_length = max(config_section_label_length, len(f'{section}'))
                with dpg.group():
                    dpg.add_text(f'{section}', tag = f'Configuration_{section}_group_label')
                    for config_key, config_value in G.CONFIG_DICT[section].items():
                        with dpg.group(horizontal=True):
                            dpg.add_text(default_value = f'{config_key}', tag = f'Configuration_{config_key}_value_label')
                            config_value_label_length = max(config_value_label_length, len(f'{config_key}'))
                            match config_value:
                                case int():
                                    dpg.add_input_int(tag = f'Configuration_{config_key}_value_input', 
                                                      default_value = config_value, 
                                                      user_data = [section, config_key],
                                                      min_value = 15, 
                                                      max_value = 10000, 
                                                      source = f'ValueRegister_Configuration_{config_key}_value',
                                                      callback=self.update_config)
                                case float():
                                    dpg.add_input_float(tag = f'Configuration_{config_key}_value_input', 
                                                        default_value = config_value, 
                                                        user_data = [section, config_key],
                                                        step = 1.0,
                                                        step_fast = 5.0,
                                                        min_value = 15.0, 
                                                        max_value = 10000.0, 
                                                        source = f'ValueRegister_Configuration_{config_key}_value',
                                                        callback=self.update_config)
                                case str():
                                    dpg.add_input_text(tag = f'Configuration_{config_key}_value_input', 
                                                       default_value = config_value, 
                                                       user_data = [section, config_key],
                                                       source = f'ValueRegister_Configuration_{config_key}_value',
                                                       callback=self.update_config)
                                    
                                case tuple():
                                    dpg.add_color_edit(tag = f'Configuration_{config_key}_value_input', 
                                                       default_value = config_value, 
                                                       user_data = [section, config_key],
                                                       source = f'ValueRegister_Configuration_{config_key}_value',
                                                       callback = self.update_config)

            # Set lengths of labels
            for section in G.CONFIG_DICT.keys():
                dpg.set_value(f'Configuration_{section}_group_label', f'{section:<{config_section_label_length}}:')
                for config_key in G.CONFIG_DICT[section].keys():
                    dpg.set_value(f'Configuration_{config_key}_value_label', f'{config_key:<{config_value_label_length}}:')

            dpg.add_button(label = 'Save Configuration', callback = self.save_configuration)


    def update_config(self, sender, app_data, user_data):
        if user_data[0] == 'default_colors':
            app_data = tuple([int(round(value*255.0)) for value in app_data])
        print(f'Config Message: Received {sender}. \n\tChanging {user_data} to {app_data}')
        G.CONFIG_DICT[user_data[0]][user_data[1]] = app_data


    def save_configuration(self, sender):
        G.save_config('current')

    def _cleanup_(self):
        pass